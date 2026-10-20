# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 against LALSimulation 6.2.0 (``SimInspiralChooseTDWaveform``,
``SimInspiralChooseTDModes`` and ``PrecessingNRSurDynamics``), its errors,
the data file and the batch function.

The waveforms must have LAL's length and epoch exactly, and agree to a
relative error ``max |h - h_LAL| / max |h_LAL|`` below 1e-10 and a mismatch
below 1e-12.
"""

import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

import waveformjax as wf
from conftest import NRSUR_DATA, compare, requires_nrsur

pytestmark = requires_nrsur

TOLERANCE = 1e-10
MISMATCH = 1e-12

BINARY = dict(
    mass_1=40.0,
    mass_2=20.0,
    spin_1=(0.4, -0.3, 0.2),
    spin_2=(0.1, 0.5, -0.3),
    luminosity_distance=100.0,
    inclination=0.7,
    phi_ref=0.1,
    f_ref=20.0,
)

# (name, binary, delta_t, f_min, mode_array)
CASES = [
    ("generic", BINARY, 1 / 4096, 20.0, None),
    ("f_min=0", BINARY, 1 / 4096, 0.0, None),
    ("f_ref=0", dict(BINARY, f_ref=0.0), 1 / 4096, 20.0, None),
    ("f_ref=f_min=0", dict(BINARY, f_ref=0.0), 1 / 4096, 0.0, None),
    ("m1<m2", dict(BINARY, mass_1=12.0, mass_2=48.0, inclination=2.1, phi_ref=-0.4), 1 / 2048, 20.0, None),
    ("q=6", dict(BINARY, mass_1=60.0, mass_2=10.0), 1 / 4096, 20.0, None),
    ("q=6.01", dict(BINARY, mass_1=60.1, mass_2=10.0, f_ref=25.0), 1 / 4096, 20.0, None),
    ("large spins", dict(BINARY, spin_1=(0.7, 0.0, 0.7), spin_2=(0.0, -0.99, 0.0)), 1 / 4096, 20.0, None),
    ("extremal aligned", dict(BINARY, spin_1=(0.0, 0.0, 1.0), spin_2=(0.0, 0.0, -1.0)), 1 / 4096, 20.0, None),
    ("aligned", dict(BINARY, spin_1=(0.0, 0.0, 0.5), spin_2=(0.0, 0.0, -0.3)), 1 / 4096, 20.0, None),
    ("non-spinning", dict(BINARY, spin_1=(0.0, 0.0, 0.0), spin_2=(0.0, 0.0, 0.0)), 1 / 4096, 20.0, None),
    ("equal masses", dict(BINARY, mass_1=30.0, mass_2=30.0), 1 / 4096, 20.0, None),
    ("reference near merger", dict(BINARY, f_ref=200.0), 1 / 4096, 20.0, None),
    ("heavy", dict(BINARY, mass_1=150.0, mass_2=50.0, f_ref=15.0), 1 / 2048, 10.0, None),
    ("delta_t=1/16384", BINARY, 1 / 16384, 20.0, None),
    ("delta_t=1/1024", BINARY, 1 / 1024, 20.0, None),
    ("mode array (2, +-2)", BINARY, 1 / 4096, 20.0, ((2, 2), (2, -2))),
    ("mode array (2, 1)", BINARY, 1 / 4096, 20.0, ((2, 1),)),
    ("mode array l=3", BINARY, 1 / 4096, 20.0, tuple((3, m) for m in range(-3, 4))),
]


def lal_dict(lal, lalsim, mode_array):
    params = lal.CreateDict()
    if mode_array is not None:
        modes = lalsim.SimInspiralCreateModeArray()
        for ell, m in mode_array:
            lalsim.SimInspiralModeArrayActivateMode(modes, ell, m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(params, modes)
    return params


def lal_polarizations(lal, lalsim, binary, delta_t, f_min, mode_array=None):
    """``SimInspiralChooseTDWaveform``."""
    s1 = binary.get("spin_1", (0.0, 0.0, 0.0))
    s2 = binary.get("spin_2", (0.0, 0.0, 0.0))
    return lalsim.SimInspiralChooseTDWaveform(
        binary["mass_1"] * lal.MSUN_SI,
        binary["mass_2"] * lal.MSUN_SI,
        *s1,
        *s2,
        binary["luminosity_distance"] * 1e6 * lal.PC_SI,
        binary["inclination"],
        binary["phi_ref"],
        0.0,
        0.0,
        0.0,
        delta_t,
        f_min,
        binary["f_ref"],
        lal_dict(lal, lalsim, mode_array),
        lalsim.NRSur7dq4,
    )


def lal_modes(lal, lalsim, binary, delta_t, f_min, mode_array=None):
    """``SimInspiralChooseTDModes`` as ``({(l, m): data}, epoch)``."""
    node = lalsim.SimInspiralChooseTDModes(
        binary["phi_ref"],
        delta_t,
        binary["mass_1"] * lal.MSUN_SI,
        binary["mass_2"] * lal.MSUN_SI,
        *binary["spin_1"],
        *binary["spin_2"],
        f_min,
        binary["f_ref"],
        binary["luminosity_distance"] * 1e6 * lal.PC_SI,
        None if mode_array is None else lal_dict(lal, lalsim, mode_array),
        4,
        lalsim.NRSur7dq4,
    )
    modes, epoch = {}, None
    while node is not None:
        modes[node.l, node.m] = np.array(node.mode.data.data)
        epoch = float(node.mode.epoch)
        node = node.next
    return modes, epoch


def check_polarizations(result, hp, hc):
    assert len(result.hplus) == hp.data.length
    assert result.epoch == float(hp.epoch)
    assert result.delta_t == hp.deltaT
    for ours, theirs in ((result.hplus, hp.data.data), (result.hcross, hc.data.data)):
        error, mismatch = compare(ours, theirs)
        assert error < TOLERANCE and mismatch < MISMATCH, (error, mismatch)


@pytest.mark.parametrize("name, binary, delta_t, f_min, mode_array", CASES, ids=[case[0] for case in CASES])
def test_td_polarizations(lal, name, binary, delta_t, f_min, mode_array):
    lal_module, lalsim = lal
    result = wf.td_polarizations("NRSur7dq4", delta_t, f_min, mode_array=mode_array, **binary)
    hp, hc = lal_polarizations(lal_module, lalsim, binary, delta_t, f_min, mode_array)
    check_polarizations(result, hp, hc)


def test_rk4_start_nodes(lal):
    """References between the first nodes: the RK4 start with the half
    nodes (node 0) and forwards with interpolated derivatives (nodes 1, 2)."""
    lal_module, lalsim = lal
    from waveformjax.models.nrsur7dq4 import _data, _driver

    binary = wf._binary.binary_from_keywords(BINARY)
    _, _, _, chi_a, chi_b, q, m_tot_sec = _driver._host_labels(binary)
    omegas = np.asarray(_driver._omegas(_data.default_data(), q, jnp.asarray(chi_a), jnp.asarray(chi_b)))
    for node, fraction in ((0, 0.3), (1, 0.4), (1, 0.6), (2, 0.5), (3, 0.5)):
        omega = omegas[node] + fraction * (omegas[node + 1] - omegas[node])
        case = dict(BINARY, f_ref=omega / (np.pi * m_tot_sec))
        result = wf.td_polarizations("NRSur7dq4", 1 / 4096, 0.0, **case)
        check_polarizations(result, *lal_polarizations(lal_module, lalsim, case, 1 / 4096, 0.0))


@pytest.mark.parametrize(
    "binary, delta_t, f_min, mode_array",
    [
        (BINARY, 1 / 2048, 20.0, None),
        (dict(BINARY, mass_1=12.0, mass_2=48.0, f_ref=0.0), 1 / 4096, 0.0, None),
        (BINARY, 1 / 4096, 20.0, ((2, 2), (2, -2), (4, 1))),
    ],
    ids=["generic", "m1<m2 f_min=0", "mode array"],
)
def test_td_modes(lal, binary, delta_t, f_min, mode_array):
    lal_module, lalsim = lal
    result = wf.td_modes("NRSur7dq4", delta_t, f_min, mode_array=mode_array, **binary)
    modes, epoch = lal_modes(lal_module, lalsim, binary, delta_t, f_min, mode_array)
    assert sorted(result.modes) == sorted(modes)
    assert result.epoch == epoch
    scale = max(np.max(np.abs(h)) for h in modes.values())
    for mode, reference in modes.items():
        ours = np.asarray(result[mode])
        assert ours.shape == reference.shape
        assert np.max(np.abs(ours - reference)) < TOLERANCE * scale, mode
    times, _ = result
    assert len(times) == len(modes[2, 2])


def test_modes_sum_to_polarizations():
    """The modes summed with -2Ylm(inclination, pi/2 - phi_ref) are the
    polarizations (h_+ - i h_x)."""
    polarizations = wf.td_polarizations("NRSur7dq4", 1 / 4096, 20.0, **BINARY)
    modes = wf.td_modes("NRSur7dq4", 1 / 4096, 20.0, **BINARY)
    total = sum(
        np.asarray(h) * complex(wf.spin_weighted_spherical_harmonic(0.7, np.pi / 2 - 0.1, ell, m))
        for (ell, m), h in modes.modes.items()
    )
    assert np.max(np.abs(total.real - polarizations.hplus)) < 1e-12 * np.max(np.abs(total))
    assert np.max(np.abs(-total.imag - polarizations.hcross)) < 1e-12 * np.max(np.abs(total))


def test_dynamics(lal):
    """The dynamics on the nodes against ``PrecessingNRSurDynamics``."""
    _, lalsim = lal
    from waveformjax.models.nrsur7dq4 import _data, _dynamics

    data = _data.default_data()
    for q, chi_a, chi_b, omega in (
        (2.0, (0.4, -0.3, 0.2), (0.1, 0.5, -0.3), 0.0),
        (3.5, (0.1, 0.2, 0.7), (-0.5, 0.1, 0.3), 0.05),
        (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.018),
    ):
        out = lalsim.PrecessingNRSurDynamics(q, *chi_a, *chi_b, omega, 1.0, 0.0, 0.0, 0.0, 0.0, None, lalsim.NRSur7dq4)
        reference = np.stack([np.asarray(series.data) for series in out[1:]], axis=1)
        states, valid = _dynamics.integrate(data, q, jnp.asarray(chi_a), jnp.asarray(chi_b), omega)
        assert bool(valid)
        np.testing.assert_allclose(np.asarray(data.t_ds), np.asarray(out[0].data), rtol=0, atol=0)
        assert np.max(np.abs(np.asarray(states) - reference)) < 1e-11


@pytest.mark.parametrize(
    "binary, f_min, mode_array, match",
    [
        (dict(BINARY, mass_1=61.0, mass_2=10.0), 20.0, None, "too much extrapolation"),
        (dict(BINARY, f_ref=15.0), 20.0, None, "fRef"),
        (BINARY, 5.0, None, "fMin should be 0 or"),
        (dict(BINARY, f_ref=5.0), 0.0, None, "smaller than the minimum"),
        (dict(BINARY, f_ref=800.0), 20.0, None, "too large"),
        (BINARY, 20.0, ((5, 5),), "not available"),
    ],
    ids=["q>6.01", "f_ref<f_min", "f_min below start", "f_ref below start", "omega_ref>0.2", "ell=5"],
)
def test_errors(lal, binary, f_min, mode_array, match):
    """LAL's failures raise InvalidParameterError (and LAL fails too)."""
    lal_module, lalsim = lal
    with pytest.raises(wf.InvalidParameterError, match=match):
        wf.td_polarizations("NRSur7dq4", 1 / 4096, f_min, mode_array=mode_array, **binary)
    with pytest.raises(wf.InvalidParameterError, match=match):
        wf.td_modes("NRSur7dq4", 1 / 4096, f_min, mode_array=mode_array, **binary)
    with pytest.raises(RuntimeError):
        lal_polarizations(lal_module, lalsim, binary, 1 / 4096, f_min, mode_array)


def test_invalid_sampling():
    with pytest.raises(wf.InvalidParameterError):
        wf.td_polarizations("NRSur7dq4", -1.0, 20.0, **BINARY)
    with pytest.raises(wf.UnsupportedError):
        wf.polarizations("NRSur7dq4", [20.0, 30.0], **BINARY)


def test_data():
    from waveformjax.models.nrsur7dq4 import _data, load

    data = load(NRSUR_DATA)
    assert load(NRSUR_DATA) is data
    assert data.n_ds == 230 and data.n_coorb == 2000
    assert data.dynamics.shape[:2] == (233, 9)
    assert data.ei_bounds[-1] == data.ei_basis.shape[0] == data.ei_time.shape[0]
    assert len(data.ei_bounds) == len(_data.PIECES) + 1
    assert data.t_ds.dtype == jnp.float64


def test_data_file_errors(tmp_path):
    from waveformjax.models.nrsur7dq4 import load

    with pytest.raises(wf.DataFileNotFoundError):
        load(tmp_path / "missing.h5")
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not an hdf5 file")
    with pytest.raises(wf.DataFileError):
        load(bad)
    h5py = pytest.importorskip("h5py")
    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w") as f:
        f["t_ds"] = np.linspace(0.0, 1.0, 20)
    with pytest.raises(wf.DataFileError, match="missing"):
        load(empty)


def test_data_file_not_found():
    """Without NRSUR7DQ4_DATA and LAL_DATA_PATH NRSur7dq4 raises
    DataFileNotFoundError (a FileNotFoundError)."""
    env = {k: v for k, v in os.environ.items() if k not in ("NRSUR7DQ4_DATA", "LAL_DATA_PATH")}
    code = (
        "import waveformjax as wf\n"
        "assert wf.nrsur7dq4_data_file() is None\n"
        "try:\n"
        "    wf.td_polarizations('NRSur7dq4', 1 / 4096, 20.0, mass_1=40.0, mass_2=20.0)\n"
        "except FileNotFoundError as error:\n"
        "    print(type(error).__name__)\n"
    )
    output = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert output.stdout.strip() == "DataFileNotFoundError"


def test_batch():
    """batch_td_polarizations agrees with td_polarizations, each binary with
    its own length and epoch, and reports failed binaries."""
    rng = np.random.default_rng(3)
    n = 7
    parameters = dict(
        mass_1=rng.uniform(40.0, 70.0, n),
        mass_2=rng.uniform(20.0, 35.0, n),
        spin_1=rng.uniform(-0.5, 0.5, (n, 3)),
        spin_2=rng.uniform(-0.5, 0.5, (n, 3)),
        luminosity_distance=rng.uniform(100.0, 500.0, n),
        inclination=rng.uniform(0.0, np.pi, n),
        phi_ref=rng.uniform(0.0, 2 * np.pi, n),
        f_ref=np.where(np.arange(n) % 2 == 0, 0.0, 25.0),
    )
    parameters["mass_1"][3] = 200.0  # q > 6.01
    parameters["mass_2"][3] = 20.0
    result = wf.batch_td_polarizations("NRSur7dq4", 1 / 2048, 20.0, parameters)
    assert not result.ok[3] and isinstance(result.errors[3], wf.InvalidParameterError) and result[3] is None
    assert result.ok.sum() == n - 1
    lengths = set()
    for i in np.flatnonzero(result.ok):
        binary = {k: v[i] for k, v in parameters.items()}
        single = wf.td_polarizations("NRSur7dq4", 1 / 2048, 20.0, **binary)
        assert result[i].epoch == single.epoch and len(result[i].hplus) == len(single.hplus)
        assert compare(result[i].hplus, single.hplus)[0] < 1e-12
        assert compare(result[i].hcross, single.hcross)[0] < 1e-12
        lengths.add(len(single.hplus))
    assert len(lengths) > 1
    with pytest.raises(wf.InvalidParameterError):
        wf.batch_td_polarizations("NRSur7dq4", 1 / 2048, 20.0, parameters, strict=True)


def test_polarizations_at_matches_td_polarizations():
    """The pure function at LAL's sample times gives td_polarizations."""
    from waveformjax.models.nrsur7dq4 import _driver

    options = wf.WaveformOptions()
    binary = wf._binary.binary_from_keywords(dict(BINARY, f_ref=0.0))
    grid = _driver.setup(binary, 1 / 4096, 20.0, options)
    times = grid.m_tot_sec * (grid.t0 + grid.dt * np.arange(grid.n))
    function = wf.td_polarizations_function("NRSur7dq4")
    hp, hc = function(grid.binary, times)
    reference = wf.td_polarizations("NRSur7dq4", 1 / 4096, 20.0, **dict(BINARY, f_ref=0.0))
    assert compare(hp, reference.hplus)[0] < 1e-12
    assert compare(hc, reference.hcross)[0] < 1e-12
    # f_ref = 0 starts at the beginning of the surrogate (LAL's f_min = 0)
    start = wf.td_polarizations("NRSur7dq4", 1 / 4096, 0.0, **dict(BINARY, f_ref=0.0))
    grid = _driver.setup(binary, 1 / 4096, 0.0, options)
    times = grid.m_tot_sec * (grid.t0 + grid.dt * np.arange(grid.n))
    hp, _ = function(binary, times)
    assert compare(hp, start.hplus)[0] < 1e-12
    # zero outside the surrogate
    hp, hc = function(binary, jnp.array([-10.0, 1.0]))
    assert np.all(np.asarray(hp) == 0) and np.all(np.asarray(hc) == 0)
