# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomD (``waveformx.models.phenomd``) against
``SimInspiralChooseFDWaveformSequence`` of LALSimulation 6.2.0 and against
waveformrs, in float64 and float32, through the public functions.

The model differs from LAL's in one place: the ringdown and damping
frequencies are closed-form fits in the final spin
(``scripts/fit_phenomd_qnm.py``) rather than LAL's spline through its
table. Those differ by a part in 10^6 or less (``test_qnm_fit``), which
leaves mismatches against LAL of ``1e-12`` or less and relative differences
of a few ``1e-6`` near the merger. With LAL's spline in place of the fits,
the model agrees with LAL to rounding (``test_matches_lal_with_its_spline``).
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformx as wf
from conftest import compare
from test_taylorf2 import keywords, lal_polarizations
from waveformx.models.phenomd import _qnm_fit, fits

MTSUN_SI = 4.925490947641267e-06

# mass_1, mass_2, chi_1, chi_2, distance, inclination, phi_ref, f_min, f_ref
# (f_ref = 0: the first frequency)
CASES = [
    (30.0, 30.0, 0.0, 0.0, 400.0, 0.4, 0.0, 20.0, 20.0),
    (36.0, 29.0, 0.3, -0.4, 500.0, 1.0, 1.2, 20.0, 30.0),
    (60.0, 15.0, 0.8, 0.5, 1000.0, 2.5, 4.0, 15.0, 20.0),
    (80.0, 10.0, -0.7, 0.2, 2000.0, 2.0, 0.3, 10.0, 25.0),
    (1.6, 1.3, 0.05, 0.02, 40.0, 0.2, 2.0, 30.0, 40.0),
    (8.0, 1.4, 0.9, 0.0, 200.0, 0.3, 5.5, 20.0, 50.0),
    (150.0, 100.0, 0.95, 0.95, 3000.0, 0.7, 3.0, 10.0, 15.0),
    (25.0, 20.0, -0.9, -0.9, 300.0, 1.5, 0.7, 20.0, 20.0),
    (20.0, 45.0, 0.3, -0.4, 300.0, 2.9, 0.7, 20.0, 0.0),
    (50.0, 2.0, 0.995, 0.3, 300.0, 1.1, 1.7, 20.0, 25.0),
    (90.0, 5.0, 1.0, 1.0, 300.0, 1.1, 1.7, 20.0, 0.0),
]


def grid(case, n=1500):
    """From ``f_min`` to beyond LAL's cutoff of the uniform grid (``Mf =
    0.2``)."""
    m1, m2, f_min = case[0], case[1], case[7]
    return np.geomspace(f_min, 0.25 / ((m1 + m2) * MTSUN_SI), n)


@pytest.mark.parametrize("case", CASES)
def test_matches_lalsimulation(lal, case):
    f = grid(case)
    hp, hc = wf.polarizations("IMRPhenomD", f, scaled_strain=False, **keywords(case))
    for ours, reference in zip((hp, hc), lal_polarizations(lal, case, f, "IMRPhenomD")):
        error, mismatch = compare(ours, reference)
        assert error < 2e-5, error
        assert mismatch < 1e-10, mismatch


def test_random_binaries(lal):
    """Random binaries (mass ratios to 20, spins to 0.99): the mismatch
    against LAL stays far below 1e-6."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(40):
        total = 10 ** rng.uniform(np.log10(3.0), np.log10(300.0))
        q = rng.uniform(1.0, 20.0)
        m1, m2 = total * q / (1 + q), total / (1 + q)
        case = (m1, m2, *rng.uniform(-0.99, 0.99, 2), 500.0, rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi),
                20.0, rng.choice([0.0, 30.0]))
        f = np.geomspace(20.0, 0.25 / (total * MTSUN_SI), 1000)
        hp, _ = wf.polarizations("IMRPhenomD", f, scaled_strain=False, **keywords(case))
        worst = max(worst, compare(hp, lal_polarizations(lal, case, f, "IMRPhenomD")[0])[1])
    assert worst < 1e-10, worst


def _lal_table():
    """LAL's QNM table (``scripts/fit_phenomd_qnm.py``), or skip."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "fit_phenomd_qnm.py"
    spec = importlib.util.spec_from_file_location("fit_phenomd_qnm", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    try:
        return script.read_table(script.find_header())
    except FileNotFoundError:
        pytest.skip("LALSuite's sources not found")


def test_qnm_fit():
    """The fits against LAL's table: within its noise (a part in 10^6, the
    table having six or seven digits) for ``a > -0.9``, within the steps
    of the table near ``a = -1``."""
    a, fring, fdamp = _lal_table()
    ring, damp = (np.asarray(x) for x in fits.qnm(jnp.asarray(a)))
    for ours, table in ((ring, fring), (damp, fdamp)):
        error = np.abs(ours / table - 1.0)
        assert np.all(error[a > -0.9] < 1e-5), error[a > -0.9].max()
        assert np.all(error < 5e-4)
    # float32: the series in the cube root holds to a few float32 ulps
    ring32, damp32 = (np.asarray(x) for x in fits.qnm(jnp.asarray(a[:-1], jnp.float32)))
    np.testing.assert_allclose(ring32, ring[:-1], rtol=2e-6)
    np.testing.assert_allclose(damp32, damp[:-1], rtol=2e-5)
    assert _qnm_fit.DEGREE == len(_qnm_fit.FRING) - 1


def test_matches_lal_with_its_spline(lal, monkeypatch):
    """With LAL's spline in place of the fits, the model is LAL's to
    rounding."""
    scipy = pytest.importorskip("scipy.interpolate")
    a, fring, fdamp = _lal_table()
    splines = [scipy.CubicSpline(a, y, bc_type="natural") for y in (fring, fdamp)]

    def spline_qnm(af):
        out = jax.ShapeDtypeStruct(jnp.shape(af), jnp.float64)
        return tuple(jax.pure_callback(lambda x, s=s: s(np.asarray(x)), out, af) for s in splines)

    monkeypatch.setattr(fits, "qnm", spline_qnm)
    function = jax.jit(wf.polarizations_function("IMRPhenomD", scaled_strain=False))
    for case in CASES[:6]:
        f = grid(case)
        binary = wf.Binary(*(jnp.asarray(v, jnp.float64) for v in wf.Binary(**keywords(case))))
        hp, hc = function(binary, jnp.asarray(f))
        for ours, reference in zip((hp, hc), lal_polarizations(lal, case, f, "IMRPhenomD")):
            error, mismatch = compare(np.asarray(ours), reference)
            assert error < 1e-10 and mismatch < 1e-14, (case, error, mismatch)


@pytest.mark.parametrize("case", CASES[:4])
def test_matches_waveformrs(case):
    waveformrs = pytest.importorskip("waveformrs")
    f = grid(case)
    hp, _ = wf.polarizations("IMRPhenomD", f, scaled_strain=False, **keywords(case))
    error, mismatch = compare(hp, np.asarray(waveformrs.polarizations("IMRPhenomD", f, **keywords(case))[0]))
    assert error < 2e-5 and mismatch < 1e-10, (error, mismatch)


@pytest.mark.parametrize("case", CASES)
def test_float32(case):
    """float32 against float64 at the float32-rounded inputs."""
    f = np.float32(grid(case, 1000)).astype(np.float64)
    binary = wf.Binary(**{k: np.float32(v) for k, v in keywords(case).items()})
    function = jax.jit(wf.polarizations_function("IMRPhenomD"))
    single = function(wf.Binary(*(jnp.asarray(v, jnp.float32) for v in binary)), jnp.asarray(f, jnp.float32))
    double = function(wf.Binary(*(jnp.asarray(v, jnp.float64) for v in binary)), jnp.asarray(f))
    assert single[0].dtype == jnp.complex64
    for a, b in zip(single, double):
        _, mismatch = compare(np.asarray(a, np.complex128), np.asarray(b))
        assert mismatch < 1e-6, mismatch


def test_float32_binary_neutron_star():
    f = np.float32(20.0 + np.arange((2048 - 20) * 256) / 256.0).astype(np.float64)
    binary = wf.Binary(1.42, 1.27, (0.0, 0.0, 0.05), (0.0, 0.0, -0.03), 100.0, 0.5, 1.0, 0.0)
    function = jax.jit(wf.polarizations_function("IMRPhenomD"))
    single = function(wf.Binary(*(jnp.asarray(np.float32(v), jnp.float32) for v in binary)),
                      jnp.asarray(f, jnp.float32))
    double = function(wf.Binary(*(jnp.asarray(np.float32(v), jnp.float64) for v in binary)), jnp.asarray(f))
    for a, b in zip(single, double):
        assert compare(np.asarray(a, np.complex128), np.asarray(b))[1] < 1e-6


def test_batch():
    f = np.geomspace(20.0, 1024.0, 700)
    columns = ("mass_1", "mass_2", "luminosity_distance", "inclination", "phi_ref", "f_ref")
    table = wf.parameter_table({**{k: np.array([keywords(c)[k] for c in CASES]) for k in columns},
                                "spin_1": np.array([keywords(c)["spin_1"] for c in CASES]),
                                "spin_2": np.array([keywords(c)["spin_2"] for c in CASES])})
    batch = wf.batch_polarizations("IMRPhenomD", f, table)
    assert batch.ok.all()
    for i, case in enumerate(CASES):
        hp, hc = wf.polarizations("IMRPhenomD", f, **keywords(case))
        np.testing.assert_allclose(np.asarray(batch.hplus[i]), np.asarray(hp), rtol=1e-12,
                                   atol=1e-12 * np.abs(hp).max())
        np.testing.assert_allclose(np.asarray(batch.hcross[i]), np.asarray(hc), rtol=1e-12,
                                   atol=1e-12 * np.abs(hc).max())


def test_lal_errors():
    f = np.geomspace(20.0, 1024.0, 100)
    good = keywords(CASES[1])
    with pytest.raises(wf.InvalidParameterError, match="in-plane"):
        wf.polarizations("IMRPhenomD", f, **{**good, "spin_2": (0.0, 0.2, 0.3)})
    with pytest.raises(wf.UnsupportedError):
        wf.modes("IMRPhenomD", f, **good)
