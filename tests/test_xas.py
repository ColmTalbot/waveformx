# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXAS (``waveformjax.models.xas``) against
``SimInspiralChooseFDWaveformSequence`` of LALSimulation 6.2.0, in float64
and float32, through the public functions.

The model is LAL's (2,2) mode with the same arithmetic in float64 and
float32 (masses in solar masses, the QNM fits in powers of ``1 - a_f``),
which leaves LAL's rounding by up to ``1e-11``: in float64 it agrees with
LAL to ``1e-10`` (relative to the largest value), in float32 as closely as
float32 allows.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformjax as wf
from conftest import compare
from waveformjax.models.xas import core

MTSUN_SI = 4.925490947641267e-06

# mass_1, mass_2, chi_1, chi_2, distance, inclination, phi_ref, f_min, f_ref
# (those of test_xpnr_xas.py, the lighter body first and f_ref = 0 among them)
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
]


def keywords(case):
    m1, m2, c1, c2, distance, inclination, phi, _, f_ref = case
    return dict(mass_1=m1, mass_2=m2, spin_1=(0.0, 0.0, c1), spin_2=(0.0, 0.0, c2), luminosity_distance=distance,
                inclination=inclination, phi_ref=phi, f_ref=f_ref)


def grid(case, n=1500):
    m1, m2, f_min = case[0], case[1], case[7]
    return np.geomspace(f_min, 0.33 / ((m1 + m2) * MTSUN_SI) * 1.05, n)


def lal_polarizations(lal, case, frequencies):
    lal_module, lalsim = lal
    m1, m2, c1, c2, distance, inclination, phi, _, f_ref = case
    vector = lal_module.CreateREAL8Vector(len(frequencies))
    vector.data = frequencies
    hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
        phi, m1 * lal_module.MSUN_SI, m2 * lal_module.MSUN_SI, 0.0, 0.0, c1, 0.0, 0.0, c2, f_ref,
        distance * 1e6 * lal_module.PC_SI, inclination, None, lalsim.IMRPhenomXAS, vector,
    )
    return hp.data.data, hc.data.data


@pytest.mark.parametrize("case", CASES)
def test_matches_lalsimulation(lal, case):
    f = grid(case)
    hp, hc = wf.polarizations("IMRPhenomXAS", f, scaled_strain=False, **keywords(case))
    for ours, reference in zip((hp, hc), lal_polarizations(lal, case, f)):
        error, mismatch = compare(ours, reference)
        assert error < 1e-10, error
        assert mismatch < 1e-14, mismatch


@pytest.mark.parametrize("case", CASES)
def test_float32(case):
    """float32 against float64 at the float32-rounded inputs: the weighted
    (uniform) mismatch of float32's rounding of the phase."""
    f = np.float32(grid(case, 1000)).astype(np.float64)
    binary = wf.Binary(**{k: np.float32(v) if np.ndim(v) == 0 else np.float32(v) for k, v in keywords(case).items()})
    function = jax.jit(wf.polarizations_function("IMRPhenomXAS"))
    single = function(wf.Binary(*(jnp.asarray(v, jnp.float32) for v in binary)), jnp.asarray(f, jnp.float32))
    double = function(wf.Binary(*(jnp.asarray(v, jnp.float64) for v in binary)), jnp.asarray(f))
    assert single[0].dtype == jnp.complex64
    for a, b in zip(single, double):
        _, mismatch = compare(np.asarray(a, np.complex128), np.asarray(b))
        assert mismatch < 1e-6, mismatch


def test_float32_binary_neutron_star():
    """A binary neutron star from 20 Hz on a uniform grid of 1/256 Hz (half
    a million frequencies): its inspiral phase reaches 10^4 radians, which
    the leading term in float64 keeps to float32's rounding."""
    f = np.float32(20.0 + np.arange((2048 - 20) * 256) / 256.0).astype(np.float64)
    binary = wf.Binary(1.42, 1.27, (0.0, 0.0, 0.05), (0.0, 0.0, -0.03), 100.0, 0.5, 1.0, 0.0)
    function = jax.jit(wf.polarizations_function("IMRPhenomXAS"))
    single = function(wf.Binary(*(jnp.asarray(np.float32(v), jnp.float32) for v in binary)), jnp.asarray(f, jnp.float32))
    double = function(wf.Binary(*(jnp.asarray(np.float32(v), jnp.float64) for v in binary)), jnp.asarray(f))
    for a, b in zip(single, double):
        assert compare(np.asarray(a, np.complex128), np.asarray(b))[1] < 1e-6


def test_uniform_grid_and_batch(lal):
    """A large uniform grid, and a batch (``jax.vmap``, shared
    frequencies) against the binaries one at a time."""
    case = CASES[1]
    f = 20.0 + np.arange(1 << 17) / 256.0
    hp, _ = wf.polarizations("IMRPhenomXAS", f, scaled_strain=False, **keywords(case))
    error, mismatch = compare(hp, lal_polarizations(lal, case, f)[0])
    assert error < 1e-10 and mismatch < 1e-14
    f = np.geomspace(20.0, 1024.0, 700)
    table = wf.parameter_table({**{k: np.array([keywords(c)[k] for c in CASES]) for k in
                                   ("mass_1", "mass_2", "luminosity_distance", "inclination", "phi_ref", "f_ref")},
                                "spin_1": np.array([keywords(c)["spin_1"] for c in CASES]),
                                "spin_2": np.array([keywords(c)["spin_2"] for c in CASES])})
    batch = wf.batch_polarizations("IMRPhenomXAS", f, table)
    assert batch.ok.all()
    for i, case in enumerate(CASES):
        hp, hc = wf.polarizations("IMRPhenomXAS", f, **keywords(case))
        np.testing.assert_allclose(np.asarray(batch.hplus[i]), np.asarray(hp), rtol=1e-12, atol=1e-12 * np.abs(hp).max())
        np.testing.assert_allclose(np.asarray(batch.hcross[i]), np.asarray(hc), rtol=1e-12, atol=1e-12 * np.abs(hc).max())


def test_lal_errors():
    f = np.geomspace(20.0, 1024.0, 100)
    good = keywords(CASES[1])
    with pytest.raises(wf.InvalidParameterError, match="in-plane"):
        wf.polarizations("IMRPhenomXAS", f, **{**good, "spin_1": (0.1, 0.0, 0.3)})
    with pytest.raises(wf.InvalidParameterError, match="1000"):
        wf.polarizations("IMRPhenomXAS", f, **{**good, "mass_1": 2000.0, "mass_2": 1.0})
    with pytest.raises(wf.UnsupportedError):
        wf.modes("IMRPhenomXAS", f, **good)
    with pytest.raises(wf.UnsupportedError):
        wf.simulate_fd("IMRPhenomXAS", 0.25, 20.0, 1024.0, **good)
    # the pure function gives NaN where LAL fails
    function = wf.polarizations_function("IMRPhenomXAS")
    binary = wf.Binary(2000.0, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 400.0, 0.4, 0.0, 20.0)
    hp, hc = function(binary, jnp.asarray(f))
    assert np.all(np.isnan(np.asarray(hp))) and np.all(np.isnan(np.asarray(hc)))


def test_qnm_fits():
    """The QNM fits in powers of ``1 - a_f``: LAL's values (to its rounding,
    which cancels near ``a_f = 1``), and float32 within ``2e-6`` where LAL's
    form loses a part in 100."""
    from waveformjax.models.xpnr._generated import qnm

    for af in [-0.99, -0.5, 0.0, 0.5, 0.9, 0.99, 0.999]:
        ring, damp = core.qnm_22(np.float64(af))
        assert float(ring) == pytest.approx(float(qnm.fring22(af)), rel=1e-10)
        assert float(damp) == pytest.approx(float(qnm.fdamp22(af)), rel=1e-10)
        ring32, damp32 = core.qnm_22(np.float32(af))
        assert float(ring32) == pytest.approx(float(ring), rel=2e-6)
        assert float(damp32) == pytest.approx(float(damp), rel=2e-6)
