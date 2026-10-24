# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""TaylorF2 (``waveformx.models.taylorf2``) against
``SimInspiralChooseFDWaveformSequence`` of LALSimulation 6.2.0 and against
waveformrs, in float64 and float32, through the public functions.

The model is LAL's with the masses in solar masses: it agrees with LAL to
rounding (``1e-10`` of the largest value, the phase of a binary neutron
star reaching ``10^4`` radians), and in float32 as closely as float32
allows.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformx as wf
from conftest import compare

MTSUN_SI = 4.925490947641267e-06

# mass_1, mass_2, chi_1, chi_2, distance, inclination, phi_ref, f_min, f_ref
# (f_ref = 0: the reference at coalescence)
CASES = [
    (30.0, 30.0, 0.0, 0.0, 400.0, 0.4, 0.0, 20.0, 20.0),
    (36.0, 29.0, 0.3, -0.4, 500.0, 1.0, 1.2, 20.0, 0.0),
    (60.0, 15.0, 0.8, 0.5, 1000.0, 2.5, 4.0, 15.0, 20.0),
    (1.6, 1.3, 0.05, 0.02, 40.0, 0.2, 2.0, 20.0, 40.0),
    (8.0, 1.4, 0.9, 0.0, 200.0, 0.3, 5.5, 20.0, 0.0),
    (25.0, 20.0, -0.9, -0.9, 300.0, 1.5, 0.7, 20.0, 20.0),
    (20.0, 45.0, 0.3, -0.4, 300.0, 2.9, 0.7, 20.0, 0.0),
    (50.0, 2.0, 0.995, 0.3, 300.0, 1.1, 1.7, 10.0, 25.0),
]


def keywords(case):
    m1, m2, c1, c2, distance, inclination, phi, _, f_ref = case
    return dict(mass_1=m1, mass_2=m2, spin_1=(0.0, 0.0, c1), spin_2=(0.0, 0.0, c2), luminosity_distance=distance,
                inclination=inclination, phi_ref=phi, f_ref=f_ref)


def grid(case, n=1500):
    """From ``f_min`` to beyond the Schwarzschild ISCO (LAL's sequence has
    no cutoff)."""
    m1, m2, f_min = case[0], case[1], case[7]
    return np.geomspace(f_min, 1.5 / (6.0 ** 1.5 * np.pi * (m1 + m2) * MTSUN_SI), n)


def lal_polarizations(lal, case, frequencies, approximant="TaylorF2"):
    lal_module, lalsim = lal
    m1, m2, c1, c2, distance, inclination, phi, _, f_ref = case
    vector = lal_module.CreateREAL8Vector(len(frequencies))
    vector.data = frequencies
    hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
        phi, m1 * lal_module.MSUN_SI, m2 * lal_module.MSUN_SI, 0.0, 0.0, c1, 0.0, 0.0, c2, f_ref,
        distance * 1e6 * lal_module.PC_SI, inclination, None, getattr(lalsim, approximant), vector,
    )
    return hp.data.data, hc.data.data


@pytest.mark.parametrize("case", CASES)
def test_matches_lalsimulation(lal, case):
    f = grid(case)
    hp, hc = wf.polarizations("TaylorF2", f, scaled_strain=False, **keywords(case))
    for ours, reference in zip((hp, hc), lal_polarizations(lal, case, f)):
        error, mismatch = compare(ours, reference)
        assert error < 1e-10, error
        assert mismatch < 1e-14, mismatch


@pytest.mark.parametrize("case", CASES[:4])
def test_matches_waveformrs(case):
    waveformrs = pytest.importorskip("waveformrs")
    f = grid(case)
    hp, hc = wf.polarizations("TaylorF2", f, scaled_strain=False, **keywords(case))
    reference = waveformrs.polarizations("TaylorF2", f, **keywords(case))
    for ours, theirs in zip((hp, hc), reference):
        error, mismatch = compare(ours, np.asarray(theirs))
        assert error < 1e-10 and mismatch < 1e-14, (error, mismatch)


@pytest.mark.parametrize("case", CASES)
def test_float32(case):
    """float32 against float64 at the float32-rounded inputs."""
    f = np.float32(grid(case, 1000)).astype(np.float64)
    binary = wf.Binary(**{k: np.float32(v) for k, v in keywords(case).items()})
    function = jax.jit(wf.polarizations_function("TaylorF2"))
    single = function(wf.Binary(*(jnp.asarray(v, jnp.float32) for v in binary)), jnp.asarray(f, jnp.float32))
    double = function(wf.Binary(*(jnp.asarray(v, jnp.float64) for v in binary)), jnp.asarray(f))
    assert single[0].dtype == jnp.complex64
    for a, b in zip(single, double):
        _, mismatch = compare(np.asarray(a, np.complex128), np.asarray(b))
        assert mismatch < 1e-6, mismatch


def test_float32_binary_neutron_star():
    """A binary neutron star from 20 Hz on a uniform grid of 1/256 Hz: the
    leading term of the phase in float64 keeps it to float32's rounding."""
    f = np.float32(20.0 + np.arange((2048 - 20) * 256) / 256.0).astype(np.float64)
    binary = wf.Binary(1.42, 1.27, (0.0, 0.0, 0.05), (0.0, 0.0, -0.03), 100.0, 0.5, 1.0, 30.0)
    function = jax.jit(wf.polarizations_function("TaylorF2"))
    single = function(wf.Binary(*(jnp.asarray(np.float32(v), jnp.float32) for v in binary)),
                      jnp.asarray(f, jnp.float32))
    double = function(wf.Binary(*(jnp.asarray(np.float32(v), jnp.float64) for v in binary)), jnp.asarray(f))
    for a, b in zip(single, double):
        assert compare(np.asarray(a, np.complex128), np.asarray(b))[1] < 1e-6


def test_batch(lal):
    """A batch (``jax.vmap``, shared frequencies) against the binaries one
    at a time, and a large uniform grid against LAL."""
    case = CASES[1]
    f = 20.0 + np.arange(1 << 17) / 256.0
    hp, _ = wf.polarizations("TaylorF2", f, scaled_strain=False, **keywords(case))
    error, mismatch = compare(hp, lal_polarizations(lal, case, f)[0])
    assert error < 1e-10 and mismatch < 1e-14
    f = np.geomspace(20.0, 1024.0, 700)
    columns = ("mass_1", "mass_2", "luminosity_distance", "inclination", "phi_ref", "f_ref")
    table = wf.parameter_table({**{k: np.array([keywords(c)[k] for c in CASES]) for k in columns},
                                "spin_1": np.array([keywords(c)["spin_1"] for c in CASES]),
                                "spin_2": np.array([keywords(c)["spin_2"] for c in CASES])})
    batch = wf.batch_polarizations("TaylorF2", f, table)
    assert batch.ok.all()
    for i, case in enumerate(CASES):
        hp, hc = wf.polarizations("TaylorF2", f, **keywords(case))
        np.testing.assert_allclose(np.asarray(batch.hplus[i]), np.asarray(hp), rtol=1e-12,
                                   atol=1e-12 * np.abs(hp).max())
        np.testing.assert_allclose(np.asarray(batch.hcross[i]), np.asarray(hc), rtol=1e-12,
                                   atol=1e-12 * np.abs(hc).max())


def test_lal_errors():
    f = np.geomspace(20.0, 1024.0, 100)
    good = keywords(CASES[1])
    with pytest.raises(wf.InvalidParameterError, match="in-plane"):
        wf.polarizations("TaylorF2", f, **{**good, "spin_1": (0.1, 0.0, 0.3)})
    with pytest.raises(wf.UnsupportedError):
        wf.modes("TaylorF2", f, **good)
    assert "TaylorF2" in wf.approximants("frequency")
    assert not wf.is_precessing("TaylorF2")
