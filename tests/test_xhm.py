# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXHM (``waveformx.models.xhm``, release 122022) against
LALSimulation 6.2.0 (``SimInspiralChooseFDWaveformSequence``,
``SimIMRPhenomXHMFrequencySequenceOneMode``, ``SimInspiralChooseFDModes``),
in float64 and float32, through the public functions.

The (2,2), (2,1), (3,3) and (4,4) modes agree with LAL to ``1e-8``
(relative to their largest value). The (3,2) mode differs by up to a few
``1e-2``: LAL fixes its intermediate phase with the second derivative of
the rotated ringdown phase taken by finite differences with a step of
``1e-7``, which amplify rounding by ``~1e14``; the model takes the exact
derivative (``tests/test_xhm.py::test_exact_derivatives`` checks it against
well-conditioned finite differences). The polarizations then agree to
mismatches below ``1e-7``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformx as wf
from conftest import compare
from waveformx.models.xhm import core, reference

MTSUN_SI = 4.925490947641267e-06

# mass_1, mass_2, chi_1, chi_2, distance, inclination, phi_ref, f_min, f_ref
CASES = [
    (36.0, 29.0, 0.3, -0.4, 500.0, 1.0, 1.2, 20.0, 30.0),
    (60.0, 15.0, 0.8, 0.5, 1000.0, 2.5, 4.0, 15.0, 20.0),
    (80.0, 10.0, -0.7, 0.2, 2000.0, 2.0, 0.3, 10.0, 25.0),
    (1.6, 1.3, 0.05, 0.02, 40.0, 0.2, 2.0, 30.0, 40.0),
    (150.0, 100.0, 0.95, 0.95, 3000.0, 0.7, 3.0, 10.0, 15.0),
    (20.0, 45.0, 0.3, -0.4, 300.0, 2.9, 0.7, 20.0, 0.0),
    (50.0, 2.0, 0.995, 0.3, 300.0, 1.1, 1.7, 20.0, 25.0),
    (100.0, 1.0, 0.5, 0.0, 300.0, 1.1, 1.7, 10.0, 0.0),
    (30.0, 30.0, 0.2, 0.2, 400.0, 0.4, 0.0, 20.0, 20.0),
]


def keywords(case):
    m1, m2, c1, c2, distance, inclination, phi, _, f_ref = case
    return dict(mass_1=m1, mass_2=m2, spin_1=(0.0, 0.0, c1), spin_2=(0.0, 0.0, c2), luminosity_distance=distance,
                inclination=inclination, phi_ref=phi, f_ref=f_ref)


def grid(case, n=1000):
    total = case[0] + case[1]
    return np.geomspace(case[7], min(0.3 / (total * MTSUN_SI), 2048.0), n)


def lal_polarizations(lal, case, f):
    lal_module, lalsim = lal
    m1, m2, c1, c2, distance, inclination, phi, _, f_ref = case
    vector = lal_module.CreateREAL8Vector(len(f))
    vector.data = f
    hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
        phi, m1 * lal_module.MSUN_SI, m2 * lal_module.MSUN_SI, 0.0, 0.0, c1, 0.0, 0.0, c2, f_ref,
        distance * 1e6 * lal_module.PC_SI, inclination, None, lalsim.IMRPhenomXHM, vector)
    return hp.data.data, hc.data.data


def lal_mode(lal, case, f, ell, emm):
    lal_module, lalsim = lal
    m1, m2, c1, c2, distance, _, phi, _, f_ref = case
    vector = lal_module.CreateREAL8Vector(len(f))
    vector.data = f
    return lalsim.SimIMRPhenomXHMFrequencySequenceOneMode(
        vector, m1 * lal_module.MSUN_SI, m2 * lal_module.MSUN_SI, c1, c2, ell, emm,
        distance * 1e6 * lal_module.PC_SI, phi, f_ref, None).data.data


_reference_modes = jax.jit(lambda p, f: _one_mode_setup(p, f))


def _one_mode_setup(p, f):
    s = reference.setup(tuple(p[i] for i in range(8)), f[0], 1.0)
    mf = s.wf.m_sec * f
    out = {}
    for mode, c in s.modes.items():
        a, phi = reference.mode_amplitude_phase(mf, c)
        out[mode] = c.amp0 * a * jnp.exp(1j * phi)
    return out


@pytest.mark.parametrize("case", CASES)
def test_polarizations_and_modes(lal, case):
    f = grid(case)
    hp, hc = wf.polarizations("IMRPhenomXHM", f, scaled_strain=False, **keywords(case))
    for ours, theirs in zip((hp, hc), lal_polarizations(lal, case, f)):
        error, mismatch = compare(ours, theirs)
        assert error < 1e-3 and mismatch < 1e-7, (error, mismatch)
    modes = wf.modes("IMRPhenomXHM", f, scaled_strain=False, **keywords(case))
    n = len(f)
    for (ell, emm) in reference.MODES:
        theirs = lal_mode(lal, case, f, ell, -emm)
        ours = np.asarray(modes[(ell, -emm)])[n:]
        if (ell, emm) == (3, 2):
            # (the (3,2) of the one-mode function does not recycle the (2,2))
            ours = np.asarray(_reference_modes(jnp.asarray([case[k] for k in (0, 1, 2, 3, 4, 5, 6, 8)]),
                                               jnp.asarray(f))[(3, 2)])
            error, mismatch = compare(ours, theirs)
            assert mismatch < 1e-4, mismatch
            continue
        if not np.any(theirs):
            assert not np.any(ours)
            continue
        error, mismatch = compare(ours, theirs)
        assert error < 1e-8, ((ell, emm), error)
    # the modes sum to the polarizations
    p2, c2 = modes.polarizations()
    assert np.max(np.abs(np.asarray(p2) - hp)) < 1e-12 * np.max(np.abs(hp))


def test_exact_derivatives():
    """The derivatives of the rotated (3,2) ringdown phase that fix its
    intermediate phase against fourth-order finite differences with steps
    at which they are well conditioned."""
    for case in (CASES[0], CASES[6]):
        p = tuple(jnp.float64(case[k]) for k in (0, 1, 2, 3, 4, 5, 6, 8))
        s = reference.setup(p, case[7], 1.0)
        c = s.modes[(3, 2)]
        fcut = float(c.phase.f_phase_match_im)

        def phase(f):
            re, im = reference.spheroidal_to_spherical(f, c.mix, c.amp)
            return jnp.arctan2(im, re)

        d1 = float(jax.grad(phase)(jnp.float64(fcut)))
        d2 = float(jax.grad(jax.grad(phase))(jnp.float64(fcut)))
        h = 1e-5
        v = np.unwrap([float(phase(jnp.float64(fcut + k * h))) for k in (-2, -1, 0, 1, 2)])
        fd1 = (-v[4] + 8 * v[3] - 8 * v[1] + v[0]) / (12 * h)
        fd2 = (-v[4] + 16 * v[3] - 30 * v[2] + 16 * v[1] - v[0]) / (12 * h * h)
        assert d1 == pytest.approx(fd1, rel=1e-7) and d2 == pytest.approx(fd2, rel=1e-5)


def test_choose_fd_modes(lal):
    """``SimInspiralChooseFDModes`` with multibanding off
    (``PhenomXHMThresholdMband = 0``; on by default for its uniform grid,
    where it interpolates the modes to ~1e-4)."""
    lal_module, lalsim = lal
    case = CASES[0]
    m1, m2, c1, c2, distance, inclination, phi, _, f_ref = case
    delta_f, f_min, f_max = 0.25, 20.0, 1024.0
    ours = wf.choose_fd_modes("IMRPhenomXHM", delta_f, f_min, f_max, scaled_strain=False, **keywords(case))
    params = lal_module.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(params, 0)
    theirs = lalsim.SimInspiralChooseFDModes(
        m1 * lal_module.MSUN_SI, m2 * lal_module.MSUN_SI, 0.0, 0.0, c1, 0.0, 0.0, c2, delta_f, f_min, f_max,
        f_ref, phi, distance * 1e6 * lal_module.PC_SI, inclination, params, lalsim.IMRPhenomXHM)
    count = 0
    while theirs is not None:
        ell, emm = theirs.l, theirs.m
        data = theirs.mode.data.data
        if abs(emm) != 2 or ell != 3:
            mine = np.asarray(ours[(ell, emm)])
            assert mine.shape == data.shape
            error, _ = compare(mine, data)
            assert error < 1e-8, ((ell, emm), error)
            count += 1
        theirs = theirs.next
    assert count == 8
    np.testing.assert_allclose(np.asarray(ours.frequencies)[-1], f_max)


def test_float32():
    """float32 against float64 at the float32-rounded inputs (uniform
    mismatch of the polarizations and of each mode), binary neutron stars
    on a uniform grid of 1/256 Hz from 20 Hz among them."""
    function = jax.jit(lambda b, f: core_modes(b, f))
    cases = [CASES[k] for k in (0, 2, 4, 6, 8)] + [(1.42, 1.27, 0.05, -0.03, 100.0, 0.5, 1.0, 20.0, 0.0)]
    for case in cases:
        bns = case[0] < 3.0
        f = (20.0 + np.arange((2048 - 20) * 256) / 256.0) if bns else grid(case)
        f = np.float32(f).astype(np.float64)
        binary = wf.Binary(**{k: np.float32(v) for k, v in keywords(case).items()})
        m32, p32 = function(wf.Binary(*(jnp.asarray(v, jnp.float32) for v in binary)), jnp.asarray(f, jnp.float32))
        m64, p64 = function(wf.Binary(*(jnp.asarray(v, jnp.float64) for v in binary)), jnp.asarray(f))
        assert p32[0].dtype == jnp.complex64
        for a, b in zip(p32, p64):
            assert compare(np.asarray(a, np.complex128), np.asarray(b))[1] < 1e-6
        for mode in m64:
            if np.any(np.asarray(m64[mode])):
                # (the (3,2) near maximal spin: the second derivative of the (2,2)
                # ringdown phase that fixes its intermediate phase loses ~1e-4)
                assert compare(np.asarray(m32[mode], np.complex128), np.asarray(m64[mode]))[1] < 2e-4, mode


def core_modes(binary, f):
    from waveformx.models.xhm import model

    return model.modes_and_polarizations(binary, f, wf.WaveformOptions())


def test_batch_and_errors():
    f = np.geomspace(20.0, 1024.0, 300)
    table = wf.parameter_table({**{k: np.array([keywords(c)[k] for c in CASES]) for k in
                                   ("mass_1", "mass_2", "luminosity_distance", "inclination", "phi_ref", "f_ref")},
                                "spin_1": np.array([keywords(c)["spin_1"] for c in CASES]),
                                "spin_2": np.array([keywords(c)["spin_2"] for c in CASES])})
    batch = wf.batch_polarizations("IMRPhenomXHM", f, table)
    assert batch.ok.all()
    for i, case in enumerate(CASES):
        hp, hc = wf.polarizations("IMRPhenomXHM", f, **keywords(case))
        np.testing.assert_allclose(np.asarray(batch.hplus[i]), np.asarray(hp), rtol=1e-10,
                                   atol=1e-10 * np.abs(hp).max())
    good = keywords(CASES[0])
    with pytest.raises(wf.InvalidParameterError, match="in-plane"):
        wf.polarizations("IMRPhenomXHM", f, **{**good, "spin_2": (0.0, 0.1, 0.3)})
    with pytest.raises(wf.UnsupportedError):
        wf.polarizations("IMRPhenomXHM", f, mode_array=((2, 2), (2, -2)), **good)
    function = wf.polarizations_function("IMRPhenomXHM")
    binary = wf.Binary(2000.0, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 400.0, 0.4, 0.0, 20.0)
    hp, _ = function(binary, jnp.asarray(f))
    assert np.all(np.isnan(np.asarray(hp)))
