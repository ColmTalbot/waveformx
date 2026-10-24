# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The matches of :mod:`waveformx.match`: analytic cases, the recovery of
injected time and phase shifts and polarisations, ``jax.jit``, ``jax.vmap``
and ``jax.grad`` against finite differences, and the optimisation over the
convention parameters of the lean IMRPhenomXPNR."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformx as wf
from waveformx.match import (
    inner_product,
    match,
    mismatch,
    optimize,
    polarization_match,
    polarization_mismatch,
    with_conventions,
)

DELTA_F = 0.25
FREQUENCIES = jnp.asarray(np.arange(80, 2049) * DELTA_F)  # 20-512 Hz
PSD = (FREQUENCIES / 100.0) ** -4 + 2 + 2 * (FREQUENCIES / 100.0) ** 2


def chirp(theta=0.0, frequencies=FREQUENCIES):
    """A stationary-phase inspiral, ``f^(-7/6) e^(-i psi(f))``, whose phase
    depends on ``theta`` (a chirp-mass-like parameter)."""
    x = (np.pi * frequencies * (1e-4 * (1 + theta))) ** (-5 / 3)
    return frequencies ** (-7 / 6) * jnp.exp(-1j * (3 / 128 * x - np.pi / 4))


def shifted(h, t, phi):
    """``h`` delayed by ``t`` with its phase shifted by ``phi``."""
    return h * jnp.exp(-2j * np.pi * FREQUENCIES * t + 1j * phi)


def test_inner_product_white():
    a = chirp()
    expected = 4 * DELTA_F * np.sum(np.abs(np.asarray(a)) ** 2)
    np.testing.assert_allclose(inner_product(a, a, FREQUENCIES, jnp.ones_like(PSD)), expected, rtol=1e-14)


def test_dirichlet_kernel():
    """A flat band shifted by ``tau``, unmaximised over time: the overlap
    is ``|sin(pi K delta_f tau) / (K sin(pi delta_f tau))|``."""
    a = jnp.ones(len(FREQUENCIES), jnp.complex128)
    flat = jnp.ones_like(PSD)
    k = len(FREQUENCIES)
    for tau in (1e-3, 0.0123, 0.3):
        expected = abs(np.sin(np.pi * k * DELTA_F * tau) / (k * np.sin(np.pi * DELTA_F * tau)))
        got = match(a, shifted(a, tau, 0.0), FREQUENCIES, flat, time=False)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-14)
        assert abs(match(a, shifted(a, tau, 0.0), FREQUENCIES, flat)) == pytest.approx(1, abs=1e-12)


def test_constant_phase():
    """``b = a e^(i phi)``: ``cos phi`` without maximisation, 1 with."""
    a = chirp()
    for phi in (0.3, 2.0, -1.0):
        b = 3.0 * shifted(a, 0.0, phi)
        np.testing.assert_allclose(match(a, b, FREQUENCIES, PSD, time=False, phase=False), np.cos(phi), rtol=1e-12)
        np.testing.assert_allclose(match(a, b, FREQUENCIES, PSD, time=False), 1, atol=1e-13)


@pytest.mark.parametrize("t, phi", [(0.0, 0.0), (0.0123456, 0.7), (-0.3141, -2.5), (1.3, 3.0)])
def test_recovers_time_and_phase(t, phi):
    """Shifts off the FFT grid, also beyond its first sample, negative and
    large (the period is ``1 / delta_f = 4 s``)."""
    a = chirp()
    assert abs(mismatch(a, shifted(a, t, phi), FREQUENCIES, PSD)) < 1e-12
    # the unmaximised mismatch of the same pair is not small
    if t != 0.0:
        assert mismatch(a, shifted(a, t, phi), FREQUENCIES, PSD, time=False) > 0.1


def test_different_waveforms():
    """A smaller match than for the same waveform, and symmetric."""
    a, b = chirp(), chirp(1e-3)
    m = mismatch(a, b, FREQUENCIES, PSD)
    assert 1e-6 < m < 1
    np.testing.assert_allclose(m, mismatch(b, a, FREQUENCIES, PSD), rtol=1e-8)
    assert m < mismatch(a, b, FREQUENCIES, PSD, time=False)


def test_polarizations_recovered():
    """Any combination of the polarizations, delayed: a match of 1."""
    hplus = chirp()
    hcross = -1j * chirp(2e-3) * 0.8
    for fp, fx, t in ((0.3, -0.9, 0.01), (1.0, 0.0, -0.2), (0.0, 0.5, 0.0)):
        signal = shifted(fp * hplus + fx * hcross, t, 0.0)
        assert abs(polarization_mismatch(signal, hplus, hcross, FREQUENCIES, PSD)) < 1e-12


def test_polarization_projection():
    """A signal with a part orthogonal to both polarizations: the norm of its
    projection over its norm."""
    hplus, hcross = chirp(), -1j * chirp(2e-3)
    noise = chirp(-5e-3) * jnp.exp(0.4j)

    def project(x, onto):
        return x - inner_product(x, onto, FREQUENCIES, PSD) / inner_product(onto, onto, FREQUENCIES, PSD) * onto

    e2 = project(hcross, hplus)
    orthogonal = project(project(noise, hplus), e2)
    signal = 0.4 * hplus + 0.7 * hcross + orthogonal
    inside = 0.4 * hplus + 0.7 * hcross
    expected = np.sqrt(inner_product(inside, inside, FREQUENCIES, PSD) / inner_product(signal, signal, FREQUENCIES, PSD))
    got = polarization_match(signal, hplus, hcross, FREQUENCIES, PSD, time=False)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_parallel_polarizations():
    """``hcross`` zero (an edge-on, non-precessing binary): the match with
    ``hplus`` alone, not NaN."""
    a = chirp()
    got = polarization_match(shifted(a, 0.05, 0.0), a, 0 * a, FREQUENCIES, PSD)
    np.testing.assert_allclose(got, 1, atol=1e-12)


def test_jit_vmap():
    a = chirp()
    thetas = jnp.asarray([0.0, 1e-4, 1e-3])
    batch = jax.vmap(lambda th: chirp(th))(thetas)
    one = [float(mismatch(a, b, FREQUENCIES, PSD)) for b in batch]
    jitted = jax.jit(jax.vmap(lambda b: mismatch(a, b, FREQUENCIES, PSD)))(batch)
    np.testing.assert_allclose(jitted, one, rtol=1e-10, atol=1e-15)
    cross = 0.5j * chirp(1e-3)
    one = [float(polarization_mismatch(b, a, cross, FREQUENCIES, PSD)) for b in batch]
    jitted = jax.jit(jax.vmap(lambda b: polarization_mismatch(b, a, cross, FREQUENCIES, PSD)))(batch)
    np.testing.assert_allclose(jitted, one, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize("polarization", [False, True])
def test_gradient(polarization):
    """``jax.grad`` with respect to a parameter of the template against
    central differences, at a time shift between the FFT samples."""
    signal = shifted(chirp(), 0.0377, 0.4)
    cross = -1j * chirp(1e-3)

    def loss(theta):
        if polarization:
            return polarization_mismatch(signal, chirp(theta), cross, FREQUENCIES, PSD)
        return mismatch(signal, chirp(theta), FREQUENCIES, PSD)

    for theta in (3e-4, -2e-4):
        step = 1e-6
        fd = (loss(theta + step) - loss(theta - step)) / (2 * step)
        np.testing.assert_allclose(jax.grad(loss)(theta), fd, rtol=1e-5)
        np.testing.assert_allclose(jax.jit(jax.grad(loss))(theta), fd, rtol=1e-5)


def test_with_conventions():
    binary = wf.Binary(30.0, 20.0, jnp.array([0.3, 0.4, 0.2]), jnp.array([-0.1, 0.0, 0.5]), 100.0, 0.3, 1.0, 20.0)
    out = with_conventions(binary, [0.5, np.pi / 2, np.pi])
    np.testing.assert_allclose(out.phi_ref, 1.5)
    np.testing.assert_allclose(out.spin_1, [-0.4, 0.3, 0.2], atol=1e-15)
    np.testing.assert_allclose(out.spin_2, [0.1, 0.0, 0.5], atol=1e-15)
    assert out.mass_1 == binary.mass_1 and out.inclination == binary.inclination


def test_optimize_multimodal():
    """The best of several starts, on a function with many minima."""
    def objective(x):
        return 1 - jnp.cos(x[0] - 1.0) * (0.6 + 0.4 * jnp.cos(3 * (x[1] + 0.5)))

    result = optimize(objective, [[3.0, 1.0], [0.5, -0.3], [-2.0, 2.0]])
    assert result.fun < 1e-12
    np.testing.assert_allclose(result.x, [1.0, -0.5], atol=1e-5)
    assert result.funs.shape == (3,) and result.xs.shape == (3, 2)


def test_optimize_xpnr_conventions():
    """The lean IMRPhenomXPNR against itself with shifted reference phase
    and spin azimuths: the optimisation over the conventions finds them."""
    frequencies = jnp.asarray(np.arange(160, 1025) / 8.0)  # 20-128 Hz
    psd = (frequencies / 100.0) ** -4 + 2 + 2 * (frequencies / 100.0) ** 2
    f = wf.polarizations_function("IMRPhenomXPNR", lean=True)
    binary = wf.Binary(jnp.float64(50.0), jnp.float64(25.0), jnp.array([0.5, 0.2, 0.3]),
                       jnp.array([-0.2, 0.4, -0.1]), jnp.float64(500.0), jnp.float64(0.9), jnp.float64(1.2),
                       jnp.float64(20.0))
    truth = jnp.array([0.4, -0.3, 0.5])
    hp, hc = f(with_conventions(binary, truth), frequencies)
    signal = 0.6 * hp + 0.8 * hc

    def objective(x):
        p, c = f(with_conventions(binary, x), frequencies)
        return polarization_mismatch(signal, p, c, frequencies, psd)

    assert objective(jnp.zeros(3)) > 1e-4
    result = optimize(objective, [[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])
    assert result.fun < 1e-9
    np.testing.assert_allclose(result.x, truth, atol=2e-3)
