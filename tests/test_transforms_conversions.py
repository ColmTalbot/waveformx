# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The conversions as pure functions (``simulate_td_function`` and
``simulate_fd_function``) under ``jax.jit``, ``jax.vmap`` and ``jax.grad``.

The functions keep the lengths, shifts, tapers and epoch of a reference
binary, so they are smooth in the binary: the gradients are checked against
central finite differences for every parameter. The time domain is
IMRPhenomXPNR's (2,2) mode (compiling the gradient of the model takes
minutes; see test_transforms_xpnr.py); its SpinTaylor angles jump by about
LAL's integration error where the number of integration steps changes, so
the steps are small (``1e-6``, relative for the masses, the distance and
``f_ref``) and the tolerance is that of test_transforms_xpnr.py. The masses
also move the starting frequency of the model and the cutoff of the first
high-pass filter, which the gradients include.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformjax as wf
from conftest import requires_nrsur

XPNR = dict(mass_1=36.0, mass_2=29.0, spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1), luminosity_distance=500.0,
            inclination=0.7, phi_ref=1.2, f_ref=25.0)
NRSUR = dict(mass_1=40.0, mass_2=20.0, spin_1=(0.4, -0.3, 0.2), spin_2=(0.1, 0.5, -0.3), luminosity_distance=100.0,
             inclination=0.7, phi_ref=0.1, f_ref=20.0)
DOMINANT = ((2, 2), (2, -2))
NAMES = ["mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z", "luminosity_distance",
         "inclination", "phi_ref", "f_ref"]


def binary_of(x):
    return wf.Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                     inclination=x[9], phi_ref=x[10], f_ref=x[11])


def vector_of(keywords):
    return jnp.asarray([keywords["mass_1"], keywords["mass_2"], *keywords["spin_1"], *keywords["spin_2"],
                        keywords["luminosity_distance"], keywords["inclination"], keywords["phi_ref"],
                        keywords["f_ref"]], dtype=jnp.float64)


@functools.lru_cache(maxsize=None)
def td_function():
    return wf.simulate_td_function("IMRPhenomXPNR", 1 / 1024, 25.0, mode_array=DOMINANT, scaled_strain=False,
                                   **XPNR)


@functools.lru_cache(maxsize=None)
def fd_function():
    return wf.simulate_fd_function("NRSur7dq4", 0.0, 20.0, 1024.0, **NRSUR)


def weights(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape))


@functools.lru_cache(maxsize=None)
def td_compiled(name):
    function = td_function()
    w = weights((2, len(function)), 3)
    scale = 1e21

    def loss(x):
        hp, hc = function(binary_of(x))
        return jnp.sum(w[0] * hp + w[1] * hc) * scale

    return {"call": jax.jit(lambda x: function(binary_of(x))), "loss": jax.jit(loss),
            "grad": jax.jit(jax.grad(loss))}[name]


def test_td_function_is_simulate_td():
    """At the reference binary the compiled function gives the waveform of
    simulate_td, which has LAL's length and epoch (test_conversions.py), to
    the rounding of another compilation amplified by the high-pass filter."""
    function = td_function()
    reference = wf.simulate_td("IMRPhenomXPNR", 1 / 1024, 25.0, mode_array=DOMINANT, scaled_strain=False, **XPNR)
    assert function.epoch == reference.epoch and len(function) == len(reference.hplus)
    hp, hc = td_compiled("call")(vector_of(XPNR))
    for a, b in ((hp, reference.hplus), (hc, reference.hcross)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12 * float(jnp.max(jnp.abs(b))))


# the parameters with relative steps of the central differences
RELATIVE = {"mass_1", "mass_2", "luminosity_distance", "f_ref"}
STEP = 1e-6
TOLERANCE = 2e-5


def test_td_gradient_matches_finite_differences():
    x = vector_of(XPNR)
    gradient = np.asarray(td_compiled("grad")(x))
    assert np.all(np.isfinite(gradient))
    f = td_compiled("loss")
    scale = float(np.max(np.abs(gradient)))
    for i, name in enumerate(NAMES):
        step = STEP * (abs(float(x[i])) if name in RELATIVE else 1.0)
        numeric = (float(f(x.at[i].add(step))) - float(f(x.at[i].add(-step)))) / (2 * step)
        error = abs(gradient[i] - numeric) / max(abs(numeric), 1e-3 * scale)
        print(f"{name}: gradient {gradient[i]:.10e} finite difference {numeric:.10e} relative {error:.1e}")
        assert error <= TOLERANCE, (name, gradient[i], numeric)


@requires_nrsur
def test_fd_function_is_simulate_fd_under_jit():
    function = fd_function()
    reference = wf.simulate_fd("NRSur7dq4", 0.0, 20.0, 1024.0, **NRSUR)
    assert function.epoch == reference.epoch and len(function) == len(reference.hplus)
    hp, hc = jax.jit(lambda x: function(binary_of(x)))(vector_of(NRSUR))
    for a, b in ((hp, reference.hplus), (hc, reference.hcross)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12 * float(jnp.max(jnp.abs(b))))


@requires_nrsur
def test_nrsur7dq4_td_function_is_simulate_td_under_jit():
    function = wf.simulate_td_function("NRSur7dq4", 1 / 2048, 20.0, **dict(NRSUR, f_ref=0.0))
    reference = wf.simulate_td("NRSur7dq4", 1 / 2048, 20.0, **dict(NRSUR, f_ref=0.0))
    assert function.epoch == reference.epoch and len(function) == len(reference.hplus)
    hp, hc = jax.jit(lambda x: function(binary_of(x)))(vector_of(dict(NRSUR, f_ref=0.0)))
    for a, b in ((hp, reference.hplus), (hc, reference.hcross)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12 * float(jnp.max(jnp.abs(b))))


@requires_nrsur
def test_fd_function_under_vmap():
    function = fd_function()
    xs = jnp.stack([vector_of(NRSUR), vector_of(dict(NRSUR, mass_1=40.5, inclination=0.9))])
    batched = jax.jit(jax.vmap(lambda x: function(binary_of(x))))(xs)
    single = jax.jit(lambda x: function(binary_of(x)))
    for i in range(xs.shape[0]):
        for a, b in zip(batched, single(xs[i])):
            np.testing.assert_allclose(np.asarray(a[i]), np.asarray(b), rtol=0, atol=1e-13 * float(jnp.max(jnp.abs(b))))


@requires_nrsur
def test_fd_gradient_matches_finite_differences():
    """The gradient of the overlap with a fixed frequency-domain waveform
    against the 4-point central difference, for every parameter."""
    function = fd_function()
    x = vector_of(NRSUR)
    reference = function(binary_of(vector_of(dict(NRSUR, mass_1=40.4, inclination=0.9))))
    norm = sum(jnp.sum(jnp.abs(h) ** 2) for h in reference)

    @jax.jit
    def overlap(x):
        hp, hc = function(binary_of(x))
        return jnp.sum(jnp.real(hp * jnp.conj(reference[0]) + hc * jnp.conj(reference[1]))) / norm

    gradient = np.asarray(jax.jit(jax.grad(overlap))(x))
    steps = np.array([1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-1, 1e-4, 1e-4, 1e-4])
    numeric = []
    for i, step in enumerate(steps):
        values = [float(overlap(x.at[i].add(k * step))) for k in (-2, -1, 1, 2)]
        numeric.append((values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step))
    numeric = np.array(numeric)
    scale = np.max(np.abs(numeric))
    for name, g, n in zip(NAMES, gradient, numeric):
        print(f"{name}: gradient {g:.10e} finite difference {n:.10e}")
        assert np.isfinite(g)
        assert abs(g - n) <= 1e-6 * abs(n) + 1e-9 * scale, (name, g, n)
