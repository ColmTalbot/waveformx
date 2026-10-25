# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR under ``jax.jit``, ``jax.vmap`` and ``jax.grad``.

The gradients are checked against central finite differences of the
compiled polarizations for every parameter of the binary, at a generic
binary and a nearly aligned one, with steps of ``1e-6`` (relative for the
masses, the distance and ``f_ref``): they agree with the gradients to
``~1e-7`` (``~1e-5`` for the in-plane spins of the nearly aligned binary,
where the waveform is far from linear in them).

Compiling the gradient of the model takes minutes: the tests use the (2,2)
mode with the PNR angles, its tuned deviations and the antisymmetric part;
the higher modes have their own gradient tests (test_xpnr_hm.py).

The NNLO and MSA prescriptions (``IMRPhenomXPNR_NNLO``,
``IMRPhenomXPNR_MSA``) are checked the same way, with finite gradients
also for aligned and vanishing spins.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import MODEL, MODEL_MSA, MODEL_NNLO

OPTIONS = WaveformOptions(mode_array=((2, 2), (2, -2)), scaled_strain=False)
FREQUENCIES = jnp.asarray(np.geomspace(20.0, 400.0, 96))
NAMES = ["mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z", "luminosity_distance",
         "inclination", "phi_ref", "f_ref"]
BINARIES = {
    "generic": [36.0, 29.0, 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 25.0],
    "nearly_aligned": [30.0, 20.0, 1e-3, 5e-4, 0.5, -5e-4, 1e-3, -0.3, 500.0, 0.9, 0.4, 25.0],
}


def binary_of(x):
    return Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                  inclination=x[9], phi_ref=x[10], f_ref=x[11])


def polarizations(x):
    return MODEL.polarizations(binary_of(x), FREQUENCIES, OPTIONS)


# a fixed, smooth real function of the polarizations
_WEIGHTS = np.random.default_rng(3).normal(size=(2, len(FREQUENCIES), 2))


def loss(x):
    hp, hc = polarizations(x)
    scale = 1e21
    w = jnp.asarray(_WEIGHTS)
    return jnp.sum(w[0, :, 0] * hp.real * scale + w[0, :, 1] * hp.imag * scale
                   + w[1, :, 0] * hc.real * scale + w[1, :, 1] * hc.imag * scale)


@functools.lru_cache(maxsize=None)
def compiled(name):
    return {"polarizations": jax.jit(polarizations), "loss": jax.jit(loss), "grad": jax.jit(jax.grad(loss)),
            "vmap": jax.jit(jax.vmap(polarizations))}[name]


def test_jit_matches_eager():
    x = jnp.asarray(BINARIES["generic"])
    eager = polarizations(x)
    jitted = compiled("polarizations")(x)
    for a, b in zip(eager, jitted):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12 * float(jnp.max(jnp.abs(b))))


def test_vmap_matches_single_calls():
    """The batched computation rounds differently from that of single
    binaries (vectorised functions, other fusions); the nearly aligned
    binary amplifies rounding to ~1e-11 (as the compiled and eager single
    computations differ), the generic ones agree to ~1e-14."""
    xs = jnp.stack([jnp.asarray(x) for x in BINARIES.values()] + [jnp.asarray(BINARIES["generic"]).at[0].set(45.0)])
    batch = compiled("vmap")(xs)
    for i in range(xs.shape[0]):
        single = compiled("polarizations")(xs[i])
        for a, b in zip(batch, single):
            scale = float(jnp.max(jnp.abs(b)))
            np.testing.assert_allclose(np.asarray(a[i]), np.asarray(b), rtol=0, atol=1e-10 * scale)


# the parameters with relative steps of the central differences
RELATIVE = {"mass_1", "mass_2", "luminosity_distance", "f_ref"}
STEP = 1e-6
TOLERANCE = 2e-5


def test_forward_mode_matches_gradient():
    """``jax.jvp`` (the model's derivatives are forward-mode Jacobians, of
    which ``jax.grad`` transposes the contraction) along a direction: the
    gradient's projection on it."""
    x = jnp.asarray(BINARIES["generic"])
    direction = jnp.asarray(np.random.default_rng(5).normal(size=x.shape)) * jnp.abs(x)
    jvp = jax.jit(lambda x, v: jax.jvp(loss, (x,), (v,))[1])
    expected = float(jnp.dot(compiled("grad")(x), direction))
    assert float(jvp(x, direction)) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("which", list(BINARIES))
def test_gradient_matches_finite_differences(which):
    x = jnp.asarray(BINARIES[which])
    gradient = np.asarray(compiled("grad")(x))
    assert np.all(np.isfinite(gradient))
    f = compiled("loss")
    scale = float(np.max(np.abs(gradient)))
    for i, name in enumerate(NAMES):
        step = STEP * (abs(float(x[i])) if name in RELATIVE else 1.0)
        numeric = (float(f(x.at[i].add(step))) - float(f(x.at[i].add(-step)))) / (2 * step)
        error = abs(gradient[i] - numeric) / max(abs(numeric), 1e-3 * scale)
        print(f"{which} {name}: gradient {gradient[i]:.10e} finite difference {numeric:.10e} relative {error:.1e}")
        assert error <= TOLERANCE, (name, gradient[i], numeric)


# ---------------------------------------------------------------------------
# the NNLO and MSA prescriptions (IMRPhenomXPNR_NNLO, IMRPhenomXPNR_MSA)
# ---------------------------------------------------------------------------

PRESCRIPTIONS = {"NNLO": MODEL_NNLO, "MSA": MODEL_MSA}
#: aligned and vanishing spins, where the waveform is not differentiable in
#: the in-plane spins: the gradients are finite
SPECIAL = {
    "aligned": [36.0, 29.0, 0.0, 0.0, 0.3, 0.0, 0.0, -0.1, 500.0, 0.7, 1.2, 25.0],
    "non_spinning": [36.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.7, 1.2, 25.0],
}


@functools.lru_cache(maxsize=None)
def compiled_prescription(name, what):
    model = PRESCRIPTIONS[name]

    def polarizations_(x):
        return model.polarizations(binary_of(x), FREQUENCIES, OPTIONS)

    def loss_(x):
        hp, hc = polarizations_(x)
        w = jnp.asarray(_WEIGHTS)
        return jnp.sum(w[0, :, 0] * hp.real * 1e21 + w[0, :, 1] * hp.imag * 1e21
                       + w[1, :, 0] * hc.real * 1e21 + w[1, :, 1] * hc.imag * 1e21)

    return {"polarizations": jax.jit(polarizations_), "loss": jax.jit(loss_), "grad": jax.jit(jax.grad(loss_)),
            "vmap": jax.jit(jax.vmap(polarizations_)),
            "jvp": jax.jit(lambda x, v: jax.jvp(loss_, (x,), (v,))[1])}[what]


@pytest.mark.parametrize("name", list(PRESCRIPTIONS))
def test_prescription_vmap_matches_single_calls(name):
    xs = jnp.stack([jnp.asarray(x) for x in BINARIES.values()] + [jnp.asarray(BINARIES["generic"]).at[0].set(45.0)])
    batch = compiled_prescription(name, "vmap")(xs)
    for i in range(xs.shape[0]):
        single = compiled_prescription(name, "polarizations")(xs[i])
        for a, b in zip(batch, single):
            assert np.all(np.isfinite(np.asarray(b)))
            scale = float(jnp.max(jnp.abs(b)))
            np.testing.assert_allclose(np.asarray(a[i]), np.asarray(b), rtol=0, atol=1e-10 * scale)


@pytest.mark.parametrize("name", list(PRESCRIPTIONS))
def test_prescription_forward_mode_matches_gradient(name):
    x = jnp.asarray(BINARIES["generic"])
    direction = jnp.asarray(np.random.default_rng(5).normal(size=x.shape)) * jnp.abs(x)
    expected = float(jnp.dot(compiled_prescription(name, "grad")(x), direction))
    assert float(compiled_prescription(name, "jvp")(x, direction)) == pytest.approx(expected, rel=1e-10)


#: the MSA angles of the nearly aligned binary round at ~1e-10 of the
#: waveform (the roots of their cubic nearly coincide), which the finite
#: differences pick up
PRESCRIPTION_TOLERANCE = {("MSA", "nearly_aligned"): 1e-4}


@pytest.mark.parametrize("which", list(BINARIES))
@pytest.mark.parametrize("name", list(PRESCRIPTIONS))
def test_prescription_gradient_matches_finite_differences(name, which):
    """Against Richardson-extrapolated central differences with steps of
    ``1e-5`` and ``5e-6`` (relative for the masses, the distance and
    ``f_ref``)."""
    x = jnp.asarray(BINARIES[which])
    gradient = np.asarray(compiled_prescription(name, "grad")(x))
    assert np.all(np.isfinite(gradient))
    f = compiled_prescription(name, "loss")
    scale = float(np.max(np.abs(gradient)))
    tolerance = PRESCRIPTION_TOLERANCE.get((name, which), TOLERANCE)
    for i, parameter in enumerate(NAMES):
        step = 1e-5 * (abs(float(x[i])) if parameter in RELATIVE else 1.0)

        def central(s):
            return (float(f(x.at[i].add(s))) - float(f(x.at[i].add(-s)))) / (2 * s)

        numeric = (4.0 * central(step / 2) - central(step)) / 3.0
        error = abs(gradient[i] - numeric) / max(abs(numeric), 1e-3 * scale)
        print(f"{name} {which} {parameter}: gradient {gradient[i]:.10e} finite difference {numeric:.10e} "
              f"relative {error:.1e}")
        assert error <= tolerance, (parameter, gradient[i], numeric)


@pytest.mark.parametrize("which", list(SPECIAL))
@pytest.mark.parametrize("name", list(PRESCRIPTIONS))
def test_prescription_gradient_finite_for_aligned_spins(name, which):
    assert np.all(np.isfinite(np.asarray(compiled_prescription(name, "grad")(jnp.asarray(SPECIAL[which])))))
