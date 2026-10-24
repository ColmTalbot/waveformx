# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXAS under ``jax.jit``, ``jax.vmap``, ``jax.grad`` and
``jax.jvp``: the gradients of a fixed real function of the polarizations
against central finite differences, for every parameter of the binary (the
in-plane spins do not enter), in float64 and float32."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from waveformx import Binary, WaveformOptions
from waveformx.models.xas import MODEL

OPTIONS = WaveformOptions()
FREQUENCIES = np.geomspace(20.0, 900.0, 300)
BINARIES = {
    "generic": [36.0, 29.0, 0.0, 0.0, 0.3, 0.0, 0.0, -0.4, 500.0, 0.7, 1.2, 25.0],
    "unequal": [50.0, 2.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3, 300.0, 2.1, 0.3, 30.0],
    # (f_ref = 0: the first frequency)
    "heavy": [150.0, 100.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.95, 3000.0, 0.4, 2.0, 0.0],
}
_WEIGHTS = np.random.default_rng(5).normal(size=(4, len(FREQUENCIES)))


def binary_of(x):
    return Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                  inclination=x[9], phi_ref=x[10], f_ref=x[11])


def loss(x):
    hp, hc = MODEL.polarizations(binary_of(x), jnp.asarray(FREQUENCIES, x.dtype), OPTIONS)
    w = jnp.asarray(_WEIGHTS, x.dtype)
    return jnp.sum(w[0] * hp.real + w[1] * hp.imag + w[2] * hc.real + w[3] * hc.imag)


@pytest.mark.parametrize("name", list(BINARIES))
def test_gradient(name):
    x = jnp.asarray(BINARIES[name])
    grad = np.asarray(jax.jit(jax.grad(loss))(x))
    assert np.all(grad[[2, 3, 5, 6]] == 0.0)
    value = jax.jit(loss)
    for k in range(12):
        if k in (2, 3, 5, 6) or (k == 11 and x[11] == 0):
            continue
        step = 1e-6 * max(abs(float(x[k])), 1.0)
        e = jnp.zeros(12).at[k].set(step)
        fd = (float(value(x + e)) - float(value(x - e))) / (2 * step)
        assert grad[k] == pytest.approx(fd, rel=1e-5, abs=1e-6 * np.max(np.abs(grad))), k
    # forward mode
    v = np.random.default_rng(1).normal(size=12)
    _, tangent = jax.jvp(loss, (x,), (jnp.asarray(v),))
    assert float(tangent) == pytest.approx(float(grad @ v), rel=1e-10)


def test_float32_gradient():
    x = jnp.asarray(BINARIES["generic"])
    g64 = np.asarray(jax.grad(loss)(x))
    g32 = np.asarray(jax.grad(loss)(x.astype(jnp.float32)))
    assert g32.dtype == np.float32
    assert np.all(np.abs(g32 - g64) <= 1e-2 * np.abs(g64) + 1e-3 * np.max(np.abs(g64)))


def test_vmap():
    xs = jnp.asarray(list(BINARIES.values()))
    batched = jax.jit(jax.vmap(lambda x: MODEL.polarizations(binary_of(x), jnp.asarray(FREQUENCIES), OPTIONS)))(xs)
    for i, x in enumerate(xs):
        single = MODEL.polarizations(binary_of(x), jnp.asarray(FREQUENCIES), OPTIONS)
        for a, b in zip(batched, single):
            np.testing.assert_allclose(np.asarray(a[i]), np.asarray(b), rtol=1e-11, atol=1e-11 * np.abs(b).max())
    grads = jax.vmap(jax.grad(loss))(xs)
    assert np.all(np.isfinite(np.asarray(grads)))
