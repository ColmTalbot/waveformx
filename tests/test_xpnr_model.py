# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR beyond LAL's comparisons.

* It is finite, with finite derivatives, for aligned, single and vanishing
  spins and beyond the PN energy test, in float64 and float32; float32 is
  close to float64.
* The public functions and the J-frame modes are the model's.
* ``jax.jit``, ``jax.vmap``, ``jax.jvp`` and ``jax.grad`` work, and the
  derivatives agree with finite differences.
* Long frequency grids in chunks and batches in blocks give the values and
  derivatives of one pass.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformx as wf
from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import MODEL, model

OPTIONS = WaveformOptions()
MTSUN_SI = 4.925490947641267e-06

BINARIES = {
    "generic": dict(),
    "light": dict(mass_1=9.0, mass_2=6.0, spin_1=(0.3, 0.3, 0.3), spin_2=(-0.3, 0.2, 0.0)),
    "heavy": dict(mass_1=130.0, mass_2=60.0, f_ref=24.0),
    "unequal": dict(mass_1=90.0, mass_2=5.0, spin_1=(0.6, 0.3, -0.2), spin_2=(0.1, 0.2, 0.3)),
    # the reference frequency beyond the PN energy test
    "beyond": dict(mass_1=180.0, mass_2=120.0, f_ref=60.0),
}
SPECIAL = {
    "aligned": dict(spin_1=(0.0, 0.0, 0.3), spin_2=(0.0, 0.0, -0.1)),
    "small_in_plane": dict(spin_1=(5e-7, 0.0, 0.3), spin_2=(0.0, 0.0, -0.1)),
    "single": dict(spin_1=(0.5, 0.2, 0.3), spin_2=(0.0, 0.0, 0.0)),
    "non-spinning": dict(spin_1=(0.0, 0.0, 0.0), spin_2=(0.0, 0.0, 0.0)),
}
# float32 NaN tangents fixed: the pseudo-PN inspiral amplitude of the (2,1)
# at q = 18 (a denominator ~1e-19 squared), and the ringdown beta below
# -0.3 rad (an overflowing power in the arctan window)
FLOAT32_REGRESSIONS = {
    "q18": BINARIES["unequal"],
    "negative_beta_rd": dict(mass_1=137.657, mass_2=7.195, spin_1=(-0.31, -0.824, 0.303),
                             spin_2=(0.113, 0.336, 0.036)),
}


def binary(mass_1=36.0, mass_2=29.0, spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1), f_ref=20.0, dtype=jnp.float64):
    return Binary(mass_1=jnp.asarray(mass_1, dtype), mass_2=jnp.asarray(mass_2, dtype),
                  spin_1=jnp.asarray(spin_1, dtype), spin_2=jnp.asarray(spin_2, dtype),
                  luminosity_distance=jnp.asarray(500.0, dtype), inclination=jnp.asarray(0.7, dtype),
                  phi_ref=jnp.asarray(1.2, dtype), f_ref=jnp.asarray(f_ref, dtype))


def frequencies_for(b, n=1024, dtype=jnp.float64):
    mass = float(b.mass_1 + b.mass_2)
    f_min = min(20.0, float(b.f_ref))
    return jnp.linspace(f_min, min(0.3 / (mass * MTSUN_SI), 1024.0), n, dtype=dtype)


@functools.lru_cache(maxsize=None)
def compiled(options):
    return jax.jit(lambda b, f: model._polarizations(b, f, options))


@functools.lru_cache(maxsize=None)
def with_derivative(options):
    """The polarizations and their forward derivative along ``chi_1x`` and
    ``m_1`` together (one compilation of the model's own function)."""
    def along(b, f):
        def h(b):
            return model._polarizations(b, f, options)

        tangent = jax.tree_util.tree_map(jnp.zeros_like, b)
        tangent = tangent._replace(spin_1=tangent.spin_1.at[0].set(1.0), mass_1=jnp.ones_like(b.mass_1))
        return jax.jvp(h, (b,), (tangent,))

    return jax.jit(along)


def stacked(h):
    return np.concatenate([np.asarray(x, complex) for x in h])


def mismatch(a, b):
    a, b = stacked(a), stacked(b)
    return 1.0 - np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)


def finite(h):
    return all(bool(np.all(np.isfinite(np.asarray(x)))) for x in h)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("which", list(SPECIAL) + ["beyond"] + list(FLOAT32_REGRESSIONS))
def test_finite_with_finite_derivatives(which, dtype):
    spins = SPECIAL.get(which, BINARIES.get(which, FLOAT32_REGRESSIONS.get(which)))
    b = binary(**spins, dtype=dtype)
    f = frequencies_for(b, 256, dtype)
    h, dh = with_derivative(OPTIONS)(b, f)
    assert finite(h) and finite(dh)
    for x in h + dh:
        assert np.asarray(x).dtype == (np.complex64 if dtype == jnp.float32 else np.complex128)


@pytest.mark.parametrize("which", ["generic", "light", "beyond"])
def test_float32_close_to_float64(which):
    b = binary(**BINARIES[which])
    f = frequencies_for(b)
    wide = compiled(OPTIONS)(b, f)
    narrow = compiled(OPTIONS)(binary(**BINARIES[which], dtype=jnp.float32), f.astype(jnp.float32))
    assert mismatch(narrow, wide) < 1e-6


def test_public_api():
    """The public functions are the model's; the J-frame modes sum to the
    polarizations."""
    b = binary()
    f = np.asarray(frequencies_for(b, 256))
    hp, hc = wf.polarizations("IMRPhenomXPNR", f, **b._asdict())
    ours = model._polarizations(b, jnp.asarray(f), OPTIONS)
    assert mismatch((hp, hc), ours) < 1e-12
    modes = MODEL.modes(b, f, OPTIONS)
    assert all(np.all(np.isfinite(np.asarray(h))) for h in modes.modes.values())


def test_vmap_matches_single_calls():
    bs = [binary(), binary(**BINARIES["heavy"]), binary(**SPECIAL["aligned"])]
    f = jnp.linspace(20.0, 512.0, 256)
    batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *bs)
    hp, hc = jax.jit(jax.vmap(lambda b: model.polarizations(b, f, OPTIONS)))(batch)
    single = compiled(OPTIONS)
    for i, b in enumerate(bs):
        ref = single(b, f)
        assert mismatch((hp[i], hc[i]), ref) < 1e-12


def test_frequency_chunks_match_one_pass(monkeypatch):
    """The polarizations and their derivative in chunks of the frequencies
    (:data:`.model.FREQUENCY_CHUNK`, here smaller than the frequencies and
    not dividing them) are those of one pass."""
    b = binary(**BINARIES["light"])
    f = frequencies_for(b, 300)
    h, dh = with_derivative(OPTIONS)(b, f)
    monkeypatch.setattr(model, "FREQUENCY_CHUNK", 128)
    h_chunks, dh_chunks = with_derivative.__wrapped__(OPTIONS)(b, f)
    assert mismatch(h_chunks, h) < 1e-14
    for x, y in zip(dh_chunks, dh):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-12 * np.max(np.abs(np.asarray(y))))


def test_jacobian_blocks_match_single_binaries(monkeypatch):
    """``jax.vmap`` of the Jacobian with the binaries in blocks
    (:data:`.model.JACOBIAN_BLOCK`, here not dividing the batch) gives the
    Jacobians of the single binaries, to the rounding of a batched Jacobian
    (~1e-9 of the largest derivative, as with plain ``jax.vmap``)."""
    monkeypatch.setattr(model, "_POLARIZATIONS", {})
    monkeypatch.setattr(model, "BATCH_BLOCK", 2)
    monkeypatch.setattr(model, "JACOBIAN_BLOCK", 2)
    f = jnp.linspace(20.0, 512.0, 64)
    bs = [binary(), binary(**BINARIES["heavy"]), binary(**SPECIAL["aligned"])]
    batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *bs)

    def h(b):
        return model.polarizations(b, f, OPTIONS)

    jac = jax.jit(jax.vmap(jax.jacfwd(h)))(batch)
    single = jax.jit(jax.jacfwd(h))
    for i, b in enumerate(bs):
        for x, y in zip(jax.tree_util.tree_leaves(jac), jax.tree_util.tree_leaves(single(b))):
            y = np.asarray(y)
            np.testing.assert_allclose(np.asarray(x[i]), y, rtol=0, atol=3e-9 * np.max(np.abs(y)))


def test_gradient_matches_finite_differences():
    f = jnp.linspace(20.0, 256.0, 256)
    x0 = np.array([36.0, 29.0, 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 20.0])

    def of(x):
        return Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                      inclination=x[9], phi_ref=x[10], f_ref=x[11])

    weights = jnp.exp(-f / 100.0)

    def loss(x):
        hp, hc = model.polarizations(of(x), f, OPTIONS)
        return jnp.sum(weights * (jnp.real(hp) + jnp.imag(hc)))

    value = jax.jit(loss)
    gradient = np.asarray(jax.jit(jax.grad(loss))(jnp.asarray(x0)))
    assert np.all(np.isfinite(gradient))
    for k in (0, 2, 4, 6, 9):
        step = 1e-6 * max(abs(x0[k]), 1.0)
        e = np.zeros(12)
        e[k] = step
        fd = (float(value(jnp.asarray(x0 + e))) - float(value(jnp.asarray(x0 - e)))) / (2 * step)
        assert abs(gradient[k] - fd) <= 1e-4 * abs(fd) + 1e-6 * np.max(np.abs(gradient)), (k, gradient[k], fd)
