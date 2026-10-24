# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomD in MLX (``waveformx.models.phenomd.mlx``) against the JAX
model of the same tables (``waveformx.models.phenomd.core``) in float64 and
float32, and against LALSimulation. Skipped without MLX.

* The polarizations are as close to float64 as JAX float32's (mismatch),
  on non-uniform and large uniform grids, and close to LAL's.
* The Jacobian and the gradient of a log-likelihood (``mx.grad``) agree
  with JAX float64 within the bounds of test_xpnr_float32.py.
* Batches, shared or per-binary frequencies, numbers or MLX arrays give
  the same values; NaN where LAL fails.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import waveformx as wf  # noqa: E402
from waveformx.models.phenomd import core  # noqa: E402
from waveformx.models.phenomd import mlx as model_mlx  # noqa: E402
from waveformx.models.xpnr._math import precision  # noqa: E402

MTSUN_SI = 4.925490947641267e-06

BINARIES = {
    "equal": wf.Binary(30.0, 30.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 400.0, 0.4, 0.0, 20.0),
    "generic": wf.Binary(36.0, 29.0, (0.0, 0.0, 0.3), (0.0, 0.0, -0.4), 500.0, 1.0, 1.2, 30.0),
    "unequal": wf.Binary(80.0, 10.0, (0.0, 0.0, -0.7), (0.0, 0.0, 0.2), 2000.0, 2.0, 0.3, 25.0),
    "heavy": wf.Binary(150.0, 100.0, (0.0, 0.0, 0.95), (0.0, 0.0, 0.95), 3000.0, 0.7, 3.0, 15.0),
    "high_spin": wf.Binary(50.0, 2.0, (0.0, 0.0, 0.995), (0.0, 0.0, 0.3), 300.0, 1.1, 1.7, 0.0),
    "light": wf.Binary(8.0, 1.4, (0.0, 0.0, 0.9), (0.0, 0.0, 0.0), 200.0, 0.3, 5.5, 50.0),
    "bns": wf.Binary(1.42, 1.27, (0.0, 0.0, 0.05), (0.0, 0.0, -0.03), 100.0, 0.5, 1.0, 0.0),
}
PARAMETERS = ("mass_1", "mass_2", "spin_1", "spin_2", "luminosity_distance", "inclination", "phi_ref", "f_ref")


def frequencies_for(binary, n=1000, uniform=False):
    mass = float(binary.mass_1 + binary.mass_2)
    f_max = min(0.25 / (mass * MTSUN_SI), 2048.0)
    if uniform:
        # (1/256 Hz for the binary neutron star: half a million frequencies)
        n = max(n, int((f_max - 20.0) * 256)) if mass < 5.0 else n
        return np.float32(20.0 + np.arange(n) * (f_max - 20.0) / n).astype(np.float64)
    return np.float32(np.geomspace(20.0, f_max, n)).astype(np.float64)


def vector(binary):
    """The twelve parameters of a binary."""
    b = binary
    return np.array([b.mass_1, b.mass_2, *b.spin_1, *b.spin_2, b.luminosity_distance, b.inclination, b.phi_ref,
                     b.f_ref], dtype=np.float64)


def jax_polarizations(p12, f, dtype=jnp.float64):
    """``(h_+, h_x)`` of :mod:`waveformx.models.phenomd.core` from the twelve
    parameters, the strain times 1e21."""
    with precision(dtype):
        p = (p12[0], p12[1], p12[4], p12[7], p12[8], p12[9], p12[10], p12[11])
        fields = core.unpack(core.table(p, f[0], 1e21))
        return core.evaluate(fields, core.frequency_powers(f))


def mismatch(a, b):
    a, b = np.asarray(a, np.complex128), np.asarray(b, np.complex128)
    return 1 - np.vdot(a, b).real / (np.linalg.norm(a) * np.linalg.norm(b))


def references(binary, f):
    p = np.float32(vector(binary)).astype(np.float64)
    h64 = jax.jit(jax_polarizations)(jnp.asarray(p), jnp.asarray(f))
    h32 = jax.jit(lambda p, f: jax_polarizations(p, f, jnp.float32))(jnp.asarray(p, jnp.float32),
                                                                        jnp.asarray(f, jnp.float32))
    return [np.asarray(h) for h in h64], [np.asarray(h, np.complex128) for h in h32]


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("name", list(BINARIES))
def test_polarizations(name, uniform):
    binary = BINARIES[name]
    f = frequencies_for(binary, 1 << 16 if uniform else 1000, uniform)
    h64, h32 = references(binary, f)
    out = model_mlx.polarizations(binary, f)
    for h, a, b in zip(out, h64, h32):
        assert h.shape == (len(f),) and h.dtype == mx.complex64
        h = np.asarray(h, np.complex128)
        # as close to float64 as JAX float32 (within a factor), or closer
        # than 1e-8 (the set-up's fast division and logarithm)
        assert mismatch(h, a) < max(4.0 * mismatch(b, a), 1e-8), (mismatch(h, a), mismatch(b, a))
        # (the target for float32: binary neutron stars from 20 Hz too)
        assert mismatch(h, a) < 1e-6
        error = np.max(np.abs(h - a))
        assert error < max(4.0 * np.max(np.abs(b - a)), 1e-5 * np.max(np.abs(a))), (error, np.max(np.abs(b - a)))


@pytest.mark.parametrize("name", ["generic", "unequal", "heavy", "high_spin"])
def test_against_lalsimulation(lal, name):
    lal_module, lalsim = lal
    binary = BINARIES[name]
    f = frequencies_for(binary)
    vec = lal_module.CreateREAL8Vector(len(f))
    vec.data = f
    b = binary
    hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
        b.phi_ref, b.mass_1 * lal_module.MSUN_SI, b.mass_2 * lal_module.MSUN_SI, 0.0, 0.0, b.spin_1[2], 0.0, 0.0,
        b.spin_2[2], b.f_ref, b.luminosity_distance * 1e6 * lal_module.PC_SI, b.inclination, None,
        lalsim.IMRPhenomD, vec,
    )
    out = model_mlx.polarizations(binary, f, scaled_strain=True)
    for h, reference in zip(out, (hp.data.data, hc.data.data)):
        h = np.asarray(h, np.complex128) * 1e-21
        assert mismatch(h, reference) < 1e-5


def test_jacobian():
    for name in ("generic", "unequal", "heavy"):
        binary = BINARIES[name]
        f = frequencies_for(binary, 500)
        p = np.float32(vector(binary)).astype(np.float64)
        jac = jax.jacfwd(lambda p: jnp.stack(jax_polarizations(p, jnp.asarray(f))))(jnp.asarray(p))
        jac = np.asarray(jac)
        (hp, hc), (jp, jc) = model_mlx.polarizations_and_jacobian(binary, f)
        assert jp.shape == (len(f), 12) and jc.shape == (len(f), 12)
        ours = np.stack([np.asarray(jp, np.complex128), np.asarray(jc, np.complex128)])
        for k in range(12):
            norm = np.linalg.norm(jac[..., k])
            error = np.linalg.norm(ours[..., k] - jac[..., k])
            if norm == 0:
                assert error == 0
            else:
                assert error < 1e-2 * norm, (name, k, error / norm)


def test_gradient():
    """``mx.grad`` of a noise-free log-likelihood (the data those of
    ``mass_1`` 1% larger) against JAX float64."""
    for name in ("generic", "high_spin", "light"):
        binary = BINARIES[name]
        f = frequencies_for(binary, 800)
        p = np.float32(vector(binary)).astype(np.float64)
        shifted = p * np.where(np.arange(12) == 0, 1.01, 1.0)
        data = [np.asarray(h) for h in jax_polarizations(jnp.asarray(shifted), jnp.asarray(f))]

        def loss_jax(p):
            hp, hc = jax_polarizations(p, jnp.asarray(f))
            return -0.5 * jnp.sum(jnp.abs(hp - data[0]) ** 2 + jnp.abs(hc - data[1]) ** 2)

        expected = np.asarray(jax.grad(loss_jax)(jnp.asarray(p)))
        d = [mx.array(x.astype(np.complex64)) for x in data]

        def loss_mlx(*fields):
            hp, hc = model_mlx.polarizations(wf.Binary(*fields), f)
            return -0.5 * mx.sum(mx.abs(hp - d[0]) ** 2 + mx.abs(hc - d[1]) ** 2)

        fields = [mx.array(np.float32(np.asarray(getattr(binary, k)))) for k in PARAMETERS]
        grads = mx.grad(loss_mlx, argnums=tuple(range(8)))(*fields)
        got = np.concatenate([np.atleast_1d(np.asarray(g, np.float64)) for g in grads])
        bound = 1e-2 * np.abs(expected) + 1e-3 * np.max(np.abs(expected))
        assert np.all(np.abs(got - expected) <= bound), (name, got, expected)


def test_batch():
    names = list(BINARIES)
    f = np.geomspace(20.0, 1024.0, 300)
    rows = [vector(BINARIES[n]) for n in names]
    stacked = np.array(rows, dtype=np.float32)
    batch = wf.Binary(stacked[:, 0], stacked[:, 1], stacked[:, 2:5], stacked[:, 5:8], stacked[:, 8], stacked[:, 9],
                      stacked[:, 10], stacked[:, 11])
    hp, hc = model_mlx.polarizations(batch, f)
    assert hp.shape == (len(names), len(f))
    # the same as one at a time, with numbers or MLX arrays, with shared or
    # per-binary frequencies
    mx_batch = wf.Binary(*(mx.array(np.asarray(v)) for v in batch))
    hp2, hc2 = model_mlx.polarizations(mx_batch, mx.array(np.tile(np.float32(f), (len(names), 1))))
    for i, n in enumerate(names):
        one = model_mlx.polarizations(BINARIES[n], f)
        for a, b, c in zip(one, (hp[i], hc[i]), (hp2[i], hc2[i])):
            np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=0, atol=1e-6 * np.max(np.abs(np.asarray(a))))
            np.testing.assert_allclose(np.asarray(c), np.asarray(a), rtol=0, atol=1e-6 * np.max(np.abs(np.asarray(a))))


def test_strain_scale():
    binary = BINARIES["generic"]
    f = frequencies_for(binary, 400)
    scaled = np.asarray(model_mlx.polarizations(binary, f)[0], np.complex128)
    plain = np.asarray(model_mlx.polarizations(binary, f, scaled_strain=False)[0], np.complex128)
    assert np.max(np.abs(plain * 1e21 - scaled)) < 1e-4 * np.max(np.abs(scaled))


def test_nan_where_lal_fails():
    """A final spin above one (spins beyond one, which the public functions
    refuse)."""
    f = np.geomspace(20.0, 1024.0, 100)
    binary = BINARIES["generic"]._replace(spin_1=(0.0, 0.0, 1.5), spin_2=(0.0, 0.0, 1.5))
    hp, hc = model_mlx.polarizations(binary, f)
    assert np.all(np.isnan(np.asarray(hp))) and np.all(np.isnan(np.asarray(hc)))
