# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXHM in MLX (``waveformx.models.xhm.mlx``) against the JAX
model of the same tables (``waveformx.models.xhm``) in float64 and float32.
Skipped without MLX.

* The polarizations and the modes are as close to float64 as JAX float32's
  (mismatch), on non-uniform and large uniform grids, binary neutron stars
  among the binaries.
* The Jacobian and the gradients of log-likelihoods of the polarizations
  and of a mode (``mx.grad``) agree with JAX float64 within the bounds of
  test_xpnr_float32.py.
* Batches and shared or per-binary frequencies give the same values; NaN
  where LAL fails.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import waveformx as wf  # noqa: E402
from waveformx.models.xhm import mlx as xhm_mlx  # noqa: E402
from waveformx.models.xhm import model  # noqa: E402

MTSUN_SI = 4.925490947641267e-06
OPTIONS = wf.WaveformOptions()

BINARIES = {
    "generic": wf.Binary(36.0, 29.0, (0.0, 0.0, 0.3), (0.0, 0.0, -0.4), 500.0, 1.0, 1.2, 30.0),
    "unequal": wf.Binary(80.0, 10.0, (0.0, 0.0, -0.7), (0.0, 0.0, 0.2), 2000.0, 2.0, 0.3, 25.0),
    "heavy": wf.Binary(150.0, 100.0, (0.0, 0.0, 0.95), (0.0, 0.0, 0.95), 3000.0, 0.7, 3.0, 15.0),
    "high_spin": wf.Binary(50.0, 2.0, (0.0, 0.0, 0.995), (0.0, 0.0, 0.3), 300.0, 1.1, 1.7, 0.0),
    "equal": wf.Binary(30.0, 30.0, (0.0, 0.0, 0.2), (0.0, 0.0, 0.2), 400.0, 0.4, 0.0, 20.0),
    "bns": wf.Binary(1.42, 1.27, (0.0, 0.0, 0.05), (0.0, 0.0, -0.03), 100.0, 0.5, 1.0, 0.0),
}
PARAMETERS = ("mass_1", "mass_2", "spin_1", "spin_2", "luminosity_distance", "inclination", "phi_ref", "f_ref")


def frequencies_for(binary, uniform=False):
    mass = float(binary.mass_1 + binary.mass_2)
    f_max = min(0.3 / (mass * MTSUN_SI) * 1.05, 2048.0)
    if uniform:
        n = int((f_max - 20.0) * 256) if mass < 5.0 else 1 << 16
        return np.float32(20.0 + np.arange(n) * (f_max - 20.0) / n).astype(np.float64)
    return np.float32(np.geomspace(20.0, f_max, 1000)).astype(np.float64)


def rounded(binary, dtype):
    return wf.Binary(*(jnp.asarray(np.float32(np.asarray(v, float)), dtype) for v in binary))


def mismatch(a, b):
    a, b = np.asarray(a, np.complex128), np.asarray(b, np.complex128)
    return 1 - np.vdot(a, b).real / (np.linalg.norm(a) * np.linalg.norm(b))


_jax = jax.jit(lambda b, f: model.modes_and_polarizations(b, f, OPTIONS))


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("name", list(BINARIES))
def test_polarizations_and_modes(name, uniform):
    binary = BINARIES[name]
    f = frequencies_for(binary, uniform)
    m64, p64 = _jax(rounded(binary, jnp.float64), jnp.asarray(f))
    m32, p32 = _jax(rounded(binary, jnp.float32), jnp.asarray(f, jnp.float32))
    hp, hc = xhm_mlx.polarizations(binary, f)
    for h, a, b in zip((hp, hc), p64, p32):
        assert h.shape == (len(f),) and h.dtype == mx.complex64
        assert mismatch(h, a) < max(4.0 * mismatch(b, a), 1e-8), (mismatch(h, a), mismatch(b, a))
        assert mismatch(h, a) < 1e-6
    modes = xhm_mlx.modes(binary, f)
    for mode, a in m64.items():
        a = np.asarray(a)
        if not np.any(a):
            assert not np.any(np.asarray(modes[mode]))
            continue
        b = np.asarray(m32[mode], np.complex128)
        assert mismatch(modes[mode], a) < max(4.0 * mismatch(b, a), 1e-7), (mode, mismatch(modes[mode], a))


def _vector(binary):
    b = binary
    return np.array([b.mass_1, b.mass_2, *b.spin_1, *b.spin_2, b.luminosity_distance, b.inclination, b.phi_ref,
                     b.f_ref], dtype=np.float64)


def _jax_outputs(p12, f):
    b = wf.Binary(p12[0], p12[1], p12[2:5], p12[5:8], p12[8], p12[9], p12[10], p12[11])
    modes, (hp, hc) = model.modes_and_polarizations(b, f, OPTIONS)
    return jnp.stack([hp, hc, modes[(3, 2)], modes[(2, 1)]])


def test_jacobian_and_gradient():
    for name in ("generic", "high_spin", "heavy"):
        binary = BINARIES[name]
        f = np.float32(np.geomspace(20.0, 1024.0, 400)).astype(np.float64)
        p = np.float32(_vector(binary)).astype(np.float64)
        jac = np.asarray(jax.jacfwd(_jax_outputs)(jnp.asarray(p), jnp.asarray(f)))[:2]
        (_, _), (jp, jc) = xhm_mlx.polarizations_and_jacobian(binary, f)
        ours = np.stack([np.asarray(jp, np.complex128), np.asarray(jc, np.complex128)])
        for k in range(12):
            norm = np.linalg.norm(jac[..., k])
            error = np.linalg.norm(ours[..., k] - jac[..., k])
            assert (error == 0) if norm == 0 else (error < 1e-2 * norm), (name, k)

        shifted = p * np.where(np.arange(12) == 0, 1.01, 1.0)
        data = np.asarray(_jax_outputs(jnp.asarray(shifted), jnp.asarray(f)))

        def loss_jax(p):
            return -0.5 * jnp.sum(jnp.abs(_jax_outputs(p, jnp.asarray(f)) - data) ** 2)

        expected = np.asarray(jax.grad(loss_jax)(jnp.asarray(p)))
        d = [mx.array(x.astype(np.complex64)) for x in data]

        def loss_mlx(*fields):
            b = wf.Binary(*fields)
            hp, hc = xhm_mlx.polarizations(b, f)
            modes = xhm_mlx.modes(b, f)
            return -0.5 * mx.sum(mx.abs(hp - d[0]) ** 2 + mx.abs(hc - d[1]) ** 2 + mx.abs(modes[(3, 2)] - d[2]) ** 2
                                 + mx.abs(modes[(2, 1)] - d[3]) ** 2)

        fields = [mx.array(np.float32(np.asarray(getattr(binary, k)))) for k in PARAMETERS]
        grads = mx.grad(loss_mlx, argnums=tuple(range(8)))(*fields)
        got = np.concatenate([np.atleast_1d(np.asarray(g, np.float64)) for g in grads])
        bound = 1e-2 * np.abs(expected) + 1e-3 * np.max(np.abs(expected))
        assert np.all(np.abs(got - expected) <= bound), (name, got, expected)


def test_batch_and_errors():
    names = list(BINARIES)
    f = np.geomspace(20.0, 1024.0, 300)
    stacked = np.array([_vector(BINARIES[n]) for n in names], dtype=np.float32)
    batch = wf.Binary(stacked[:, 0], stacked[:, 1], stacked[:, 2:5], stacked[:, 5:8], stacked[:, 8], stacked[:, 9],
                      stacked[:, 10], stacked[:, 11])
    hp, hc = xhm_mlx.polarizations(batch, f)
    mx_batch = wf.Binary(*(mx.array(np.asarray(v)) for v in batch))
    hp2, _ = xhm_mlx.polarizations(mx_batch, mx.array(np.tile(np.float32(f), (len(names), 1))))
    for i, n in enumerate(names):
        one, _ = xhm_mlx.polarizations(BINARIES[n], f)
        scale = np.max(np.abs(np.asarray(one)))
        np.testing.assert_allclose(np.asarray(hp[i]), np.asarray(one), rtol=0, atol=1e-6 * scale)
        np.testing.assert_allclose(np.asarray(hp2[i]), np.asarray(one), rtol=0, atol=1e-6 * scale)
    bad = wf.Binary(1500.0, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 400.0, 0.4, 0.0, 20.0)
    hp, _ = xhm_mlx.polarizations(bad, f)
    assert np.all(np.isnan(np.asarray(hp)))


def test_reference_mf():
    """The mass-scaled reference: the polarizations and modes against the
    JAX model (float64) with it, the Jacobian against JAX's, NaN for a
    non-zero ``f_ref``."""
    options = wf.WaveformOptions(reference_mf=0.01)

    def outputs(p12, f):
        b = wf.Binary(p12[0], p12[1], p12[2:5], p12[5:8], p12[8], p12[9], p12[10], p12[11])
        modes, (hp, hc) = model.modes_and_polarizations(b, f, options)
        return jnp.stack([hp, hc, modes[(3, 2)], modes[(2, 1)], modes[(3, 3)]])

    for name in ("generic", "high_spin", "bns"):
        binary = BINARIES[name]._replace(f_ref=0.0)
        f = np.float32(np.geomspace(20.0, min(0.3 / (float(binary.mass_1 + binary.mass_2) * MTSUN_SI), 1024.0),
                                    400)).astype(np.float64)
        p = np.float32(_vector(binary)).astype(np.float64)
        expected = np.asarray(outputs(jnp.asarray(p), jnp.asarray(f)))
        hp, hc = xhm_mlx.polarizations(binary, f, reference_mf=0.01)
        modes = xhm_mlx.modes(binary, f, reference_mf=0.01)
        # (the (3,2) of the high-spin binary is ~5e-5 from float64 with
        # LAL's reference too: float32's limit of its rotated ringdown)
        for h, a, bound in zip((hp, hc, modes[(3, 2)], modes[(2, 1)], modes[(3, 3)]), expected,
                               (1e-6, 1e-6, 1e-4, 1e-6, 1e-6)):
            assert mismatch(h, a) < bound, name
        jac = np.asarray(jax.jacfwd(outputs)(jnp.asarray(p), jnp.asarray(f)))[:2]
        (_, _), (jp, jc) = xhm_mlx.polarizations_and_jacobian(binary, f, reference_mf=0.01)
        ours = np.stack([np.asarray(jp, np.complex128), np.asarray(jc, np.complex128)])
        for k in range(12):
            norm = np.linalg.norm(jac[..., k])
            error = np.linalg.norm(ours[..., k] - jac[..., k])
            assert (error == 0) if norm == 0 else (error < 1e-2 * norm), (name, k)
        bad = xhm_mlx.polarizations(binary._replace(f_ref=20.0), f, reference_mf=0.01)[0]
        assert np.all(np.isnan(np.asarray(bad)))
