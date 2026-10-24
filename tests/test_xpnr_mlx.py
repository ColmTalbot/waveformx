# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The lean IMRPhenomXPNR in MLX (``waveformjax.models.xpnr.mlx``) against
the JAX lean model, stage by stage (the stages of
``waveformjax.models.xpnr.mlx._stages``) and end to end. Skipped without
MLX.

* The stages composed in JAX reproduce the lean model to rounding (the
  reference of everything else).
* The lowering of jaxprs to scalar programs (``_program``) and the
  generated Metal kernels (``_kernels``) reproduce JAX, values and JVPs.
* The scalar stages agree with JAX float32 from the same inputs (most
  leaves to rounding; the rest, ill-conditioned in float32, as far from
  float64 as JAX float32 is), values and derivatives.
* The SpinTaylor kernel takes JAX's steps, within float32 integration
  error, and its derivatives of the end of the inspiral match float64
  (better than JAX float32, whose Newton step uses a secant float32 cannot
  resolve).
* End to end: the polarizations are as close to float64 as JAX float32's
  (mismatch), their Jacobian and the gradient of a log-likelihood within
  the bounds of test_xpnr_float32.py; batches and shared or per-binary
  frequencies give the same values; NaN where JAX gives NaN.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import waveformjax as wf  # noqa: E402
from waveformjax.models.xpnr import lean, model  # noqa: E402
from waveformjax.models.xpnr.mlx import _stages  # noqa: E402

MTSUN_SI = 4.925490947641267e-06

BINARIES = {
    "generic": wf.Binary(36.0, 29.0, (0.4, 0.2, 0.3), (-0.2, 0.3, -0.1), 500.0, 0.7, 1.2, 20.0),
    "aligned": wf.Binary(36.0, 29.0, (0.0, 0.0, 0.3), (0.0, 0.0, -0.1), 500.0, 0.7, 1.2, 20.0),
    "unequal": wf.Binary(90.0, 5.0, (0.6, 0.3, -0.2), (0.1, 0.2, 0.3), 500.0, 0.7, 1.2, 20.0),
    "beyond": wf.Binary(180.0, 120.0, (0.4, 0.2, 0.3), (-0.2, 0.3, -0.1), 500.0, 0.7, 1.2, 60.0),
    "light": wf.Binary(9.0, 6.0, (0.3, 0.3, 0.3), (-0.3, 0.2, 0.0), 500.0, 0.7, 1.2, 20.0),
}
#: The binaries of the stage-by-stage comparisons.
STAGE_BINARIES = ("generic", "beyond", "light")
P = 12


def as_dtype(binary, dtype):
    return wf.Binary(*(jnp.asarray(value, dtype) for value in binary))


def frequencies_for(binary, n=256):
    mass = float(binary.mass_1 + binary.mass_2)
    f_min = min(20.0, float(binary.f_ref))
    return np.linspace(f_min, min(0.3 / (mass * MTSUN_SI), 1024.0), n)


def mismatch(a, b):
    a, b = np.asarray(a, np.complex128), np.asarray(b, np.complex128)
    return 1 - abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


@functools.lru_cache(maxsize=None)
def jax_reference(name, dtype_name):
    """The intermediates of the stages composed in JAX, with their
    derivatives along the twelve parameters (dicts of
    :class:`._stages.Dual`)."""
    binary = as_dtype(BINARIES[name], getattr(jnp, dtype_name))
    f = jnp.asarray(frequencies_for(BINARIES[name]), getattr(jnp, dtype_name))
    _, values = _stages.reference(binary, f, keep=True, tangents=True)
    return values


# ---------------------------------------------------------------------------
# stage 0: the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["generic", "aligned", "unequal", "beyond"])
def test_stages_reproduce_lean(name):
    """The stages composed in JAX are the lean model (float64, to
    rounding)."""
    binary = as_dtype(BINARIES[name], jnp.float64)
    f = jnp.asarray(frequencies_for(binary, 512))
    expected = jax.jit(lambda b, f: lean.polarizations(b, f))(binary, f)
    got = _stages.reference(binary, f)
    for a, b in zip(got, expected):
        a, b = np.asarray(a), np.asarray(b)
        assert np.all(np.isfinite(b))
        assert np.max(np.abs(a - b)) < 1e-12 * np.max(np.abs(b))


def test_reference_tangents():
    """The derivatives of the stages composed in JAX are the lean model's
    (``jax.jacfwd``)."""
    name = "generic"
    values = jax_reference(name, "float64")
    f = jnp.asarray(frequencies_for(BINARIES[name]))

    def of(x):
        return wf.Binary(x[0], x[1], x[2:5], x[5:8], x[8], x[9], x[10], x[11])

    x = jnp.asarray(np.concatenate([np.ravel(np.asarray(v, float)) for v in BINARIES[name]]))
    jp, _ = jax.jit(jax.jacfwd(lambda x: lean.polarizations(of(x), f)))(x)
    got = values["hp"].tangents[0]
    assert np.max(np.abs(got.T - jp)) < 1e-12 * np.max(np.abs(jp))


# ---------------------------------------------------------------------------
# stage 1: the lowering and the generated kernels
# ---------------------------------------------------------------------------


def _probe(x, y, i, z):
    """Primitives of the model: elementwise, small arrays, loops, conds,
    solves, sorts, gathers, complex numbers."""
    a = jnp.stack([x, y, x * y, jnp.sin(x)])
    m = jnp.outer(a, a) + 3.0 * jnp.eye(4)
    sol = jnp.linalg.solve(m, a)
    s = jnp.sort(a)
    k = jnp.clip(i, 0, 3)
    g = a[k] + a[jnp.argmax(a)]
    loop = jax.lax.fori_loop(0, 5, lambda j, c: c * 0.9 + jnp.cos(c + j), x)
    c = jax.lax.cond(y > 0.5, lambda v: v * 2.0, lambda v: -v, x)
    w = jnp.exp(1j * x) * (y + 2j)
    r = jnp.cbrt(z) + jnp.arctan2(y, x) + jnp.maximum(x, y) + jnp.where(x > y, x ** 3, jnp.sqrt(y))
    return (jnp.sum(sol), s, g, loop, c, jnp.real(w), jnp.imag(jnp.sqrt(w) * jnp.log(w)), r,
            jnp.round(x * 7.0), jnp.floor(y * 3.0), x % 0.3)


def test_program_and_kernel():
    """A jaxpr lowered and run as MLX operations and as a Metal kernel, and
    its JVP, agree with JAX (float32)."""
    from waveformjax.models.xpnr.mlx import _kernels, _mlxrun, _program

    rng = np.random.default_rng(1)
    x = rng.uniform(0.1, 1.0, 64).astype(np.float32)
    y = rng.uniform(0.1, 1.0, 64).astype(np.float32)
    i = rng.integers(-1, 5, 64).astype(np.int32)
    z = rng.uniform(-2.0, 2.0, 64).astype(np.float32)
    expected = jax.vmap(_probe)(x, y, i, z)
    closed = jax.make_jaxpr(_probe)(x[0], y[0], i[0], z[0])
    program = _program.lower(closed)
    outs = _mlxrun.run(program, [[mx.array(v)] for v in (x, y, i, z)])
    for got, want in zip(outs, expected):
        want = np.asarray(want, np.float64).reshape(64, -1)
        got = np.stack([np.broadcast_to(np.asarray(g, np.float64), (64,)) for g in got], axis=1)
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)
    s = jax.ShapeDtypeStruct
    kernel = _kernels.StageKernel(_probe, (s((), np.float32), s((), np.float32), s((), np.int32),
                                           s((), np.float32)), 0, "probe")
    args = [mx.array(v[None]) for v in (x, y, i, z)]
    got = kernel.values(*args, batch=1, n_points=64)
    for g, want in zip(got, expected):
        np.testing.assert_allclose(np.asarray(g[0], np.float64).reshape(64, -1),
                                   np.asarray(want, np.float64).reshape(64, -1), rtol=2e-5, atol=2e-5)
    # the JVP along x and z
    tangents = [mx.array(np.stack([np.ones(64), np.zeros(64)], -1)[None].astype(np.float32)), None, None,
                mx.array(np.stack([np.zeros(64), np.ones(64)], -1)[None].astype(np.float32))]
    _, dout = kernel.tangents(args, tangents, batch=1, n_directions=2, n_points=64)
    for k, dx, dz in ((0, 1.0, 0.0), (1, 0.0, 1.0)):
        _, want = jax.vmap(lambda *v: jax.jvp(_probe, v[:4], (jnp.float32(dx), jnp.float32(0.0), np.zeros((),
                                              jax.dtypes.float0), jnp.float32(dz))))(x, y, i, z)
        for g, w in zip(dout, want):
            if g is None:
                continue
            np.testing.assert_allclose(np.asarray(g[0, ..., k], np.float64).reshape(64, -1),
                                       np.asarray(w, np.float64).reshape(64, -1), rtol=5e-4, atol=5e-4)
