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


# ---------------------------------------------------------------------------
# stages 2-3, 5: the stages of scalars and points against JAX
# ---------------------------------------------------------------------------


def _stack(trees):
    from waveformjax.models.xpnr.mlx._stages import combine, partition

    parts = [partition(t) for t in trees]
    leaves = []
    for k in range(len(parts[0][0])):
        a = np.stack([np.asarray(p[0][k]) for p in parts])
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        elif a.dtype == np.int64:
            a = a.astype(np.int32)
        leaves.append(mx.array(a))
    return combine(leaves, parts[0][1])


def _stack_tangents(tangent_lists, trees):
    from waveformjax.models.xpnr.mlx._stages import combine, partition

    out = []
    for k in range(len(tangent_lists[0])):
        ts = [t[k] for t in tangent_lists]
        if all(t is None for t in ts):
            out.append(None)
            continue
        shape = next(t.shape for t in ts if t is not None)
        a = np.stack([np.zeros(shape) if t is None else np.asarray(t) for t in ts])
        out.append(mx.array(np.moveaxis(a, 1, -1).astype(np.float32)))
    return combine(out, partition(trees[0])[1])


def _leaf_errors(mlx_tree, mlx_tangents, key, names):
    """Per leaf and binary: the relative error of MLX against float64, and
    JAX float32's (values, then tangents)."""
    from waveformjax.models.xpnr.mlx._kernels import _flatten_like
    from waveformjax.models.xpnr.mlx._stages import partition

    values = partition(mlx_tree)[0]
    tangents = _flatten_like(mlx_tangents, mlx_tree) if mlx_tangents is not None else [None] * len(values)
    out = []
    for b, name in enumerate(names):
        r64, r32 = jax_reference(name, "float64")[key], jax_reference(name, "float32")[key]
        v64, v32 = partition(r64.tree)[0], partition(r32.tree)[0]
        for k, (m, a, c) in enumerate(zip(values, v64, v32)):
            a = np.asarray(a)
            if not np.issubdtype(a.dtype, np.floating):
                continue
            m, a, c = np.asarray(m[b], np.float64), a.astype(np.float64), np.asarray(c, np.float64)
            fin = np.isfinite(a)
            if not fin.any():
                continue
            scale = np.max(np.abs(a[fin])) or 1.0
            out.append(("value", k, name, np.max(np.abs(m[fin] - a[fin])) / scale,
                        np.max(np.abs(c[fin] - a[fin])) / scale, np.max(np.abs(m[fin] - c[fin])) / scale))
            t64, t32 = r64.tangents[k], r32.tangents[k]
            if tangents[k] is None or t64 is None:
                continue
            tm = np.asarray(tangents[k][b], np.float64)
            t64 = np.moveaxis(np.asarray(t64, np.float64), 0, -1)
            t32 = np.moveaxis(np.asarray(t32, np.float64), 0, -1) if t32 is not None else np.zeros_like(t64)
            fin = np.isfinite(t64) & np.isfinite(t32)
            if not fin.any():
                continue
            scale = np.max(np.abs(t64[fin])) or 1.0
            out.append(("tangent", k, name, np.max(np.abs(tm[fin] - t64[fin])) / scale,
                        np.max(np.abs(t32[fin] - t64[fin])) / scale, np.max(np.abs(tm[fin] - t32[fin])) / scale))
    return out


def _check(errors):
    """Most leaves as JAX float32 to rounding; every leaf within 100 times
    JAX float32's distance to float64 (or 1e-2)."""
    for kind, tol in (("value", 1e-4), ("tangent", 1e-3)):
        rows = [e for e in errors if e[0] == kind]
        if not rows:
            continue
        close = np.mean([e[5] <= tol for e in rows])
        assert close >= 0.9, (kind, close)
        bad = [e for e in rows if not e[3] <= max(100 * e[4], 1e-2)]
        assert not bad, bad[:5]


def _inputs(keys, names, dtype="float32"):
    trees, ttrees = [], []
    for key in keys:
        refs = [jax_reference(n, dtype)[key] for n in names]
        trees.append(_stack([r.tree for r in refs]))
        ttrees.append(_stack_tangents([r.tangents for r in refs], [r.tree for r in refs]))
    return trees, ttrees


@pytest.fixture(scope="module")
def built():
    from waveformjax.models.xpnr.mlx import _build

    return _build.stages(model.DEFAULT_MODES, 1e21)


@pytest.mark.parametrize("stage,inputs,outputs", [
    ("c1", ("base", "wf0", "summary"), ("prec_c1", "wf_c1", "points", "rotation", "st_range")),
    ("c2", ("prec_c1", "wf_c1", "points", "bank", "st_range"), ("prec_c2", "wf_c2", "grid", "error_c")),
])
def test_scalar_stage(built, stage, inputs, outputs):
    """A stage of the set-up from JAX float32's inputs: its values and
    derivatives."""
    names = STAGE_BINARIES
    trees, ttrees = _inputs(inputs, names)
    out, tout = getattr(built, stage).tangents(trees, ttrees, batch=len(names), n_directions=P)
    errors = []
    for i, key in enumerate(outputs):
        errors += _leaf_errors(out[i], tout[i], key, names)
    _check(errors)


def binary_tangents(leaves):
    """The tangents of the fields of the binaries along their twelve
    parameters."""
    eye = np.eye(P, dtype=np.float32)
    columns = {"mass_1": eye[0], "mass_2": eye[1], "spin_1": eye[2:5], "spin_2": eye[5:8],
               "luminosity_distance": eye[8], "inclination": eye[9], "phi_ref": eye[10], "f_ref": eye[11]}
    batch = leaves["mass_1"].shape[0]
    return {k: mx.broadcast_to(mx.array(v), (batch,) + v.shape) for k, v in columns.items()}


def test_stage_a(built):
    """The common set-up from the binaries."""
    names = STAGE_BINARIES
    leaves = {k: mx.array(np.stack([np.asarray(getattr(BINARIES[n], k), np.float32) for n in names]))
              for k in _stages.BINARY_FIELDS}
    f = [frequencies_for(BINARIES[n]) for n in names]
    f0 = mx.array(np.array([x[0] for x in f], np.float32))
    f1 = mx.array(np.array([x[-1] for x in f], np.float32))
    out, tout = built.a.tangents((leaves, f0, f1), (binary_tangents(leaves), None, None),
                                 batch=len(names), n_directions=P)
    errors = []
    for i, key in enumerate(("base", "wf0", "ode", "error_a")):
        errors += _leaf_errors(out[i], tout[i], key, names)
    _check(errors)


def test_stage_e(built):
    """The offsets, the remapped J frame and the coefficients of the modes."""
    names = STAGE_BINARIES
    trees, ttrees = _inputs(("prec_c2", "wf_c2", "at_ref"), names)
    at_ref, dat_ref = trees.pop(), ttrees.pop()
    out, tout = built.e.tangents(trees + [at_ref[0], at_ref[1]], ttrees + [dat_ref[0], None], batch=len(names),
                                 n_directions=P)
    errors = []
    for i, key in enumerate(("prec_e", "wf_e", "extra", "error_e")):
        errors += _leaf_errors(out[i], tout[i], key, names)
    _check(errors)
