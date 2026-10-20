# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Constants and small numerical helpers shared by the modules of
IMRPhenomXPNR.

The arithmetic follows waveformrs, which writes LAL's expressions with the
fused multiply-adds that clang emits for the arm64 builds of LALSimulation.
JAX has no portable fused multiply-add, so :func:`fma` and its relatives are
plain ``a * b + c``: the results agree with LAL to rounding, not bit for bit.

Everything here works on traced values: branches are ``jnp.where`` and the
"safe" helpers keep NaNs (and NaN gradients) out of the branches that are not
taken.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.interpreters import partial_eval as _partial_eval

__all__ = [
    "PI", "FRAC_PI_2", "FRAC_PI_4", "LN_2", "EGAMMA", "DBL_MIN", "DBL_EPSILON", "NAN", "INF",
    "MSUN_SI", "MTSUN_SI", "MRSUN_SI", "PC_SI", "MPC_SI", "G_SI", "C_SI", "REAL4_EPS",
    "fma", "fms", "fnms", "Struct", "powers", "POWERS_OF_LALPI", "safe_div", "safe_sqrt",
    "safe_log", "safe_arccos", "where", "dot", "cross", "norm", "rotate_y", "rotate_z",
    "hz_to_mf", "mf_to_hz",
]

PI = float(np.pi)
FRAC_PI_2 = float(np.pi / 2)
FRAC_PI_4 = float(np.pi / 4)
LN_2 = float(np.log(2.0))
EGAMMA = 0.577215664901532860606512090082402431
DBL_MIN = float(np.finfo(np.float64).tiny)
DBL_EPSILON = float(np.finfo(np.float64).eps)
NAN = float("nan")
INF = float("inf")

# LAL's constants (lal 7.7)
PC_SI = 3.085677581491367278913937957796471611e16
MPC_SI = 1.0e6 * PC_SI
MSUN_SI = 1.988409870698050731911960804878414216e30
MRSUN_SI = 1.476625038050124729627979840144936351e3
MTSUN_SI = 4.925490947641266978197229498498379006e-6
G_SI = 6.67430e-11
C_SI = 299792458.0
REAL4_EPS = 1.1920928955078125e-07


def fma(a, b, c):
    """``a * b + c`` (fused in LAL's arm64 build; rounded twice here)."""
    return a * b + c


def fms(a, b, c):
    """``a * b - c``."""
    return a * b - c


def fnms(a, b, c):
    """``c - a * b``."""
    return c - a * b


@jax.tree_util.register_pytree_node_class
class Struct:
    """A record of named values (LAL's structs), a JAX pytree.

    Attributes can be set and read like those of a C struct; ``replace``
    returns a modified copy. Leaves are the values, in the order of the
    sorted names.
    """

    def __init__(self, **fields):
        object.__setattr__(self, "_fields", dict(fields))

    def __getattr__(self, name):
        try:
            return self.__dict__["_fields"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self._fields[name] = value

    def __contains__(self, name):
        return name in self._fields

    def keys(self):
        return self._fields.keys()

    def items(self):
        return self._fields.items()

    def replace(self, **fields):
        new = dict(self._fields)
        new.update(fields)
        return Struct(**new)

    def copy(self):
        return Struct(**self._fields)

    def __repr__(self):
        return "Struct(" + ", ".join(f"{k}={v!r}" for k, v in self._fields.items()) + ")"

    def tree_flatten(self):
        names = tuple(sorted(self._fields))
        return [self._fields[name] for name in names], names

    @classmethod
    def tree_unflatten(cls, names, values):
        return cls(**dict(zip(names, values)))


def powers(number):
    """``IMRPhenomX_Initialize_Powers`` (``IMRPhenomX_UsefulPowers``): the
    powers of ``number`` used by the IMRPhenomX ansatzes, from one
    ``cbrt`` (waveformrs's ``UsefulPowers::fast``; LAL derives them from
    ``pow(x, 1/6)``, the same to a few ulp)."""
    number = jnp.asarray(number)
    one_third = jnp.cbrt(number)
    sixth = jnp.sqrt(one_third)
    m_one = 1.0 / number
    log = jnp.log(number)
    two_thirds = one_third * one_third
    m_one_third = two_thirds * m_one
    m_one_sixth = sixth * m_one_third
    four_thirds = two_thirds * two_thirds
    five_thirds = four_thirds * one_third
    seven_thirds = four_thirds * number
    eight_thirds = seven_thirds * one_third
    m_two_thirds = m_one_third * m_one_third
    m_four_thirds = m_two_thirds * m_two_thirds
    m_five_thirds = m_four_thirds * m_one_third
    m_seven_thirds = m_four_thirds * m_one
    m_eight_thirds = m_seven_thirds * m_one_third
    two = number * number
    three = two * number
    m_two = m_one * m_one
    m_three = m_two * m_one
    m_four = m_two * m_two
    return Struct(
        one_sixth=sixth,
        m_one_sixth=m_one_sixth,
        itself=number,
        m_one=m_one,
        one_third=one_third,
        m_one_third=m_one_third,
        two_thirds=two_thirds,
        m_two_thirds=m_two_thirds,
        four_thirds=four_thirds,
        m_four_thirds=m_four_thirds,
        five_thirds=five_thirds,
        m_five_thirds=m_five_thirds,
        seven_thirds=seven_thirds,
        m_seven_thirds=m_seven_thirds,
        eight_thirds=eight_thirds,
        m_eight_thirds=m_eight_thirds,
        two=two,
        three=three,
        four=two * two,
        five=three * two,
        m_two=m_two,
        m_three=m_three,
        m_four=m_four,
        m_five=m_three * m_two,
        seven_sixths=sixth * number,
        m_seven_sixths=m_one_sixth * m_one,
        log=log,
        sqrt=sixth * one_third,
    )


class _LazyPi:
    """``powers_of_lalpi`` with Python floats (computed with numpy)."""

    def __getattr__(self, name):
        with jax.ensure_compile_time_eval():
            value = getattr(powers(np.float64(PI)), name)
        value = float(value)
        setattr(self, name, value)
        return value


POWERS_OF_LALPI = _LazyPi()


def where(condition, x, y):
    """``jnp.where``."""
    return jnp.where(condition, x, y)


def safe_div(numerator, denominator, fallback=1.0):
    """``numerator / denominator`` where the denominator is non-zero; the
    division is done with ``fallback`` in its place elsewhere (and the result
    is then the division by ``fallback``), so that neither branch gives NaN
    or infinite gradients."""
    ok = denominator != 0
    return numerator / jnp.where(ok, denominator, fallback)


def safe_sqrt(x, floor=0.0):
    """``sqrt(x)`` for ``x > floor``, ``sqrt(floor)`` below: finite gradients
    where the argument vanishes (pass a small positive ``floor`` when the
    gradient at zero matters)."""
    return jnp.sqrt(jnp.maximum(x, floor))


def safe_log(x, fallback=1.0):
    """``log(x)`` for ``x > 0``, ``log(fallback)`` elsewhere."""
    return jnp.log(jnp.where(x > 0, x, fallback))


def safe_arccos(x):
    """``arccos`` of ``x`` clipped to ``[-1, 1]`` (finite values; the
    gradient at the end points is infinite, as for ``arccos``)."""
    return jnp.arccos(jnp.clip(x, -1.0, 1.0))


def dot(a, b):
    """``ax bx + ay by + az bz``, as written in LAL."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    """The cross product of two 3-vectors (sequences of arrays)."""
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def norm(a):
    return jnp.sqrt(dot(a, a))


def rotate_z(angle, v):
    """``IMRPhenomX_rotate_z``: ``v`` rotated by ``angle`` about ``z``."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]]


def rotate_y(angle, v):
    """``IMRPhenomX_rotate_y``: ``v`` rotated by ``angle`` about ``y``."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]


def hz_to_mf(f, m_tot):
    """``XLALSimIMRPhenomXUtilsHztoMf``."""
    return f * (MTSUN_SI * m_tot)


def mf_to_hz(mf, m_tot):
    """``XLALSimIMRPhenomXUtilsMftoHz``."""
    return mf / (MTSUN_SI * m_tot)


def _split_leaves(tree):
    """The leaves of ``tree`` in three lists: inexact arrays, other arrays,
    and the rest (Python values), with what rebuilds the tree."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    kinds = []
    for leaf in leaves:
        if isinstance(leaf, (jax.Array, np.ndarray)):
            kinds.append("f" if jnp.issubdtype(leaf.dtype, jnp.inexact) else "i")
        else:
            kinds.append("s")
    floats = [leaf for leaf, k in zip(leaves, kinds) if k == "f"]
    others = [leaf for leaf, k in zip(leaves, kinds) if k == "i"]
    static = [leaf if k == "s" else None for leaf, k in zip(leaves, kinds)]
    return floats, others, (treedef, kinds, static)


def _join_leaves(floats, others, meta):
    treedef, kinds, static = meta
    floats, others = iter(floats), iter(others)
    leaves = [next(floats) if k == "f" else next(others) if k == "i" else s for k, s in zip(kinds, static)]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _index(tree, i):
    return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_index_in_dim(x, i, keepdims=False), tree)


def _blocked_forward(fn, n_blocks, blocks, args):
    count = jax.tree_util.tree_leaves(blocks)[0].shape[0]
    out_shape = jax.eval_shape(fn, _index(blocks, 0), args)
    out = jax.tree_util.tree_map(lambda s: jnp.zeros((count,) + s.shape, s.dtype), out_shape)

    def cond(carry):
        return carry[0] < n_blocks

    def body(carry):
        i, out = carry
        value = fn(_index(blocks, i), args)
        out = jax.tree_util.tree_map(lambda o, v: jax.lax.dynamic_update_index_in_dim(o, v, i, 0), out, value)
        return i + 1, out

    return jax.lax.while_loop(cond, body, (jnp.zeros((), jnp.int32), out))[1]


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _blocked(fn, meta, n_blocks, block_floats, block_others, arg_floats, arg_others):
    def call(block, args):
        return fn(_join_leaves(*block, meta[0]), _join_leaves(*args, meta[1]))

    return _blocked_forward(call, n_blocks, (block_floats, block_others), (arg_floats, arg_others))


def _blocked_fwd(fn, meta, n_blocks, block_floats, block_others, arg_floats, arg_others):
    out = _blocked(fn, meta, n_blocks, block_floats, block_others, arg_floats, arg_others)
    return out, (n_blocks, block_floats, block_others, arg_floats, arg_others)


def _float0_like(tree):
    return jax.tree_util.tree_map(lambda x: np.zeros(np.shape(x), dtype=jax.dtypes.float0), tree)


def _blocked_bwd(fn, meta, residuals, cotangents):
    n_blocks, block_floats, block_others, arg_floats, arg_others = residuals

    def body(carry):
        i, ct_blocks, ct_args = carry
        others = _index(block_others, i)

        def local(floats, arg_floats):
            return fn(_join_leaves(floats, others, meta[0]), _join_leaves(arg_floats, arg_others, meta[1]))

        _, vjp = jax.vjp(local, _index(block_floats, i), arg_floats)
        ct_block, ct_arg = vjp(_index(cotangents, i))
        ct_blocks = jax.tree_util.tree_map(lambda o, v: jax.lax.dynamic_update_index_in_dim(o, v, i, 0), ct_blocks, ct_block)
        ct_args = jax.tree_util.tree_map(jnp.add, ct_args, ct_arg)
        return i + 1, ct_blocks, ct_args

    zeros = jax.tree_util.tree_map(jnp.zeros_like, (block_floats, arg_floats))
    _, ct_blocks, ct_args = jax.lax.while_loop(lambda c: c[0] < n_blocks, body, (jnp.zeros((), jnp.int32),) + zeros)
    return (jnp.zeros_like(n_blocks), ct_blocks, _float0_like(block_others), ct_args, _float0_like(arg_others))


_blocked.defvjp(_blocked_fwd, _blocked_bwd)


def blocked_map(function, blocks, n_blocks, args=()):
    """``function(block, args)`` for the first ``n_blocks`` (traced) blocks
    of ``blocks`` (a pytree with a leading axis of blocks); the results of
    the other blocks are zeros. ``function`` must not close over traced
    values: they are passed in ``args``.

    The blocks run in a ``jax.lax.while_loop``, so that the cost is that of
    the blocks needed (under ``jax.vmap``, of the largest number of blocks
    of the batch), with the reverse-mode derivative from ``jax.custom_vjp``
    (a second loop over the blocks, each recomputed and differentiated with
    ``jax.vjp``). Forward-mode differentiation is not available."""
    block_floats, block_others, block_meta = _split_leaves(blocks)
    arg_floats, arg_others, arg_meta = _split_leaves(args)
    count = jax.tree_util.tree_leaves(blocks)[0].shape[0]
    # the arguments that function does not use: their cotangents would be
    # zeros, and zeros multiplied by the (possibly infinite) derivatives of
    # whatever computed them (padding, unused branches) can be NaN; behind
    # stop_gradient they get none
    first = [jax.lax.dynamic_index_in_dim(x, 0, keepdims=False) for x in block_floats]
    others_first = [jax.lax.dynamic_index_in_dim(x, 0, keepdims=False) for x in block_others]
    jaxpr = jax.make_jaxpr(lambda floats: function(_join_leaves(first, others_first, block_meta),
                                                   _join_leaves(floats, arg_others, arg_meta)))(arg_floats).jaxpr
    _, used = _partial_eval.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars))
    arg_floats = [x if u else jax.lax.stop_gradient(x) for x, u in zip(arg_floats, used)]
    n_blocks = jnp.minimum(jnp.asarray(n_blocks).astype(jnp.int32), count)
    return _blocked(function, (block_meta, arg_meta), n_blocks, block_floats, block_others, arg_floats, arg_others)


def chunked_map(function, xs, n, chunk, args=()):
    """``function(xs, args)`` for arrays ``xs`` (a pytree with a leading
    axis of static length ``capacity``) of which only the first ``n``
    (traced) entries are needed: evaluated in chunks of ``chunk`` by
    :func:`blocked_map` (the entries of the skipped chunks are zeros)."""
    leaves = jax.tree_util.tree_leaves(xs)
    capacity = leaves[0].shape[0]
    n_chunks = -(-capacity // chunk)
    pad = n_chunks * chunk - capacity

    def pad_leaf(x):
        if pad:
            x = jnp.concatenate([x, jnp.broadcast_to(x[:1], (pad,) + x.shape[1:])])
        return x.reshape((n_chunks, chunk) + x.shape[1:])

    chunks = jax.tree_util.tree_map(pad_leaf, xs)
    out = blocked_map(function, chunks, jnp.ceil(n / chunk), args)
    return jax.tree_util.tree_map(lambda y: y.reshape((n_chunks * chunk,) + y.shape[2:])[:capacity], out)


def barrier(tree):
    """``jax.lax.optimization_barrier`` on the traced scalars (and small
    arrays) of ``tree``; Python values and arrays pass unchanged (the same
    objects). At the boundaries between the set-up (scalars) and the
    evaluations at many frequencies it keeps XLA from duplicating the set-up
    into every consumer, which costs compile time."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    # one barrier per value: a barrier mixing differentiated values with
    # others (constants, integers) cannot be transposed
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jax.Array) and leaf.size <= 16 and jnp.issubdtype(leaf.dtype, jnp.inexact):
            leaves[i] = jax.lax.optimization_barrier(leaf)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def fenced(function):
    """``function`` with :func:`barrier` on its results: the set-up
    functions are islands for XLA's fusion (which otherwise copies their
    scalar arithmetic, and that of their derivatives, into every kernel that
    uses it)."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        return barrier(function(*args, **kwargs))

    return wrapped


def _fenced_both(function):
    """``function`` with :func:`barrier` on its arguments and results (and
    so on the cotangents of both: its derivative is an island too)."""

    @functools.wraps(function)
    def wrapped(*args):
        return barrier(function(*barrier(args)))

    return wrapped


class FencedModule:
    """A module whose functions are :func:`fenced` (the generated fits of
    the set-up: each value is computed once, not copied into its users)."""

    def __init__(self, module):
        self._module = module
        self._cache = {}

    def __getattr__(self, name):
        try:
            return self._cache[name]
        except KeyError:
            value = getattr(self._module, name)
            if callable(value):
                value = _fenced_both(value)
            self._cache[name] = value
            return value


def sqrt_nonneg(x):
    """``sqrt(x)`` for ``x > 0`` and 0 otherwise, with a zero (rather than
    infinite) derivative at ``x = 0``: ``sqrt(1 - 4 eta)`` at equal masses."""
    positive = x > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)
