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

__all__ = [
    "PI", "FRAC_PI_2", "FRAC_PI_4", "LN_2", "EGAMMA", "DBL_MIN", "DBL_EPSILON", "NAN", "INF",
    "MSUN_SI", "MTSUN_SI", "MRSUN_SI", "PC_SI", "MPC_SI", "G_SI", "C_SI", "REAL4_EPS",
    "fma", "fms", "fnms", "Struct", "powers", "POWERS_OF_LALPI", "safe_div", "safe_sqrt",
    "safe_log", "safe_arccos", "where", "dot", "cross", "norm", "rotate_y", "rotate_z",
    "hz_to_mf", "mf_to_hz", "materialized",
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


def _index(tree, i):
    return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_index_in_dim(x, i, keepdims=False), tree)


def isolated(function, *args):
    """``function(*args)``, computed in a ``jax.lax.while_loop`` of one
    iteration.

    XLA's CPU runtime executes the body of a loop thunk by thunk on the
    calling thread only when the body has few thunks or touches no buffer
    larger than 512 bytes; otherwise it dispatches the thunks to a thread
    pool, which costs microseconds per iteration. A loop whose body updates
    a large array (a record of its iterations) keeps its many small
    operations on the calling thread when they are isolated in a nested
    loop like this one. The loop is differentiable in forward mode."""
    out_shape = jax.eval_shape(function, *args)
    init = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), out_shape)
    # (behind a barrier, XLA cannot see that the loop runs once and inline it)
    first = jax.lax.optimization_barrier(jnp.asarray(True))

    def body(carry):
        return jnp.zeros((), bool), function(*args)

    return jax.lax.while_loop(lambda carry: carry[0], body, (first, init))[1]


def blocked_map(function, blocks, n_blocks, args=()):
    """``function(block, args)`` for the first ``n_blocks`` (traced) blocks
    of ``blocks`` (a pytree with a leading axis of blocks); the results of
    the other blocks are zeros. ``function`` must not close over traced
    values: they are passed in ``args``.

    The blocks run in a loop (:func:`._transforms.while_loop`), so that the
    cost is that of the blocks needed (under ``jax.vmap``, of the largest
    number of blocks of the batch). The loop is differentiable in forward
    mode (the model's reverse-mode derivatives come from its forward-mode
    Jacobian)."""
    from ._transforms import while_loop

    count = jax.tree_util.tree_leaves(blocks)[0].shape[0]
    n_blocks = jnp.minimum(jnp.asarray(n_blocks).astype(jnp.int32), count)
    out_shape = jax.eval_shape(function, _index(blocks, 0), args)
    out = jax.tree_util.tree_map(lambda s: jnp.zeros((count,) + s.shape, s.dtype), out_shape)

    def test(consts, i):
        return i < consts[2]

    def body(consts, i, out, active):
        blocks, args, _ = consts
        value = function(_index(blocks, i), args)
        # (inactive under jax.vmap: the block written back unchanged)
        value = jax.tree_util.tree_map(lambda o, v: jnp.where(active, v, _index(o, i)), out, value)
        out = jax.tree_util.tree_map(lambda o, v: jax.lax.dynamic_update_index_in_dim(o, v, i, 0), out, value)
        return i + 1, out

    return while_loop(test, body, (blocks, args, n_blocks), jnp.zeros((), jnp.int32), out)[1]


def blocked_scan(function, blocks, n_blocks, init, args=()):
    """``value, carry = function(block, carry, args)`` for the first
    ``n_blocks`` (traced) blocks of ``blocks`` in turn, from ``carry =
    init``: the values stacked (zeros for the other blocks) and the final
    carry. As :func:`blocked_map`, with a carry from block to block."""
    from ._transforms import while_loop

    count = jax.tree_util.tree_leaves(blocks)[0].shape[0]
    n_blocks = jnp.minimum(jnp.asarray(n_blocks).astype(jnp.int32), count)
    out_shape, _ = jax.eval_shape(function, _index(blocks, 0), init, args)
    out = jax.tree_util.tree_map(lambda s: jnp.zeros((count,) + s.shape, s.dtype), out_shape)

    def test(consts, small):
        return small[0] < consts[2]

    def body(consts, small, out, active):
        blocks, args, _ = consts
        i, carry = small
        value, carry = function(_index(blocks, i), carry, args)
        value = jax.tree_util.tree_map(lambda o, v: jnp.where(active, v, _index(o, i)), out, value)
        out = jax.tree_util.tree_map(lambda o, v: jax.lax.dynamic_update_index_in_dim(o, v, i, 0), out, value)
        return (i + 1, carry), out

    (_, carry), out = while_loop(test, body, (blocks, args, n_blocks), (jnp.zeros((), jnp.int32), init), out)
    return out, carry


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


def materialized(tree):
    """``jax.lax.optimization_barrier`` on every traced array of ``tree``
    (one barrier per value, as :func:`barrier`): values at many frequencies
    that several consumers read (the co-precessing modes, the factors of the
    twist) are computed once, rather than copied by XLA's fusion into each
    consumer (``h_+`` and ``h_x``, every column of a Jacobian), which costs
    both compile time and run time."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.inexact):
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
