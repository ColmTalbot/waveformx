# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""A :class:`._program.Program` evaluated with MLX operations.

Each scalar of the program is an MLX array (of the shape of the batch, or
of the batch and the points: the inputs broadcast against each other), so
that MLX's automatic differentiation (``mx.grad``, ``mx.vjp``) and
``mx.compile`` apply to it.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ._program import BOOL, FLOAT, INT, Complex, Var

__all__ = ["run", "scalar"]

_MX = {FLOAT: mx.float32, INT: mx.int32, BOOL: mx.bool_}


def scalar(x):
    """A constant of a program as a Python number."""
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    return float(x)


def _div(a, b, kind):
    if kind == INT:
        q = mx.floor_divide(a, b)
        fix = mx.logical_and(mx.not_equal(mx.remainder(a, b), 0), mx.not_equal(mx.less(a, 0), mx.less(b, 0)))
        return mx.where(fix, q + 1, q)
    return mx.divide(a, b)


def _cbrt(x, kind):
    # (exp(log|x| / 3) and one Newton step, whose derivative is exact at the
    # root; sign restored)
    ax = mx.abs(x)
    y = mx.exp(mx.log(ax) / 3.0)
    y = y - (y * y * y - ax) / (3.0 * y * y)
    y = mx.where(ax == 0, 0.0, mx.where(mx.isinf(ax), ax, y))
    return mx.where(x < 0, -y, y)


def _round_away(x, kind):
    return mx.sign(x) * mx.floor(mx.abs(x) + 0.5)


def _rem(a, b, kind):
    # C's fmod: the sign of the dividend
    if kind == INT:
        r = mx.remainder(a, b)
        return mx.where(mx.logical_and(r != 0, mx.less(a, 0) != mx.less(b, 0)), r - b, r)
    return a - b * _trunc(a / b)


def _trunc(x):
    return mx.where(x < 0, mx.ceil(x), mx.floor(x))


def _logical(logical, bitwise):
    def op(a, b, kind):
        return logical(a, b) if kind == BOOL else bitwise(a, b)
    return op


def _not(a, kind):
    return mx.logical_not(a) if kind == BOOL else mx.bitwise_invert(a)


def _convert(a, kind):
    if isinstance(a, mx.array):
        return a.astype(_MX[kind])
    return mx.array(a, dtype=_MX[kind])


def _bitcast(a, kind):
    return mx.view(a, _MX[kind])


def _unary(fn):
    return lambda a, kind: fn(a)


def _binary(fn):
    return lambda a, b, kind: fn(a, b)


OPS = {
    "add": _binary(mx.add), "sub": _binary(mx.subtract), "mul": _binary(mx.multiply), "div": _div,
    "neg": _unary(mx.negative), "abs": _unary(mx.abs), "sqrt": _unary(mx.sqrt), "exp": _unary(mx.exp),
    "log": _unary(mx.log), "sin": _unary(mx.sin), "cos": _unary(mx.cos), "tan": _unary(mx.tan),
    "tanh": _unary(mx.tanh), "atan": _unary(mx.arctan), "asin": _unary(mx.arcsin), "acos": _unary(mx.arccos),
    "atan2": _binary(mx.arctan2), "pow": _binary(mx.power), "cbrt": _cbrt, "floor": _unary(mx.floor),
    "ceil": _unary(mx.ceil), "round_even": _unary(mx.round), "round_away": _round_away, "sign": _unary(mx.sign),
    "is_finite": _unary(mx.isfinite), "lt": _binary(mx.less), "le": _binary(mx.less_equal),
    "gt": _binary(mx.greater), "ge": _binary(mx.greater_equal), "eq": _binary(mx.equal),
    "ne": _binary(mx.not_equal), "and": _logical(mx.logical_and, mx.bitwise_and),
    "or": _logical(mx.logical_or, mx.bitwise_or), "not": _not, "rem": _rem,
    "shift_right_arithmetic": _binary(mx.right_shift), "convert": _convert, "bitcast": _bitcast,
    "stop_gradient": _unary(mx.stop_gradient), "expm1": _unary(mx.expm1), "log1p": _unary(mx.log1p),
}


def run(program, inputs):
    """The outputs of ``program`` for ``inputs``: one list per input of the
    jaxpr, of MLX arrays (or numbers), one per element (row-major).
    Returns one list per output of the elements: MLX arrays, Python
    numbers for constants, ``(re, im)`` pairs for complex values."""
    values = [None] * len(program.instructions)
    for i, (op, args, kind) in enumerate(program.instructions):
        if op == "input":
            index, k = args[0]
            values[i] = inputs[index][k]
            continue
        resolved = [values[a.id] if isinstance(a, Var) else scalar(a) for a in args]
        if op == "select":
            values[i] = mx.where(*resolved)
        else:
            values[i] = OPS[op](*resolved, kind)

    def resolve(x):
        if isinstance(x, Var):
            return values[x.id]
        if isinstance(x, Complex):
            return (resolve(x.re), resolve(x.im))
        return scalar(x)

    return [[resolve(x) for x in out.reshape(-1)] for out in program.outputs]
