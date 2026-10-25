# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""JAX functions of scalars as straight-line programs of scalar operations.

The stages of the model (:mod:`._stages`) are functions of one
binary's scalars, or of one binary's scalars and one point: their jaxprs
hold scalars and small arrays of static shape only. :func:`lower` turns
such a jaxpr into a :class:`Program`: a list of operations on scalars
(``float32``, ``int32``, ``bool``), each array of the jaxpr becoming an
array of scalars of its shape (so that reshapes, slices, transposes,
gathers at constant indices and the like cost nothing), loops of static
length unrolled, branches evaluated both ways and selected, small linear
solves written out, complex numbers as pairs of reals, and the values that
do not depend on the inputs computed in numpy (constant folding). Equal
operations are computed once and unused ones dropped.

A program runs on MLX arrays of any (broadcastable) shapes
(:func:`._mlxrun.run`): the scalars of one binary become arrays over the
batch, or over the batch and the points, and MLX's automatic
differentiation follows the operations. The operations keep JAX's
semantics where they matter for the values or the derivatives (``max`` and
``min`` split the derivative at ties, ``clamp`` passes it strictly inside
its bounds, ``integer_pow`` multiplies in JAX's order).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Var", "Complex", "Program", "lower", "FLOAT", "INT", "BOOL"]

FLOAT, INT, BOOL = "f", "i", "b"
_NUMPY = {FLOAT: np.float32, INT: np.int32, BOOL: np.bool_}


class Var:
    """A value computed by the program: the result of instruction ``id``."""

    __slots__ = ("id",)

    def __init__(self, id_):
        self.id = id_

    def __repr__(self):
        return f"v{self.id}"


class Complex:
    """A complex scalar as a pair of real scalars (constants or
    :class:`Var`)."""

    __slots__ = ("re", "im")

    def __init__(self, re, im):
        self.re = re
        self.im = im


def kind_of(dtype):
    """The kind of scalar (:data:`FLOAT`, :data:`INT`, :data:`BOOL`) of a
    numpy dtype (complex types have the kind of their parts)."""
    dtype = np.dtype(dtype)
    if dtype.kind == "V":
        return FLOAT
    if dtype == np.bool_:
        return BOOL
    if np.issubdtype(dtype, np.integer):
        return INT
    return FLOAT


def is_const(x):
    return not isinstance(x, (Var, Complex))


def const(value, kind):
    return _NUMPY[kind](value)


def _key(x):
    if isinstance(x, Var):
        return x.id
    return (np.asarray(x).dtype.char, np.asarray(x).tobytes())


# ---------------------------------------------------------------------------
# folding: the operations in numpy (float32 arithmetic)
# ---------------------------------------------------------------------------


def _trunc_div(a, b):
    if isinstance(a, np.integer):
        if b == 0:
            return np.int32(0)
        q = abs(int(a)) // abs(int(b))
        return np.int32(q if (a >= 0) == (b >= 0) else -q)
    with np.errstate(all="ignore"):
        return a / b


def _round_even(x):
    return np.round(x)


def _round_away(x):
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def _cbrt(x):
    return np.cbrt(x)


def _max(a, b):
    if np.isnan(a) or np.isnan(b):
        return a + b
    return np.maximum(a, b)


def _min(a, b):
    if np.isnan(a) or np.isnan(b):
        return a + b
    return np.minimum(a, b)


def _logical(numpy_bool, numpy_bitwise):
    def op(a, b):
        if isinstance(a, np.bool_):
            return numpy_bool(a, b)
        return numpy_bitwise(a, b)
    return op


def _not(a):
    if isinstance(a, np.bool_):
        return np.logical_not(a)
    return np.invert(a)


def _shift_right_arithmetic(a, b):
    return np.right_shift(a, b)


FOLD = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": _trunc_div, "max": _max, "min": _min,
    "neg": np.negative, "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "tanh": np.tanh, "atan": np.arctan, "asin": np.arcsin, "acos": np.arccos, "atan2": np.arctan2,
    "pow": np.power, "cbrt": _cbrt, "floor": np.floor, "ceil": np.ceil, "round_even": _round_even,
    "round_away": _round_away, "sign": np.sign, "is_finite": np.isfinite, "lt": np.less, "le": np.less_equal,
    "gt": np.greater, "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal,
    "and": _logical(np.logical_and, np.bitwise_and), "or": _logical(np.logical_or, np.bitwise_or), "not": _not,
    "rem": np.fmod, "shift_right_arithmetic": _shift_right_arithmetic, "expm1": np.expm1, "log1p": np.log1p,
}


def _fold(op, args, kind):
    with np.errstate(all="ignore"):
        if op == "select":
            p, a, b = args
            return a if p else b
        if op == "convert":
            return const(args[0], kind) if not (kind == INT and not np.isfinite(args[0])) else np.int32(0)
        if op == "bitcast":
            return np.asarray(args[0]).view(_NUMPY[kind])[()]
        if op == "stop_gradient":
            return args[0]
        value = FOLD[op](*args)
    return const(value, kind)


# ---------------------------------------------------------------------------
# the program
# ---------------------------------------------------------------------------


@dataclass
class Program:
    """Straight-line scalar operations: ``instructions[i] = (op, args,
    kind)`` computes value ``i`` (``op == "input"`` for the inputs, with
    ``args = (index,)``); ``inputs`` are the shapes and kinds of the
    jaxpr's inputs and ``input_vars`` their scalars (object arrays of
    :class:`Var`); ``outputs`` the object arrays of the outputs (of
    :class:`Var`, constants or :class:`Complex`)."""

    instructions: list = field(default_factory=list)
    inputs: list = field(default_factory=list)
    input_vars: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    output_dtypes: list = field(default_factory=list)

    def counts(self):
        out = {}
        for op, _, _ in self.instructions:
            out[op] = out.get(op, 0) + 1
        return out

    def __len__(self):
        return len(self.instructions)


class Builder:
    """Emits the instructions of a program, folding constants and sharing
    equal operations."""

    def __init__(self):
        self.program = Program()
        self.seen = {}

    def input(self, index, kind):
        var = Var(len(self.program.instructions))
        self.program.instructions.append(("input", (index,), kind))
        return var

    def kind(self, x):
        if isinstance(x, Var):
            return self.program.instructions[x.id][2]
        return kind_of(np.asarray(x).dtype)

    def emit(self, op, args, kind):
        if all(is_const(a) for a in args):
            return _fold(op, args, kind)
        simplified = self._simplify(op, args, kind)
        if simplified is not None:
            return simplified
        key = (op, kind) + tuple(_key(a) for a in args)
        var = self.seen.get(key)
        if var is None:
            var = Var(len(self.program.instructions))
            self.program.instructions.append((op, tuple(args), kind))
            self.seen[key] = var
        return var

    def _simplify(self, op, args, kind):
        if op == "select":
            p, a, b = args
            if is_const(p):
                return a if p else b
            if _key(a) == _key(b):
                return a
            if kind == BOOL and is_const(a) and is_const(b):
                # (select(p, True, False) = p, select(p, False, True) = not p)
                if bool(a) and not bool(b):
                    return p
                if not bool(a) and bool(b):
                    return self.emit("not", (p,), BOOL)
        if op == "and" and kind == BOOL:
            for x, y in (args, args[::-1]):
                if is_const(x):
                    return y if bool(x) else np.bool_(False)
        if op == "or" and kind == BOOL:
            for x, y in (args, args[::-1]):
                if is_const(x):
                    return np.bool_(True) if bool(x) else y
        if op == "convert" and not is_const(args[0]) and self.kind(args[0]) == kind:
            return args[0]
        if op == "stop_gradient" and kind != FLOAT:
            return args[0]
        return None


# ---------------------------------------------------------------------------
# lowering a jaxpr
# ---------------------------------------------------------------------------


def _obj(shape, values=None):
    out = np.empty(shape, dtype=object)
    if values is not None:
        flat = out.reshape(-1)
        for i, v in enumerate(values):
            flat[i] = v
    return out


def _const_array(value, dtype):
    if np.dtype(dtype).kind == "V":
        # (float0: the tangent of an integer, zero)
        return _const_array(np.zeros(np.shape(value), np.float32), np.float32)
    value = np.asarray(value)
    kind = kind_of(dtype)
    out = np.empty(value.shape, dtype=object)
    flat_out = out.reshape(-1)
    for i, v in enumerate(value.reshape(-1)):
        if np.iscomplexobj(value):
            flat_out[i] = Complex(np.float32(v.real), np.float32(v.imag))
        else:
            flat_out[i] = const(v, kind)
    return out


def _take(x, i, axis):
    """``x`` at index ``i`` of ``axis`` (an array, also for 1-d ``x``)."""
    index = [slice(None)] * x.ndim
    index[axis] = i
    value = x[tuple(index)]
    if isinstance(value, np.ndarray):
        return value.copy()
    out = np.empty((), dtype=object)
    out[()] = value
    return out


def _is_complex(dtype):
    return np.issubdtype(np.dtype(dtype), np.complexfloating)


class Lowering:
    """Evaluates a jaxpr symbolically into a :class:`Builder`."""

    def __init__(self, builder):
        self.b = builder

    # --- scalar helpers ----------------------------------------------------

    def op(self, name, *args, kind=FLOAT):
        return self.b.emit(name, args, kind)

    def select(self, p, a, b):
        if isinstance(a, Complex) or isinstance(b, Complex):
            a, b = self.as_complex(a), self.as_complex(b)
            return Complex(self.select(p, a.re, b.re), self.select(p, a.im, b.im))
        return self.b.emit("select", (p, a, b), self.b.kind(a) if not is_const(a) else self.b.kind(b))

    def as_complex(self, x):
        if isinstance(x, Complex):
            return x
        return Complex(x, np.float32(0.0))

    # --- real arithmetic with JAX's derivative conventions ------------------

    def max_(self, a, b, kind):
        if kind != FLOAT:
            return self.select(self.op("gt", a, b, kind=BOOL), a, b)
        # (the derivative split at ties, NaN propagated, as lax.max)
        gt = self.op("gt", a, b, kind=BOOL)
        lt = self.op("lt", a, b, kind=BOOL)
        tie = self.op("mul", np.float32(0.5), self.op("add", a, b))
        return self.select(gt, a, self.select(lt, b, tie))

    def min_(self, a, b, kind):
        if kind != FLOAT:
            return self.select(self.op("lt", a, b, kind=BOOL), a, b)
        lt = self.op("lt", a, b, kind=BOOL)
        gt = self.op("gt", a, b, kind=BOOL)
        tie = self.op("mul", np.float32(0.5), self.op("add", a, b))
        return self.select(lt, a, self.select(gt, b, tie))

    def integer_pow(self, x, y):
        if y == 0:
            return np.float32(1.0)
        reciprocal = y < 0
        y = abs(y)
        acc = None
        while y > 0:
            if y & 1:
                acc = x if acc is None else self.op("mul", acc, x)
            y >>= 1
            if y > 0:
                x = self.op("mul", x, x)
        return self.op("div", np.float32(1.0), acc) if reciprocal else acc

    # --- complex arithmetic -------------------------------------------------

    def cadd(self, a, b):
        a, b = self.as_complex(a), self.as_complex(b)
        return Complex(self.op("add", a.re, b.re), self.op("add", a.im, b.im))

    def csub(self, a, b):
        a, b = self.as_complex(a), self.as_complex(b)
        return Complex(self.op("sub", a.re, b.re), self.op("sub", a.im, b.im))

    def cmul(self, a, b):
        a, b = self.as_complex(a), self.as_complex(b)
        re = self.op("sub", self.op("mul", a.re, b.re), self.op("mul", a.im, b.im))
        im = self.op("add", self.op("mul", a.re, b.im), self.op("mul", a.im, b.re))
        return Complex(re, im)

    def cdiv(self, a, b):
        a, b = self.as_complex(a), self.as_complex(b)
        den = self.op("add", self.op("mul", b.re, b.re), self.op("mul", b.im, b.im))
        re = self.op("add", self.op("mul", a.re, b.re), self.op("mul", a.im, b.im))
        im = self.op("sub", self.op("mul", a.im, b.re), self.op("mul", a.re, b.im))
        return Complex(self.op("div", re, den), self.op("div", im, den))

    def cexp(self, a):
        a = self.as_complex(a)
        r = self.op("exp", a.re)
        return Complex(self.op("mul", r, self.op("cos", a.im)), self.op("mul", r, self.op("sin", a.im)))

    def cabs(self, a):
        a = self.as_complex(a)
        return self.op("sqrt", self.op("add", self.op("mul", a.re, a.re), self.op("mul", a.im, a.im)))

    def clog(self, a):
        a = self.as_complex(a)
        return Complex(self.op("log", self.cabs(a)), self.op("atan2", a.im, a.re))

    def csqrt(self, a):
        # principal branch: sqrt((|z| + x) / 2) + i sign(y) sqrt((|z| - x) / 2)
        a = self.as_complex(a)
        r = self.cabs(a)
        re = self.op("sqrt", self.op("mul", np.float32(0.5), self.op("add", r, a.re)))
        im = self.op("sqrt", self.op("mul", np.float32(0.5), self.op("sub", r, a.re)))
        negative = self.op("lt", a.im, np.float32(0.0), kind=BOOL)
        return Complex(re, self.select(negative, self.op("neg", im), im))

    # --- evaluation ----------------------------------------------------------

    def read(self, env, v):
        from jax._src.core import Literal

        if isinstance(v, Literal):
            return _const_array(v.val, v.aval.dtype)
        return env[v]

    def run(self, jaxpr, consts, args):
        env = {}
        for var, value in zip(jaxpr.constvars, consts):
            env[var] = value
        for var, value in zip(jaxpr.invars, args):
            env[var] = value
        for eqn in jaxpr.eqns:
            invals = [self.read(env, v) for v in eqn.invars]
            rule = getattr(self, "p_" + eqn.primitive.name, None)
            if rule is None:
                raise NotImplementedError(f"primitive {eqn.primitive.name} ({eqn.params})")
            outvals = rule(eqn, *invals)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            for var, value in zip(eqn.outvars, outvals):
                if type(var).__name__ != "DropVar":
                    env[var] = value
        return [self.read(env, v) for v in jaxpr.outvars]

    def run_closed(self, closed, args):
        consts = [_const_array(c, np.asarray(c).dtype) for c in closed.consts]
        return self.run(closed.jaxpr, consts, args)

    # --- elementwise ---------------------------------------------------------

    def elementwise(self, fn, *arrays):
        arrays = np.broadcast_arrays(*arrays)
        out = np.empty(arrays[0].shape, dtype=object)
        flat = [a.reshape(-1) for a in arrays]
        o = out.reshape(-1)
        for i in range(o.shape[0]):
            o[i] = fn(*[a[i] for a in flat])
        return out

    def _unary(name, complex_rule=None):
        def rule(self, eqn, x):
            kind = kind_of(eqn.outvars[0].aval.dtype)
            if _is_complex(eqn.invars[0].aval.dtype):
                return self.elementwise(lambda a: getattr(self, complex_rule)(a), x)
            return self.elementwise(lambda a: self.op(name, a, kind=kind), x)
        return rule

    def _binary(name, complex_rule=None):
        def rule(self, eqn, x, y):
            kind = kind_of(eqn.outvars[0].aval.dtype)
            if complex_rule is not None and any(_is_complex(v.aval.dtype) for v in eqn.invars):
                return self.elementwise(lambda a, b: getattr(self, complex_rule)(a, b), x, y)
            return self.elementwise(lambda a, b: self.op(name, a, b, kind=kind), x, y)
        return rule

    p_add = _binary("add", "cadd")
    p_add_any = _binary("add", "cadd")
    p_sub = _binary("sub", "csub")
    p_mul = _binary("mul", "cmul")
    p_div = _binary("div", "cdiv")
    p_atan2 = _binary("atan2")
    p_pow = _binary("pow")
    p_rem = _binary("rem")
    p_and = _binary("and")
    p_or = _binary("or")
    p_shift_right_arithmetic = _binary("shift_right_arithmetic")
    p_lt = _binary("lt")
    p_le = _binary("le")
    p_gt = _binary("gt")
    p_ge = _binary("ge")
    p_ne = _binary("ne")
    p_neg = _unary("neg", "cneg")
    p_sqrt = _unary("sqrt", "csqrt")
    p_exp = _unary("exp", "cexp")
    p_log = _unary("log", "clog")
    p_sin = _unary("sin")
    p_cos = _unary("cos")
    p_tan = _unary("tan")
    p_tanh = _unary("tanh")
    p_atan = _unary("atan")
    p_asin = _unary("asin")
    p_acos = _unary("acos")
    p_cbrt = _unary("cbrt")
    p_floor = _unary("floor")
    p_ceil = _unary("ceil")
    p_sign = _unary("sign")
    p_not = _unary("not")
    p_expm1 = _unary("expm1")
    p_log1p = _unary("log1p")

    def p_one_minus_square(self, eqn, x):
        one = np.float32(1.0)
        return self.elementwise(lambda a: self.op("mul", self.op("add", one, a), self.op("sub", one, a)), x)

    def p_rsqrt(self, eqn, x):
        return self.elementwise(lambda a: self.op("div", np.float32(1.0), self.op("sqrt", a)), x)

    def p_logistic(self, eqn, x):
        one = np.float32(1.0)
        return self.elementwise(lambda a: self.op("div", one, self.op("add", one, self.op("exp", self.op("neg", a)))),
                                x)

    def p_sinh(self, eqn, x):
        half = np.float32(0.5)
        return self.elementwise(lambda a: self.op("mul", half, self.op("sub", self.op("exp", a),
                                                                       self.op("exp", self.op("neg", a)))), x)

    def p_cosh(self, eqn, x):
        half = np.float32(0.5)
        return self.elementwise(lambda a: self.op("mul", half, self.op("add", self.op("exp", a),
                                                                       self.op("exp", self.op("neg", a)))), x)

    def cneg(self, a):
        a = self.as_complex(a)
        return Complex(self.op("neg", a.re), self.op("neg", a.im))

    def p_eq(self, eqn, x, y):
        if any(_is_complex(v.aval.dtype) for v in eqn.invars):
            def ceq(a, b):
                a, b = self.as_complex(a), self.as_complex(b)
                return self.op("and", self.op("eq", a.re, b.re, kind=BOOL), self.op("eq", a.im, b.im, kind=BOOL),
                               kind=BOOL)
            return self.elementwise(ceq, x, y)
        return self.elementwise(lambda a, b: self.op("eq", a, b, kind=BOOL), x, y)

    def p_abs(self, eqn, x):
        if _is_complex(eqn.invars[0].aval.dtype):
            return self.elementwise(self.cabs, x)
        kind = kind_of(eqn.outvars[0].aval.dtype)
        return self.elementwise(lambda a: self.op("abs", a, kind=kind), x)

    def p_max(self, eqn, x, y):
        kind = kind_of(eqn.outvars[0].aval.dtype)
        return self.elementwise(lambda a, b: self.max_(a, b, kind), x, y)

    def p_min(self, eqn, x, y):
        kind = kind_of(eqn.outvars[0].aval.dtype)
        return self.elementwise(lambda a, b: self.min_(a, b, kind), x, y)

    def p_square(self, eqn, x):
        if _is_complex(eqn.invars[0].aval.dtype):
            return self.elementwise(lambda a: self.cmul(a, a), x)
        return self.elementwise(lambda a: self.op("mul", a, a), x)

    def p_integer_pow(self, eqn, x):
        y = eqn.params["y"]
        if _is_complex(eqn.invars[0].aval.dtype):
            def cpow(a):
                acc = Complex(np.float32(1.0), np.float32(0.0))
                for _ in range(abs(y)):
                    acc = self.cmul(acc, a)
                return self.cdiv(np.float32(1.0), acc) if y < 0 else acc
            return self.elementwise(cpow, x)
        return self.elementwise(lambda a: self.integer_pow(a, y), x)

    def p_is_finite(self, eqn, x):
        return self.elementwise(lambda a: self.op("is_finite", a, kind=BOOL), x)

    def p_round(self, eqn, x):
        method = int(eqn.params["rounding_method"])
        name = "round_even" if method == 1 else "round_away"
        return self.elementwise(lambda a: self.op(name, a), x)

    def p_stop_gradient(self, eqn, x):
        kind = kind_of(eqn.outvars[0].aval.dtype)
        if _is_complex(eqn.invars[0].aval.dtype):
            return self.elementwise(lambda a: Complex(self.op("stop_gradient", self.as_complex(a).re),
                                                      self.op("stop_gradient", self.as_complex(a).im)), x)
        return self.elementwise(lambda a: self.op("stop_gradient", a, kind=kind), x)

    def p_select_n(self, eqn, pred, *cases):
        pred_kind = kind_of(eqn.invars[0].aval.dtype)

        def pick(p, *values):
            if pred_kind == BOOL:
                return self.select(p, values[1], values[0])
            out = values[-1]
            for i in range(len(values) - 2, -1, -1):
                out = self.select(self.op("eq", p, np.int32(i), kind=BOOL), values[i], out)
            return out

        return self.elementwise(pick, pred, *cases)

    def p_clamp(self, eqn, lo, x, hi):
        kind = kind_of(eqn.outvars[0].aval.dtype)

        def clamp(lo, x, hi):
            if kind != FLOAT:
                return self.min_(self.max_(x, lo, kind), hi, kind)
            # (the derivative passes strictly inside the bounds, as lax.clamp)
            inside = self.op("and", self.op("gt", x, lo, kind=BOOL), self.op("lt", x, hi, kind=BOOL), kind=BOOL)
            low = self.op("le", x, lo, kind=BOOL)
            outside = self.select(low, lo, hi)
            nan = self.op("ne", x, x, kind=BOOL)
            return self.select(inside, x, self.select(nan, x, outside))

        return self.elementwise(clamp, lo, x, hi)

    def p_convert_element_type(self, eqn, x):
        dtype = eqn.params["new_dtype"]
        src = eqn.invars[0].aval.dtype
        if _is_complex(dtype):
            if _is_complex(src):
                return x
            return self.elementwise(lambda a: Complex(self.op("convert", a, kind=FLOAT), np.float32(0.0)), x)
        if _is_complex(src):
            return self.elementwise(lambda a: self.op("convert", self.as_complex(a).re, kind=kind_of(dtype)), x)
        kind = kind_of(dtype)
        if kind == BOOL and kind_of(src) != BOOL:
            return self.elementwise(lambda a: self.op("ne", a, const(0, kind_of(src)), kind=BOOL), x)
        return self.elementwise(lambda a: self.op("convert", a, kind=kind), x)

    def p_bitcast_convert_type(self, eqn, x):
        kind = kind_of(eqn.params["new_dtype"])
        return self.elementwise(lambda a: self.op("bitcast", a, kind=kind), x)

    def p_real(self, eqn, x):
        return self.elementwise(lambda a: self.as_complex(a).re, x)

    def p_imag(self, eqn, x):
        return self.elementwise(lambda a: self.as_complex(a).im, x)

    def p_conj(self, eqn, x):
        return self.elementwise(lambda a: Complex(self.as_complex(a).re, self.op("neg", self.as_complex(a).im)), x)

    def p_complex(self, eqn, x, y):
        return self.elementwise(lambda a, b: Complex(a, b), x, y)

    # --- structure -----------------------------------------------------------

    def p_optimization_barrier(self, eqn, *xs):
        return list(xs)

    def p_copy(self, eqn, x):
        return x

    p_copy_p = p_copy

    def p_broadcast_in_dim(self, eqn, x):
        shape = eqn.params["shape"]
        dims = eqn.params["broadcast_dimensions"]
        expanded = [1] * len(shape)
        for i, d in enumerate(dims):
            expanded[d] = x.shape[i]
        return np.broadcast_to(x.reshape(expanded), shape).copy()

    def p_reshape(self, eqn, x):
        return x.reshape(eqn.params["new_sizes"]).copy()

    def p_squeeze(self, eqn, x):
        return np.squeeze(x, axis=tuple(eqn.params["dimensions"])).copy() if eqn.params["dimensions"] else x

    def p_expand_dims(self, eqn, x):
        return np.expand_dims(x, eqn.params["dimensions"]).copy()

    def p_transpose(self, eqn, x):
        return np.transpose(x, eqn.params["permutation"]).copy()

    def p_slice(self, eqn, x):
        start = eqn.params["start_indices"]
        limit = eqn.params["limit_indices"]
        strides = eqn.params["strides"] or (1,) * len(start)
        return x[tuple(slice(s, l, st) for s, l, st in zip(start, limit, strides))].copy()

    def p_concatenate(self, eqn, *xs):
        return np.concatenate(xs, axis=eqn.params["dimension"])

    def p_stack(self, eqn, *xs):
        return np.stack(xs, axis=eqn.params["axis"])

    def p_unstack(self, eqn, x):
        axis = eqn.params["axis"]
        return [_take(x, i, axis) for i in range(x.shape[axis])]

    def p_iota(self, eqn):
        shape = eqn.params["shape"]
        dim = eqn.params["dimension"]
        kind = kind_of(eqn.params["dtype"])
        values = np.broadcast_to(np.arange(shape[dim]).reshape([-1 if i == dim else 1 for i in range(len(shape))]),
                                 shape)
        return _const_array(values.astype(_NUMPY[kind]), _NUMPY[kind])

    def p_rev(self, eqn, x):
        return np.flip(x, axis=tuple(eqn.params["dimensions"])).copy()

    def p_pad(self, eqn, x, value):
        config = eqn.params["padding_config"]
        out = x
        for axis, (lo, hi, interior) in enumerate(config):
            if interior:
                raise NotImplementedError("interior padding")
            pad_value = value.reshape(())[()] if hasattr(value, "reshape") else value
            parts = []
            if lo > 0:
                shape = list(out.shape)
                shape[axis] = lo
                parts.append(np.full(shape, pad_value, dtype=object))
            body = out
            if lo < 0:
                body = np.take(body, range(-lo, body.shape[axis]), axis=axis)
            if hi < 0:
                body = np.take(body, range(0, body.shape[axis] + hi), axis=axis)
            parts.append(body)
            if hi > 0:
                shape = list(out.shape)
                shape[axis] = hi
                parts.append(np.full(shape, pad_value, dtype=object))
            out = np.concatenate(parts, axis=axis)
        return out

    # --- reductions ----------------------------------------------------------

    def _reduce(self, eqn, x, combine):
        axes = tuple(eqn.params["axes"])
        if not axes:
            return x
        moved = np.moveaxis(x, list(axes), list(range(len(axes))))
        n = int(np.prod([x.shape[a] for a in axes]))
        flat = moved.reshape((n,) + moved.shape[len(axes):])
        out = np.empty(flat.shape[1:], dtype=object)
        for idx in np.ndindex(*out.shape) if out.shape else [()]:
            acc = flat[(0,) + idx]
            for k in range(1, n):
                acc = combine(acc, flat[(k,) + idx])
            out[idx] = acc
        return out

    def p_reduce_sum(self, eqn, x):
        if _is_complex(eqn.invars[0].aval.dtype):
            return self._reduce(eqn, x, self.cadd)
        kind = kind_of(eqn.outvars[0].aval.dtype)
        return self._reduce(eqn, x, lambda a, b: self.op("add", a, b, kind=kind))

    def p_reduce_max(self, eqn, x):
        kind = kind_of(eqn.outvars[0].aval.dtype)
        return self._reduce(eqn, x, lambda a, b: self.max_(a, b, kind))

    def p_reduce_min(self, eqn, x):
        kind = kind_of(eqn.outvars[0].aval.dtype)
        return self._reduce(eqn, x, lambda a, b: self.min_(a, b, kind))

    def p_reduce_or(self, eqn, x):
        return self._reduce(eqn, x, lambda a, b: self.op("or", a, b, kind=BOOL))

    def p_reduce_and(self, eqn, x):
        return self._reduce(eqn, x, lambda a, b: self.op("and", a, b, kind=BOOL))

    def _arg_reduce(self, eqn, x, better):
        axes = eqn.params["axes"]
        assert len(axes) == 1
        axis = axes[0]
        moved = np.moveaxis(x, axis, 0)
        out = np.empty(moved.shape[1:], dtype=object)
        kind = kind_of(eqn.invars[0].aval.dtype)
        for idx in np.ndindex(*out.shape) if out.shape else [()]:
            best, best_i = moved[(0,) + idx], np.int32(0)
            for k in range(1, moved.shape[0]):
                value = moved[(k,) + idx]
                take = better(value, best, kind)
                best = self.select(take, value, best)
                best_i = self.select(take, np.int32(k), best_i)
            out[idx] = best_i
        return out

    def p_argmax(self, eqn, x):
        def better(value, best, kind):
            if kind == BOOL:
                return self.op("and", value, self.op("not", best, kind=BOOL), kind=BOOL)
            # (NaN wins, the first of equal values is kept)
            gt = self.op("gt", value, best, kind=BOOL)
            if kind == FLOAT:
                nan = self.op("and", self.op("ne", value, value, kind=BOOL),
                              self.op("eq", best, best, kind=BOOL), kind=BOOL)
                return self.op("or", gt, nan, kind=BOOL)
            return gt
        return self._arg_reduce(eqn, x, better)

    def p_argmin(self, eqn, x):
        def better(value, best, kind):
            if kind == BOOL:
                return self.op("and", self.op("not", value, kind=BOOL), best, kind=BOOL)
            lt = self.op("lt", value, best, kind=BOOL)
            if kind == FLOAT:
                nan = self.op("and", self.op("ne", value, value, kind=BOOL),
                              self.op("eq", best, best, kind=BOOL), kind=BOOL)
                return self.op("or", lt, nan, kind=BOOL)
            return lt
        return self._arg_reduce(eqn, x, better)

    def p_dot_general(self, eqn, x, y):
        (cx, cy), (bx, by) = eqn.params["dimension_numbers"]
        complex_ = any(_is_complex(v.aval.dtype) for v in eqn.invars)
        kind = kind_of(eqn.outvars[0].aval.dtype)
        fx = [d for d in range(x.ndim) if d not in cx and d not in bx]
        fy = [d for d in range(y.ndim) if d not in cy and d not in by]
        xt = np.transpose(x, list(bx) + fx + list(cx))
        yt = np.transpose(y, list(by) + fy + list(cy))
        nb = len(bx)
        batch_shape = xt.shape[:nb]
        fx_shape = xt.shape[nb:nb + len(fx)]
        fy_shape = yt.shape[nb:nb + len(fy)]
        contract = int(np.prod(xt.shape[nb + len(fx):])) if cx else 1
        xr = xt.reshape(batch_shape + fx_shape + (contract,))
        yr = yt.reshape(batch_shape + fy_shape + (contract,))
        out = np.empty(batch_shape + fx_shape + fy_shape, dtype=object)
        mul = self.cmul if complex_ else (lambda a, b: self.op("mul", a, b, kind=kind))
        add = self.cadd if complex_ else (lambda a, b: self.op("add", a, b, kind=kind))
        for bi in np.ndindex(*batch_shape) if batch_shape else [()]:
            for i in np.ndindex(*fx_shape) if fx_shape else [()]:
                for j in np.ndindex(*fy_shape) if fy_shape else [()]:
                    acc = None
                    for k in range(contract):
                        term = mul(xr[bi + i + (k,)], yr[bi + j + (k,)])
                        acc = term if acc is None else add(acc, term)
                    out[bi + i + j] = acc if acc is not None else const(0, kind)
        return out

    # --- indexing ------------------------------------------------------------

    def _index_value(self, x):
        """A constant Python int, or None for a traced index."""
        if is_const(x):
            return int(x)
        return None

    def _dynamic_pick(self, values, index):
        """``values[index]`` (a list) for a traced ``index`` (clamped)."""
        out = values[-1]
        for k in range(len(values) - 2, -1, -1):
            out = self.select(self.op("le", index, np.int32(k), kind=BOOL), values[k], out)
        return out

    def p_dynamic_slice(self, eqn, x, *starts):
        sizes = eqn.params["slice_sizes"]
        starts = [s.reshape(())[()] for s in starts]
        out = x
        for axis, (start, size) in enumerate(zip(starts, sizes)):
            n = out.shape[axis]
            s = self._index_value(start)
            if s is not None:
                s = min(max(s, 0), n - size)
                out = np.take(out, range(s, s + size), axis=axis)
            else:
                start = self.op("convert", start, kind=INT)
                start = self.min_(self.max_(start, np.int32(0), INT), np.int32(n - size), INT)
                pieces = []
                for k in range(size):
                    options = [_take(out, j + k, axis) for j in range(n - size + 1)]
                    pieces.append(self._pick_arrays(options, start))
                out = np.stack(pieces, axis=axis)
        return out

    def _pick_arrays(self, arrays, index):
        """``arrays[index]`` element by element for a traced ``index``."""
        shape = arrays[0].shape
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            out[idx] = self._dynamic_pick([a[idx] for a in arrays], index)
        return out

    def p_dynamic_update_slice(self, eqn, x, update, *starts):
        starts = [s.reshape(())[()] for s in starts]
        out = x.copy()
        consts = [self._index_value(s) for s in starts]
        if all(c is not None for c in consts):
            index = tuple(slice(min(max(c, 0), x.shape[a] - update.shape[a]), min(max(c, 0), x.shape[a]
                                - update.shape[a]) + update.shape[a]) for a, c in enumerate(consts))
            out[index] = update
            return out
        # traced starts: every element chooses between itself and the update
        clamped = []
        for a, s in enumerate(starts):
            s = self.op("convert", s, kind=INT)
            clamped.append(self.min_(self.max_(s, np.int32(0), INT), np.int32(x.shape[a] - update.shape[a]), INT))
        for idx in np.ndindex(*x.shape):
            value = x[idx]
            for uidx in np.ndindex(*update.shape):
                # x[idx] is update[uidx] where start + uidx == idx in every axis
                hit = None
                for a in range(x.ndim):
                    cond = self.op("eq", clamped[a], np.int32(idx[a] - uidx[a]), kind=BOOL)
                    hit = cond if hit is None else self.op("and", hit, cond, kind=BOOL)
                value = self.select(hit, update[uidx], value)
            out[idx] = value
        return out

    def p_gather(self, eqn, operand, indices):
        dn = eqn.params["dimension_numbers"]
        slice_sizes = eqn.params["slice_sizes"]
        offset_dims = dn.offset_dims
        collapsed = tuple(dn.collapsed_slice_dims) + tuple(dn.operand_batching_dims)
        start_index_map = dn.start_index_map
        batching = list(zip(dn.operand_batching_dims, dn.start_indices_batching_dims))
        batch_shape = indices.shape[:-1]
        out_shape = list(eqn.outvars[0].aval.shape)
        out = np.empty(out_shape, dtype=object)
        offset_sizes = [slice_sizes[d] for d in range(operand.ndim) if d not in collapsed]
        batch_dims_out = [d for d in range(len(out_shape)) if d not in offset_dims]
        for bidx in np.ndindex(*batch_shape) if batch_shape else [()]:
            starts = [indices[bidx + (k,)] for k in range(indices.shape[-1])]
            start_full = [np.int32(0)] * operand.ndim
            for k, d in enumerate(start_index_map):
                start_full[d] = starts[k]
            for operand_dim, indices_dim in batching:
                start_full[operand_dim] = np.int32(bidx[indices_dim])
            # clamp the starts (XLA's semantics)
            consts = []
            for d in range(operand.ndim):
                s = self._index_value(start_full[d])
                if s is not None:
                    consts.append(min(max(s, 0), operand.shape[d] - slice_sizes[d]))
                else:
                    consts.append(None)
            for oidx in np.ndindex(*offset_sizes) if offset_sizes else [()]:
                full = []
                it = iter(oidx)
                for d in range(operand.ndim):
                    full.append(0 if d in collapsed else next(it))
                if all(c is not None for c in consts):
                    value = operand[tuple(c + f for c, f in zip(consts, full))]
                else:
                    value = self._gather_dynamic(operand, start_full, consts, full, slice_sizes)
                out_index = [0] * len(out_shape)
                for k, d in enumerate(batch_dims_out):
                    out_index[d] = bidx[k]
                for k, d in enumerate(offset_dims):
                    out_index[d] = oidx[k]
                out[tuple(out_index)] = value
        return out

    def _gather_dynamic(self, operand, starts, consts, offsets, slice_sizes):
        # positions along each axis: constant, or a choice among the valid starts
        candidates = [[]]
        for d in range(operand.ndim):
            if consts[d] is not None:
                candidates = [c + [consts[d] + offsets[d]] for c in candidates]
            else:
                candidates = [c + [None] for c in candidates]
        dynamic_axes = [d for d in range(operand.ndim) if consts[d] is None]
        clamped = {}
        for d in dynamic_axes:
            s = self.op("convert", starts[d], kind=INT)
            clamped[d] = self.min_(self.max_(s, np.int32(0), INT), np.int32(operand.shape[d] - slice_sizes[d]), INT)

        def pick(axis_list, fixed):
            if not axis_list:
                return operand[tuple(fixed)]
            d = axis_list[0]
            values = []
            for j in range(operand.shape[d] - slice_sizes[d] + 1):
                f = list(fixed)
                f[d] = j + offsets[d]
                values.append(pick(axis_list[1:], f))
            return self._dynamic_pick(values, clamped[d])

        return pick(dynamic_axes, [c if c is not None else 0 for c in candidates[0]])

    def p_scatter(self, eqn, operand, indices, updates):
        return self._scatter(eqn, operand, indices, updates, None)

    def p_scatter_add(self, eqn, operand, indices, updates):
        kind = kind_of(eqn.outvars[0].aval.dtype)
        return self._scatter(eqn, operand, indices, updates, lambda a, b: self.op("add", a, b, kind=kind))

    def _scatter(self, eqn, operand, indices, updates, combine):
        dn = eqn.params["dimension_numbers"]
        update_window_dims = dn.update_window_dims
        inserted = dn.inserted_window_dims
        sdims = dn.scatter_dims_to_operand_dims
        out = operand.copy()
        scatter_shape = [updates.shape[d] for d in range(updates.ndim) if d not in update_window_dims]
        window_shape = [updates.shape[d] for d in update_window_dims]
        for sidx in np.ndindex(*scatter_shape) if scatter_shape else [()]:
            starts = [indices[sidx + (k,)] if indices.ndim > len(sidx) else indices[sidx] for k in
                      range(indices.shape[-1])]
            consts = [self._index_value(s) for s in starts]
            if any(c is None for c in consts):
                raise NotImplementedError("scatter at traced indices")
            start_full = [0] * operand.ndim
            for k, d in enumerate(sdims):
                start_full[d] = consts[k]
            for widx in np.ndindex(*window_shape) if window_shape else [()]:
                full = []
                it = iter(widx)
                for d in range(operand.ndim):
                    full.append(0 if d in inserted else next(it))
                target = tuple(s + f for s, f in zip(start_full, full))
                if any(t < 0 or t >= n for t, n in zip(target, operand.shape)):
                    continue
                uidx = [0] * updates.ndim
                si = iter(sidx)
                wi = iter(widx)
                for d in range(updates.ndim):
                    uidx[d] = next(wi) if d in update_window_dims else next(si)
                value = updates[tuple(uidx)]
                out[target] = value if combine is None else combine(out[target], value)
        return out

    # --- linear algebra ---------------------------------------------------------

    def p_lu(self, eqn, a):
        """LAPACK's ``getrf`` (partial pivoting) on a small matrix: the
        factors, the pivots and the permutation."""
        n, m = a.shape[-2], a.shape[-1]
        if a.ndim > 2:
            results = [self.p_lu(eqn, a[idx]) for idx in np.ndindex(*a.shape[:-2])]
            return [np.stack([r[k] for r in results]).reshape(a.shape[:-2] + results[0][k].shape) for k in range(3)]
        lu = a.copy()
        perm = [np.int32(i) for i in range(n)]
        pivots = []
        for k in range(min(n, m)):
            # the row of the largest |a[i, k]|, i >= k
            best = self.op("abs", lu[k, k])
            best_i = np.int32(k)
            for i in range(k + 1, n):
                value = self.op("abs", lu[i, k])
                take = self.op("gt", value, best, kind=BOOL)
                best = self.select(take, value, best)
                best_i = self.select(take, np.int32(i), best_i)
            pivots.append(best_i)
            # swap rows k and best_i
            rows = [lu[i].copy() for i in range(n)]
            perm_rows = list(perm)
            for i in range(k, n):
                is_i = self.op("eq", best_i, np.int32(i), kind=BOOL)
                for j in range(m):
                    lu[k, j] = self.select(is_i, rows[i][j], lu[k, j]) if i != k else lu[k, j]
                perm[k] = self.select(is_i, perm_rows[i], perm[k]) if i != k else perm[k]
            for i in range(k + 1, n):
                is_i = self.op("eq", best_i, np.int32(i), kind=BOOL)
                for j in range(m):
                    lu[i, j] = self.select(is_i, rows[k][j], rows[i][j])
                perm[i] = self.select(is_i, perm_rows[k], perm_rows[i])
            # eliminate
            for i in range(k + 1, n):
                lu[i, k] = self.op("div", lu[i, k], lu[k, k])
                for j in range(k + 1, m):
                    lu[i, j] = self.op("sub", lu[i, j], self.op("mul", lu[i, k], lu[k, j]))
        return [lu, _obj((len(pivots),), pivots), _obj((n,), perm)]

    def p_triangular_solve(self, eqn, a, b):
        p = eqn.params
        if p["conjugate_a"] and any(isinstance(x, Complex) for x in a.reshape(-1)):
            raise NotImplementedError("complex triangular solve")
        if a.ndim > 2:
            results = [self.p_triangular_solve(eqn, a[idx], b[idx]) for idx in np.ndindex(*a.shape[:-2])]
            return np.stack(results).reshape(a.shape[:-2] + results[0].shape)
        lower = p["lower"]
        mat = a.T.copy() if p["transpose_a"] else a
        if p["transpose_a"]:
            lower = not lower
        unit = p["unit_diagonal"]
        rhs = b if p["left_side"] else b.T
        n = mat.shape[0]
        x = np.empty(rhs.shape, dtype=object)
        order = range(n) if lower else range(n - 1, -1, -1)
        for col in range(rhs.shape[1]):
            for i in order:
                acc = rhs[i, col]
                js = range(i) if lower else range(i + 1, n)
                for j in js:
                    acc = self.op("sub", acc, self.op("mul", mat[i, j], x[j, col]))
                x[i, col] = acc if unit else self.op("div", acc, mat[i, i])
        if not p["left_side"]:
            # x A = b: solved as A^T x^T = b^T
            raise NotImplementedError("right-side triangular solve")
        return x

    def p_custom_linear_solve(self, eqn, *args):
        lengths = eqn.params["const_lengths"]
        jaxprs = eqn.params["jaxprs"]
        n_consts = [lengths.matvec, lengths.vecmat, lengths.solve, lengths.transpose_solve]
        offsets = np.cumsum([0] + n_consts)
        solve_consts = args[offsets[2]:offsets[3]]
        b = args[offsets[4]:]
        return self._run_with_consts(jaxprs.solve, solve_consts, b)

    def _run_with_consts(self, closed, consts, args):
        inner_consts = [_const_array(c, np.asarray(c).dtype) for c in closed.consts]
        return self.run(closed.jaxpr, inner_consts, list(consts) + list(args))

    def p_sort(self, eqn, *operands):
        dim = eqn.params["dimension"]
        num_keys = eqn.params["num_keys"]
        if num_keys != 1:
            raise NotImplementedError("sort with several keys")
        moved = [np.moveaxis(o, dim, -1).copy() for o in operands]
        n = moved[0].shape[-1]
        kind = kind_of(eqn.invars[0].aval.dtype)
        for idx in np.ndindex(*moved[0].shape[:-1]) if moved[0].ndim > 1 else [()]:
            rows = [list(m[idx]) for m in moved]
            # odd-even transposition sort (stable for distinct keys); NaN last
            for rnd in range(n):
                for i in range(rnd % 2, n - 1, 2):
                    a, b = rows[0][i], rows[0][i + 1]
                    swap = self.op("lt", b, a, kind=BOOL)
                    if kind == FLOAT:
                        nan_a = self.op("ne", a, a, kind=BOOL)
                        nan_b = self.op("ne", b, b, kind=BOOL)
                        swap = self.op("or", swap, self.op("and", nan_a, self.op("not", nan_b, kind=BOOL),
                                                           kind=BOOL), kind=BOOL)
                    for r in rows:
                        x, y = r[i], r[i + 1]
                        r[i] = self.select(swap, y, x)
                        r[i + 1] = self.select(swap, x, y)
            for m, r in zip(moved, rows):
                m[idx] = _obj((n,), r) if m.ndim == 1 else r
        return [np.moveaxis(m, -1, dim).copy() for m in moved]

    # --- control flow ----------------------------------------------------------

    def p_jit(self, eqn, *args):
        return self.run_closed(eqn.params["jaxpr"], list(args))

    p_pjit = p_jit

    def p_custom_vmap_call(self, eqn, *args):
        call = eqn.params.get("call", eqn.params.get("call_jaxpr"))
        if hasattr(call, "consts"):
            return self.run_closed(call, list(args))
        return self.run(call, [], list(args))

    def p_custom_jvp_call(self, eqn, *args):
        return self.run_closed(eqn.params["call_jaxpr"], list(args))

    def p_custom_vjp_call(self, eqn, *args):
        return self.run_closed(eqn.params["call_jaxpr"], list(args))

    def p_closed_call(self, eqn, *args):
        return self.run_closed(eqn.params["call_jaxpr"], list(args))

    def p_cond(self, eqn, index, *operands):
        branches = eqn.params["branches"]
        index = index.reshape(())[()]
        i = self._index_value(index)
        if i is not None:
            i = min(max(i, 0), len(branches) - 1)
            return self.run_closed(branches[i], list(operands))
        results = [self.run_closed(branch, list(operands)) for branch in branches]
        kind = kind_of(eqn.invars[0].aval.dtype)
        out = []
        for k in range(len(results[0])):
            values = [r[k] for r in results]
            if kind == BOOL:
                out.append(self.elementwise(lambda a, b: self.select(index, b, a), *values))
            else:
                def pick(*vs):
                    acc = vs[-1]
                    for j in range(len(vs) - 2, -1, -1):
                        acc = self.select(self.op("le", index, np.int32(j), kind=BOOL), vs[j], acc)
                    return acc
                out.append(self.elementwise(pick, *values))
        return out

    def p_scan(self, eqn, *args):
        p = eqn.params
        closed = p["jaxpr"]
        length = p["length"]
        if "num_consts" in p:
            n_consts, n_carry = p["num_consts"], p["num_carry"]
        else:
            # (JAX >= 0.10: flat trees of the consts, the carry and the xs)
            n_consts, n_carry = len(p["ft_in"].elts[0]), len(p["ft_in"].elts[1])
        consts = list(args[:n_consts])
        carry = list(args[n_consts:n_consts + n_carry])
        xs = list(args[n_consts + n_carry:])
        ys = []
        steps = range(length - 1, -1, -1) if p["reverse"] else range(length)
        for i in steps:
            x_i = [x[i] if x.ndim else x for x in xs]
            out = self.run_closed(closed, consts + carry + x_i)
            carry = out[:n_carry]
            ys.append(out[n_carry:])
        if p["reverse"]:
            ys = ys[::-1]
        stacked = []
        n_ys = len(eqn.outvars) - n_carry
        for k in range(n_ys):
            stacked.append(np.stack([y[k] for y in ys]) if ys else np.empty((0,), dtype=object))
        return carry + stacked

    def p_while(self, eqn, *args):
        p = eqn.params
        cond_consts = list(args[:p["cond_nconsts"]])
        body_consts = list(args[p["cond_nconsts"]:p["cond_nconsts"] + p["body_nconsts"]])
        carry = list(args[p["cond_nconsts"] + p["body_nconsts"]:])
        for _ in range(100000):
            (go,) = self.run_closed_with(p["cond_jaxpr"], cond_consts, carry)
            go = go.reshape(())[()]
            if not is_const(go):
                raise NotImplementedError("while loop with a traced condition")
            if not go:
                return carry
            carry = self.run_closed_with(p["body_jaxpr"], body_consts, carry)
        raise RuntimeError("while loop did not end")

    def run_closed_with(self, closed, consts, args):
        inner = [_const_array(c, np.asarray(c).dtype) for c in closed.consts]
        return self.run(closed.jaxpr, inner, list(consts) + list(args))


def lower(closed_jaxpr):
    """The :class:`Program` of a closed jaxpr (its inputs become the
    program's inputs, element by element), with the unused instructions
    removed."""
    builder = Builder()
    args = []
    for index, var in enumerate(closed_jaxpr.jaxpr.invars):
        aval = var.aval
        kind = kind_of(aval.dtype)
        if _is_complex(aval.dtype):
            raise NotImplementedError("complex inputs")
        n = int(np.prod(aval.shape)) if aval.shape else 1
        elements = [builder.input((index, k), kind) for k in range(n)]
        arr = _obj(tuple(aval.shape), elements)
        args.append(arr)
        builder.program.inputs.append((tuple(aval.shape), kind))
        builder.program.input_vars.append(arr)
    outs = Lowering(builder).run_closed(closed_jaxpr, args)
    builder.program.outputs = outs
    builder.program.output_dtypes = [np.dtype(v.aval.dtype) for v in closed_jaxpr.jaxpr.outvars]
    return prune(builder.program)


def _uses(x):
    if isinstance(x, Var):
        yield x.id
    elif isinstance(x, Complex):
        yield from _uses(x.re)
        yield from _uses(x.im)


def prune(program):
    """The program without the instructions no output depends on
    (renumbered; the inputs are kept)."""
    live = set()
    stack = []
    for out in program.outputs:
        for x in out.reshape(-1):
            for i in _uses(x):
                stack.append(i)
    while stack:
        i = stack.pop()
        if i in live:
            continue
        live.add(i)
        for a in program.instructions[i][1]:
            if isinstance(a, Var):
                stack.append(a.id)
    mapping = {}
    instructions = []
    for i, (op, args, kind) in enumerate(program.instructions):
        if op == "input" or i in live:
            mapping[i] = len(instructions)
            instructions.append((op, tuple(Var(mapping[a.id]) if isinstance(a, Var) else a for a in args), kind))

    def remap(x):
        if isinstance(x, Var):
            return Var(mapping[x.id])
        if isinstance(x, Complex):
            return Complex(remap(x.re), remap(x.im))
        return x

    outputs = []
    for out in program.outputs:
        new = np.empty(out.shape, dtype=object)
        for idx in np.ndindex(*out.shape) if out.shape else [()]:
            new[idx] = remap(out[idx])
        outputs.append(new)
    input_vars = []
    for arr in program.input_vars:
        new = np.empty(arr.shape, dtype=object)
        for idx in np.ndindex(*arr.shape) if arr.shape else [()]:
            new[idx] = remap(arr[idx])
        input_vars.append(new)
    return Program(instructions, program.inputs, input_vars, outputs, program.output_dtypes)
