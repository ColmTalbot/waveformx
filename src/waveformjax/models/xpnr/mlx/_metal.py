# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""A :class:`._program.Program` as Metal source: straight-line code on
scalars, one thread per binary (or per binary and point).

:func:`function_source` writes a program as a Metal function of its input
and output scalars, for the kernels written by hand (the integration of the
precession equations calls the derivatives of :mod:`..spintaylor.dynamics`
this way); :class:`PackedKernel` runs a program as a kernel of its own,
with the inputs and outputs packed into one float array each (a kernel
has a limited number of arguments).
"""

from __future__ import annotations

import math

import numpy as np

from ._program import BOOL, FLOAT, INT, Complex, Var

__all__ = ["function_source", "literal", "PackedKernel"]

_TYPES = {FLOAT: "float", INT: "int", BOOL: "bool"}

_UNARY = {
    "neg": "(-({0}))", "abs": "metal::abs({0})", "sqrt": "metal::precise::sqrt({0})", "exp": "metal::precise::exp({0})",
    "log": "metal::precise::log({0})", "sin": "metal::precise::sin({0})", "cos": "metal::precise::cos({0})",
    "tan": "metal::precise::tan({0})", "tanh": "metal::precise::tanh({0})", "atan": "metal::precise::atan({0})",
    "asin": "metal::precise::asin({0})", "acos": "metal::precise::acos({0})", "cbrt": "wj_cbrt({0})",
    "floor": "metal::floor({0})", "ceil": "metal::ceil({0})", "round_even": "metal::rint({0})",
    "round_away": "metal::round({0})", "sign": "wj_sign({0})", "is_finite": "wj_isfinite({0})",
    "stop_gradient": "({0})", "expm1": "(metal::precise::exp({0}) - 1.0f)", "log1p": "metal::precise::log(1.0f + {0})",
}
_BINARY = {
    "add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})", "div": "({0} / {1})",
    "atan2": "metal::precise::atan2({0}, {1})", "pow": "metal::precise::pow({0}, {1})", "lt": "({0} < {1})",
    "le": "({0} <= {1})", "gt": "({0} > {1})", "ge": "({0} >= {1})", "eq": "({0} == {1})", "ne": "({0} != {1})",
    "shift_right_arithmetic": "({0} >> {1})",
}

#: Helpers of the generated code.
HEADER = """
inline float wj_sign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : x); }
inline int wj_sign(int x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }
inline bool wj_isfinite(float x) { return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u; }
inline float wj_cbrt(float x) {
    float a = metal::abs(x);
    if (a == 0.0f || !wj_isfinite(a)) return x;
    float y = metal::precise::exp(metal::precise::log(a) / 3.0f);
    y = y - (y * y * y - a) / (3.0f * y * y);
    return x < 0.0f ? -y : y;
}
inline float wj_fmod(float a, float b) { return metal::precise::fmod(a, b); }
"""


def literal(x, kind=None):
    """A constant of a program in Metal."""
    if isinstance(x, (np.bool_, bool)):
        return "true" if x else "false"
    if isinstance(x, (np.integer, int)) and not isinstance(x, (bool, np.bool_)):
        return f"{int(x)}"
    value = float(x)
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "(-INFINITY)"
    return f"{np.float32(value)!r}".replace("np.float32(", "").rstrip(")") + "f" if False else _float_literal(value)


def _float_literal(value):
    text = repr(float(np.float32(value)))
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def _expression(op, args, kind, name_of):
    a = [name_of(x) for x in args]
    if op == "select":
        return f"({a[0]} ? {a[1]} : {a[2]})"
    if op == "convert":
        return f"static_cast<{_TYPES[kind]}>({a[0]})"
    if op == "bitcast":
        return f"as_type<{_TYPES[kind]}>({a[0]})"
    if op == "and":
        return f"({a[0]} && {a[1]})" if kind == BOOL else f"({a[0]} & {a[1]})"
    if op == "or":
        return f"({a[0]} || {a[1]})" if kind == BOOL else f"({a[0]} | {a[1]})"
    if op == "not":
        return f"(!{a[0]})" if kind == BOOL else f"(~{a[0]})"
    if op == "rem":
        return f"wj_fmod({a[0]}, {a[1]})" if kind == FLOAT else f"({a[0]} % {a[1]})"
    if op == "max" or op == "min":
        raise NotImplementedError(op)
    if op in _UNARY:
        return _UNARY[op].format(*a)
    if op in _BINARY:
        return _BINARY[op].format(*a)
    raise NotImplementedError(op)


def _use_counts(program):
    counts = [0] * len(program.instructions)

    def use(x):
        if isinstance(x, Var):
            counts[x.id] += 1
        elif isinstance(x, Complex):
            use(x.re)
            use(x.im)

    for _, args, _ in program.instructions:
        for a in args:
            use(a)
    for out in program.outputs:
        for x in out.reshape(-1):
            use(x)
    return counts


def _fusions(program):
    """The additions and subtractions whose operand is a product used only
    there: written as fused multiply-adds (as XLA's CPU backend, whose
    float32 evaluation the JAX model is, contracts them; the Metal compiler
    does not contract across statements). Returns ``{add index: (mul
    index, which operand)}`` and the set of the products absorbed."""
    counts = _use_counts(program)
    fused, absorbed = {}, set()
    instructions = program.instructions
    for i, (op, args, kind) in enumerate(instructions):
        if kind != FLOAT or op not in ("add", "sub"):
            continue
        for position, a in enumerate(args):
            if (isinstance(a, Var) and instructions[a.id][0] == "mul" and instructions[a.id][2] == FLOAT
                    and counts[a.id] == 1 and a.id not in absorbed):
                fused[i] = (a.id, position)
                absorbed.add(a.id)
                break
    return fused, absorbed


#: Whether :func:`body_lines` fuses multiply-adds by default.
CONTRACT = True


def body_lines(program, input_name, prefix="v", contract=None):
    """The statements of ``program``: ``input_name(index, k)`` gives the
    expression of element ``k`` of input ``index``. Returns the lines and
    a function giving the expression of a value (a :class:`Var` or a
    constant). With ``contract``, products added once are fused
    (:func:`_fusions`)."""
    lines = []

    def name_of(x):
        if isinstance(x, Var):
            return f"{prefix}{x.id}"
        return literal(x)

    if contract is None:
        contract = CONTRACT
    fused, absorbed = _fusions(program) if contract else ({}, set())
    for i, (op, args, kind) in enumerate(program.instructions):
        if op == "input":
            index, k = args[0]
            lines.append(f"{_TYPES[kind]} {prefix}{i} = {input_name(index, k, kind)};")
        elif i in absorbed:
            continue
        elif i in fused:
            mul, position = fused[i]
            a, b = (name_of(x) for x in program.instructions[mul][1])
            other = name_of(args[1 - position])
            if op == "add":
                expression = f"metal::fma({a}, {b}, {other})"
            elif position == 0:
                expression = f"metal::fma({a}, {b}, -({other}))"
            else:
                expression = f"metal::fma(-({a}), {b}, {other})"
            lines.append(f"float {prefix}{i} = {expression};")
        else:
            lines.append(f"{_TYPES[kind]} {prefix}{i} = {_expression(op, args, kind, name_of)};")
    return lines, name_of


def function_source(program, name, n_primal=None):
    """``program`` as a Metal function ``void name(thread const float*
    in0, ..., thread float* out0, ...)``: one pointer per input and output
    of the jaxpr (bool and int values converted to float).

    With ``n_primal`` (the number of primal inputs of a program lowered
    from ``jax.jvp``, whose inputs are the primal values then the tangents
    of all of them, and whose outputs are the primal values then their
    tangents), the function takes and returns dual numbers
    (``WjDual``: ``.v`` the value, ``.d`` the derivative) instead: one
    pointer per primal input and output."""
    n_in = len(program.inputs)
    n_out = len(program.outputs)
    dual = n_primal is not None
    if dual:
        n_primal_out = n_out // 2
        params = ([f"thread const WjDual* in{i}" for i in range(n_primal)]
                  + [f"thread WjDual* out{i}" for i in range(n_primal_out)])
    else:
        params = [f"thread const float* in{i}" for i in range(n_in)] + [f"thread float* out{i}" for i in range(n_out)]

    def input_name(index, k, kind):
        if dual:
            value = f"in{index}[{k}].v" if index < n_primal else f"in{index - n_primal}[{k}].d"
        else:
            value = f"in{index}[{k}]"
        return value if kind == FLOAT else f"static_cast<{_TYPES[kind]}>({value})"

    lines, name_of = body_lines(program, input_name)
    for i, out in enumerate(program.outputs):
        for k, x in enumerate(out.reshape(-1)):
            if isinstance(x, Complex):
                raise NotImplementedError("complex outputs")
            if dual:
                target = f"out{i}[{k}].v" if i < n_primal_out else f"out{i - n_primal_out}[{k}].d"
            else:
                target = f"out{i}[{k}]"
            lines.append(f"{target} = static_cast<float>({name_of(x)});")
    body = "\n    ".join(lines)
    return f"inline void {name}({', '.join(params)}) {{\n    {body}\n}}\n"


#: Dual numbers of the generated functions.
DUAL = """
struct WjDual {
    float v;
    float d;
    WjDual() = default;
    WjDual(float value, float derivative = 0.0f) : v(value), d(derivative) {}
};
"""


class PackedKernel:
    """A program as a Metal kernel over ``n`` threads (one per row): the
    inputs packed into a float array of shape ``(n, n_inputs)`` and the
    outputs into one of shape ``(n, n_outputs)`` (bool and int values as
    floats)."""

    def __init__(self, program, name):
        import mlx.core as mx

        self.program = program
        self.in_sizes = [int(np.prod(shape)) if shape else 1 for shape, _ in program.inputs]
        self.in_offsets = np.concatenate([[0], np.cumsum(self.in_sizes)]).astype(int)
        self.n_in = int(self.in_offsets[-1])
        self.out_sizes = [out.size for out in program.outputs]
        self.out_offsets = np.concatenate([[0], np.cumsum(self.out_sizes)]).astype(int)
        self.n_out = int(self.out_offsets[-1])
        offsets = self.in_offsets

        def input_name(index, k, kind):
            value = f"x[base_in + {offsets[index] + k}]"
            return value if kind == FLOAT else f"static_cast<{_TYPES[kind]}>({value})"

        lines, name_of = body_lines(program, input_name)
        for i, out in enumerate(program.outputs):
            for k, x in enumerate(out.reshape(-1)):
                if isinstance(x, Complex):
                    raise NotImplementedError("complex outputs")
                lines.append(f"y[base_out + {self.out_offsets[i] + k}] = static_cast<float>({name_of(x)});")
        source = (
            "uint row = thread_position_in_grid.x;\n"
            "if (row >= n_rows[0]) return;\n"
            f"size_t base_in = (size_t)row * {self.n_in};\n"
            f"size_t base_out = (size_t)row * {self.n_out};\n" + "\n".join(lines)
        )
        self.source = source
        self.kernel = mx.fast.metal_kernel(name=name, input_names=["x", "n_rows"], output_names=["y"],
                                           source=source, header=HEADER)

    def __call__(self, x):
        import mlx.core as mx

        n = x.shape[0]
        threads = 64
        (y,) = self.kernel(inputs=[x, mx.array([n], dtype=mx.uint32)], grid=(-(-n // threads) * threads, 1, 1),
                           threadgroup=(threads, 1, 1), output_shapes=[(n, self.n_out)], output_dtypes=[mx.float32])
        return y
