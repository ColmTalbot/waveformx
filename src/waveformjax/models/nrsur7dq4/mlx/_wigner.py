# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The rotation of the modes from the coprecessing to the inertial frame
(:func:`._frames.coprecessing_to_inertial`) as Metal kernels.

Written with MLX operations, the rotation makes hundreds of arrays of the
size of the batch times the coorbital times (one or more per element of the
Wigner D-matrices), which dominate the time and memory of a batch and of
its gradient. Here one thread rotates the 21 modes of one time sample,
with the sums of :mod:`.._frames` written out as straight-line Metal code
generated from the same terms (``_TERMS``), in the same order, one scope
per ``l`` so that a thread holds the modes of one ``l`` at a time.

The derivatives are a second kernel, called by the VJP of :func:`rotate`,
an ``mx.custom_function``, which reads the rotated modes ``y = D(q) x``
rather than ``x``:

* the rotation is linear in the modes, so their cotangents are the
  conjugate-transposed rotation of the cotangents ``c`` of ``y``;
* ``D`` is a representation of the quaternions, ``D(p q) = D(p) D(q)``
  for any ``p``, ``q`` (each block ``D^l`` is homogeneous of degree
  ``2 l``), so a change ``dq = u q`` of the quaternion, with ``u = dq
  conj(q) / |q|^2``, changes the modes by ``dy = sum_k u_k G_k y``, where
  ``G_k = dD/dq_k`` at the identity are fixed, sparse matrices
  (:func:`generators`: ``2 l`` on the diagonal for ``k = 0``, diagonal for
  ``k = 3``, tridiagonal otherwise). The cotangent of ``q`` is thus
  ``sum_k Re<c, G_k y> du_k/dq``: four sums per time sample rather than the
  derivatives of every element of ``D`` (which, carried as dual numbers,
  made the kernel 20 times slower than the rotation).
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np

from .._data import LMAX, MODES
from .._frames import _TERMS, _ComplexPowers, _Powers

__all__ = ["generators", "rotate", "forward", "backward"]

N_MODES = len(MODES)


def _number(value: float) -> str:
    return f"{float(value)!r}f"


def _elements(ell: int, complex_power, power):
    """The elements ``D^l_{m, m'}`` (``l = ell``) as
    :func:`.._frames.coprecessing_to_inertial` computes them, in the
    arithmetic of ``complex_power(base, k)`` and ``power(base, k)`` (of the
    bases ``ra``, ``rb``, ``ra_abs``, ``rb_abs``): yields ``(m + l, m' + l,
    d_re, d_im)``, with ``d_im`` ``None`` where it vanishes and ``d_re`` a
    Python float where it is a constant."""
    for m in range(-ell, ell + 1):
        for mp in range(-ell, ell + 1):
            a, b = m + mp, m - mp
            if a == 0 and b == 0:
                z = None
            elif a == 0:
                z = complex_power("rb", b)
            elif b == 0:
                z = complex_power("ra", a)
            else:
                (ar, ai), (br, bi) = complex_power("ra", a), complex_power("rb", b)
                z = (ar * br - ai * bi, ar * bi + ai * br)
            total = None
            for coef, alpha, beta in _TERMS[ell, m, mp]:
                if alpha and beta:
                    term = coef * (power("ra_abs", alpha) * power("rb_abs", beta))
                elif alpha:
                    term = coef * power("ra_abs", alpha)
                elif beta:
                    term = coef * power("rb_abs", beta)
                else:
                    term = coef
                total = term if total is None else total + term
            if z is None:
                yield m + ell, mp + ell, total, None
            else:
                yield m + ell, mp + ell, z[0] * total, z[1] * total


class _Code:
    """Metal code in the arithmetic of :func:`_elements`: every intermediate
    is a ``float`` defined once in :attr:`lines`."""

    def __init__(self):
        self.lines: List[str] = []
        self.names: Dict[str, str] = {}
        for name, expression in (("ra_re1", "q0"), ("ra_im1", "-q3"), ("rb_re1", "-q2"), ("rb_im1", "-q1"),
                                 ("ra_abs1", "q0 * q0 + q3 * q3"), ("rb_abs1", "q2 * q2 + q1 * q1")):
            self.define(name, expression)

    def define(self, name: str, expression: str) -> "_Expression":
        if name not in self.names:
            self.lines.append(f"float {name} = {expression};")
            self.names[name] = expression
        return _Expression(self, name)

    def complex_power(self, base: str, k: int):
        """``z^k`` (``k > 0``) or ``conj(z)^-k`` (``k < 0``) by repeated
        multiplication (``ComplexPowers_Compute``)."""
        for j in range(2, abs(k) + 1):
            a, b, x, y = f"{base}_re{j - 1}", f"{base}_im{j - 1}", f"{base}_re1", f"{base}_im1"
            self.define(f"{base}_re{j}", f"{a} * {x} - {b} * {y}")
            self.define(f"{base}_im{j}", f"{a} * {y} + {b} * {x}")
        re, im = _Expression(self, f"{base}_re{abs(k)}"), _Expression(self, f"{base}_im{abs(k)}")
        return (re, im) if k >= 0 else (re, -im)

    def power(self, base: str, k: int):
        for j in range(2, k + 1):
            self.define(f"{base}{j}", f"{base}{j - 1} * {base}1")
        return _Expression(self, f"{base}{k}")


class _Expression:
    """A named Metal value of :class:`_Code`; arithmetic defines new ones."""

    def __init__(self, code: _Code, text: str):
        self.code, self.text = code, text

    def _binary(self, other, op: str, reverse: bool = False):
        other = other.text if isinstance(other, _Expression) else _number(other)
        a, b = (other, self.text) if reverse else (self.text, other)
        return self.code.define(f"t{len(self.code.lines)}", f"{a} {op} {b}")

    def __add__(self, other):
        return self._binary(other, "+")

    def __radd__(self, other):
        return self._binary(other, "+", True)

    def __sub__(self, other):
        return self._binary(other, "-")

    def __mul__(self, other):
        return self._binary(other, "*")

    def __rmul__(self, other):
        return self._binary(other, "*", True)

    def __neg__(self):
        return self.code.define(f"t{len(self.code.lines)}", f"-{self.text}")


class _Dual:
    """A float64 number with its derivatives with respect to the four
    components of the quaternion."""

    def __init__(self, value, derivative=(0.0, 0.0, 0.0, 0.0)):
        self.v, self.d = float(value), np.asarray(derivative, dtype=float)

    @staticmethod
    def lift(x):
        return x if isinstance(x, _Dual) else _Dual(x)

    def __add__(self, other):
        other = _Dual.lift(other)
        return _Dual(self.v + other.v, self.d + other.d)

    __radd__ = __add__

    def __sub__(self, other):
        other = _Dual.lift(other)
        return _Dual(self.v - other.v, self.d - other.d)

    def __mul__(self, other):
        other = _Dual.lift(other)
        return _Dual(self.v * other.v, self.d * other.v + self.v * other.d)

    __rmul__ = __mul__

    def __neg__(self):
        return _Dual(-self.v, -self.d)


@functools.lru_cache(maxsize=None)
def generators() -> np.ndarray:
    """``G_k = dD/dq_k`` at the identity quaternion (``(4, n_modes,
    n_modes)``, complex128, block diagonal in ``l``; ``dy = G_k y`` for the
    modes ``y``), exactly, from the terms of the D-matrices on dual
    numbers."""
    q = [_Dual(1.0 if k == 0 else 0.0, np.eye(4)[k]) for k in range(4)]
    bases = {
        "ra": _ComplexPowers(q[0], -q[3]),
        "rb": _ComplexPowers(-q[2], -q[1]),
        "ra_abs": _Powers(q[0] * q[0] + q[3] * q[3]),
        "rb_abs": _Powers(q[2] * q[2] + q[1] * q[1]),
    }
    out = np.zeros((4, N_MODES, N_MODES), dtype=complex)
    offset = 0
    for ell in range(2, LMAX + 1):
        for i, j, d_re, d_im in _elements(ell, lambda base, k: bases[base][k], lambda base, k: bases[base][k]):
            d_re, d_im = _Dual.lift(d_re), _Dual.lift(0.0 if d_im is None else d_im)
            out[:, offset + i, offset + j] = d_re.d + 1j * d_im.d
        offset += 2 * ell + 1
    return out


def _rotation_code(ell: int, accumulate) -> str:
    """The code of the elements of ``D^l``, each passed to
    ``accumulate(i, j, d_re, d_im)`` (texts, ``d_im`` ``None`` where it
    vanishes)."""
    code = _Code()
    for i, j, d_re, d_im in _elements(ell, code.complex_power, code.power):
        d_re = _number(d_re) if isinstance(d_re, float) else d_re.text
        code.lines.append(accumulate(i, j, d_re, None if d_im is None else d_im.text))
    return "\n        ".join(code.lines)


def _forward_accumulate(i: int, j: int, d_re: str, d_im: Optional[str]) -> str:
    if d_im is None:
        return f"y_re[{i}] += {d_re} * x_re[{j}]; y_im[{i}] += {d_re} * x_im[{j}];"
    return (f"y_re[{i}] += {d_re} * x_re[{j}] - {d_im} * x_im[{j}]; "
            f"y_im[{i}] += {d_re} * x_im[{j}] + {d_im} * x_re[{j}];")


def _backward_accumulate(i: int, j: int, d_re: str, d_im: Optional[str]) -> str:
    """The conjugate-transposed rotation of the cotangents ``c``."""
    if d_im is None:
        return f"g_re[{j}] += {d_re} * c_re[{i}]; g_im[{j}] += {d_re} * c_im[{i}];"
    return (f"g_re[{j}] += {d_re} * c_re[{i}] + {d_im} * c_im[{i}]; "
            f"g_im[{j}] += {d_re} * c_im[{i}] - {d_im} * c_re[{i}];")


def _generator_code(ell: int, offset: int) -> str:
    """``s_k += Re<c, G_k y>`` over the modes of ``l = ell`` (the non-zero
    entries of :func:`generators`)."""
    size = 2 * ell + 1
    block = generators()[:, offset:offset + size, offset:offset + size]
    lines = []
    for k in range(4):
        for i, j in zip(*np.nonzero(np.abs(block[k]) > 1e-12)):
            g = block[k, i, j]
            terms = []
            if abs(g.real) > 1e-12:
                terms.append(f"{_number(g.real)} * (c_re[{i}] * y_re[{j}] + c_im[{i}] * y_im[{j}])")
            if abs(g.imag) > 1e-12:
                terms.append(f"{_number(g.imag)} * (c_im[{i}] * y_re[{j}] - c_re[{i}] * y_im[{j}])")
            lines.append(f"s[{k}] += {' + '.join(terms)};")
    return "\n        ".join(lines)


_PROLOGUE = """
    uint gid = thread_position_in_grid.x;
    uint b = gid / N;
    uint t = gid % N;
    const device float* qp = quat + ((size_t)b * N + t) * 4;
    float q0 = qp[0], q1 = qp[1], q2 = qp[2], q3 = qp[3];
"""


def _blocks(body) -> str:
    """One scope per ``l`` (the modes of different ``l`` do not mix):
    ``body(ell, size, offset)`` gives the code of the block of the ``size =
    2 l + 1`` modes from index ``offset`` of :data:`MODES`."""
    blocks, offset = [], 0
    for ell in range(2, LMAX + 1):
        blocks.append("{" + body(ell, 2 * ell + 1, offset) + "    }")
        offset += 2 * ell + 1
    return "\n    ".join(blocks)


def _forward_block(ell: int, size: int, offset: int) -> str:
    return f"""
        float x_re[{size}], x_im[{size}], y_re[{size}], y_im[{size}];
        for (int i = 0; i < {size}; ++i) {{
            size_t at = ((size_t)b * {N_MODES} + {offset} + i) * N + t;
            x_re[i] = re[at];
            x_im[i] = im[at];
            y_re[i] = 0.0f;
            y_im[i] = 0.0f;
        }}
        {_rotation_code(ell, _forward_accumulate)}
        for (int i = 0; i < {size}; ++i) {{
            size_t at = ((size_t)b * {N_MODES} + {offset} + i) * N + t;
            out_re[at] = y_re[i];
            out_im[at] = y_im[i];
        }}
"""


def _backward_block(ell: int, size: int, offset: int) -> str:
    return f"""
        float y_re[{size}], y_im[{size}], c_re[{size}], c_im[{size}], g_re[{size}], g_im[{size}];
        for (int i = 0; i < {size}; ++i) {{
            size_t at = ((size_t)b * {N_MODES} + {offset} + i) * N + t;
            y_re[i] = out_re[at];
            y_im[i] = out_im[at];
            c_re[i] = cot_re[at];
            c_im[i] = cot_im[at];
            g_re[i] = 0.0f;
            g_im[i] = 0.0f;
        }}
        {_generator_code(ell, offset)}
        {_rotation_code(ell, _backward_accumulate)}
        for (int i = 0; i < {size}; ++i) {{
            size_t at = ((size_t)b * {N_MODES} + {offset} + i) * N + t;
            grad_re[at] = g_re[i];
            grad_im[at] = g_im[i];
        }}
"""


_FORWARD = _PROLOGUE + f"""
    {_blocks(_forward_block)}
"""

_BACKWARD = _PROLOGUE + f"""
    float s[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    {_blocks(_backward_block)}
    // the cotangent of q: sum_k s_k du_k/dq with u = dq conj(q) / |q|^2
    float norm = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
    device float* gq = grad_quat + ((size_t)b * N + t) * 4;
    gq[0] = (s[0] * q0 - s[1] * q1 - s[2] * q2 - s[3] * q3) / norm;
    gq[1] = (s[0] * q1 + s[1] * q0 + s[2] * q3 - s[3] * q2) / norm;
    gq[2] = (s[0] * q2 - s[1] * q3 + s[2] * q0 + s[3] * q1) / norm;
    gq[3] = (s[0] * q3 + s[1] * q2 - s[2] * q1 + s[3] * q0) / norm;
"""

_FORWARD_KERNEL = mx.fast.metal_kernel(
    name="nrsur7dq4_wigner",
    input_names=["re", "im", "quat"],
    output_names=["out_re", "out_im"],
    source=_FORWARD,
)
_BACKWARD_KERNEL = mx.fast.metal_kernel(
    name="nrsur7dq4_wigner_vjp",
    input_names=["quat", "out_re", "out_im", "cot_re", "cot_im"],
    output_names=["grad_re", "grad_im", "grad_quat"],
    source=_BACKWARD,
)


def _launch(kernel, inputs, size: int, n: int, outputs):
    threads = size * n
    return kernel(
        inputs=[x.astype(mx.float32) for x in inputs],
        template=[("N", n)],
        grid=(threads, 1, 1),
        threadgroup=(min(threads, 256), 1, 1),
        output_shapes=[shape for shape, _ in outputs],
        output_dtypes=[dtype for _, dtype in outputs],
    )


def forward(re, im, quat):
    """The inertial modes (real and imaginary parts, ``(B, n_modes, n)``)
    from the coprecessing modes ``re``, ``im`` and the quaternion ``quat``
    (``(B, n, 4)``)."""
    size, _, n = re.shape
    return tuple(_launch(_FORWARD_KERNEL, [re, im, quat], size, n, [(re.shape, mx.float32)] * 2))


def backward(quat, out_re, out_im, cot_re, cot_im):
    """The cotangents of the coprecessing modes and of ``quat`` from the
    inertial modes ``out_re``, ``out_im`` (:func:`forward`) and their
    cotangents."""
    size, _, n = out_re.shape
    shapes = [(out_re.shape, mx.float32), (out_re.shape, mx.float32), (quat.shape, mx.float32)]
    return tuple(_launch(_BACKWARD_KERNEL, [quat, out_re, out_im, cot_re, cot_im], size, n, shapes))


@mx.custom_function
def rotate(re, im, quat):
    """:func:`._frames.coprecessing_to_inertial` in a Metal kernel,
    differentiable (``mx.grad``, ``mx.vjp``)."""
    return forward(re, im, quat)


@rotate.vjp
def _rotate_vjp(primals, cotangents, outputs):
    _, _, quat = primals
    out_re, out_im = outputs
    cot_re, cot_im = cotangents
    return backward(quat, out_re, out_im, cot_re, cot_im)
