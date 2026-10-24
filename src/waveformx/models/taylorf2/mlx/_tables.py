# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""MLX for models written as a table per binary and a formula per
frequency (TaylorF2, IMRPhenomD), as IMRPhenomXAS's MLX port
(:mod:`...xas.mlx`), which this module generalises.

A model gives the JAX function of its table (the inputs: the eight
aligned-spin parameters and the first frequency), the layout of the table
(``OFFSETS``, ``SIZE``) and, in Metal, its waveform at one frequency: a
``struct Point``, ``model_point(f, T, p)`` (the amplitude ``p.amp`` and the
cosine and sine ``p.c``, ``p.sn`` of the phase) and ``model_tangent(p, T,
dT)`` (the derivatives of the amplitude and of the phase along one
direction, from the derivatives ``dT`` of the table), and which reference
frequency anchors the leading term of the phase.

* The set-up (one thread per binary) is generated from the JAX function:
  traced in float32, lowered to straight-line code
  (:mod:`...xpnr.mlx._program`) and written as a Metal function, with its
  forward-mode derivative as a function of dual numbers (one thread per
  binary and parameter).
* The waveform at the frequencies is in three kernels of one thread per
  binary and frequency: the polarizations; with their Jacobian along the
  eight parameters of the aligned-spin binary; and the derivatives
  contracted with cotangents and summed over the frequencies (the VJP).

The leading term of the phase, ``T0 f^(-5/3)``, and its value at the
reference frequency, which the JAX models compute in float64, are computed
in double-float arithmetic (pairs of floats with exact sums and products by
fused multiply-adds): ``T0`` and the anchor by the set-up kernels (the
float32 values of the generated code are replaced), the term at each
frequency, its sum with the rest of the phase and the reduction to ``[-pi,
pi]`` by the point kernels.
"""

from __future__ import annotations

import functools
import re

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np

from ...._options import STRAIN_SCALE
from ...xas.core import K0
from ...xpnr._math import precision
from ...xpnr.mlx import _metal
from ...xpnr.mlx._api import _frequencies, batched
from ...xpnr.mlx._program import lower

__all__ = ["N_INPUTS", "N_DIRECTIONS", "COLUMNS", "DOUBLE_FLOAT", "TableKernels", "polarizations",
           "polarizations_and_jacobian", "converters", "reference"]

#: The inputs of the set-up: the eight parameters of the tables and the
#: first frequency.
N_INPUTS = 9
#: The parameters whose derivatives are computed (all but ``f_min``).
N_DIRECTIONS = 8
#: Their columns among the twelve parameters of a binary (``mass_1``,
#: ``mass_2``, ``spin_1`` (3), ``spin_2`` (3), ``luminosity_distance``,
#: ``inclination``, ``phi_ref``, ``f_ref``): the in-plane spins do not enter.
COLUMNS = (0, 1, 4, 7, 8, 9, 10, 11)
GROUP = 256
TWO_PI_HI = float(np.float32(2.0 * np.pi))
TWO_PI_LO = float(2.0 * np.pi - np.float64(np.float32(2.0 * np.pi)))

#: Double-float arithmetic, the leading term of the phase and the
#: reduction to ``[-pi, pi]`` (as :mod:`...xas.mlx._kernels`).
DOUBLE_FLOAT = """
// Double-float arithmetic (an unevaluated sum hi + lo of floats): exact
// sums and products (Metal keeps them: no reassociation, fma exact).
struct DF {
    float hi, lo;
};

inline DF df_two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return DF{s, (a - (s - bb)) + (b - bb)};
}

inline DF df_quick(float a, float b) {
    float s = a + b;
    return DF{s, b - (s - a)};
}

inline DF df_two_prod(float a, float b) {
    float p = a * b;
    return DF{p, metal::fma(a, b, -p)};
}

inline DF df_add(DF a, DF b) {
    DF s = df_two_sum(a.hi, b.hi);
    return df_quick(s.hi, s.lo + a.lo + b.lo);
}

inline DF df_mul(DF a, DF b) {
    DF p = df_two_prod(a.hi, b.hi);
    return df_quick(p.hi, metal::fma(a.hi, b.lo, metal::fma(a.lo, b.hi, p.lo)));
}

inline DF df_mul(DF a, float b) {
    DF p = df_two_prod(a.hi, b);
    return df_quick(p.hi, metal::fma(a.lo, b, p.lo));
}

// f^(-5/3): one Newton step on y^3 f^5 = 1 from the float value
inline DF df_m53(float f) {
    float y = metal::precise::exp(-5.0f / 3.0f * metal::precise::log(f));
    DF f2 = df_two_prod(f, f);
    DF f5 = df_mul(df_mul(f2, f2), f);
    DF y3 = df_mul(df_two_prod(y, y), y);
    DF w = df_mul(y3, f5);
    float eps = (1.0f - w.hi) - w.lo;
    return df_quick(y, y * eps / 3.0f);
}

// T0 = K0 (m1 + m2)^(1/3) / (m1 m2) and A = T0 f_ref^(-5/3) (zero unless
// anchored), as the tables compute them in float64: hi, lo of {T0, A}. With
// the mass-scaled reference (mass_scaled non-zero: -3 / 128 (pi M f_ref)^(-5/3)),
// A = mass_scaled (m1 + m2)^2 / (m1 m2).
inline void leading_term(float m1, float m2, float f_ref, bool anchored, DF mass_scaled, thread float* hi,
                         thread float* lo) {
    DF mass = df_two_sum(m1, m2);
    float c = metal::precise::exp(metal::precise::log(mass.hi) / 3.0f);
    DF c3 = df_mul(df_two_prod(c, c), c);
    float eps = ((mass.hi - c3.hi) + (mass.lo - c3.lo)) / mass.hi;
    DF cbrt_mass = df_quick(c, c * eps / 3.0f);
    DF product = df_two_prod(m1, m2);
    float q = cbrt_mass.hi / product.hi;
    DF qp = df_mul(product, q);
    float r = ((cbrt_mass.hi - qp.hi) + (cbrt_mass.lo - qp.lo)) / product.hi;
    DF t0 = df_mul(df_quick(q, r), DF{K0_HI, K0_LO});
    DF anchor = DF{0.0f, 0.0f};
    if (anchored && mass_scaled.hi != 0.0f) {
        DF square = df_mul(mass, mass);
        float s = square.hi / product.hi;
        DF sp = df_mul(product, s);
        float sr = ((square.hi - sp.hi) + (square.lo - sp.lo)) / product.hi;
        anchor = df_mul(df_quick(s, sr), mass_scaled);
    } else if (anchored) {
        anchor = df_mul(t0, df_m53(f_ref));
    }
    hi[0] = t0.hi;
    lo[0] = t0.lo;
    hi[1] = anchor.hi;
    lo[1] = anchor.lo;
}

// a double-float phase reduced to [-pi, pi] (2 pi in two parts)
inline float reduce_phase(DF phi) {
    float k = metal::rint(phi.hi * INV_TWO_PI);
    float r = metal::fma(-k, TWO_PI_HI, phi.hi);
    return metal::fma(-k, TWO_PI_LO, r) + phi.lo;
}

// h_+, h_x (re, im) along one direction from the derivatives of the
// amplitude and of the phase
inline float4 polarization_tangent(float amp, float c, float sn, float damp, float dphi, const device float* T,
                                   const device float* dT) {
    float kp = T[O_KP];
    float kc = T[O_KC];
    float gp = dT[O_KP * ND] * amp + kp * damp;
    float gc = dT[O_KC * ND] * amp + kc * damp;
    float rp = kp * amp * dphi;
    float rc = kc * amp * dphi;
    return float4(gp * c - rp * sn, gp * sn + rp * c, -gc * sn - rc * c, gc * c - rc * sn);
}
"""

_POINT_HEAD = """
uint n = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
uint n_points = dims[0];
uint shared_f = dims[2];
bool active = n < n_points;
uint nn = active ? n : n_points - 1;
float f = freq[(shared_f ? 0 : (size_t)b * n_points) + nn];
const device float* T = table + (size_t)b * NT;
Point p;
model_point(f, T, p);
size_t at = (size_t)b * n_points + nn;
"""

_VALUES = _POINT_HEAD + """
if (!active) return;
float kp = T[O_KP] * p.amp;
float kc = T[O_KC] * p.amp;
hp[2 * at] = kp * p.c;
hp[2 * at + 1] = kp * p.sn;
hc[2 * at] = -kc * p.sn;
hc[2 * at + 1] = kc * p.c;
"""

_JACOBIAN = _VALUES.replace("if (!active) return;\n", "if (!active) return;\nconst device float* dT = dtable + (size_t)b * NT * ND;\n") + """
for (uint q = 0; q < 12; ++q) {
    jp[(at * 12 + q) * 2] = 0.0f; jp[(at * 12 + q) * 2 + 1] = 0.0f;
    jc[(at * 12 + q) * 2] = 0.0f; jc[(at * 12 + q) * 2 + 1] = 0.0f;
}
for (uint d = 0; d < ND; ++d) {
    float4 g = model_tangent(p, T, dT + d);
    size_t o = (at * 12 + COLUMN[d]) * 2;
    jp[o] = g.x; jp[o + 1] = g.y; jc[o] = g.z; jc[o + 1] = g.w;
}
"""

_VJP = _POINT_HEAD + """
const device float* dT = dtable + (size_t)b * NT * ND;
float4 ct = active ? float4(cot_p[2 * at], cot_p[2 * at + 1], cot_c[2 * at], cot_c[2 * at + 1]) : float4(0.0f);
threadgroup float partial[(GROUP / 32) * ND];
uint lane = thread_index_in_simdgroup;
uint warp = simdgroup_index_in_threadgroup;
for (uint d = 0; d < ND; ++d) {
    float v = 0.0f;
    if (active && any(ct != 0.0f)) {
        // (no contribution where the cotangent vanishes, whatever the
        // derivative, as in reverse mode)
        float4 g = model_tangent(p, T, dT + d);
        v = metal::dot(g, ct);
    }
    v = metal::simd_sum(v);
    if (lane == 0) partial[warp * ND + d] = v;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (thread_index_in_threadgroup < ND) {
    uint d = thread_index_in_threadgroup;
    float total = 0.0f;
    for (uint w = 0; w < GROUP / 32; ++w) total += partial[w * ND + d];
    uint groups = (n_points + GROUP - 1) / GROUP;
    out[((size_t)b * groups + threadgroup_position_in_grid.x) * ND + d] = total;
}
"""

# ``ANCHOR`` (the model's) sets ``float f_ref``, ``bool anchored`` and ``DF
# mass_scaled`` (:func:`anchor`) from the inputs ``x`` and the table ``y``
# (``{v}``: ``.v`` for dual numbers).
_SETUP = """
uint b = thread_position_in_grid.x;
if (b >= dims[0]) return;
float x[NI];
for (uint k = 0; k < NI; ++k) x[k] = params[(size_t)b * NI + k];
float y[NT];
model_table(x, y);
{anchor}
float hi[2], lo[2];
leading_term(x[0], x[1], f_ref, anchored, mass_scaled, hi, lo);
for (uint k = 0; k < 2; ++k) {{
    y[O_BIG + k] = hi[k];
    y[O_BIG_LO + k] = lo[k];
}}
for (uint k = 0; k < NT; ++k) table[(size_t)b * NT + k] = y[k];
"""

_SETUP_JVP = """
uint tid = thread_position_in_grid.x;
if (tid >= dims[0] * ND) return;
uint b = tid / ND;
uint d = tid % ND;
WjDual x[NI];
for (uint k = 0; k < NI; ++k) x[k] = WjDual(params[(size_t)b * NI + k], k == d ? 1.0f : 0.0f);
WjDual y[NT];
model_table_jvp(x, y);
{anchor}
float hi[2], lo[2];
leading_term(x[0].v, x[1].v, f_ref, anchored, mass_scaled, hi, lo);
for (uint k = 0; k < 2; ++k) {{
    y[O_BIG + k].v = hi[k];
    y[O_BIG_LO + k].v = lo[k];
}}
for (uint k = 0; k < NT; ++k) {{
    if (d == 0) table[(size_t)b * NT + k] = y[k].v;
    dtable[((size_t)b * NT + k) * ND + d] = y[k].d;
}}
"""


def mass_scaled(reference_mf):
    """The Metal ``DF`` constant of the anchor of the mass-scaled reference,
    ``-3 / 128 (pi reference_mf)^(-5/3)`` (zero without)."""
    if reference_mf is None:
        return "DF{0.0f, 0.0f}"
    value = -3.0 / 128.0 * (np.pi * reference_mf) ** (-5.0 / 3.0)
    hi = float(np.float32(value))
    return f"DF{{{hi!r}f, {float(np.float32(value - hi))!r}f}}"


def kernel_name(name, reference_mf):
    """A kernel's name for the reference convention (MLX caches kernels by
    name)."""
    return name if reference_mf is None else f"{name}_mf{repr(float(reference_mf)).replace('.', 'p').replace('-', 'm')}"


def _fast(source):
    """``source`` (generated) with Metal's fast division, exponential and
    logarithm: on a single thread, as the set-up of one binary runs, the
    precise ones take hundreds of nanoseconds each."""
    source = re.sub(r"= \(([\w.+-]+) / ([\w.+-]+)\);", r"= metal::fast::divide(\1, \2);", source)
    return source.replace("metal::precise::exp(", "metal::fast::exp(").replace("metal::precise::log(", "metal::fast::log(")


def _round(n, group):
    return max(-(-n // group) * group, group)


class TableKernels:
    """The kernels of a model for one scale of the strain (built on first
    use).

    ``name``: the prefix of the kernels; ``core``: the model's module of
    tables (``OFFSETS``, ``SIZE``); ``table(params, f_min, amp_scale)``: its
    JAX function of the table; ``point``: its Metal code at one frequency;
    ``anchor``: the Metal code that sets ``f_ref`` and ``anchored``;
    ``constants``: more Metal constants of the model. With ``fast``, the
    set-up uses Metal's fast division, exponential and logarithm."""

    def __init__(self, name, core, table, point, anchor, constants, amp_scale, fast=True):
        self.name, self.core, self.table_function = name, core, table
        self.point, self.anchor, self.constants = point, anchor, constants
        self.amp_scale = float(amp_scale)
        self.fast = bool(fast)

    def _function(self):
        def function(x):
            with precision(np.float32):
                params = tuple(x[i] for i in range(8))
                return self.table_function(params, x[8], self.amp_scale)

        return function

    def _programs(self):
        function = self._function()
        aval = jax.ShapeDtypeStruct((N_INPUTS,), np.float32)
        values = lower(jax.make_jaxpr(function)(aval))

        def jvp(x, dx):
            return jax.jvp(function, (x,), (dx,))

        tangents = lower(jax.make_jaxpr(jvp)(aval, aval))
        return values, tangents

    def _constants(self):
        lines = [f"constant constexpr uint O_{name.upper()} = {start};"
                 for name, (start, _) in self.core.OFFSETS.items()]
        lines.append(f"constant constexpr uint NT = {self.core.SIZE};")
        lines.append(f"constant constexpr uint ND = {N_DIRECTIONS};")
        lines.append(f"constant constexpr uint NI = {N_INPUTS};")
        lines.append(f"constant constexpr uint GROUP = {GROUP};")
        lines.append(f"constant constexpr float TWO_PI_HI = {_metal.literal(TWO_PI_HI)};")
        lines.append(f"constant constexpr float TWO_PI_LO = {TWO_PI_LO!r}f;")
        lines.append(f"constant constexpr float INV_TWO_PI = {_metal.literal(1.0 / (2.0 * np.pi))};")
        k0_hi = float(np.float32(K0))
        lines.append(f"constant constexpr float K0_HI = {k0_hi!r}f;")
        lines.append(f"constant constexpr float K0_LO = {float(np.float32(K0 - k0_hi))!r}f;")
        lines.append(f"constant constexpr uint COLUMN[ND] = {{{', '.join(str(c) for c in COLUMNS)}}};")
        return "\n".join(lines) + "\n" + self.constants

    @functools.cached_property
    def _header(self):
        values, tangents = self._programs()
        setup = (_metal.function_source(values, "model_table")
                 + _metal.function_source(tangents, "model_table_jvp", n_primal=1))
        if self.fast:
            setup = _fast(setup)
        return _metal.HEADER + _metal.DUAL + self._constants() + setup + DOUBLE_FLOAT + self.point

    def _kernel(self, name, inputs, outputs, source):
        name = f"{self.name}_{name}_fast" if self.fast else f"{self.name}_{name}"
        return mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source,
                                    header=self._header)

    @functools.cached_property
    def setup(self):
        return self._kernel("setup", ["params", "dims"], ["table"], _SETUP.format(anchor=self.anchor.format(v="")))

    @functools.cached_property
    def setup_jvp(self):
        return self._kernel("setup_jvp", ["params", "dims"], ["table", "dtable"],
                            _SETUP_JVP.format(anchor=self.anchor.format(v=".v")))

    @functools.cached_property
    def values(self):
        return self._kernel("values", ["table", "freq", "dims"], ["hp", "hc"], _VALUES)

    @functools.cached_property
    def jacobian(self):
        return self._kernel("jacobian", ["table", "dtable", "freq", "dims"], ["hp", "hc", "jp", "jc"], _JACOBIAN)

    @functools.cached_property
    def vjp(self):
        return self._kernel("vjp", ["table", "dtable", "freq", "dims", "cot_p", "cot_c"], ["out"], _VJP)

    # --- running ---------------------------------------------------------

    def table(self, params):
        """The tables ``(B, SIZE)`` of the inputs ``(B, N_INPUTS)``."""
        batch = params.shape[0]
        (table,) = self.setup(inputs=[params, mx.array([batch], dtype=mx.uint32)], grid=(_round(batch, 64), 1, 1),
                              threadgroup=(64, 1, 1), output_shapes=[(batch, self.core.SIZE)],
                              output_dtypes=[mx.float32])
        return table

    def table_jvp(self, params):
        """The tables and their derivatives ``(B, SIZE, N_DIRECTIONS)``."""
        batch = params.shape[0]
        threads = batch * N_DIRECTIONS
        return self.setup_jvp(inputs=[params, mx.array([batch], dtype=mx.uint32)], grid=(_round(threads, 64), 1, 1),
                              threadgroup=(64, 1, 1),
                              output_shapes=[(batch, self.core.SIZE), (batch, self.core.SIZE, N_DIRECTIONS)],
                              output_dtypes=[mx.float32, mx.float32])

    @staticmethod
    def _dims(freq, batch):
        n = freq.shape[-1]
        return n, mx.array([n, batch, 1 if freq.shape[0] == 1 else 0], dtype=mx.uint32)

    def polarizations(self, table, freq):
        """``h_+``, ``h_x`` as float32 ``(B, N, 2)`` (real, imaginary)."""
        batch = table.shape[0]
        n, dims = self._dims(freq, batch)
        return self.values(inputs=[table, freq, dims], grid=(_round(n, GROUP), batch, 1), threadgroup=(GROUP, 1, 1),
                           output_shapes=[(batch, n, 2)] * 2, output_dtypes=[mx.float32] * 2)

    def polarizations_and_jacobian(self, table, dtable, freq):
        """``h_+``, ``h_x`` ``(B, N, 2)`` and their Jacobians ``(B, N, 12, 2)``."""
        batch = table.shape[0]
        n, dims = self._dims(freq, batch)
        return self.jacobian(inputs=[table, dtable, freq, dims], grid=(_round(n, GROUP), batch, 1),
                             threadgroup=(GROUP, 1, 1),
                             output_shapes=[(batch, n, 2), (batch, n, 2), (batch, n, 12, 2), (batch, n, 12, 2)],
                             output_dtypes=[mx.float32] * 4)

    def cotangent_product(self, table, dtable, freq, cot_p, cot_c):
        """``sum_n Re(conj(J) cot)`` per binary: ``(B, N_DIRECTIONS)``."""
        batch = table.shape[0]
        n, dims = self._dims(freq, batch)
        groups = -(-n // GROUP)
        (out,) = self.vjp(inputs=[table, dtable, freq, dims, cot_p, cot_c], grid=(groups * GROUP, batch, 1),
                          threadgroup=(GROUP, 1, 1), output_shapes=[(batch, groups, N_DIRECTIONS)],
                          output_dtypes=[mx.float32])
        return mx.sum(out, axis=1) if groups > 1 else out[:, 0]


# --- the pure functions ------------------------------------------------------


def _host_inputs(binary, frequencies):
    """The inputs of the set-up ``(B, 9)``, the frequencies ``(B or 1, N)``
    and whether there is a batch axis, for a binary of numbers or numpy
    arrays: the parameters made on the host (one transfer). ``None`` if
    any field of the binary is an MLX array."""
    if any(isinstance(value, mx.array) for value in binary):
        return None
    fields = [np.asarray(value, dtype=np.float32) for value in binary]
    spin = [name in ("spin_1", "spin_2") for name in binary._fields]
    sizes = {value.shape[0] for value, s in zip(fields, spin) if value.ndim == (2 if s else 1)}
    if len(sizes) > 1:
        raise ValueError(f"the fields of the binary have different batch sizes {sorted(sizes)}")
    batch = next(iter(sizes), 1)
    b = dict(zip(binary._fields, fields))
    columns = [b["mass_1"], b["mass_2"], b["spin_1"][..., 2], b["spin_2"][..., 2], b["luminosity_distance"],
               b["inclination"], b["phi_ref"], b["f_ref"]]
    x = np.empty((batch, N_INPUTS - 1), dtype=np.float32)
    for i, column in enumerate(columns):
        x[:, i] = column
    if isinstance(frequencies, mx.array):
        f = _frequencies(frequencies, batch)
        x = mx.concatenate([mx.array(x), mx.broadcast_to(f[:, :1], (batch, 1))], axis=1)
    else:
        f = np.asarray(frequencies, dtype=np.float32)
        f = f[None] if f.ndim == 1 else f
        if f.shape[0] not in (1, batch):
            raise ValueError(f"frequencies of shape {f.shape} for {batch} binaries")
        x = mx.array(np.concatenate([x, np.broadcast_to(f[:, :1], (batch, 1))], axis=1))
        f = mx.array(f)
    return x, f, bool(sizes)


def _inputs(binary, frequencies):
    """The inputs of the set-up ``(B, 9)`` (the aligned-spin parameters and
    the first frequency), differentiable with respect to the fields of
    ``binary``, the frequencies ``(B or 1, N)`` and whether the binary had
    a batch axis."""
    host = _host_inputs(binary, frequencies)
    if host is not None:
        return host
    binary, has_batch = batched(binary)
    b = binary
    batch = b.mass_1.shape[0]
    f = _frequencies(frequencies, batch)
    x = mx.stack([b.mass_1, b.mass_2, b.spin_1[:, 2], b.spin_2[:, 2], b.luminosity_distance, b.inclination,
                  b.phi_ref, b.f_ref, mx.broadcast_to(f[:, 0], (batch,))], axis=1)
    return x, f, has_batch


def _complex(x):
    """``(B, N, 2)`` float32 as ``(B, N)`` complex64."""
    return x.view(mx.complex64).reshape(x.shape[:-1])


def _real(z):
    """``(B, N)`` complex64 as ``(B, N, 2)`` float32."""
    return z.reshape(z.shape + (1,)).view(mx.float32)


@functools.lru_cache(maxsize=None)
def _core(kernels):
    """``f(inputs (B, 9), frequencies) -> (h_+, h_x)`` with its custom VJP."""
    k = kernels

    @mx.custom_function
    def core(x, frequencies):
        hp, hc = k.polarizations(k.table(x), frequencies)
        return _complex(hp), _complex(hc)

    @core.vjp
    def core_vjp(primals, cotangents, outputs):
        x, frequencies = primals
        table, dtable = k.table_jvp(mx.stop_gradient(x))
        cot_p, cot_c = (mx.zeros_like(o) if c is None else c for c, o in zip(cotangents, outputs))
        grad = k.cotangent_product(table, dtable, frequencies, _real(cot_p.astype(mx.complex64)),
                                   _real(cot_c.astype(mx.complex64)))
        # (none for the first frequency, which is not a parameter)
        grad = mx.concatenate([grad, mx.zeros((grad.shape[0], 1), dtype=grad.dtype)], axis=1)
        return grad, mx.zeros_like(frequencies)

    return core


def reference(reference_mf):
    """``reference_mf`` checked as ``WaveformOptions`` does (``None``: LAL's
    convention)."""
    from ...._options import WaveformOptions

    return WaveformOptions(reference_mf=reference_mf).reference_mf


def scale(scaled_strain):
    """The scale of the strain."""
    return STRAIN_SCALE if scaled_strain else 1.0


def polarizations(kernels, binary, frequencies):
    """``(h_+, h_x)`` (complex64) of the model of ``kernels`` (a
    :class:`TableKernels`) at ``frequencies``: ``(N,)`` each for one
    binary, ``(B, N)`` for a batch. Differentiable with ``mx.grad`` with
    respect to the fields of the binary."""
    x, f, has_batch = _inputs(binary, frequencies)
    hp, hc = _core(kernels)(x, f)
    return (hp, hc) if has_batch else (hp[0], hc[0])


def polarizations_and_jacobian(kernels, binary, frequencies):
    """``(h_+, h_x)`` and their Jacobians with respect to the twelve
    parameters of the binary: each ``(..., N, 12)`` complex64 (zero for
    the in-plane spins)."""
    x, f, has_batch = _inputs(binary, frequencies)
    table, dtable = kernels.table_jvp(mx.stop_gradient(x))
    hp, hc, jp, jc = kernels.polarizations_and_jacobian(table, dtable, f)
    out = (_complex(hp), _complex(hc)), (_complex(jp), _complex(jc))
    if has_batch:
        return out
    return (out[0][0][0], out[0][1][0]), (out[1][0][0], out[1][1][0])


# --- the reference conventions -----------------------------------------------


def _field(value):
    return value if isinstance(value, mx.array) else mx.array(np.asarray(value, dtype=np.float32))


def converters(to_mass_scaled, to_physical):
    """The MLX versions of a model's converters (JAX functions of
    :mod:`..reference`): the JAX function of the new ``phi_ref``, traced in
    float32 and lowered to MLX operations (:mod:`...xpnr.mlx._program`),
    so that ``mx.grad`` follows them. In float32: for binary neutron stars
    from low frequencies the phase between the references (``~10^4``
    radians) costs ``~10^-3`` radians of ``phi_ref``; convert in float64
    with the JAX functions where that matters."""
    from ...._binary import Binary
    from ...xpnr.mlx._mlxrun import run

    def jax_binary(x):
        z = jnp.zeros((), x.dtype)
        return Binary(x[0], x[1], jnp.stack([z, z, x[2]]), jnp.stack([z, z, x[3]]), 1.0, 0.0, x[4], x[5])

    @functools.lru_cache(maxsize=None)
    def program(kind, reference_mf, with_f_min):
        if kind == "scaled":
            def function(x):
                kw = dict(f_min=x[6]) if with_f_min else {}
                return to_mass_scaled(jax_binary(x), reference_mf, **kw).phi_ref
        else:
            def function(x):
                return to_physical(jax_binary(x), x[6], reference_mf).phi_ref
        return lower(jax.make_jaxpr(function)(jax.ShapeDtypeStruct((7,), np.float32)))

    def inputs(binary, last):
        fields = [_field(binary.mass_1), _field(binary.mass_2), _field(binary.spin_1)[..., 2],
                  _field(binary.spin_2)[..., 2], _field(binary.phi_ref), _field(binary.f_ref), _field(last)]
        return [fields]

    def to_mass_scaled_reference(binary, reference_mf, f_min=None):
        with_f_min = f_min is not None
        phi = run(program("scaled", float(reference_mf), with_f_min),
                  inputs(binary, f_min if with_f_min else 0.0))[0][0]
        return binary._replace(phi_ref=phi, f_ref=mx.zeros_like(_field(binary.f_ref)))

    def to_physical_reference(binary, f_ref, reference_mf):
        phi = run(program("physical", float(reference_mf), False), inputs(binary, f_ref))[0][0]
        return binary._replace(phi_ref=phi, f_ref=mx.broadcast_to(_field(f_ref), phi.shape))

    return to_mass_scaled_reference, to_physical_reference
