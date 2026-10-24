# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The Metal kernels of IMRPhenomXAS.

* The set-up (:func:`..core.table`, one thread per binary) is generated
  from the JAX function: traced in float32, lowered to straight-line code
  (:mod:`...xpnr.mlx._program`) and written as a Metal function, with its
  forward-mode derivative as a function of dual numbers (one thread per
  binary and parameter).
* The waveform at the frequencies (:func:`..core.evaluate`) is written by
  hand, in three kernels of one thread per binary and frequency: the
  polarizations; with their derivatives along the eight parameters of the
  aligned-spin binary (the Jacobian); and the derivatives contracted with
  cotangents and summed over the frequencies in the kernel (the VJP, one
  vector of eight per binary). The derivatives at a frequency are linear in
  those of the table, which the set-up kernel gives.

The leading term of the inspiral phase, which :func:`..core.phase`
computes in float64, is computed here in double-float arithmetic (pairs of
floats, with exact sums and products by fused multiply-adds): its
coefficient and its value at the reference frequency by the set-up kernels
(the values of the generated code, float32, are replaced), its value at
each frequency, the sum with the rest of the phase and the reduction to
``[-pi, pi]`` by the point kernels.
"""

from __future__ import annotations

import functools
import re

import jax
import mlx.core as mx
import numpy as np

from ...xpnr._math import precision
from ...xpnr.mlx import _metal
from ...xpnr.mlx._program import lower
from .. import core

__all__ = ["N_INPUTS", "N_DIRECTIONS", "Kernels", "kernels"]

#: The inputs of the set-up: the eight parameters of :func:`..core.table`
#: and the first frequency.
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


def _table_function(amp_scale):
    def function(x):
        with precision(np.float32):
            params = tuple(x[i] for i in range(8))
            return core.table(params, x[8], amp_scale)

    return function


def _programs(amp_scale):
    """The programs of the set-up and of its JVP."""
    function = _table_function(amp_scale)
    aval = jax.ShapeDtypeStruct((N_INPUTS,), np.float32)
    values = lower(jax.make_jaxpr(function)(aval))

    def jvp(x, dx):
        out, tangent = jax.jvp(function, (x,), (dx,))
        return out, tangent

    tangents = lower(jax.make_jaxpr(jvp)(aval, aval))
    return values, tangents


def _constants():
    lines = []
    for name, (start, _) in core.OFFSETS.items():
        lines.append(f"constant constexpr uint O_{name.upper()} = {start};")
    lines.append(f"constant constexpr uint NT = {core.SIZE};")
    lines.append(f"constant constexpr uint ND = {N_DIRECTIONS};")
    lines.append(f"constant constexpr int NB = {len(core.PHASE_BASIS)};")
    lines.append(f"constant constexpr float TWO_PI_HI = {_metal.literal(TWO_PI_HI)};")
    lines.append(f"constant constexpr float TWO_PI_LO = {TWO_PI_LO!r}f;")
    lines.append(f"constant constexpr float INV_TWO_PI = {_metal.literal(1.0 / (2.0 * np.pi))};")
    k0_hi = float(np.float32(core.K0))
    lines.append(f"constant constexpr float K0_HI = {k0_hi!r}f;")
    lines.append(f"constant constexpr float K0_LO = {float(np.float32(core.K0 - k0_hi))!r}f;")
    const = core.PHASE_BASIS.index((0, 0))
    for r, terms in enumerate(core.REGION_TERMS):
        terms = [k for k in terms if k != const]
        lines.append(f"constant constexpr int NTERMS{r} = {len(terms)};")
        lines.append(f"constant constexpr int TERMS{r}[NTERMS{r}] = {{{', '.join(str(k) for k in terms)}}};")
    lines.append(f"constant constexpr int CONST_TERM = {const};")
    columns = ", ".join(str(c) for c in COLUMNS)
    lines.append(f"constant constexpr uint COLUMN[ND] = {{{columns}}};")
    return "\n".join(lines) + "\n"


#: The waveform at one frequency (shared by the point kernels): the basis
#: of the phase, the regions, the phase and the amplitude, and what their
#: derivatives need; the leading term of the inspiral phase in double-float
#: arithmetic (as core.phase in float64).
POINT = """
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

// T0 = K0 (m1 + m2)^(1/3) / (m1 m2) and A = T0 f_ref^(-5/3) (zero past the
// inspiral), as core.table computes them in float64: out = {T0, A}, lo
inline void xas_big(float m1, float m2, float f_ref, float f_phase_in, thread float* hi, thread float* lo) {
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
    DF anchor = f_ref >= f_phase_in ? DF{0.0f, 0.0f} : df_mul(t0, df_m53(f_ref));
    hi[0] = t0.hi;
    lo[0] = t0.lo;
    hi[1] = anchor.hi;
    lo[1] = anchor.lo;
}

struct XasPoint {
    float f, t, m76, y;
    float basis[NB];
    int rp, ra;
    float z, at;
    float poly, den, u, e, s;
    float amp, phi, c, sn;
};

// sum_k row[k stride] basis[k] over the terms of region r (and the
// constant)
inline float xas_phase_sum(int r, const device float* row, thread const float* b, uint stride) {
    float total = row[CONST_TERM * stride];
    if (r == 0) {
        for (int i = 0; i < NTERMS0; ++i) total = metal::fma(row[TERMS0[i] * stride], b[TERMS0[i]], total);
    } else if (r == 1) {
        for (int i = 0; i < NTERMS1; ++i) total = metal::fma(row[TERMS1[i] * stride], b[TERMS1[i]], total);
    } else {
        for (int i = 0; i < NTERMS2; ++i) total = metal::fma(row[TERMS2[i] * stride], b[TERMS2[i]], total);
    }
    return total;
}

inline void xas_point(float f, const device float* T, thread XasPoint& p) {
    p.f = f;
    float m1 = 1.0f / f;
    float lf = metal::precise::log(f);
    // (the cube root as in core.frequency_powers)
    float t = metal::precise::exp(lf / 3.0f);
    t = t * (4.0f - t * t * t * m1) / 3.0f;
    float m76 = m1 / metal::precise::sqrt(t);
    float t2 = t * t;
    float mt1 = t2 * m1;
    float mt2 = mt1 * mt1;
    float m2 = m1 * m1;
    p.t = t;
    p.m76 = m76;
    thread float* b = p.basis;
    b[0] = m1 * mt2; b[1] = m1; b[2] = mt2; b[3] = mt1; b[4] = lf; b[5] = 1.0f; b[6] = t; b[7] = t * lf;
    b[8] = t2; b[9] = f; b[10] = f * lf; b[11] = f * t; b[12] = f * t2; b[13] = f * f; b[14] = m2; b[15] = m2 * m1;
    int rp = int(f >= T[O_F_PHASE_IN]) + int(f > T[O_F_PHASE_IM]);
    int ra = int(f >= T[O_F_AMP_IN]) + int(f > T[O_F_AMP_RD]);
    p.rp = rp;
    p.ra = ra;
    const device float* row = T + O_PHASE + rp * NB;
    float rest = xas_phase_sum(rp, row, b, 1);
    float y = f - T[O_SHIFT];
    p.y = y;
    p.z = 0.0f;
    p.at = 0.0f;
    if (rp > 0) {
        p.z = T[O_ATAN_W + rp] * y;
        p.at = metal::precise::atan(p.z);
        rest = metal::fma(T[O_ATAN_A + rp], p.at, rest);
    }
    DF big = DF{-T[O_BIG + 1], -T[O_BIG_LO + 1]};
    if (rp == 0) {
        big = df_add(df_mul(DF{T[O_BIG], T[O_BIG_LO]}, df_m53(f)), big);
    }
    DF phi = df_add(big, DF{rest, 0.0f});
    float amp;
    if (ra == 0) {
        float poly = T[O_AMP_INS + 9];
        for (int k = 8; k >= 0; --k) {
            poly = metal::fma(poly, t, T[O_AMP_INS + k]);
        }
        p.poly = poly;
        amp = m76 * poly;
    } else if (ra == 1) {
        float den = T[O_AMP_INT_D + 4];
        for (int j = 3; j >= 0; --j) {
            den = metal::fma(den, f, T[O_AMP_INT_D + j]);
        }
        p.den = den;
        amp = T[O_AMP_INT_N] / den;
    } else {
        float u = T[O_AMP_RD_S] * y;
        float e = metal::precise::exp(-T[O_AMP_RD_GR] * u);
        float s = metal::fma(u, u, T[O_AMP_RD_G]);
        p.u = u;
        p.e = e;
        p.s = s;
        amp = T[O_AMP_RD_N] * m76 * e / s;
    }
    if (f > T[O_F_CUT]) {
        amp = 0.0f;
    }
    // (the phase reduced to [-pi, pi] with 2 pi in two parts)
    float k = metal::rint(phi.hi * INV_TWO_PI);
    float r = metal::fma(-k, TWO_PI_HI, phi.hi);
    r = metal::fma(-k, TWO_PI_LO, r) + phi.lo;
    p.amp = amp;
    p.phi = r;
    p.c = metal::precise::cos(r);
    p.sn = metal::precise::sin(r);
}

// The derivatives of h_+ and h_x (re, im) along one direction, from the
// derivatives dT of the table (stride ND).
inline float4 xas_tangent(thread const XasPoint& p, const device float* T, const device float* dT) {
    int rp = p.rp;
    const device float* drow = dT + (O_PHASE + rp * NB) * ND;
    float dphi = xas_phase_sum(rp, drow, p.basis, ND);
    dphi -= dT[(O_BIG + 1) * ND];
    if (rp == 0) {
        dphi = metal::fma(dT[O_BIG * ND], p.basis[0], dphi);
    }
    float dshift = dT[O_SHIFT * ND];
    if (rp > 0) {
        float a = T[O_ATAN_A + rp];
        float w = T[O_ATAN_W + rp];
        float da = dT[(O_ATAN_A + rp) * ND];
        float dw = dT[(O_ATAN_W + rp) * ND];
        float dz = dw * p.y - w * dshift;
        dphi += da * p.at + a * dz / metal::fma(p.z, p.z, 1.0f);
    }
    float damp;
    if (p.ra == 0) {
        float dpoly = dT[(O_AMP_INS + 9) * ND];
        for (int k = 8; k >= 0; --k) {
            dpoly = metal::fma(dpoly, p.t, dT[(O_AMP_INS + k) * ND]);
        }
        damp = p.m76 * dpoly;
    } else if (p.ra == 1) {
        float dden = dT[(O_AMP_INT_D + 4) * ND];
        for (int j = 3; j >= 0; --j) {
            dden = metal::fma(dden, p.f, dT[(O_AMP_INT_D + j) * ND]);
        }
        damp = (dT[O_AMP_INT_N * ND] - p.amp * dden) / p.den;
    } else {
        float s = T[O_AMP_RD_S];
        float gr = T[O_AMP_RD_GR];
        float du = dT[O_AMP_RD_S * ND] * p.y - s * dshift;
        float dlog = -dT[O_AMP_RD_GR * ND] * p.u - gr * du - (2.0f * p.u * du + dT[O_AMP_RD_G * ND]) / p.s;
        damp = p.m76 * p.e / p.s * dT[O_AMP_RD_N * ND] + p.amp * dlog;
    }
    if (p.f > T[O_F_CUT]) {
        damp = 0.0f;
    }
    float kp = T[O_KP];
    float kc = T[O_KC];
    float dkp = dT[O_KP * ND];
    float dkc = dT[O_KC * ND];
    float gp = dkp * p.amp + kp * damp;
    float gc = dkc * p.amp + kc * damp;
    float rp_ = kp * p.amp * dphi;
    float rc_ = kc * p.amp * dphi;
    return float4(gp * p.c - rp_ * p.sn, gp * p.sn + rp_ * p.c, -gc * p.sn - rc_ * p.c, gc * p.c - rc_ * p.sn);
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
XasPoint p;
xas_point(f, T, p);
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
    float4 g = xas_tangent(p, T, dT + d);
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
        float4 g = xas_tangent(p, T, dT + d);
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

_SETUP = """
uint b = thread_position_in_grid.x;
if (b >= dims[0]) return;
float x[NI];
for (uint k = 0; k < NI; ++k) x[k] = params[(size_t)b * NI + k];
float y[NT];
xas_table(x, y);
float hi[2], lo[2];
xas_big(x[0], x[1], x[7] == 0.0f ? x[8] : x[7], y[O_F_PHASE_IN], hi, lo);
for (uint k = 0; k < 2; ++k) {
    y[O_BIG + k] = hi[k];
    y[O_BIG_LO + k] = lo[k];
}
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
xas_table_jvp(x, y);
float hi[2], lo[2];
xas_big(x[0].v, x[1].v, x[7].v == 0.0f ? x[8].v : x[7].v, y[O_F_PHASE_IN].v, hi, lo);
for (uint k = 0; k < 2; ++k) {
    y[O_BIG + k].v = hi[k];
    y[O_BIG_LO + k].v = lo[k];
}
for (uint k = 0; k < NT; ++k) {
    if (d == 0) table[(size_t)b * NT + k] = y[k].v;
    dtable[((size_t)b * NT + k) * ND + d] = y[k].d;
}
"""


def _fast(source):
    """``source`` (generated) with Metal's fast division, exponential and
    logarithm: on a single thread, as the set-up of one binary runs, the
    precise ones take hundreds of nanoseconds each."""
    source = re.sub(r"= \(([\w.+-]+) / ([\w.+-]+)\);", r"= metal::fast::divide(\1, \2);", source)
    return source.replace("metal::precise::exp(", "metal::fast::exp(").replace("metal::precise::log(", "metal::fast::log(")


class Kernels:
    """The kernels for one scale of the strain (built on first use); with
    ``fast``, the set-up uses Metal's fast division, exponential and
    logarithm (:func:`_fast`)."""

    def __init__(self, amp_scale, fast=True):
        self.amp_scale = float(amp_scale)
        self.fast = bool(fast)

    @functools.cached_property
    def _header(self):
        values, tangents = _programs(self.amp_scale)
        consts = _constants() + f"constant constexpr uint NI = {N_INPUTS};\nconstant constexpr uint GROUP = {GROUP};\n"
        setup = (_metal.function_source(values, "xas_table")
                 + _metal.function_source(tangents, "xas_table_jvp", n_primal=1))
        if self.fast:
            setup = _fast(setup)
        return _metal.HEADER + _metal.DUAL + consts + setup + POINT

    def _kernel(self, name, inputs, outputs, source):
        name = f"{name}_fast" if self.fast else name
        return mx.fast.metal_kernel(name=f"xas_{name}", input_names=inputs, output_names=outputs, source=source,
                                    header=self._header)

    @functools.cached_property
    def setup(self):
        return self._kernel("setup", ["params", "dims"], ["table"], _SETUP)

    @functools.cached_property
    def setup_jvp(self):
        return self._kernel("setup_jvp", ["params", "dims"], ["table", "dtable"], _SETUP_JVP)

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
                              threadgroup=(64, 1, 1), output_shapes=[(batch, core.SIZE)],
                              output_dtypes=[mx.float32])
        return table

    def table_jvp(self, params):
        """The tables and their derivatives ``(B, SIZE, N_DIRECTIONS)``."""
        batch = params.shape[0]
        threads = batch * N_DIRECTIONS
        return self.setup_jvp(inputs=[params, mx.array([batch], dtype=mx.uint32)], grid=(_round(threads, 64), 1, 1),
                              threadgroup=(64, 1, 1),
                              output_shapes=[(batch, core.SIZE), (batch, core.SIZE, N_DIRECTIONS)],
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


def _round(n, group):
    return max(-(-n // group) * group, group)


@functools.lru_cache(maxsize=None)
def kernels(amp_scale, fast=True):
    """The :class:`Kernels` for the scale of the strain ``amp_scale``."""
    return Kernels(amp_scale, fast)
