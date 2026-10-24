# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The Metal kernels of IMRPhenomXHM.

* The set-up (:mod:`..core`) is generated from the JAX functions, as for
  IMRPhenomXAS (:mod:`...xas.mlx._kernels`): one kernel for the head of the
  table (the (2,2), the factors of the modes in the polarizations) and one
  for each higher mode, which run in parallel (each recomputes the (2,2)
  set-up its mode needs), with their forward-mode derivatives (dual numbers,
  one thread per binary and parameter). The double-float leading term of the
  (2,2) is written by hand into the head kernel.
* The waveform at the frequencies is a kernel written by hand, one thread per
  binary and frequency: the (2,2) as IMRPhenomXAS's kernel computes it, each
  higher mode from a function generated from :func:`..core.mode_point` (its
  amplitude and its phase but for the leading term and the anchor), the
  double-float leading terms and the reduction of the phases by hand, and
  the sum with the spherical harmonics. Its derivatives along the eight
  parameters come from the derivatives of the table: those of the (2,2) by
  hand, those of the higher modes from the dual-number functions generated
  with :func:`..core.mode_point`. A third kernel contracts them with the
  cotangents and sums over the frequencies (the VJP).
"""

from __future__ import annotations

import functools

import jax
import mlx.core as mx
import numpy as np

from ...xas.mlx import _kernels as xas_kernels
from ...xpnr._math import precision
from ...xpnr.mlx import _metal
from ...xpnr.mlx._program import lower
from .. import core

__all__ = ["Kernels", "kernels", "N_INPUTS", "N_DIRECTIONS", "COLUMNS"]

N_INPUTS = xas_kernels.N_INPUTS
N_DIRECTIONS = xas_kernels.N_DIRECTIONS
COLUMNS = xas_kernels.COLUMNS
GROUP = xas_kernels.GROUP
TAGS = tuple(10 * ell + emm for ell, emm in core.HIGHER_MODES)


def _setup_function(function):
    def wrapped(x):
        with precision(np.float32):
            return function(tuple(x[i] for i in range(8)), x[8])

    return wrapped


def _lowered(function, n_inputs, avals):
    """The programs of ``function`` and of its JVP (tangents of every input)."""
    values = lower(jax.make_jaxpr(function)(*avals))

    def jvp(*args):
        return jax.jvp(function, args[:n_inputs], args[n_inputs:])

    tangents = lower(jax.make_jaxpr(jvp)(*avals, *avals))
    return values, tangents


@functools.lru_cache(maxsize=None)
def _programs(amp_scale):
    """The set-up programs (head, modes) and the point programs of the modes."""
    x = jax.ShapeDtypeStruct((N_INPUTS,), np.float32)
    out = {"head": _lowered(_setup_function(lambda p, f: core.head_table(p, f, amp_scale)), 1, [x])}
    scalar = jax.ShapeDtypeStruct((), np.float32)
    table = jax.ShapeDtypeStruct((core.MODE_SIZE,), np.float32)
    for (ell, emm), tag in zip(core.HIGHER_MODES, TAGS):
        out[f"mode{tag}"] = _lowered(
            _setup_function(lambda p, f, ell=ell, emm=emm: core.mode_table(p, f, amp_scale, ell, emm)), 1, [x])

        def point(m, f, s, a0, a22, phi22p, tag=tag):
            with precision(np.float32):
                return core.mode_point(m, f, s, a0, a22, phi22p, tag)

        out[f"point{tag}"] = _lowered(point, 6, [table] + [scalar] * 5)

        def point_jvp8(m, f, s, a0, a22, phi22p, dm, ds, da0, da22, dphi22p, point=point):
            # the values once and their derivatives along the N_DIRECTIONS
            # directions (a JVP vectorised over the tangents)
            primals = (m, f, s, a0, a22, phi22p)

            def one(dm, ds, da0, da22, dphi22p):
                return jax.jvp(point, primals, (dm, jax.numpy.zeros_like(f), ds, da0, da22, dphi22p))[1]

            return point(*primals), jax.vmap(one, in_axes=(1, 0, 0, 0, 0), out_axes=-1)(dm, ds, da0, da22, dphi22p)

        def point_vjp(m, f, s, a0, a22, phi22p, g_amp, g_rest, point=point):
            # the cotangents of the table and of the scalar inputs for the
            # cotangents of the amplitude and of the phase
            def fn(m, s, a0, a22, phi22p):
                return point(m, f, s, a0, a22, phi22p)[:2]

            return jax.vjp(fn, m, s, a0, a22, phi22p)[1]((g_amp, g_rest))

        out[f"point{tag}_vjp"] = lower(jax.make_jaxpr(point_vjp)(table, *[scalar] * 7))
        directions = jax.ShapeDtypeStruct((N_DIRECTIONS,), np.float32)
        table_t = jax.ShapeDtypeStruct((core.MODE_SIZE, N_DIRECTIONS), np.float32)
        out[f"point{tag}_jvp8"] = lower(jax.make_jaxpr(point_jvp8)(table, *[scalar] * 5, table_t, *[directions] * 4))
    return out


def _constants():
    lines = [f"constant constexpr uint NH = {core.HEAD_SIZE};", f"constant constexpr uint NM = {core.MODE_SIZE};",
             f"constant constexpr uint NI = {N_INPUTS};", f"constant constexpr uint GROUP = {GROUP};"]
    for name, (start, _) in core.HEAD_OFFSETS.items():
        lines.append(f"constant constexpr uint OH_{name.upper()} = {start};")
    lines.append(f"constant constexpr uint OM_ON = {core.MODE_OFFSETS['on'][0]};")
    for k, (ell, emm) in enumerate(core.HIGHER_MODES):
        m_over_2 = emm / 2.0
        c = m_over_2 ** (8.0 / 3.0)
        hi = float(np.float32(c))
        lines.append(f"constant constexpr float C83_{10 * ell + emm}_HI = {hi!r}f;")
        lines.append(f"constant constexpr float C83_{10 * ell + emm}_LO = {float(np.float32(c - hi))!r}f;")
    return "\n".join(lines) + "\n"


def _device_table(source, name, indices=(0,)):
    """A generated point function whose inputs ``indices`` (the table of the
    mode, its derivatives) are read from device memory."""
    for i in indices:
        old = f"thread const float* in{i}"
        assert old in source
        source = source.replace(old, f"const device float* in{i}", 1)
    return source


#: The waveform at one frequency: the (2,2) (``xas_point``), each higher
#: mode, the polarizations, and their derivatives along one direction.
XHM_POINT = """
struct XhmModes {
    float re[5], im[5];          // the modes h_{l,-m} (22, 21, 33, 32, 44)
    float amp[5], phi[5];        // their amplitudes (with the sign) and phases
    float rest[4];               // the phase of the higher modes but for the big part
    float ins[4];                // whether in their inspiral
    float hm_amp[4];             // their amplitudes (without the sign)
};

inline float2 xhm_factor(const device float* H, uint k, uint which) {
    return float2(H[OH_FACTORS + k * 4 + 2 * which], H[OH_FACTORS + k * 4 + 2 * which + 1]);
}

// the big part of the phase of a higher mode: [inspiral] (m/2)^(8/3) T0
// f^(-5/3) - (m/2) A, plus the rest, reduced
inline float xhm_phase(const device float* H, thread const XasPoint& p, float c_hi, float c_lo, float m_over_2,
                       float inspiral, float rest) {
    DF big = df_mul(DF{H[O_BIG + 1], H[O_BIG_LO + 1]}, -m_over_2);
    if (inspiral > 0.5f) {
        big = df_add(df_mul(df_mul(DF{H[O_BIG], H[O_BIG_LO]}, DF{c_hi, c_lo}), p.b0), big);
    }
    return xas_reduce(df_add(big, DF{rest, 0.0f}));
}
"""


def _mode_calls():
    """The Metal statements computing the higher modes into ``XhmModes q``."""
    lines = []
    for k, ((ell, emm), tag) in enumerate(zip(core.HIGHER_MODES, TAGS), start=1):
        sign = -1.0 if ell % 2 else 1.0
        lines.append(f"""
{{
    const device float* M = mode{tag} + (size_t)b * NM;
    float fs = f, ss = H[OH_S], a0 = H[OH_AMP0], a22 = p.amp, p22 = p.phi_p;
    float amp, rest, ins;
    xhm_point_{tag}(M, &fs, &ss, &a0, &a22, &p22, &amp, &rest, &ins);
    float phi = xhm_phase(H, p, C83_{tag}_HI, C83_{tag}_LO, {emm / 2.0!r}f, ins, rest);
    float a = {sign!r}f * M[OM_ON] * amp;
    q.amp[{k}] = a; q.phi[{k}] = phi; q.rest[{k - 1}] = rest; q.ins[{k - 1}] = ins; q.hm_amp[{k - 1}] = amp;
    q.re[{k}] = a * metal::precise::cos(phi); q.im[{k}] = a * metal::precise::sin(phi);
}}""")
    return "\n".join(lines)


def _mode_tangents():
    """The Metal statements of the derivatives of the modes along every
    direction into ``dre[5][ND]``, ``dim[5][ND]``."""
    lines = ["""
float dre[5][ND], dim[5][ND];
float d_amp22[ND], d_phi22p[ND], d_s[ND], d_a0[ND];
{
    float c = metal::precise::cos(p.phi), sn = metal::precise::sin(p.phi);
    for (uint d = 0; d < ND; ++d) {
        float damp22, dphi22p;
        xas_amp_phase_tangent(p, H, dH + d, damp22, dphi22p);
        float dphi22 = dphi22p - dH[(O_BIG + 1) * ND + d];
        dre[0][d] = damp22 * c - p.amp * dphi22 * sn;
        dim[0][d] = damp22 * sn + p.amp * dphi22 * c;
        d_amp22[d] = damp22; d_phi22p[d] = dphi22p;
        d_s[d] = dH[OH_S * ND + d]; d_a0[d] = dH[OH_AMP0 * ND + d];
    }
}"""]
    for k, ((ell, emm), tag) in enumerate(zip(core.HIGHER_MODES, TAGS), start=1):
        sign = -1.0 if ell % 2 else 1.0
        lines.append(f"""
{{
    const device float* M = mode{tag} + (size_t)b * NM;
    const device float* dM = dmode{tag} + (size_t)b * NM * ND;
    float fs = f, ss = H[OH_S], a0 = H[OH_AMP0], a22 = p.amp, p22 = p.phi_p;
    float amp, rest, ins, damp[ND], drest[ND], dins[ND];
    xhm_point_{tag}_jvp8(M, &fs, &ss, &a0, &a22, &p22, dM, d_s, d_a0, d_amp22, d_phi22p, &amp, &rest, &ins,
                         damp, drest, dins);
    float c = metal::precise::cos(q.phi[{k}]), sn = metal::precise::sin(q.phi[{k}]);
    for (uint d = 0; d < ND; ++d) {{
        float dphi = drest[d] - {emm / 2.0!r}f * dH[(O_BIG + 1) * ND + d];
        if (q.ins[{k - 1}] > 0.5f) {{
            dphi = metal::fma({(emm / 2.0) ** (8.0 / 3.0)!r}f * dH[O_BIG * ND + d], p.basis[0], dphi);
        }}
        float da = {sign!r}f * M[OM_ON] * damp[d];
        dre[{k}][d] = da * c - q.amp[{k}] * dphi * sn;
        dim[{k}][d] = da * sn + q.amp[{k}] * dphi * c;
    }}
}}""")
    return "\n".join(lines)


_HEAD = """
uint n = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
uint n_points = dims[0];
uint shared_f = dims[2];
bool active = n < n_points;
uint nn = active ? n : n_points - 1;
float f = freq[(shared_f ? 0 : (size_t)b * n_points) + nn];
const device float* H = head + (size_t)b * NH;
XasPoint p;
xas_point(f, H, p);
XhmModes q;
q.amp[0] = p.amp; q.phi[0] = p.phi;
q.re[0] = p.amp * p.c; q.im[0] = p.amp * p.sn;
"""

_POLARIZATIONS = """
float2 hp = float2(0.0f), hc = float2(0.0f);
for (uint k = 0; k < 5; ++k) {
    float2 fp = xhm_factor(H, k, 0), fc = xhm_factor(H, k, 1);
    hp += float2(fp.x * q.re[k] - fp.y * q.im[k], fp.x * q.im[k] + fp.y * q.re[k]);
    hc += float2(fc.x * q.re[k] - fc.y * q.im[k], fc.x * q.im[k] + fc.y * q.re[k]);
}
"""


def _values_source(with_modes):
    body = _HEAD + _mode_calls() + _POLARIZATIONS + """
if (!active) return;
size_t at = (size_t)b * n_points + nn;
out_hp[2 * at] = hp.x; out_hp[2 * at + 1] = hp.y;
out_hc[2 * at] = hc.x; out_hc[2 * at + 1] = hc.y;
"""
    if with_modes:
        body += """
for (uint k = 0; k < 5; ++k) {
    out_modes[(at * 5 + k) * 2] = q.re[k];
    out_modes[(at * 5 + k) * 2 + 1] = q.im[k];
}
"""
    return body


_DIRECTION = """
float2 dhp = float2(0.0f), dhc = float2(0.0f);
for (uint k = 0; k < 5; ++k) {
    float2 fp = xhm_factor(H, k, 0), fc = xhm_factor(H, k, 1);
    float2 dfp = float2(dH[(OH_FACTORS + k * 4) * ND + d], dH[(OH_FACTORS + k * 4 + 1) * ND + d]);
    float2 dfc = float2(dH[(OH_FACTORS + k * 4 + 2) * ND + d], dH[(OH_FACTORS + k * 4 + 3) * ND + d]);
    dhp += float2(fp.x * dre[k][d] - fp.y * dim[k][d] + dfp.x * q.re[k] - dfp.y * q.im[k],
                  fp.x * dim[k][d] + fp.y * dre[k][d] + dfp.x * q.im[k] + dfp.y * q.re[k]);
    dhc += float2(fc.x * dre[k][d] - fc.y * dim[k][d] + dfc.x * q.re[k] - dfc.y * q.im[k],
                  fc.x * dim[k][d] + fc.y * dre[k][d] + dfc.x * q.im[k] + dfc.y * q.re[k]);
}
"""


def _jacobian_source():
    return _HEAD + _mode_calls() + _POLARIZATIONS + """
if (!active) return;
const device float* dH = dhead + (size_t)b * NH * ND;
size_t at = (size_t)b * n_points + nn;
out_hp[2 * at] = hp.x; out_hp[2 * at + 1] = hp.y;
out_hc[2 * at] = hc.x; out_hc[2 * at + 1] = hc.y;
for (uint j = 0; j < 12; ++j) {
    jp[(at * 12 + j) * 2] = 0.0f; jp[(at * 12 + j) * 2 + 1] = 0.0f;
    jc[(at * 12 + j) * 2] = 0.0f; jc[(at * 12 + j) * 2 + 1] = 0.0f;
}
""" + _mode_tangents() + """
for (uint d = 0; d < ND; ++d) {
""" + _DIRECTION + """
    size_t o = (at * 12 + COLUMN[d]) * 2;
    jp[o] = dhp.x; jp[o + 1] = dhp.y; jc[o] = dhc.x; jc[o + 1] = dhc.y;
}
"""


def _vjp_source(with_modes):
    """The VJP: the (2,2) and the factors of the polarizations in forward
    mode (their derivatives are a few operations per direction), the higher
    modes in reverse mode (:func:`_programs`, ``point<tag>_vjp``): the
    cotangents of their tables summed over the frequencies (``out_tables``,
    contracted with the derivatives of the tables by the caller) and those
    of their scalar inputs contracted here."""
    lines = [_HEAD, _mode_calls(), """
const device float* dH = dhead + (size_t)b * NH * ND;
size_t at = (size_t)b * n_points + nn;
float2 cp = active ? float2(cot_p[2 * at], cot_p[2 * at + 1]) : float2(0.0f);
float2 cc = active ? float2(cot_c[2 * at], cot_c[2 * at + 1]) : float2(0.0f);
float2 g[5];
for (uint k = 0; k < 5; ++k) {
    float2 fp = xhm_factor(H, k, 0), fc = xhm_factor(H, k, 1);
    // conj(F+) cp + conj(Fx) cc: the cotangent of the mode
    g[k] = float2(fp.x * cp.x + fp.y * cp.y + fc.x * cc.x + fc.y * cc.y,
                  fp.x * cp.y - fp.y * cp.x + fc.x * cc.y - fc.y * cc.x);
"""]
    if with_modes:
        lines.append("""    if (active) g[k] += float2(cot_modes[(at * 5 + k) * 2], cot_modes[(at * 5 + k) * 2 + 1]);
""")
    lines.append("""}
bool any_cot = false;
for (uint k = 0; k < 5; ++k) any_cot = any_cot || any(g[k] != 0.0f);
any_cot = active && any_cot;
float vs[ND];
float d_amp22[ND], d_phi22p[ND], d_s[ND], d_a0[ND];
{
    float c = metal::precise::cos(p.phi), sn = metal::precise::sin(p.phi);
    for (uint d = 0; d < ND; ++d) {
        float damp22, dphi22p;
        xas_amp_phase_tangent(p, H, dH + d, damp22, dphi22p);
        float dphi22 = dphi22p - dH[(O_BIG + 1) * ND + d];
        float v = g[0].x * (damp22 * c - p.amp * dphi22 * sn) + g[0].y * (damp22 * sn + p.amp * dphi22 * c);
        for (uint k = 0; k < 5; ++k) {
            float2 dfp = float2(dH[(OH_FACTORS + k * 4) * ND + d], dH[(OH_FACTORS + k * 4 + 1) * ND + d]);
            float2 dfc = float2(dH[(OH_FACTORS + k * 4 + 2) * ND + d], dH[(OH_FACTORS + k * 4 + 3) * ND + d]);
            float2 hp = float2(dfp.x * q.re[k] - dfp.y * q.im[k], dfp.x * q.im[k] + dfp.y * q.re[k]);
            float2 hc = float2(dfc.x * q.re[k] - dfc.y * q.im[k], dfc.x * q.im[k] + dfc.y * q.re[k]);
            v += dot(hp, cp) + dot(hc, cc);
        }
        vs[d] = any_cot ? v : 0.0f;
        d_amp22[d] = damp22; d_phi22p[d] = dphi22p;
        d_s[d] = dH[OH_S * ND + d]; d_a0[d] = dH[OH_AMP0 * ND + d];
    }
}
threadgroup float partial[(GROUP / 32) * NM];
uint lane = thread_index_in_simdgroup;
uint warp = simdgroup_index_in_threadgroup;
uint groups = (n_points + GROUP - 1) / GROUP;
size_t group_index = (size_t)b * groups + threadgroup_position_in_grid.x;
""")
    for k, ((ell, emm), tag) in enumerate(zip(core.HIGHER_MODES, TAGS), start=1):
        sign = -1.0 if ell % 2 else 1.0
        lines.append(f"""
{{
    const device float* M = mode{tag} + (size_t)b * NM;
    float c = metal::precise::cos(q.phi[{k}]), sn = metal::precise::sin(q.phi[{k}]);
    float g_amp = {sign!r}f * M[OM_ON] * (g[{k}].x * c + g[{k}].y * sn);
    float g_phi = q.amp[{k}] * (g[{k}].y * c - g[{k}].x * sn);
    float cot_m[NM];
    float cs = 0.0f, ca0 = 0.0f, ca22 = 0.0f, cp22 = 0.0f;
    if (any_cot) {{
        float fs = f, ss = H[OH_S], a0 = H[OH_AMP0], a22 = p.amp, p22 = p.phi_p;
        xhm_point_{tag}_vjp(M, &fs, &ss, &a0, &a22, &p22, &g_amp, &g_phi, cot_m, &cs, &ca0, &ca22, &cp22);
        for (uint d = 0; d < ND; ++d) {{
            float big = -{emm / 2.0!r}f * dH[(O_BIG + 1) * ND + d];
            if (q.ins[{k - 1}] > 0.5f) big = metal::fma({(emm / 2.0) ** (8.0 / 3.0)!r}f * dH[O_BIG * ND + d], p.basis[0], big);
            vs[d] += cs * d_s[d] + ca0 * d_a0[d] + ca22 * d_amp22[d] + cp22 * d_phi22p[d] + g_phi * big;
        }}
    }} else {{
        for (uint j = 0; j < NM; ++j) cot_m[j] = 0.0f;
    }}
    // the cotangents of the table summed over the frequencies of the group
    for (uint j = 0; j < NM; ++j) {{
        float x = metal::simd_sum(cot_m[j]);
        if (lane == 0) partial[warp * NM + j] = x;
    }}
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint j = thread_index_in_threadgroup; j < NM; j += GROUP) {{
        float total = 0.0f;
        for (uint w = 0; w < GROUP / 32; ++w) total += partial[w * NM + j];
        out_tables[(group_index * 4 + {k - 1}) * NM + j] = total;
    }}
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}}""")
    lines.append("""
threadgroup float partial_d[(GROUP / 32) * ND];
for (uint d = 0; d < ND; ++d) {
    float v = metal::simd_sum(vs[d]);
    if (lane == 0) partial_d[warp * ND + d] = v;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (thread_index_in_threadgroup < ND) {
    uint d = thread_index_in_threadgroup;
    float total = 0.0f;
    for (uint w = 0; w < GROUP / 32; ++w) total += partial_d[w * ND + d];
    out[group_index * ND + d] = total;
}
""")
    return "\n".join(lines)


_SETUP_HEAD = """
uint b = thread_position_in_grid.x;
if (b >= dims[0]) return;
float x[NI];
for (uint k = 0; k < NI; ++k) x[k] = params[(size_t)b * NI + k];
float y[NH];
xhm_head(x, y);
float hi[2], lo[2];
xas_big(x[0], x[1], x[7] == 0.0f ? x[8] : x[7], y[O_F_PHASE_IN], hi, lo);
for (uint k = 0; k < 2; ++k) {
    y[O_BIG + k] = hi[k];
    y[O_BIG_LO + k] = lo[k];
}
for (uint k = 0; k < NH; ++k) table[(size_t)b * NH + k] = y[k];
"""

_SETUP_HEAD_JVP = """
uint tid = thread_position_in_grid.x;
if (tid >= dims[0] * ND) return;
uint b = tid / ND;
uint d = tid % ND;
WjDual x[NI];
for (uint k = 0; k < NI; ++k) x[k] = WjDual(params[(size_t)b * NI + k], k == d ? 1.0f : 0.0f);
WjDual y[NH];
xhm_head_jvp(x, y);
float hi[2], lo[2];
xas_big(x[0].v, x[1].v, x[7].v == 0.0f ? x[8].v : x[7].v, y[O_F_PHASE_IN].v, hi, lo);
for (uint k = 0; k < 2; ++k) {
    y[O_BIG + k].v = hi[k];
    y[O_BIG_LO + k].v = lo[k];
}
for (uint k = 0; k < NH; ++k) {
    if (d == 0) table[(size_t)b * NH + k] = y[k].v;
    dtable[((size_t)b * NH + k) * ND + d] = y[k].d;
}
"""


def _setup_mode(tag, jvp):
    if jvp:
        return f"""
uint tid = thread_position_in_grid.x;
if (tid >= dims[0] * ND) return;
uint b = tid / ND;
uint d = tid % ND;
WjDual x[NI];
for (uint k = 0; k < NI; ++k) x[k] = WjDual(params[(size_t)b * NI + k], k == d ? 1.0f : 0.0f);
WjDual y[NM];
xhm_mode{tag}_jvp(x, y);
for (uint k = 0; k < NM; ++k) {{
    if (d == 0) table[(size_t)b * NM + k] = y[k].v;
    dtable[((size_t)b * NM + k) * ND + d] = y[k].d;
}}
"""
    return f"""
uint b = thread_position_in_grid.x;
if (b >= dims[0]) return;
float x[NI];
for (uint k = 0; k < NI; ++k) x[k] = params[(size_t)b * NI + k];
float y[NM];
xhm_mode{tag}(x, y);
for (uint k = 0; k < NM; ++k) table[(size_t)b * NM + k] = y[k];
"""


class Kernels:
    """The kernels for one scale of the strain (built on first use)."""

    def __init__(self, amp_scale, fast=True):
        self.amp_scale = float(amp_scale)
        self.fast = bool(fast)

    @functools.cached_property
    def _common(self):
        """Constants, the double-float and point code of IMRPhenomXAS and the
        point functions of the higher modes."""
        programs = _programs(self.amp_scale)
        functions = []
        for tag in TAGS:
            values, tangents = programs[f"point{tag}"]
            functions.append(_device_table(_metal.function_source(values, f"xhm_point_{tag}"), f"xhm_point_{tag}"))
            name = f"xhm_point_{tag}_jvp8"
            functions.append(_device_table(_metal.function_source(programs[f"point{tag}_jvp8"], name), name, (0, 6)))
            name = f"xhm_point_{tag}_vjp"
            functions.append(_device_table(_metal.function_source(programs[f"point{tag}_vjp"], name), name))
        consts = xas_kernels._constants() + _constants()
        return _metal.HEADER + _metal.DUAL + consts + xas_kernels.POINT + "".join(functions) + XHM_POINT

    def _setup_header(self, name):
        values, tangents = _programs(self.amp_scale)[name]
        fn = f"xhm_{name}"
        source = _metal.function_source(values, fn) + _metal.function_source(tangents, fn + "_jvp", n_primal=1)
        if self.fast:
            source = xas_kernels._fast(source)
        return self._common + source

    def _kernel(self, name, inputs, outputs, source, header):
        suffix = "_fast" if self.fast else ""
        return mx.fast.metal_kernel(name=f"xhm_{name}{suffix}", input_names=inputs, output_names=outputs,
                                    source=source, header=header)

    @functools.lru_cache(maxsize=None)
    def setup(self, name, jvp):
        """The set-up kernel of the head (``"head"``) or of a mode
        (``"mode21"``, ...)."""
        header = self._setup_header(name)
        if name == "head":
            source = _SETUP_HEAD_JVP if jvp else _SETUP_HEAD
        else:
            source = _setup_mode(int(name[4:]), jvp)
        outputs = ["table", "dtable"] if jvp else ["table"]
        return self._kernel(f"{name}{'_jvp' if jvp else ''}", ["params", "dims"], outputs, source, header)

    @functools.lru_cache(maxsize=None)
    def point(self, kind, with_modes=False):
        """The point kernels: ``"values"``, ``"jacobian"``, ``"vjp"``."""
        tables = ["head"] + [f"mode{tag}" for tag in TAGS]
        if kind == "values":
            source = _values_source(with_modes)
            inputs = tables + ["freq", "dims"]
            outputs = ["out_hp", "out_hc"] + (["out_modes"] if with_modes else [])
        elif kind == "jacobian":
            source = _jacobian_source()
            inputs = tables + ["dhead"] + [f"dmode{tag}" for tag in TAGS] + ["freq", "dims"]
            outputs = ["out_hp", "out_hc", "jp", "jc"]
        else:
            source = _vjp_source(with_modes)
            inputs = (tables + ["dhead"] + [f"dmode{tag}" for tag in TAGS] + ["freq", "dims", "cot_p", "cot_c"]
                      + (["cot_modes"] if with_modes else []))
            outputs = ["out", "out_tables"]
        name = f"point_{kind}{'_modes' if with_modes else ''}"
        return self._kernel(name, inputs, outputs, source, self._common)

    # --- running ---------------------------------------------------------

    def tables(self, params):
        """The head ``(B, NH)`` and the tables of the modes ``(B, NM)``."""
        batch = params.shape[0]
        dims = mx.array([batch], dtype=mx.uint32)
        out = []
        for name, size in [("head", core.HEAD_SIZE)] + [(f"mode{tag}", core.MODE_SIZE) for tag in TAGS]:
            (table,) = self.setup(name, False)(
                inputs=[params, dims], grid=(_round(batch, 64), 1, 1), threadgroup=(64, 1, 1),
                output_shapes=[(batch, size)], output_dtypes=[mx.float32])
            out.append(table)
        return out

    def tables_jvp(self, params):
        """The tables and their derivatives ``(B, size, N_DIRECTIONS)``."""
        batch = params.shape[0]
        dims = mx.array([batch], dtype=mx.uint32)
        values, tangents = [], []
        for name, size in [("head", core.HEAD_SIZE)] + [(f"mode{tag}", core.MODE_SIZE) for tag in TAGS]:
            table, dtable = self.setup(name, True)(
                inputs=[params, dims], grid=(_round(batch * N_DIRECTIONS, 64), 1, 1), threadgroup=(64, 1, 1),
                output_shapes=[(batch, size), (batch, size, N_DIRECTIONS)], output_dtypes=[mx.float32] * 2)
            values.append(table)
            tangents.append(dtable)
        return values, tangents

    @staticmethod
    def _dims(freq, batch):
        n = freq.shape[-1]
        return n, mx.array([n, batch, 1 if freq.shape[0] == 1 else 0], dtype=mx.uint32)

    def polarizations(self, tables, freq, with_modes=False):
        """``h_+``, ``h_x`` ``(B, N, 2)`` (and the modes ``(B, N, 5, 2)``)."""
        batch = tables[0].shape[0]
        n, dims = self._dims(freq, batch)
        shapes = [(batch, n, 2)] * 2 + ([(batch, n, 5, 2)] if with_modes else [])
        return self.point("values", with_modes)(
            inputs=list(tables) + [freq, dims], grid=(_round(n, GROUP), batch, 1), threadgroup=(GROUP, 1, 1),
            output_shapes=shapes, output_dtypes=[mx.float32] * len(shapes))

    def polarizations_and_jacobian(self, tables, dtables, freq):
        """``h_+``, ``h_x`` ``(B, N, 2)`` and their Jacobians ``(B, N, 12, 2)``."""
        batch = tables[0].shape[0]
        n, dims = self._dims(freq, batch)
        return self.point("jacobian")(
            inputs=list(tables) + list(dtables) + [freq, dims], grid=(_round(n, GROUP), batch, 1),
            threadgroup=(GROUP, 1, 1), output_shapes=[(batch, n, 2), (batch, n, 2), (batch, n, 12, 2), (batch, n, 12, 2)],
            output_dtypes=[mx.float32] * 4)

    def cotangent_product(self, tables, dtables, freq, cot_p, cot_c, cot_modes=None):
        """``sum_n Re(conj(J) cot)`` per binary: ``(B, N_DIRECTIONS)``."""
        batch = tables[0].shape[0]
        n, dims = self._dims(freq, batch)
        groups = -(-n // GROUP)
        with_modes = cot_modes is not None
        inputs = list(tables) + list(dtables) + [freq, dims, cot_p, cot_c] + ([cot_modes] if with_modes else [])
        out, out_tables = self.point("vjp", with_modes)(
            inputs=inputs, grid=(groups * GROUP, batch, 1), threadgroup=(GROUP, 1, 1),
            output_shapes=[(batch, groups, N_DIRECTIONS), (batch, groups, 4, core.MODE_SIZE)],
            output_dtypes=[mx.float32] * 2)
        grad = mx.sum(out, axis=1)
        cot_tables = mx.sum(out_tables, axis=1)
        for k, dtable in enumerate(dtables[1:]):
            grad = grad + mx.sum(cot_tables[:, k, :, None] * dtable, axis=1)
        return grad


def _round(n, group):
    return max(-(-n // group) * group, group)


@functools.lru_cache(maxsize=None)
def kernels(amp_scale, fast=True):
    """The :class:`Kernels` for the scale of the strain ``amp_scale``."""
    return Kernels(amp_scale, fast)
