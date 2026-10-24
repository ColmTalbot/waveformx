# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The lean SpinTaylor solution (:func:`..spintaylor.lean.solution`) as
Metal kernels: the loop of DOP853 steps (backward, then forward until the
stopping conditions) and the end of the inspiral (``_end``: a bisection on
the dense output and a Newton step). The pieces of arithmetic are the JAX
functions of :mod:`._st_parts`, translated; the loops, the stages of the
method and the bookkeeping are written here.

Every binary is integrated by one SIMD group of 32 threads. Lane ``c <
14`` holds component ``c`` of the state, of the 16 stages of a step and of
the dense output (so that a thread holds a few dozen numbers rather than
the 16 x 14 stages, which spilled from the registers and made a step ten
times slower); before each derivative the lanes exchange the state by
shuffles and every lane evaluates the translated derivative of the whole
state, keeping its own component.

* :meth:`SpinTaylor.solve` runs the values: per step (``rows``) ``y[1:5]``,
  the dense output of those components, the time at the start of the
  step, the step and ``M omega`` at its end (:data:`._st_parts.N_ROW`);
  per binary the counts, the end of the inspiral and the decisions of the
  bisection (``info``) and what the set-up reads (``summary``).
* :meth:`SpinTaylor.tangents` runs the same steps on dual numbers, one
  SIMD group per binary and direction, from the decisions of
  :meth:`SpinTaylor.solve` (the numbers of steps, the steps holding the
  end, the last point and slope of the Newton step), with JAX's
  derivatives: those of the end of the inspiral are those of the root
  (the bisection is not differentiated, the Newton step is), as ``_end``
  takes them. The slope of the Newton step is the exact derivative of the
  stopping conditions in time (JAX takes the secant of the bisection's
  bracket, which float32 does not resolve: its derivatives of the end of
  the inspiral are off by up to 90% in float32).

Steps go to buffers of a capacity chosen per call; a binary needing more
steps than the capacity is flagged (``overflow``) and the caller runs it
again with more room, up to JAX's capacity (8192, beyond which the model
fails as the JAX model does).
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from ..spintaylor import lean as st_lean
from ..spintaylor import trajectory
from . import _metal, _st_parts

__all__ = ["INFO", "SUMMARY", "SpinTaylor", "spintaylor"]

#: The fields of ``info`` (floats, one row per binary).
INFO = ("n", "n_back", "stopped", "beyond", "failed", "incomplete", "overflow", "t_end", "x_last", "slope",
        "root_valid", "row3_0", "row3_1", "row3_2", "row_end", "row_low", "omega_first", "omega_ref", "msec", "pi_gm")
#: The fields of ``summary``: the spins and LNhat at the end of the
#: inspiral, ``f_min_pn``, ``M omega`` at the end and at the start.
SUMMARY = ("s1x", "s1y", "s1z", "s2x", "s2y", "s2z", "lx", "ly", "lz", "f_min_pn", "omega_last", "omega_first")

CAPACITY = st_lean.CAPACITY
SIMD = 32


def _tables():
    """The DOP853 tables (float32, as JAX takes them) as Metal constants."""
    a = np.asarray(trajectory._A, np.float32)
    b = np.asarray(trajectory._B, np.float32)
    d = np.asarray(trajectory._D, np.float32)

    def table(name, values):
        items = ", ".join(_metal._float_literal(v) for v in np.ravel(values))
        return f"constant float {name}[{np.size(values)}] = {{{items}}};"

    return "\n".join([table("WJ_A", a), table("WJ_B", b), table("WJ_D", d)])


_HELPERS = """
inline WjDual operator+(WjDual a, WjDual b) { return WjDual(a.v + b.v, a.d + b.d); }
inline WjDual operator-(WjDual a, WjDual b) { return WjDual(a.v - b.v, a.d - b.d); }
inline WjDual operator-(WjDual a) { return WjDual(-a.v, -a.d); }
inline WjDual operator*(WjDual a, WjDual b) { return WjDual(a.v * b.v, a.d * b.v + a.v * b.d); }
inline WjDual operator/(WjDual a, WjDual b) { float r = a.v / b.v; return WjDual(r, (a.d - r * b.d) / b.v); }
inline WjDual operator*(float c, WjDual a) { return WjDual(c * a.v, c * a.d); }
inline float wj_max(float a, float b) { return a > b ? a : (b > a ? b : 0.5f * (a + b)); }
inline WjDual wj_max(WjDual a, WjDual b) {
    return a.v > b.v ? a : (b.v > a.v ? b : WjDual(0.5f * (a.v + b.v), 0.5f * (a.d + b.d)));
}
inline float wj_fma(float a, float b, float c) { return metal::fma(a, b, c); }
// a b + c fused in the values (as the float kernel and XLA compute it)
inline float wj_fma3(float a, float b, float c) { return metal::fma(a, b, c); }
inline WjDual wj_fma3(WjDual a, WjDual b, WjDual c) {
    return WjDual(metal::fma(a.v, b.v, c.v), a.d * b.v + a.v * b.d + c.d);
}
inline WjDual wj_fma(float a, WjDual b, WjDual c) { return WjDual(metal::fma(a, b.v, c.v), metal::fma(a, b.d, c.d)); }
inline float wj_shuffle(float x, ushort lane) { return metal::simd_shuffle(x, lane); }
inline WjDual wj_shuffle(WjDual x, ushort lane) {
    return WjDual(metal::simd_shuffle(x.v, lane), metal::simd_shuffle(x.d, lane));
}

// the state of all lanes (component j from lane j)
template <typename T>
inline void wj_gather(T value, thread T* full) {
    for (ushort j = 0; j < 14; ++j) full[j] = wj_shuffle(value, j);
}

// component `lane` of an array of the 14 components
template <typename T>
inline T wj_pick(thread const T* values, ushort lane) {
    T out = values[0];
    for (ushort j = 1; j < 14; ++j) out = lane == j ? values[j] : out;
    return out;
}

// the dense output at x in the step (trajectory._dense_eval)
template <typename T>
inline T wj_dense_eval(thread const T* dense, T y0, T x) {
    T y = T(0.0f);
    for (int i = 6; i >= 0; --i) {
        y = y + dense[i];
        y = y * (((6 - i) % 2 == 0) ? x : (T(1.0f) - x));
    }
    return y + y0;
}
"""

_STAGES = """
// the 16 stages of a DOP853 step of h from the state (this lane's
// component y, derivative k0) and the new component (_stages_looped)
template <typename T>
inline void wj_stages(thread const T* c, T y, T k0, T h, ushort lane, thread T* k, thread T& y_new) {
    k[0] = k0;
    for (int s = 1; s < 16; ++s) {
        T acc = T(0.0f);
        for (int j = 0; j < s; ++j) acc = wj_fma(WJ_A[s * 16 + j], k[j], acc);
        T ys = wj_fma3(h, acc, y);
        T full[14], d[14];
        wj_gather(ys, full);
        st_rhs(c, full, d);
        k[s] = wj_pick(d, lane);
    }
    T acc = T(0.0f);
    for (int j = 0; j < 16; ++j) acc = wj_fma(WJ_B[j], k[j], acc);
    y_new = wj_fma3(h, acc, y);
}

// the dense output of this lane's component over the step (_dense_looped)
template <typename T>
inline void wj_dense(T y, T h, thread const T* k, T y_new, thread T* dense) {
    T dy = y_new - y;
    dense[0] = dy;
    dense[1] = wj_fma3(h, k[0], -dy);
    dense[2] = wj_fma3(-h, k[12] + k[0], 2.0f * dy);
    for (int i = 0; i < 4; ++i) {
        T acc = T(0.0f);
        for (int j = 0; j < 16; ++j) acc = wj_fma(WJ_D[i * 16 + j], k[j], acc);
        dense[3 + i] = h * acc;
    }
}
"""


def _primal_source(n_c):
    n_row = _st_parts.N_ROW
    n_info = len(INFO)
    n_in = _st_parts.N_INPUTS
    pi = _metal._float_literal(np.float32(np.pi))
    return f"""
uint b = threadgroup_position_in_grid.x;
ushort lane = thread_index_in_simdgroup;
uint n_binaries = dims[0];
int cap = (int)dims[1];
if (b >= n_binaries) return;
float xin[{n_in}];
for (int j = 0; j < {n_in}; ++j) xin[j] = x[b * {n_in} + j];
float c[{n_c}], y_ref[14], k_ref[14], inf[{len(_st_parts.SETUP_FIELDS)}];
st_setup(xin, c, y_ref, k_ref, inf);
bool beyond = inf[0] != 0.0f;
float omega_ref = inf[1];
bool setup_failed = inf[3] != 0.0f;
float du_step = inf[4];
int n_back = (int)inf[5];
float du_back = inf[6];
float dt_fine = inf[7];
float msec = inf[8];
float pi_gm = inf[9];
int limit = active[b] != 0.0f ? cap : 0;
device float* R = rows + (size_t)b * cap * {n_row};
device float* S = scratch + (size_t)b * cap * 28;
ushort me = min(lane, (ushort)13);
float yc = y_ref[me], k0c = k_ref[me];
float k[16], dense[7], full[14], mrg[1], hh[1];
int i = 0;
bool stopped = beyond;
float t_back = 0.0f, t_fwd = 0.0f;
while (i < limit && (i < n_back || !stopped)) {{
    if (i == n_back) {{ yc = y_ref[me]; k0c = k_ref[me]; }}
    float y1 = wj_shuffle(yc, 1), k01 = wj_shuffle(k0c, 1);
    float du[1];
    du[0] = i < n_back ? du_back : wj_max(-du_step, -0.1f / y1);
    float om[1] = {{y1}};
    float dom[1] = {{k01}};
    st_step_in_time(du, om, dom, hh);
    float h = hh[0];
    float ync;
    wj_stages(c, yc, k0c, h, me, k, ync);
    wj_dense(yc, h, k, ync, dense);
    wj_gather(ync, full);
    st_margin(c, full, mrg);
    float kn1 = wj_shuffle(k[12], 1);
    bool stop = (i >= n_back) && (!(mrg[0] > 0.0f) || !(kn1 > k01));
    float t0;
    if (i < n_back) {{ t_back += h; t0 = t_back - h; }} else {{ t_fwd += h; t0 = t_fwd - h; }}
    device float* row = R + (size_t)i * {n_row};
    if (lane >= 1 && lane <= 4) {{
        row[lane - 1] = yc;
        for (int r = 0; r < 7; ++r) row[4 + r * 4 + lane - 1] = dense[r];
    }}
    if (lane == 0) {{ row[32] = t0; row[33] = h; row[34] = full[1]; }}
    if (lane < 14) {{ S[(size_t)i * 28 + lane] = yc; S[(size_t)i * 28 + 14 + lane] = k0c; }}
    yc = ync;
    k0c = k[12];
    stopped = stop;
    i += 1;
}}
threadgroup_barrier(mem_flags::mem_device);
int n = i;
bool overflow = (n >= cap) && !stopped && cap < {CAPACITY};
float omega_first = n > 0 ? (n_back > 0 ? R[(size_t)(n_back - 1) * {n_row} + 34] : R[0]) : INFINITY;
// the steps searched for the end of the inspiral
int r3[3];
if (beyond) {{
    int lo = -1, hi = max(n_back - 2, 0);
    for (int it = 0; it < 14; ++it) {{
        int mid = (lo + hi) >> 1;
        int idx = min(max(mid + 1, 0), max(n_back - 1, 0));
        float ys[14];
        for (int j = 0; j < 14; ++j) ys[j] = S[(size_t)idx * 28 + j];
        float mm[1];
        st_margin(c, ys, mm);
        bool ok = mm[0] > 0.0f;
        lo = ok ? lo : mid;
        hi = ok ? mid : hi;
    }}
    int jb = max(hi, 0);
    for (int q = 0; q < 3; ++q) r3[q] = min(max(jb + (2 - q), 0), max(n_back - 1, 0));
}} else {{
    for (int q = 0; q < 3; ++q) r3[q] = min(max(n - 1 - (2 - q), n_back), max(n - 1, n_back));
}}
float y3[3], d3[3][7], t03[3], h3[3], lo_t[3], hi_t[3], hs[3];
for (int q = 0; q < 3; ++q) {{
    float yy = S[(size_t)r3[q] * 28 + me], kk0 = S[(size_t)r3[q] * 28 + 14 + me];
    t03[q] = R[(size_t)r3[q] * {n_row} + 32];
    h3[q] = R[(size_t)r3[q] * {n_row} + 33];
    float ynq;
    wj_stages(c, yy, kk0, h3[q], me, k, ynq);
    wj_dense(yy, h3[q], k, ynq, d3[q]);
    y3[q] = yy;
    lo_t[q] = metal::min(t03[q], t03[q] + h3[q]);
    hi_t[q] = metal::max(t03[q], t03[q] + h3[q]);
    hs[q] = h3[q] != 0.0f ? h3[q] : 1.0f;
}}
float t_a = lo_t[0], t_b = hi_t[2];
const int last = {st_lean.BISECTIONS + 2};
float blo = t_a, bhi = t_b, g_lo = 0.0f, g_hi = 0.0f, slope = 0.0f, root = 0.5f * (t_a + t_b), x_last = root;
for (int it = 0; it <= last; ++it) {{
    float xx = it == 0 ? t_b : (it == 1 ? t_a : (it == last ? root : 0.5f * (blo + bhi)));
    int cnt = 0;
    for (int q = 0; q < 3; ++q) cnt += (lo_t[q] <= xx) ? 1 : 0;
    int kq = min(max(cnt - 1, 0), 2);
    float value = wj_dense_eval(d3[kq], y3[kq], (xx - t03[kq]) / hs[kq]);
    float g[1];
    wj_gather(value, full);
    st_going(c, full, g);
    bool bisecting = it >= 2 && it < last;
    bool ok = g[0] > 0.0f;
    g_hi = it == 0 ? g[0] : ((bisecting && !ok) ? g[0] : g_hi);
    g_lo = it == 1 ? g[0] : ((bisecting && ok) ? g[0] : g_lo);
    blo = (bisecting && ok) ? xx : blo;
    bhi = (bisecting && !ok) ? xx : bhi;
    slope = it == {st_lean.SECANT} ? (g_hi - g_lo) / (bhi - blo) : slope;
    if (it == last) {{
        x_last = xx;
        // the slope of the Newton step: d going / dt at the point, exact
        // (dual numbers), the secant where it is not finite
        WjDual cd[{n_c}], fd[14], gd[1];
        for (int j = 0; j < {n_c}; ++j) cd[j] = WjDual(c[j]);
        WjDual dd[7];
        for (int j = 0; j < 7; ++j) dd[j] = WjDual(d3[kq][j]);
        WjDual xd = WjDual((xx - t03[kq]) / hs[kq], 1.0f / hs[kq]);
        WjDual vd = wj_dense_eval(dd, WjDual(y3[kq]), xd);
        wj_gather(vd, fd);
        st_going(cd, fd, gd);
        if (gd[0].d != 0.0f && wj_isfinite(gd[0].d)) slope = gd[0].d;
    }}
    float newton = xx - g[0] / (slope != 0.0f ? slope : 1.0f);
    root = it == last ? newton : 0.5f * (blo + bhi);
}}
bool root_valid = g_lo > 0.0f;
root = root_valid ? root : t_a;
float back_off = 0.5f * dt_fine;
int row_low = n_back > 0 ? n_back - 1 : 0;
float t_low0 = n > 0 ? (n_back > 0 ? R[(size_t)row_low * {n_row} + 32] + R[(size_t)row_low * {n_row} + 33]
                                    : R[32]) : INFINITY;
float t_end = wj_max(root - back_off, t_low0);
// the step (in the order of time) holding t_end
int qlo = 0, qhi = n;
while (qlo < qhi) {{
    int q = (qlo + qhi) >> 1;
    int r = q < n_back ? n_back - 1 - q : q;
    float tl = q < n_back ? R[(size_t)r * {n_row} + 32] + R[(size_t)r * {n_row} + 33] : R[(size_t)r * {n_row} + 32];
    if (tl <= t_end) qlo = q + 1; else qhi = q;
}}
int p = min(max(qlo - 1, 0), max(n - 1, 0));
int row_end = p < n_back ? n_back - 1 - p : p;
float last_state[14];
{{
    float yy = S[(size_t)row_end * 28 + me], kk0 = S[(size_t)row_end * 28 + 14 + me];
    float t0e = R[(size_t)row_end * {n_row} + 32];
    float he = R[(size_t)row_end * {n_row} + 33];
    float yne, de[7];
    wj_stages(c, yy, kk0, he, me, k, yne);
    wj_dense(yy, he, k, yne, de);
    float value = wj_dense_eval(de, yy, (t_end - t0e) / (he != 0.0f ? he : 1.0f));
    wj_gather(value, last_state);
}}
if (lane == 0) {{
    float of[1] = {{omega_first}};
    float sm[11];
    st_finish(last_state, of, xin, sm);
    device float* SO = summary + (size_t)b * {len(SUMMARY)};
    for (int j = 0; j < 11; ++j) SO[j] = sm[j];
    SO[11] = omega_first;
    bool failed = setup_failed || !(xin[8] > 0.0f);
    bool incomplete = ((n >= {CAPACITY}) && !stopped) || (omega_first > {pi} * msec * (xin[9] - 0.5f));
    device float* I = info + (size_t)b * {n_info};
    I[0] = (float)n; I[1] = (float)n_back; I[2] = stopped ? 1.0f : 0.0f; I[3] = beyond ? 1.0f : 0.0f;
    I[4] = failed ? 1.0f : 0.0f; I[5] = incomplete ? 1.0f : 0.0f; I[6] = overflow ? 1.0f : 0.0f; I[7] = t_end;
    I[8] = x_last; I[9] = slope; I[10] = root_valid ? 1.0f : 0.0f;
    I[11] = (float)r3[0]; I[12] = (float)r3[1]; I[13] = (float)r3[2]; I[14] = (float)row_end; I[15] = (float)row_low;
    I[16] = omega_first; I[17] = omega_ref; I[18] = msec; I[19] = pi_gm;
}}
"""


def _tangent_source(n_c):
    n_rt = _st_parts.N_ROW_TANGENT
    n_info = len(INFO)
    n_in = _st_parts.N_INPUTS
    return f"""
uint g = threadgroup_position_in_grid.x;
ushort lane = thread_index_in_simdgroup;
uint n_binaries = dims[0];
int cap = (int)dims[1];
uint n_dir = dims[2];
if (g >= n_binaries * n_dir) return;
uint b = g / n_dir;
uint d = g % n_dir;
WjDual xin[{n_in}];
for (int j = 0; j < {n_in}; ++j) xin[j] = WjDual(x[b * {n_in} + j], dx[((size_t)b * {n_in} + j) * n_dir + d]);
WjDual c[{n_c}], y_ref[14], k_ref[14], inf[{len(_st_parts.SETUP_FIELDS)}];
st_setup(xin, c, y_ref, k_ref, inf);
const device float* I = info + (size_t)b * {n_info};
int n = (int)I[0];
int n_back = (int)I[1];
float x_last = I[8], slope = I[9];
bool root_valid = I[10] != 0.0f;
int keep[5] = {{(int)I[11], (int)I[12], (int)I[13], (int)I[14], (int)I[15]}};
WjDual du_step = inf[4], du_back = inf[6], dt_fine = inf[7];
device float* DR = drows + (size_t)b * cap * {n_rt} * n_dir + d;
ushort me = min(lane, (ushort)13);
WjDual yc = y_ref[me], k0c = k_ref[me];
WjDual k[16], dense[7], full[14], hh[1];
WjDual t_back = WjDual(0.0f), t_fwd = WjDual(0.0f);
// the steps kept for the end: the three of the bisection, the one holding
// the end, and the first in the order of time
WjDual cy[5], ck[5], ct0[5], ch[5], comega[5];
for (int i = 0; i < n; ++i) {{
    if (i == n_back) {{ yc = y_ref[me]; k0c = k_ref[me]; }}
    WjDual y1 = wj_shuffle(yc, 1), k01 = wj_shuffle(k0c, 1);
    WjDual du[1];
    du[0] = i < n_back ? du_back : wj_max(-du_step, WjDual(-0.1f) / y1);
    WjDual om[1] = {{y1}};
    WjDual dom[1] = {{k01}};
    st_step_in_time(du, om, dom, hh);
    WjDual h = hh[0];
    WjDual ync;
    wj_stages(c, yc, k0c, h, me, k, ync);
    wj_dense(yc, h, k, ync, dense);
    WjDual omega_end = wj_shuffle(ync, 1);
    WjDual t0;
    if (i < n_back) {{ t_back = t_back + h; t0 = t_back - h; }} else {{ t_fwd = t_fwd + h; t0 = t_fwd - h; }}
    device float* row = DR + (size_t)i * {n_rt} * n_dir;
    if (lane >= 1 && lane <= 4) {{
        row[(lane - 1) * n_dir] = yc.d;
        for (int r = 0; r < 7; ++r) row[(4 + r * 4 + lane - 1) * n_dir] = dense[r].d;
    }}
    if (lane == 0) {{ row[32 * n_dir] = t0.d; row[33 * n_dir] = h.d; }}
    for (int s = 0; s < 5; ++s) {{
        if (keep[s] == i) {{ cy[s] = yc; ck[s] = k0c; ct0[s] = t0; ch[s] = h; comega[s] = omega_end; }}
    }}
    yc = ync;
    k0c = k[12];
}}
// the end of the inspiral: the Newton step from the bisection's last point
float lo_t[3];
for (int q = 0; q < 3; ++q) lo_t[q] = metal::min(ct0[q].v, ct0[q].v + ch[q].v);
int cnt = 0;
for (int q = 0; q < 3; ++q) cnt += (lo_t[q] <= x_last) ? 1 : 0;
int kq = min(max(cnt - 1, 0), 2);
WjDual root;
{{
    WjDual ynq, dq[7], gg[1];
    wj_stages(c, cy[kq], ck[kq], ch[kq], me, k, ynq);
    wj_dense(cy[kq], ch[kq], k, ynq, dq);
    WjDual hsafe = ch[kq].v != 0.0f ? ch[kq] : WjDual(1.0f);
    WjDual value = wj_dense_eval(dq, cy[kq], (WjDual(x_last) - ct0[kq]) / hsafe);
    wj_gather(value, full);
    st_going(c, full, gg);
    float sl = slope != 0.0f ? slope : 1.0f;
    root = root_valid ? WjDual(x_last) - gg[0] / WjDual(sl) : WjDual(lo_t[0]);
}}
WjDual back_off = 0.5f * dt_fine;
WjDual t_low0 = n > 0 ? (n_back > 0 ? ct0[4] + ch[4] : ct0[4]) : WjDual(INFINITY);
WjDual t_end = wj_max(root - back_off, t_low0);
WjDual last_state[14];
{{
    WjDual yne, de[7];
    wj_stages(c, cy[3], ck[3], ch[3], me, k, yne);
    wj_dense(cy[3], ch[3], k, yne, de);
    WjDual hsafe = ch[3].v != 0.0f ? ch[3] : WjDual(1.0f);
    WjDual value = wj_dense_eval(de, cy[3], (t_end - ct0[3]) / hsafe);
    wj_gather(value, last_state);
}}
WjDual y_first = wj_shuffle(cy[4], 1);
if (lane == 0) {{
    WjDual omega_first = n > 0 ? (n_back > 0 ? comega[4] : y_first) : WjDual(INFINITY);
    WjDual of[1] = {{omega_first}};
    WjDual sm[11];
    st_finish(last_state, of, xin, sm);
    device float* DS = dsummary + (size_t)b * {len(SUMMARY)} * n_dir + d;
    for (int j = 0; j < 11; ++j) DS[j * n_dir] = sm[j].d;
    DS[11 * n_dir] = omega_first.d;
    dt_end[(size_t)b * n_dir + d] = t_end.d;
}}
"""


class SpinTaylor:
    """The kernels of the SpinTaylor solution."""

    @functools.cached_property
    def n_c(self):
        return _st_parts.c_size()

    def _header(self, dual):
        names = ("setup", "step_in_time", "rhs", "margin", "going", "finish")
        src, _ = _st_parts.device_source(dual, names)
        if not dual:
            # (the exact slope of the Newton step takes the dual derivative)
            extra, _ = _st_parts.device_source(True, ("going",))
            src = src + extra
        return _metal.HEADER + _metal.DUAL + _tables() + _HELPERS + src + _STAGES

    @functools.cached_property
    def primal_kernel(self):
        return mx.fast.metal_kernel(name="xpnr_spintaylor", input_names=["x", "active", "dims"],
                                    output_names=["rows", "scratch", "info", "summary"],
                                    source=_primal_source(self.n_c), header=self._header(False))

    @functools.cached_property
    def tangent_kernel(self):
        return mx.fast.metal_kernel(name="xpnr_spintaylor_jvp", input_names=["x", "dx", "info", "dims"],
                                    output_names=["drows", "dsummary", "dt_end"],
                                    source=_tangent_source(self.n_c), header=self._header(True))

    def solve(self, x, active, capacity):
        """``(rows, info, summary)`` for the inputs ``x`` (``(B, 11)``:
        ``m1``, ``m2``, ``chi1``, ``chi2``, ``f_ref``, ``f_low``, ``f_cut``)
        and ``active`` (``(B,)``), with room for ``capacity`` steps."""
        batch = x.shape[0]
        dims = mx.array([batch, capacity], dtype=mx.uint32)
        rows, _, info, summary = self.primal_kernel(
            inputs=[x.astype(mx.float32), active.astype(mx.float32), dims],
            grid=(batch * SIMD, 1, 1), threadgroup=(SIMD, 1, 1),
            output_shapes=[(batch, capacity, _st_parts.N_ROW), (batch, capacity, 28), (batch, len(INFO)),
                           (batch, len(SUMMARY))],
            output_dtypes=[mx.float32] * 4, init_value=0.0)
        return rows, info, summary

    def tangents(self, x, dx, info, capacity):
        """The tangents ``(drows, dsummary, dt_end)`` along the directions of
        ``dx`` (``(B, 11, P)``) of the rows (all but ``M omega`` at the end
        of the step), the summary and the end of the inspiral, from the
        decisions of :meth:`solve` (``info``)."""
        batch, _, n_dir = dx.shape
        dims = mx.array([batch, capacity, n_dir], dtype=mx.uint32)
        drows, dsummary, dt_end = self.tangent_kernel(
            inputs=[x.astype(mx.float32), dx.astype(mx.float32), info, dims],
            grid=(batch * n_dir * SIMD, 1, 1), threadgroup=(SIMD, 1, 1),
            output_shapes=[(batch, capacity, _st_parts.N_ROW_TANGENT, n_dir), (batch, len(SUMMARY), n_dir),
                           (batch, n_dir)],
            output_dtypes=[mx.float32] * 3, init_value=0.0)
        return drows, dsummary, dt_end


@functools.lru_cache(maxsize=None)
def spintaylor():
    """The kernels (built once)."""
    return SpinTaylor()
