# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The lean SpinTaylor solution (:func:`..spintaylor.lean.solution`) as
Metal kernels, one thread per binary: the loop of DOP853 steps (backward,
then forward until the stopping conditions), and the end of the inspiral
(``_end``: a bisection on the dense output and a Newton step). The pieces
of arithmetic are the JAX functions of :mod:`._st_parts`, translated; the
loops and the bookkeeping are written here.

* :meth:`SpinTaylor.solve` runs the values: per step (``rows``) ``y[1:5]``,
  the dense output of those components, the time at the start of the
  step, the step and ``M omega`` at its end (:data:`._st_parts.N_ROW`);
  per binary the counts, the end of the inspiral and the decisions of the
  bisection (``info``) and what the set-up reads (``summary``).
* :meth:`SpinTaylor.tangents` runs the same steps on dual numbers, one
  thread per binary and direction, from the decisions of
  :meth:`SpinTaylor.solve` (the numbers of steps, the steps holding the
  end, the bracket and slope of the bisection), with JAX's derivatives:
  those of the end of the inspiral are those of the root (the bisection is
  not differentiated, the Newton step is), as ``_end`` takes them.

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
from . import _metal, _st_parts

__all__ = ["INFO", "SUMMARY", "SpinTaylor", "spintaylor"]

#: The fields of ``info`` (floats, one row per binary).
INFO = ("n", "n_back", "stopped", "beyond", "failed", "incomplete", "overflow", "t_end", "x_last", "slope",
        "root_valid", "row3_0", "row3_1", "row3_2", "row_end", "row_low", "omega_first", "omega_ref", "msec", "pi_gm")
#: The fields of ``summary``: the spins and LNhat at the end of the
#: inspiral, ``f_min_pn``, ``M omega`` at the end and at the start.
SUMMARY = ("s1x", "s1y", "s1z", "s2x", "s2y", "s2z", "lx", "ly", "lz", "f_min_pn", "omega_last", "omega_first")

N_Y = 14
N_C = None  # set at build time
CAPACITY = st_lean.CAPACITY

_COMMON = """
inline float wj_max(float a, float b) { return a > b ? a : (b > a ? b : 0.5f * (a + b)); }
inline WjDual wj_max(WjDual a, WjDual b) {
    return a.v > b.v ? a : (b.v > a.v ? b : WjDual(0.5f * (a.v + b.v), 0.5f * (a.d + b.d)));
}
inline WjDual operator+(WjDual a, WjDual b) { return WjDual(a.v + b.v, a.d + b.d); }
inline WjDual operator-(WjDual a, WjDual b) { return WjDual(a.v - b.v, a.d - b.d); }
inline WjDual operator-(WjDual a) { return WjDual(-a.v, -a.d); }
inline WjDual operator*(WjDual a, WjDual b) { return WjDual(a.v * b.v, a.d * b.v + a.v * b.d); }
inline WjDual operator/(WjDual a, WjDual b) { float r = a.v / b.v; return WjDual(r, (a.d - r * b.d) / b.v); }
inline WjDual operator*(float c, WjDual a) { return WjDual(c * a.v, c * a.d); }
inline float wj_v(float a) { return a; }
inline float wj_v(WjDual a) { return a.v; }
"""


def _primal_source(n_c):
    n_row = _st_parts.N_ROW
    n_info = len(INFO)
    return f"""
uint b = thread_position_in_grid.x;
uint n_binaries = dims[0];
int cap = (int)dims[1];
if (b >= n_binaries) return;
float xin[{_st_parts.N_INPUTS}];
for (int j = 0; j < {_st_parts.N_INPUTS}; ++j) xin[j] = x[b * {_st_parts.N_INPUTS} + j];
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
float y[14], k0[14], yn[14], kn[14], dense[98], mrg[1], hh[1];
for (int j = 0; j < 14; ++j) {{ y[j] = y_ref[j]; k0[j] = k_ref[j]; }}
int i = 0;
bool stopped = beyond;
float t_back = 0.0f, t_fwd = 0.0f;
while (i < limit && (i < n_back || !stopped)) {{
    if (i == n_back) {{
        for (int j = 0; j < 14; ++j) {{ y[j] = y_ref[j]; k0[j] = k_ref[j]; }}
    }}
    float du[1];
    du[0] = i < n_back ? du_back : wj_max(-du_step, -0.1f / y[1]);
    float om[1] = {{y[1]}};
    float dom[1] = {{k0[1]}};
    st_step_in_time(du, om, dom, hh);
    st_step(c, y, k0, hh, yn, kn, dense, mrg);
    bool stop = (i >= n_back) && (!(mrg[0] > 0.0f) || !(kn[1] > k0[1]));
    float h = hh[0];
    float t0;
    if (i < n_back) {{ t_back += h; t0 = t_back - h; }} else {{ t_fwd += h; t0 = t_fwd - h; }}
    device float* row = R + (size_t)i * {n_row};
    for (int j = 0; j < 4; ++j) row[j] = y[1 + j];
    for (int r = 0; r < 7; ++r)
        for (int j = 0; j < 4; ++j) row[4 + r * 4 + j] = dense[r * 14 + 1 + j];
    row[32] = t0;
    row[33] = h;
    row[34] = yn[1];
    device float* srow = S + (size_t)i * 28;
    for (int j = 0; j < 14; ++j) {{ srow[j] = y[j]; srow[14 + j] = k0[j]; }}
    for (int j = 0; j < 14; ++j) {{ y[j] = yn[j]; k0[j] = kn[j]; }}
    stopped = stop;
    i += 1;
}}
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
    for (int k = 0; k < 3; ++k) r3[k] = min(max(jb + (2 - k), 0), max(n_back - 1, 0));
}} else {{
    for (int k = 0; k < 3; ++k) r3[k] = min(max(n - 1 - (2 - k), n_back), max(n - 1, n_back));
}}
float y3[3][14], d3[3][98], t03[3], h3[3], lo_t[3], hi_t[3], hs[3];
for (int k = 0; k < 3; ++k) {{
    float yy[14], kk[14];
    for (int j = 0; j < 14; ++j) {{ yy[j] = S[(size_t)r3[k] * 28 + j]; kk[j] = S[(size_t)r3[k] * 28 + 14 + j]; }}
    t03[k] = R[(size_t)r3[k] * {n_row} + 32];
    h3[k] = R[(size_t)r3[k] * {n_row} + 33];
    float hk[1] = {{h3[k]}};
    st_step(c, yy, kk, hk, yn, kn, d3[k], mrg);
    for (int j = 0; j < 14; ++j) y3[k][j] = yy[j];
    lo_t[k] = metal::min(t03[k], t03[k] + h3[k]);
    hi_t[k] = metal::max(t03[k], t03[k] + h3[k]);
    hs[k] = h3[k] != 0.0f ? h3[k] : 1.0f;
}}
float t_a = lo_t[0], t_b = hi_t[2];
const int last = {st_lean.BISECTIONS + 2};
float blo = t_a, bhi = t_b, g_lo = 0.0f, g_hi = 0.0f, slope = 0.0f, root = 0.5f * (t_a + t_b), x_last = root;
for (int it = 0; it <= last; ++it) {{
    float xx = it == 0 ? t_b : (it == 1 ? t_a : (it == last ? root : 0.5f * (blo + bhi)));
    if (it == last) x_last = xx;
    int cnt = 0;
    for (int k = 0; k < 3; ++k) cnt += (lo_t[k] <= xx) ? 1 : 0;
    int kk = min(max(cnt - 1, 0), 2);
    float xs[1] = {{(xx - t03[kk]) / hs[kk]}};
    float st[14], g[1];
    st_dense_eval(d3[kk], y3[kk], xs, st);
    st_going(c, st, g);
    bool bisecting = it >= 2 && it < last;
    bool ok = g[0] > 0.0f;
    g_hi = it == 0 ? g[0] : ((bisecting && !ok) ? g[0] : g_hi);
    g_lo = it == 1 ? g[0] : ((bisecting && ok) ? g[0] : g_lo);
    blo = (bisecting && ok) ? xx : blo;
    bhi = (bisecting && !ok) ? xx : bhi;
    slope = it == {st_lean.SECANT} ? (g_hi - g_lo) / (bhi - blo) : slope;
    if (it == last) {{
        // the slope of the Newton step: d going / dt at the point (exact,
        // from dual numbers; JAX takes the secant of a narrow bracket, which
        // float32 cannot resolve), the secant where it is not finite
        WjDual cd[{n_c}], yd[14], ddd[98], sd[14], gd[1];
        for (int j = 0; j < {n_c}; ++j) cd[j] = WjDual(c[j]);
        for (int j = 0; j < 14; ++j) yd[j] = WjDual(y3[kk][j]);
        for (int j = 0; j < 98; ++j) ddd[j] = WjDual(d3[kk][j]);
        WjDual xd[1] = {{WjDual((xx - t03[kk]) / hs[kk], 1.0f / hs[kk])}};
        st_dense_eval(ddd, yd, xd, sd);
        st_going(cd, sd, gd);
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
    float yy[14], kk[14], dd[98];
    for (int j = 0; j < 14; ++j) {{ yy[j] = S[(size_t)row_end * 28 + j]; kk[j] = S[(size_t)row_end * 28 + 14 + j]; }}
    float t0e = R[(size_t)row_end * {n_row} + 32];
    float he = R[(size_t)row_end * {n_row} + 33];
    float hk[1] = {{he}};
    st_step(c, yy, kk, hk, yn, kn, dd, mrg);
    float xs[1] = {{(t_end - t0e) / (he != 0.0f ? he : 1.0f)}};
    st_dense_eval(dd, yy, xs, last_state);
}}
float of[1] = {{omega_first}};
float sm[11];
st_finish(last_state, of, xin, sm);
device float* SO = summary + (size_t)b * {len(SUMMARY)};
for (int j = 0; j < 11; ++j) SO[j] = sm[j];
SO[11] = omega_first;
bool failed = setup_failed || !(xin[8] > 0.0f);
bool incomplete = ((n >= {CAPACITY}) && !stopped) || (omega_first > {_metal._float_literal(np.float32(np.pi))} * msec * (xin[9] - 0.5f));
device float* I = info + (size_t)b * {n_info};
I[0] = (float)n; I[1] = (float)n_back; I[2] = stopped ? 1.0f : 0.0f; I[3] = beyond ? 1.0f : 0.0f;
I[4] = failed ? 1.0f : 0.0f; I[5] = incomplete ? 1.0f : 0.0f; I[6] = overflow ? 1.0f : 0.0f; I[7] = t_end;
I[8] = x_last; I[9] = slope; I[10] = root_valid ? 1.0f : 0.0f;
I[11] = (float)r3[0]; I[12] = (float)r3[1]; I[13] = (float)r3[2]; I[14] = (float)row_end; I[15] = (float)row_low;
I[16] = omega_first; I[17] = omega_ref; I[18] = msec; I[19] = pi_gm;
"""


def _tangent_source(n_c):
    n_rt = _st_parts.N_ROW_TANGENT
    n_info = len(INFO)
    n_in = _st_parts.N_INPUTS
    return f"""
uint tid = thread_position_in_grid.x;
uint n_binaries = dims[0];
int cap = (int)dims[1];
uint n_dir = dims[2];
if (tid >= n_binaries * n_dir) return;
uint b = tid / n_dir;
uint d = tid % n_dir;
WjDual xin[{n_in}];
for (int j = 0; j < {n_in}; ++j) xin[j] = WjDual(x[b * {n_in} + j], dx[((size_t)b * {n_in} + j) * n_dir + d]);
WjDual c[{n_c}], y_ref[14], k_ref[14], inf[{len(_st_parts.SETUP_FIELDS)}];
st_setup(xin, c, y_ref, k_ref, inf);
const device float* I = info + (size_t)b * {n_info};
int n = (int)I[0];
int n_back = (int)I[1];
float x_last = I[8], slope = I[9];
bool root_valid = I[10] != 0.0f;
int r3[3] = {{(int)I[11], (int)I[12], (int)I[13]}};
int row_end = (int)I[14];
int row_low = (int)I[15];
WjDual du_step = inf[4], du_back = inf[6], dt_fine = inf[7];
device float* DR = drows + (size_t)b * cap * {n_rt} * n_dir;
WjDual y[14], k0[14], yn[14], kn[14], dense[98], mrg[1], hh[1];
for (int j = 0; j < 14; ++j) {{ y[j] = y_ref[j]; k0[j] = k_ref[j]; }}
WjDual t_back = WjDual(0.0f), t_fwd = WjDual(0.0f);
// the steps kept for the end: the three of the bisection, the one holding
// the end, and the first in the order of time
WjDual cy[5][14], ck[5][14], ct0[5], ch[5], comega[5];
int keep[5] = {{r3[0], r3[1], r3[2], row_end, row_low}};
for (int i = 0; i < n; ++i) {{
    if (i == n_back) {{
        for (int j = 0; j < 14; ++j) {{ y[j] = y_ref[j]; k0[j] = k_ref[j]; }}
    }}
    WjDual du[1];
    du[0] = i < n_back ? du_back : wj_max(-du_step, WjDual(-0.1f) / y[1]);
    WjDual om[1] = {{y[1]}};
    WjDual dom[1] = {{k0[1]}};
    st_step_in_time(du, om, dom, hh);
    st_step(c, y, k0, hh, yn, kn, dense, mrg);
    WjDual h = hh[0];
    WjDual t0;
    if (i < n_back) {{ t_back = t_back + h; t0 = t_back - h; }} else {{ t_fwd = t_fwd + h; t0 = t_fwd - h; }}
    device float* row = DR + (size_t)i * {n_rt} * n_dir + d;
    for (int j = 0; j < 4; ++j) row[j * n_dir] = y[1 + j].d;
    for (int r = 0; r < 7; ++r)
        for (int j = 0; j < 4; ++j) row[(4 + r * 4 + j) * n_dir] = dense[r * 14 + 1 + j].d;
    row[32 * n_dir] = t0.d;
    row[33 * n_dir] = h.d;
    for (int s = 0; s < 5; ++s) {{
        if (keep[s] == i) {{
            for (int j = 0; j < 14; ++j) {{ cy[s][j] = y[j]; ck[s][j] = k0[j]; }}
            ct0[s] = t0;
            ch[s] = h;
            comega[s] = yn[1];
        }}
    }}
    for (int j = 0; j < 14; ++j) {{ y[j] = yn[j]; k0[j] = kn[j]; }}
}}
// the end of the inspiral: the Newton step from the bisection's last point
float lo_t[3];
for (int k = 0; k < 3; ++k) lo_t[k] = metal::min(ct0[k].v, ct0[k].v + ch[k].v);
int cnt = 0;
for (int k = 0; k < 3; ++k) cnt += (lo_t[k] <= x_last) ? 1 : 0;
int kk = min(max(cnt - 1, 0), 2);
WjDual root;
{{
    WjDual dd[98], st[14], g[1];
    WjDual hk[1] = {{ch[kk]}};
    st_step(c, cy[kk], ck[kk], hk, yn, kn, dd, mrg);
    WjDual hsafe = ch[kk].v != 0.0f ? ch[kk] : WjDual(1.0f);
    WjDual xs[1] = {{(WjDual(x_last) - ct0[kk]) / hsafe}};
    st_dense_eval(dd, cy[kk], xs, st);
    st_going(c, st, g);
    float sl = slope != 0.0f ? slope : 1.0f;
    root = root_valid ? WjDual(x_last) - g[0] / WjDual(sl) : WjDual(lo_t[0]);
}}
WjDual back_off = 0.5f * dt_fine;
WjDual t_low0 = n > 0 ? (n_back > 0 ? ct0[4] + ch[4] : ct0[4]) : WjDual(INFINITY);
WjDual t_end = wj_max(root - back_off, t_low0);
WjDual last_state[14];
{{
    WjDual dd[98];
    WjDual hk[1] = {{ch[3]}};
    st_step(c, cy[3], ck[3], hk, yn, kn, dd, mrg);
    WjDual hsafe = ch[3].v != 0.0f ? ch[3] : WjDual(1.0f);
    WjDual xs[1] = {{(t_end - ct0[3]) / hsafe}};
    st_dense_eval(dd, cy[3], xs, last_state);
}}
WjDual omega_first = n > 0 ? (n_back > 0 ? comega[4] : cy[4][1]) : WjDual(INFINITY);
WjDual of[1] = {{omega_first}};
WjDual sm[11];
st_finish(last_state, of, xin, sm);
device float* DS = dsummary + (size_t)b * {len(SUMMARY)} * n_dir + d;
for (int j = 0; j < 11; ++j) DS[j * n_dir] = sm[j].d;
DS[11 * n_dir] = omega_first.d;
device float* DT = dt_end + (size_t)b * n_dir + d;
DT[0] = t_end.d;
"""


class SpinTaylor:
    """The kernels of the SpinTaylor solution."""

    @functools.cached_property
    def n_c(self):
        return _st_parts.c_size()

    @functools.cached_property
    def primal_kernel(self):
        src, _ = _st_parts.device_source(False)
        dual, _ = _st_parts.device_source(True, ("dense_eval", "going"))
        header = _metal.HEADER + _metal.DUAL + _COMMON + src + dual
        return mx.fast.metal_kernel(name="xpnr_spintaylor", input_names=["x", "active", "dims"],
                                    output_names=["rows", "scratch", "info", "summary"],
                                    source=_primal_source(self.n_c), header=header)

    @functools.cached_property
    def tangent_kernel(self):
        src, _ = _st_parts.device_source(True)
        header = _metal.HEADER + _metal.DUAL + _COMMON + src
        return mx.fast.metal_kernel(name="xpnr_spintaylor_jvp", input_names=["x", "dx", "info", "dims"],
                                    output_names=["drows", "dsummary", "dt_end"],
                                    source=_tangent_source(self.n_c), header=header)

    def solve(self, x, active, capacity):
        """``(rows, info, summary)`` for the inputs ``x`` (``(B, 11)``:
        ``m1``, ``m2``, ``chi1``, ``chi2``, ``f_ref``, ``f_low``, ``f_cut``)
        and ``active`` (``(B,)``), with room for ``capacity`` steps."""
        batch = x.shape[0]
        dims = mx.array([batch, capacity], dtype=mx.uint32)
        threads = 32
        rows, _, info, summary = self.primal_kernel(
            inputs=[x.astype(mx.float32), active.astype(mx.float32), dims],
            grid=(-(-batch // threads) * threads, 1, 1), threadgroup=(threads, 1, 1),
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
        threads = 32
        total = batch * n_dir
        drows, dsummary, dt_end = self.tangent_kernel(
            inputs=[x.astype(mx.float32), dx.astype(mx.float32), info, dims],
            grid=(-(-total // threads) * threads, 1, 1), threadgroup=(threads, 1, 1),
            output_shapes=[(batch, capacity, _st_parts.N_ROW_TANGENT, n_dir), (batch, len(SUMMARY), n_dir),
                           (batch, n_dir)],
            output_dtypes=[mx.float32] * 3, init_value=0.0)
        return drows, dsummary, dt_end


@functools.lru_cache(maxsize=None)
def spintaylor():
    """The kernels (built once)."""
    return SpinTaylor()
