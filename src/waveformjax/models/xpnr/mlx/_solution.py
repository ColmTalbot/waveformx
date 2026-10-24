# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The SpinTaylor solution of a batch (:mod:`._spintaylor`) and the
SpinTaylor angles at points from its dense output (``lean.angles``): the
searches in the tables of each binary's steps (:func:`._arrays.count_le`),
the gathers, and the arithmetic at each point as generated kernels
(:func:`._stages.st_newton`, :func:`._stages.st_point_angles`), with the
tangents along ``P`` directions when asked for. Arrays have a leading
batch axis; tangents a trailing axis of directions, or are ``None``."""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import mlx.core as mx
import numpy as np

from .._math import PI, Struct
from . import _arrays, _build, _spintaylor
from ._spintaylor import INFO, SUMMARY

__all__ = ["Solution", "Tables", "integrate", "tables", "st_angles", "CAPACITIES"]

#: The capacities of the SpinTaylor solution tried in turn (the last is
#: the JAX model's).
CAPACITIES = (512, 2048, 8192)
_INFO = {name: i for i, name in enumerate(INFO)}
_SUMMARY = {name: i for i, name in enumerate(SUMMARY)}
_TWO_PI = 2.0 * PI


def _run(kernel, args, targs, batch, n_directions, n_points=None, shared=False):
    """A generated kernel: its outputs as :class:`._kernels.Packed` (with
    their tangents if ``n_directions``)."""
    return kernel.run(args, targs if n_directions else None, batch=batch, n_directions=n_directions,
                      n_points=n_points, shared=shared)


def _call(kernel, trees, ttrees, batch, n_directions, n_points=None, shared=False):
    """A generated kernel: ``(outputs, tangents)`` as pytrees (the tangents
    ``None`` without ``n_directions``)."""
    out = _run(kernel, trees, ttrees, batch, n_directions, n_points, shared)
    if isinstance(out, tuple):
        values = tuple(p.tree() for p in out)
        return values, (tuple(p.tangent_tree() for p in out) if n_directions else None)
    return out.tree(), (out.tangent_tree() if n_directions else None)


def _next_power_of_two(n, minimum=64):
    return max(minimum, 1 << int(math.ceil(math.log2(max(int(n), 1)))))


# ---------------------------------------------------------------------------
# the SpinTaylor solution
# ---------------------------------------------------------------------------


class Solution(NamedTuple):
    rows: mx.array
    drows: Optional[mx.array]
    info: mx.array
    n: mx.array
    n_back: mx.array


def _ode_inputs(ode, ode_t, n_directions):
    x = mx.concatenate([ode.m1[:, None], ode.m2[:, None], ode.chi1, ode.chi2, ode.f_ref[:, None],
                        ode.f_low[:, None], ode.f_cut[:, None]], axis=1)
    if not n_directions:
        return x, None
    batch = x.shape[0]

    def t(value, shape):
        return mx.zeros((batch,) + shape + (n_directions,)) if value is None else value

    dx = mx.concatenate([t(ode_t.m1, ())[:, None], t(ode_t.m2, ())[:, None], t(ode_t.chi1, (3,)),
                         t(ode_t.chi2, (3,)), t(ode_t.f_ref, ())[:, None], t(ode_t.f_low, ())[:, None],
                         t(ode_t.f_cut, ())[:, None]], axis=1)
    return x, dx


def integrate(ode, ode_t, n_directions):
    """The SpinTaylor solution of each binary, and the summary the set-up
    reads (a :class:`~.._math.Struct` like :func:`._stages.solution_summary`)
    with its tangents."""
    st = _spintaylor.spintaylor()
    x, dx = _ode_inputs(ode, ode_t, n_directions)
    active = ode.active.astype(mx.float32)
    batch = x.shape[0]
    rows, info, summary = st.solve(x, active, CAPACITIES[0])
    capacity = CAPACITIES[0]
    for bigger in CAPACITIES[1:]:
        overflow = np.asarray(info[:, _INFO["overflow"]]) > 0
        if not overflow.any():
            break
        index = mx.array(np.nonzero(overflow)[0])
        r2, i2, s2 = st.solve(x[index], active[index], bigger)
        rows = mx.concatenate([rows, mx.zeros((batch, bigger - capacity, rows.shape[2]))], axis=1)
        rows[index] = r2
        info[index] = i2
        summary[index] = s2
        capacity = bigger
    n = info[:, _INFO["n"]].astype(mx.int32)
    n_back = info[:, _INFO["n_back"]].astype(mx.int32)
    used = _next_power_of_two(int(np.max(np.asarray(n))) if batch else 1)
    used = min(used, rows.shape[1])
    rows = rows[:, :used]
    drows = dsummary = None
    if n_directions:
        drows, dsummary, _ = st.tangents(x, dx, info, used)
    failed = (info[:, _INFO["failed"]] > 0) | ~ode.flow_ok
    incomplete = info[:, _INFO["incomplete"]] > 0
    s = _SUMMARY

    def summarize(a, flags):
        return Struct(s1_end=a[:, s["s1x"]:s["s1z"] + 1], s2_end=a[:, s["s2x"]:s["s2z"] + 1],
                      lnhat_end=a[:, s["lx"]:s["lz"] + 1], f_min_pn=a[:, s["f_min_pn"]],
                      omega_first=a[:, s["omega_first"]], omega_last=a[:, s["omega_last"]],
                      failed=flags[0], incomplete=flags[1])

    out = summarize(summary, (failed, incomplete))
    out_t = summarize(dsummary, (None, None)) if n_directions else None
    return Solution(rows, drows, info, n, n_back), out, out_t


# ---------------------------------------------------------------------------
# the SpinTaylor angles at points
# ---------------------------------------------------------------------------


class Tables(NamedTuple):
    """The steps of the solution in the order of time, the search tables
    and the continuous ``alpha`` at the nodes."""

    order: mx.array
    omega_low: mx.array
    node_t: mx.array
    alpha: mx.array
    dalpha: Optional[mx.array]


def _small_atan2(y, x, dy=None, dx=None):
    small = x * x + y * y < 1e-30
    value = mx.arctan2(mx.where(small, 0.0, y), mx.where(small, 1.0, x))
    if dy is None:
        return value, None
    r2 = mx.where(small, 1.0, x * x + y * y)
    d = (x[..., None] * dy - y[..., None] * dx) / r2[..., None]
    return value, mx.where(small[..., None], 0.0, d)


def tables(solution, rotation, rotation_t, n_directions):
    rows, drows, _, n, n_back = solution
    batch, capacity = rows.shape[0], rows.shape[1]
    p = mx.arange(capacity)[None, :]
    nb = n_back[:, None]
    order = mx.clip(mx.where(p < nb, nb - 1 - p, p), 0, capacity - 1)
    valid = p < n[:, None]
    ordered, dordered = _arrays.take(rows, order, drows)
    inf = float("inf")
    omega_low = mx.where(valid, mx.where(p < nb, ordered[..., 34], ordered[..., 0]), inf)
    node_t = mx.where(valid, ordered[..., 32], inf)
    lnhat = ordered[..., 1:4]
    dlnhat = dordered[..., 1:4, :] if dordered is not None else None
    parts = _build.parts()
    raw, draw = _call(parts.st_node_raw, (rotation, lnhat), (rotation_t, dlnhat), batch, n_directions,
                      n_points=capacity)
    raw = mx.where(valid, raw, 0.0)
    d = raw[:, 1:] - raw[:, :-1]
    shift = mx.where(d > PI, -_TWO_PI, mx.where(d < -PI, _TWO_PI, 0.0))
    alpha = mx.concatenate([raw[:, :1], raw[:, :1] + mx.cumsum(d + shift, axis=1)], axis=1)
    dalpha = None
    if draw is not None:
        draw = mx.where(valid[..., None], draw, 0.0)
        dd = draw[:, 1:] - draw[:, :-1]
        dalpha = mx.concatenate([draw[:, :1], draw[:, :1] + mx.cumsum(dd, axis=1)], axis=1)
    return Tables(order, omega_low, node_t, alpha, dalpha)


def st_angles(solution, tab, st_range, st_range_t, rotation, rotation_t, mf, mf_t, n_directions):
    """The SpinTaylor angles ``(alpha, dalpha, cosbeta, dcosbeta, inside)``
    at ``mf`` (``(B, M)``), and their tangents. The kernels read the steps
    holding the points from the tables of the solution (no copies of the
    steps for every point)."""
    rows, drows, _, n, _ = solution
    batch, m = mf.shape
    parts = _build.parts()
    omega = mx.stop_gradient(mf / st_range[:, 0:1])
    last = mx.maximum(n - 1, 0)[:, None]
    p = mx.clip(_arrays.count_le(tab.omega_low, n, omega) - 1, 0, last)
    r = mx.take_along_axis(tab.order, p, axis=1)
    # (y[1:5] in columns 0-3 of a step, the dense output in 4-31, the time
    # at its start in 32, the step in 33, M omega at its end in 34)
    gather = dict(args={1: (0, None), 2: (4, None), 3: (34, None)}, table=rows, index=r)
    x = parts.st_newton.run((omega, None, None, None), batch=batch, n_points=m, gather=gather).tree()
    times, _ = _arrays.take(rows[..., 32:34], r)
    t = times[..., 0] + x * times[..., 1]
    k = mx.clip(_arrays.count_le(tab.node_t, n, t) - 1, 0, last)
    node, dnode = _arrays.take(tab.alpha, k, tab.dalpha)
    gather = dict(args={3: (0, 0), 4: (4, 4), 5: (32, 32), 6: (33, 33)}, table=rows, table_t=drows, index=r)
    args = (st_range, rotation, mf, None, None, None, None, x, node)
    targs = (st_range_t, rotation_t, mf_t, None, None, None, None, None, dnode) if n_directions else None
    out = parts.st_point_angles.run(args, targs, batch=batch, n_directions=n_directions, n_points=m, gather=gather)
    values = tuple(q.tree() for q in out)
    return values, (tuple(q.tangent_tree() for q in out) if n_directions else None)
