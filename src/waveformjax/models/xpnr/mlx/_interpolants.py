# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The interpolants of the smooth PNR angles of a batch
(:func:`..pnr._geometric_interpolants`) and their values at points
(:func:`..pnr._spline_at`): the vector ``(u, v, w)`` of the angles on the
grid, the natural splines (:func:`._arrays.natural_curvatures`), the
integrals of ``gamma`` by Boole's rule interval by interval (a generated
kernel, :func:`._stages.boole_interval`) summed by ``cumsum``, with the
tangents along ``P`` directions when asked for."""

from __future__ import annotations

from typing import NamedTuple, Optional

import mlx.core as mx

from . import _arrays, _build
from ._solution import _call, _small_atan2

__all__ = ["Splines", "interpolants", "spline_values"]


# ---------------------------------------------------------------------------
# the interpolants of the PNR angles
# ---------------------------------------------------------------------------


class Splines(NamedTuple):
    values: mx.array  # (B, N, 6): u, v, w, plus, minus, unwrapped
    curvatures: mx.array
    dvalues: Optional[mx.array]
    dcurvatures: Optional[mx.array]
    grid: mx.array
    dgrid: Optional[mx.array]
    n: mx.array


def interpolants(alpha, beta, dalpha, dbeta, grid, dgrid, n_grid, n_directions):
    """:func:`..pnr._geometric_interpolants` of ``alpha`` and ``beta`` on the
    grid (``(B, N)``, zero beyond ``n_grid``)."""
    batch, size = alpha.shape
    parts = _build.parts()
    h = grid[:, 2]
    dh = dgrid[:, 2, :] if dgrid is not None else None
    vec, dvec = _call(parts.uvw, (alpha, beta), (dalpha, dbeta), batch, n_directions, n_points=size)
    u, v, w = vec[..., 0], vec[..., 1], vec[..., 2]
    du = dv = dw = None
    if dvec is not None:
        du, dv, dw = dvec[..., 0, :], dvec[..., 1, :], dvec[..., 2, :]
    curv = [_arrays.natural_curvatures(y, n_grid, h, dy, dh) for y, dy in ((u, du), (v, dv), (w, dw))]
    raw, draw = _small_atan2(v, u, dv, du)
    after = mx.minimum(mx.arange(size) + 1, size - 1)[None, :]

    def ends(y):
        nxt = mx.take_along_axis(y, mx.broadcast_to(after.reshape(after.shape + (1,) * (y.ndim - 2)),
                                                     y.shape), axis=1)
        return mx.stack([y, nxt], axis=2)

    values = mx.stack([ends(u), ends(v), ends(w)], axis=2)  # (B, N, 3, 2)
    curvatures = mx.stack([ends(c[0]) for c in curv], axis=2)
    raws = ends(raw)
    trees = (grid, mx.arange(size, dtype=mx.float32)[None, :], values, curvatures, raws)
    ttrees = None
    if n_directions:
        dvalues = mx.stack([ends(du), ends(dv), ends(dw)], axis=2)  # (B, N, 2, P) -> stacked (B, N, 3, 2, P)
        dcurv = mx.stack([ends(c[1]) for c in curv], axis=2)
        ttrees = (dgrid, None, dvalues, dcurv, ends(draw))
    inc, dinc = _boole(parts, trees, ttrees, batch, n_directions, size)
    d_gamma, d_alpha = inc
    zero = mx.zeros((batch, 1))
    gamma = mx.concatenate([zero, mx.cumsum(d_gamma, axis=1)[:, :-1]], axis=1)
    unwrapped = raw[:, :1] + mx.concatenate([zero, mx.cumsum(d_alpha, axis=1)[:, :-1]], axis=1)
    plus, minus = gamma + unwrapped, gamma - unwrapped
    rest = [_arrays.natural_curvatures(y, n_grid, h) for y in (plus, minus, unwrapped)]
    out_values = mx.stack([u, v, w, plus, minus, unwrapped], axis=2)
    out_curv = mx.stack([c[0] for c in curv] + [c[0] for c in rest], axis=2)
    dvals = dcurvs = None
    if n_directions:
        zero_t = mx.zeros((batch, 1, n_directions))
        dgamma = mx.concatenate([zero_t, mx.cumsum(dinc[0], axis=1)[:, :-1]], axis=1)
        dunwrapped = draw[:, :1] + mx.concatenate([zero_t, mx.cumsum(dinc[1], axis=1)[:, :-1]], axis=1)
        dplus, dminus = dgamma + dunwrapped, dgamma - dunwrapped
        rest_t = [_arrays.natural_curvatures(y, n_grid, h, dy, dh)[1]
                  for y, dy in ((plus, dplus), (minus, dminus), (unwrapped, dunwrapped))]
        dvals = mx.stack([du, dv, dw, dplus, dminus, dunwrapped], axis=2)
        dcurvs = mx.stack([c[1] for c in curv] + rest_t, axis=2)
    return Splines(out_values, out_curv, dvals, dcurvs, grid, dgrid, n_grid)


def _boole(parts, trees, ttrees, batch, n_directions, size):
    out, dout = _call(parts.boole, trees, ttrees, batch, n_directions, n_points=size)
    if dout is None:
        return out, None
    return out, dout


def spline_values(splines, x, dx, n_directions):
    """The six interpolants at ``x`` (Hz, ``(B, M)``): ``(values (B, M, 6),
    inside (B, M), tangents (B, M, 6, P))``."""
    batch, m = x.shape
    grid = splines.grid
    i_start, h = grid[:, 0:1], grid[:, 2:3]
    n = splines.n[:, None]
    x0 = i_start * h
    xs = mx.stop_gradient(x)
    k = mx.clip(mx.floor((xs - x0) / h), 0, (n - 2).astype(mx.float32))
    inside = (xs >= x0) & (xs <= x0 + (n.astype(mx.float32) - 1.0) * h)
    ki = k.astype(mx.int32)
    size = splines.values.shape[1]
    k1 = mx.minimum(ki + 1, size - 1)
    y0, dy0 = _arrays.take(splines.values, ki, splines.dvalues)
    y1, dy1 = _arrays.take(splines.values, k1, splines.dvalues)
    c0, dc0 = _arrays.take(splines.curvatures, ki, splines.dcurvatures)
    c1, dc1 = _arrays.take(splines.curvatures, k1, splines.dcurvatures)
    # one point per (x, spline)
    flat = m * 6
    ends = mx.stack([y0, y1], axis=-1).reshape(batch, flat, 2)
    curv = mx.stack([c0, c1], axis=-1).reshape(batch, flat, 2)
    xx = mx.broadcast_to(x[:, :, None], (batch, m, 6)).reshape(batch, flat)
    kk = mx.broadcast_to(k[:, :, None], (batch, m, 6)).reshape(batch, flat)
    trees = (grid, xx, kk, ends, curv)
    ttrees = None
    if n_directions:
        dxx = None if dx is None else mx.broadcast_to(dx[:, :, None, :], (batch, m, 6, n_directions)).reshape(
            batch, flat, n_directions)
        dends = mx.stack([dy0, dy1], axis=-2).reshape(batch, flat, 2, n_directions)
        dcurv = mx.stack([dc0, dc1], axis=-2).reshape(batch, flat, 2, n_directions)
        ttrees = (splines.dgrid, dxx, None, dends, dcurv)
    out, dout = _call(_build.parts().spline_point, trees, ttrees, batch, n_directions, n_points=flat)
    out = out.reshape(batch, m, 6)
    if dout is not None:
        dout = dout.reshape(batch, m, 6, n_directions)
    return out, inside, dout


