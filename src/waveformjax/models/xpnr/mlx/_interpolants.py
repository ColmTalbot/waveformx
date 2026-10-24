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
    """The six interpolants (:data:`._stages.SPLINES`) of a batch: their
    values and second derivatives on the grid, ``table`` (``(B, N, 12)``:
    the six values, then the six second derivatives) and its tangents
    ``dtable`` (``(B, N, 12, P)``), the grid ``(i_start, n, h)`` and its
    tangents, and ``n`` (int)."""

    table: mx.array
    dtable: Optional[mx.array]
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
    # the kernel reads both ends of each interval from one table of the grid
    table = mx.stack([u, v, w, curv[0][0], curv[1][0], curv[2][0], raw], axis=2)
    dtable = None
    if n_directions:
        dtable = mx.stack([du, dv, dw, curv[0][1], curv[1][1], curv[2][1], draw], axis=2)
    here = mx.broadcast_to(mx.arange(size, dtype=mx.int32)[None, :], (batch, size))
    gather = dict(args={2: (0, 0, 0, 0), 3: (0, 0, 1, 0)}, table=table, table_t=dtable, index=here)
    out = parts.boole.run((grid, mx.arange(size, dtype=mx.float32)[None, :], None, None),
                          (dgrid, None, None, None) if n_directions else None, batch=batch,
                          n_directions=n_directions, n_points=size, gather=gather)
    d_gamma, d_alpha = out[0].tree(), out[1].tree()
    zero = mx.zeros((batch, 1))
    gamma = mx.concatenate([zero, mx.cumsum(d_gamma, axis=1)[:, :-1]], axis=1)
    unwrapped = raw[:, :1] + mx.concatenate([zero, mx.cumsum(d_alpha, axis=1)[:, :-1]], axis=1)
    plus, minus = gamma + unwrapped, gamma - unwrapped
    rest = [_arrays.natural_curvatures(y, n_grid, h) for y in (plus, minus, unwrapped)]
    out_table = mx.stack([u, v, w, plus, minus, unwrapped] + [c[0] for c in curv] + [c[0] for c in rest], axis=2)
    out_dtable = None
    if n_directions:
        zero_t = mx.zeros((batch, 1, n_directions))
        dgamma = mx.concatenate([zero_t, mx.cumsum(out[0].tangent_tree(), axis=1)[:, :-1]], axis=1)
        dunwrapped = draw[:, :1] + mx.concatenate([zero_t, mx.cumsum(out[1].tangent_tree(), axis=1)[:, :-1]],
                                                  axis=1)
        dplus, dminus = dgamma + dunwrapped, dgamma - dunwrapped
        rest_t = [_arrays.natural_curvatures(y, n_grid, h, dy, dh)[1]
                  for y, dy in ((plus, dplus), (minus, dminus), (unwrapped, dunwrapped))]
        out_dtable = mx.stack([du, dv, dw, dplus, dminus, dunwrapped] + [c[1] for c in curv] + rest_t, axis=2)
    return Splines(out_table, out_dtable, grid, dgrid, n_grid)


def intervals(splines, x):
    """The interval ``k`` (real and int) of the grid holding each of ``x``
    (Hz, ``(B, ...)``), clipped to the grid (:func:`..pnr._spline_at`)."""
    grid = splines.grid
    shape = (-1,) + (1,) * (x.ndim - 1)
    i_start, h = grid[:, 0].reshape(shape), grid[:, 2].reshape(shape)
    n = splines.n.reshape(shape)
    x0 = i_start * h
    k = mx.clip(mx.floor((mx.stop_gradient(x) - x0) / h), 0, (n - 2).astype(mx.float32))
    return k, k.astype(mx.int32)


#: The gathered arguments of the values and second derivatives at both ends
#: of an interval (``(column, tangent column, row offset, slot)``).
def ends_gather(first_arg, slot=0):
    return {first_arg: (0, 0, 0, slot), first_arg + 1: (0, 0, 1, slot), first_arg + 2: (6, 6, 0, slot),
            first_arg + 3: (6, 6, 1, slot)}


def spline_values(splines, x, dx, n_directions):
    """The six interpolants at ``x`` (Hz, ``(B, M)``): ``(values (B, M, 6),
    inside (B, M), tangents (B, M, 6, P))``."""
    batch, m = x.shape
    k, ki = intervals(splines, x)
    gather = dict(args=ends_gather(3), table=splines.table, table_t=splines.dtable, index=ki)
    args = (splines.grid, x, k, None, None, None, None)
    targs = (splines.dgrid, dx, None, None, None, None, None) if n_directions else None
    values, inside = _build.parts().spline_point6.run(args, targs, batch=batch, n_directions=n_directions,
                                                      n_points=m, gather=gather)
    return values.tree(), inside.tree(), (values.tangent_tree() if n_directions else None)
