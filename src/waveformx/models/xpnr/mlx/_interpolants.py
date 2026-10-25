# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The interpolants of the PNR angles of a batch
(:func:`..pnr._lal_interpolants`) and their values at points
(:func:`..pnr._spline_at`): the natural splines of ``alpha`` and ``beta``
(:func:`._arrays.natural_curvatures`), the integral of ``gamma`` by Boole's
rule interval by interval (a generated kernel,
:func:`._stages.boole_interval`) summed by ``cumsum`` and its spline, with
the tangents along ``P`` directions when asked for."""

from __future__ import annotations

from typing import NamedTuple, Optional

import mlx.core as mx

from . import _arrays, _build, _stages

__all__ = ["Splines", "interpolants", "spline_values"]


# ---------------------------------------------------------------------------
# the interpolants of the PNR angles
# ---------------------------------------------------------------------------


class Splines(NamedTuple):
    """The interpolants (:data:`._stages.SPLINES`) of a batch: their values
    and second derivatives on the grid, ``table`` (``(B, N, 2 S)``: the
    ``S`` values, then the ``S`` second derivatives) and its tangents
    ``dtable`` (``(B, N, 2 S, P)``), the grid ``(i_start, n, h)`` and its
    tangents, and ``n`` (int)."""

    table: mx.array
    dtable: Optional[mx.array]
    grid: mx.array
    dgrid: Optional[mx.array]
    n: mx.array


def interpolants(alpha, beta, dalpha, dbeta, grid, dgrid, n_grid, n_directions):
    """:func:`..pnr._lal_interpolants` of ``alpha`` and ``beta`` on the grid
    (``(B, N)``, zero beyond ``n_grid``)."""
    batch, size = alpha.shape
    parts = _build.parts()
    h = grid[:, 2]
    dh = dgrid[:, 2, :] if dgrid is not None else None
    curv = [_arrays.natural_curvatures(y, n_grid, h, dy, dh) for y, dy in ((alpha, dalpha), (beta, dbeta))]
    # the kernel reads both ends of each interval from one table of the grid
    table = mx.stack([alpha, beta, curv[0][0], curv[1][0]], axis=2)
    dtable = None
    if n_directions:
        dtable = mx.stack([dalpha, dbeta, curv[0][1], curv[1][1]], axis=2)
    here = mx.broadcast_to(mx.arange(size, dtype=mx.int32)[None, :], (batch, size))
    gather = dict(args={2: (0, 0, 0, 0), 3: (0, 0, 1, 0)}, table=table, table_t=dtable, index=here)
    out = parts.boole.run((grid, mx.arange(size, dtype=mx.float32)[None, :], None, None),
                          (dgrid, None, None, None) if n_directions else None, batch=batch,
                          n_directions=n_directions, n_points=size, gather=gather)
    zero = mx.zeros((batch, 1))
    gamma = mx.concatenate([zero, mx.cumsum(out[0].tree(), axis=1)[:, :-1]], axis=1)
    curv_gamma = _arrays.natural_curvatures(gamma, n_grid, h)
    out_table = mx.stack([alpha, beta, gamma, curv[0][0], curv[1][0], curv_gamma[0]], axis=2)
    out_dtable = None
    if n_directions:
        zero_t = mx.zeros((batch, 1, n_directions))
        dgamma = mx.concatenate([zero_t, mx.cumsum(out[0].tangent_tree(), axis=1)[:, :-1]], axis=1)
        dcurv_gamma = _arrays.natural_curvatures(gamma, n_grid, h, dgamma, dh)[1]
        out_dtable = mx.stack([dalpha, dbeta, dgamma, curv[0][1], curv[1][1], dcurv_gamma], axis=2)
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
    s = len(_stages.SPLINES)
    return {first_arg: (0, 0, 0, slot), first_arg + 1: (0, 0, 1, slot), first_arg + 2: (s, s, 0, slot),
            first_arg + 3: (s, s, 1, slot)}


def spline_values(splines, x, dx, n_directions):
    """The interpolants at ``x`` (Hz, ``(B, M)``): ``(values (B, M, S),
    inside (B, M), tangents (B, M, S, P))``."""
    batch, m = x.shape
    k, ki = intervals(splines, x)
    gather = dict(args=ends_gather(3), table=splines.table, table_t=splines.dtable, index=ki)
    args = (splines.grid, x, k, None, None, None, None)
    targs = (splines.dgrid, dx, None, None, None, None, None) if n_directions else None
    values, inside = _build.parts().spline_point6.run(args, targs, batch=batch, n_directions=n_directions,
                                                      n_points=m, gather=gather)
    return values.tree(), inside.tree(), (values.tangent_tree() if n_directions else None)
