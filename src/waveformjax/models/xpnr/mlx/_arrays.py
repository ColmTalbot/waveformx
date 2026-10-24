# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Operations on whole arrays of a batch of binaries that the lean model
does with loops or searches, in MLX:

* :func:`count_le`: the binary searches (``jnp.searchsorted``) in sorted
  tables of different lengths, as a Metal kernel;
* :func:`natural_curvatures`: the second derivatives of the natural cubic
  splines through uniformly spaced points (GSL's, as
  :func:`..pnr._uniform_natural_splines` computes them by local solves of
  the tridiagonal equations): the inverse of the tridiagonal matrix is a
  filter whose weights decay as ``(2 - sqrt(3))^|k|``, applied to the
  second differences extended oddly beyond both ends (which makes the
  natural end conditions exact). It is linear in the values, so their
  derivatives go through the same filter;
* :func:`take`: gathers along the points of each binary, of values and
  their tangents alike.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

__all__ = ["count_le", "natural_curvatures", "take", "FILTER_HALF_WIDTH"]

#: Half-width of the filter of :func:`natural_curvatures`: the weights
#: beyond are below ``(2 - sqrt(3))^24 / (2 sqrt(3)) ~ 6e-15``.
FILTER_HALF_WIDTH = 24
_LAMBDA = np.sqrt(3.0) - 2.0
_WEIGHTS = [float(_LAMBDA ** abs(d) / (2.0 * np.sqrt(3.0))) for d in range(-FILTER_HALF_WIDTH, FILTER_HALF_WIDTH + 1)]


@functools.lru_cache(maxsize=None)
def _count_kernel():
    source = """
    uint tid = thread_position_in_grid.x;
    uint n_rows = dims[0];
    uint n_table = dims[1];
    uint n_queries = dims[2];
    uint shared = dims[3];
    if (tid >= n_rows * n_queries) return;
    uint b = tid / n_queries;
    uint q = tid % n_queries;
    int n = min(max(lengths[b], 0), (int)n_table);
    float value = queries[(size_t)(shared ? 0 : b) * n_queries + q];
    const device float* t = table + (size_t)b * n_table;
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (t[mid] <= value) lo = mid + 1; else hi = mid;
    }
    out[tid] = lo;
    """
    return mx.fast.metal_kernel(name="xpnr_count_le", input_names=["table", "lengths", "queries", "dims"],
                                output_names=["out"], source=source)


def count_le(table, lengths, queries):
    """The number of the first ``lengths[b]`` entries of ``table[b]``
    (sorted ascending) that are at most ``queries[b, q]``
    (``searchsorted(..., side="right")``): ``(B, Q)`` int32. ``queries`` may
    be ``(1, Q)`` (shared)."""
    batch, n_table = table.shape
    shared = queries.shape[0] == 1 and batch > 1
    n_queries = queries.shape[1]
    total = batch * n_queries
    threads = 256
    (out,) = _count_kernel()(
        inputs=[mx.stop_gradient(table).astype(mx.float32), lengths.astype(mx.int32),
                mx.stop_gradient(queries).astype(mx.float32),
                mx.array([batch, n_table, n_queries, 1 if shared else 0], dtype=mx.uint32)],
        grid=(max(-(-total // threads) * threads, threads), 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(batch, n_queries)], output_dtypes=[mx.int32])
    return out


def take(values, index, tangents=None):
    """``values[b, index[b, ...]]`` along axis 1 (``values`` of shape ``(B,
    M, *rest)``, ``index`` of shape ``(B, Q)``) and the same of the tangents
    (``(B, M, *rest, P)``)."""
    index = mx.stop_gradient(index).astype(mx.int32)
    extra = values.ndim - 2

    def gather(x, extra_dims):
        idx = index.reshape(index.shape + (1,) * extra_dims)
        idx = mx.broadcast_to(idx, index.shape + x.shape[2:])
        return mx.take_along_axis(x, idx, axis=1)

    out = gather(values, extra)
    if tangents is None:
        return out, None
    return out, gather(tangents, extra + 1)


def natural_curvatures(y, n, h, y_tangents=None, h_tangents=None):
    """The second derivatives of the natural cubic spline through ``y[b,
    k]`` at ``k h[b]``, ``k < n[b]`` (zero beyond): ``y`` of shape ``(B,
    N)``, ``n`` (``(B,)``, int) and ``h`` (``(B,)``). With tangents
    (``(B, N, P)`` and ``(B, P)``), also those of the result."""
    batch, size = y.shape[0], y.shape[1]
    half = FILTER_HALF_WIDTH
    n = mx.stop_gradient(n).astype(mx.int32)
    k = mx.arange(size)[None, :]
    interior = (k >= 1) & (k <= (n - 2)[:, None])

    def second_differences(x):
        before = mx.concatenate([x[:, :1], x[:, :-1]], axis=1)
        after = mx.concatenate([x[:, 1:], x[:, -1:]], axis=1)
        mask = interior.reshape(interior.shape + (1,) * (x.ndim - 2))
        return mx.where(mask, (after - x) - (x - before), 0.0)

    # the odd extension of the second differences about 0 and n - 1
    period = mx.maximum(2 * (n - 1), 2)[:, None]
    q = mx.arange(-half, size + half)[None, :]
    qq = mx.remainder(q, period)
    upper = period // 2
    index = mx.where(qq <= upper, qq, period - qq)
    sign = mx.where(qq <= upper, 1.0, -1.0)
    index = mx.minimum(index, size - 1)

    def filtered(r):
        extra = r.ndim - 2
        idx = mx.broadcast_to(index.reshape(index.shape + (1,) * extra), (batch, size + 2 * half) + r.shape[2:])
        s = sign.reshape(sign.shape + (1,) * extra)
        ext = s * mx.take_along_axis(r, idx, axis=1)
        out = None
        for j, w in enumerate(_WEIGHTS):
            term = w * ext[:, j:j + size]
            out = term if out is None else out + term
        mask = interior.reshape(interior.shape + (1,) * extra)
        return mx.where(mask, out, 0.0)

    scale = (6.0 / (h * h))[:, None]
    m = filtered(second_differences(y)) * scale
    if y_tangents is None and h_tangents is None:
        return m, None
    dm = None
    if y_tangents is not None:
        dm = filtered(second_differences(y_tangents)) * scale[..., None]
    if h_tangents is not None:
        term = -2.0 * m[..., None] * (h_tangents / h[:, None])[:, None, :]
        dm = term if dm is None else dm + term
    return m, dm
