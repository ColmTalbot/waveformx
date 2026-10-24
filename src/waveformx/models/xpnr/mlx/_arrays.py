# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
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


@functools.lru_cache(maxsize=None)
def _filter_kernel():
    weights = ", ".join(f"{w!r}f" for w in _WEIGHTS)
    source = f"""
    uint tid = thread_position_in_grid.x;
    uint n_rows = dims[0];
    int size = (int)dims[1];
    uint per = dims[2];
    if (tid >= n_rows * (uint)size) return;
    uint r = tid / size;
    int k = (int)(tid % size);
    int n = min(lengths[r / per], size);
    if (k < 1 || k > n - 2) {{ out[tid] = 0.0f; return; }}
    const float w[{2 * FILTER_HALF_WIDTH + 1}] = {{{weights}}};
    const device float* yr = y + (size_t)r * size;
    int period = max(2 * (n - 1), 2);
    int upper = period / 2;
    float acc = 0.0f;
    for (int d = -{FILTER_HALF_WIDTH}; d <= {FILTER_HALF_WIDTH}; ++d) {{
        int q = (k + d) % period;
        if (q < 0) q += period;
        float sign = 1.0f;
        if (q > upper) {{ q = period - q; sign = -1.0f; }}
        if (q < 1 || q > n - 2) continue;
        float second = (yr[q + 1] - yr[q]) - (yr[q] - yr[q - 1]);
        acc = metal::fma(w[d + {FILTER_HALF_WIDTH}], sign * second, acc);
    }}
    out[tid] = acc * scale[r / per];
    """
    return mx.fast.metal_kernel(name="xpnr_natural_curvatures", input_names=["y", "lengths", "scale", "dims"],
                                output_names=["out"], source=source)


def _filtered(rows, lengths, scale, per):
    """The filter of :func:`natural_curvatures` on ``rows`` (``(R, N)``),
    row ``r`` of length ``lengths[r // per]`` scaled by ``scale[r //
    per]``."""
    n_rows, size = rows.shape
    total = n_rows * size
    threads = 256
    (out,) = _filter_kernel()(
        inputs=[rows.astype(mx.float32), lengths.astype(mx.int32), scale.astype(mx.float32),
                mx.array([n_rows, size, per], dtype=mx.uint32)],
        grid=(max(-(-total // threads) * threads, threads), 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(n_rows, size)], output_dtypes=[mx.float32])
    return out


def natural_curvatures(y, n, h, y_tangents=None, h_tangents=None):
    """The second derivatives of the natural cubic spline through ``y[b,
    k]`` at ``k h[b]``, ``k < n[b]`` (zero beyond): ``y`` of shape ``(B,
    N)``, ``n`` (``(B,)``, int) and ``h`` (``(B,)``). With tangents
    (``(B, N, P)`` and ``(B, P)``), also those of the result (one Metal
    kernel for the values, one for the tangents)."""
    batch, size = y.shape[0], y.shape[1]
    n = mx.stop_gradient(n).astype(mx.int32)
    scale = 6.0 / (h * h)
    m = _filtered(y, n, scale, 1)
    if y_tangents is None and h_tangents is None:
        return m, None
    dm = None
    if y_tangents is not None:
        p = y_tangents.shape[-1]
        rows = mx.moveaxis(y_tangents, 2, 1).reshape(batch * p, size)
        dm = mx.moveaxis(_filtered(rows, n, scale, p).reshape(batch, p, size), 1, 2)
    if h_tangents is not None:
        term = -2.0 * m[..., None] * (h_tangents / h[:, None])[:, None, :]
        dm = term if dm is None else dm + term
    return m, dm
