# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The moving average of the amplitude ratio of the antisymmetric mode
(:func:`..antisymmetric.smooth_amp_ratio`) as a Metal kernel: LAL's
in-place average over 81 frequencies, each window reading the averages
already written for the 40 frequencies before its centre (an order-40
recurrence), one SIMD group per row (a binary, or the derivative of a
binary along a direction: the average is linear in the ratio). The window
lives in the registers of the 32 lanes (three values each), shifted by
shuffles from one frequency to the next, and its sums are taken with
``simd_sum``; the sums are direct (JAX's float32 evaluation takes them as
differences of running sums, which lose digits on long grids)."""

from __future__ import annotations

import functools

import mlx.core as mx

from ..antisymmetric import WIDTH

__all__ = ["smooth_amp_ratio"]


@functools.lru_cache(maxsize=None)
def _kernel():
    source = """
    uint r = threadgroup_position_in_grid.x;
    ushort lane = thread_index_in_simdgroup;
    uint n_rows = dims[0];
    int length = (int)dims[1];
    uint shared = dims[2];
    int width = (int)dims[3];
    if (r >= n_rows) return;
    uint fr = shared ? 0 : r / dims[4];
    const device float* f = frequencies + (size_t)fr * length;
    const device float* k = kappa + (size_t)r * length;
    device float* o = out + (size_t)r * length;
    for (int i = lane; i < length; i += 32) o[i] = k[i];
    int hw = width / 2;
    int end = length - width - 1;
    // the window w[j] = o[id + j], j <= width: lane j % 32, slot j / 32
    float w[3];
    for (int s = 0; s < 3; ++s) {
        int j = lane + 32 * s;
        w[s] = j <= width ? k[j] : 0.0f;
    }
    for (int id = 0; id < end; ++id) {
        float sum = 0.0f, total = 0.0f;
        for (int s = 0; s < 3; ++s) {
            int j = lane + 32 * s;
            if (j <= width) {
                float df = f[id + j + 1] - f[id + j];
                total += df;
                sum = metal::fma(w[s], df, sum);
            }
        }
        float value = metal::simd_sum(sum) / metal::simd_sum(total);
        if (lane == 0) o[id + hw] = value;
        // the next window: shifted by one, the new value at hw - 1 and the
        // next of the original values at width
        float first1 = metal::simd_broadcast(w[1], 0);
        float first2 = metal::simd_broadcast(w[2], 0);
        float next[3];
        next[0] = metal::simd_shuffle_down(w[0], 1);
        next[1] = metal::simd_shuffle_down(w[1], 1);
        next[2] = metal::simd_shuffle_down(w[2], 1);
        if (lane == 31) { next[0] = first1; next[1] = first2; next[2] = 0.0f; }
        for (int s = 0; s < 3; ++s) {
            int j = lane + 32 * s;
            if (j == hw - 1) next[s] = value;
            if (j == width) next[s] = id + 1 + width < length ? k[id + 1 + width] : 0.0f;
            w[s] = next[s];
        }
    }
    """
    return mx.fast.metal_kernel(name="xpnr_smooth", input_names=["kappa", "frequencies", "dims"],
                                output_names=["out"], source=source)


def smooth_amp_ratio(kappa, frequencies, rows_per_frequency=1):
    """The moving average of ``kappa`` (``(R, N)``) at ``frequencies``
    (``(1, N)``, shared, or ``(R / rows_per_frequency, N)``: row ``r`` of
    ``kappa`` at row ``r // rows_per_frequency`` of the frequencies)."""
    n_rows, length = kappa.shape
    width = WIDTH
    if width > length - 1:
        width = length // 2
    if length - width - 1 <= 0:
        return kappa
    shared = frequencies.shape[0] == 1
    (out,) = _kernel()(
        inputs=[kappa.astype(mx.float32), mx.stop_gradient(frequencies).astype(mx.float32),
                mx.array([n_rows, length, 1 if shared else 0, width, rows_per_frequency], dtype=mx.uint32)],
        grid=(n_rows * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n_rows, length)], output_dtypes=[mx.float32])
    return out
