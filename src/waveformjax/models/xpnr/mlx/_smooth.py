# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The moving average of the amplitude ratio of the antisymmetric mode
(:func:`..antisymmetric.smooth_amp_ratio`) as a Metal kernel: LAL's
in-place average over 81 frequencies, each window reading the averages
already written for the 40 frequencies before its centre, one thread per
row (a binary, or the derivative of a binary along a direction: the
average is linear in the ratio). The sums are direct (JAX's float32
evaluation takes them as differences of running sums, which lose digits
on long grids)."""

from __future__ import annotations

import functools

import mlx.core as mx

from ..antisymmetric import WIDTH

__all__ = ["smooth_amp_ratio"]


@functools.lru_cache(maxsize=None)
def _kernel():
    source = """
    uint r = thread_position_in_grid.x;
    uint n_rows = dims[0];
    int length = (int)dims[1];
    uint shared = dims[2];
    int width = (int)dims[3];
    if (r >= n_rows) return;
    uint fr = shared ? 0 : r / dims[4];
    const device float* f = frequencies + (size_t)fr * length;
    const device float* k = kappa + (size_t)r * length;
    device float* o = out + (size_t)r * length;
    for (int i = 0; i < length; ++i) o[i] = k[i];
    int hw = width / 2;
    int end = length - width - 1;
    for (int id = 0; id < end; ++id) {
        float total = 0.0f, sum = 0.0f;
        for (int j = 0; j <= width; ++j) {
            float df = f[id + j + 1] - f[id + j];
            total += df;
            sum = metal::fma(o[id + j], df, sum);
        }
        o[id + hw] = sum / total;
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
    threads = 32
    (out,) = _kernel()(
        inputs=[kappa.astype(mx.float32), mx.stop_gradient(frequencies).astype(mx.float32),
                mx.array([n_rows, length, 1 if shared else 0, width, rows_per_frequency], dtype=mx.uint32)],
        grid=(-(-n_rows // threads) * threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(n_rows, length)], output_dtypes=[mx.float32])
    return out
