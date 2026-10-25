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
differences of running sums, which lose digits on long grids).

On a uniform grid every weight is ``1 / 81``: the recurrence has constant
coefficients, and is solved for all the frequencies at once rather than one
after the other (:func:`_uniform`), as JAX's ``uniform`` branch does it, with
the recurrence between blocks of 40 averages solved by products with powers
of its matrix (:func:`_block_recurrence`)."""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

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


#: The blocks of the recurrence between blocks combined at each level of
#: :func:`_block_recurrence`.
SUPERBLOCK = 16


@functools.lru_cache(maxsize=None)
def _uniform_matrices(width):
    """For equal frequency steps (weights ``1 / (width + 1)``), the
    matrices of JAX's ``uniform`` branch of ``smooth_amp_ratio`` (in
    float64): the averages of a block of ``half`` are ``a @ previous + m
    @ r`` with ``previous`` those of the block before and ``r`` the sums of
    the original values the block's windows read."""
    half = width // 2
    w = 1.0 / (width + 1)
    i_index = np.arange(half)[:, None]
    k_index = np.arange(half)[None, :]
    from_previous = np.where(k_index - i_index >= 0, w, 0.0)
    lower = np.where(k_index < i_index, w, 0.0)
    m = np.linalg.inv(np.eye(half) - lower)
    return m @ from_previous, m * w


@functools.lru_cache(maxsize=None)
def _powers(a_bytes, d):
    """The powers ``a^0 .. a^S`` of the matrix ``a`` (float64, from its
    bytes) and the block lower-triangular Toeplitz matrix of ``a^(t - i)``
    (``t >= i``), as float32 MLX arrays."""
    a = np.frombuffer(a_bytes, np.float64).reshape(d, d)
    powers = [np.eye(d)]
    for _ in range(SUPERBLOCK):
        powers.append(a @ powers[-1])
    toeplitz = np.zeros((SUPERBLOCK * d, SUPERBLOCK * d))
    for t in range(SUPERBLOCK):
        for i in range(t + 1):
            toeplitz[t * d:(t + 1) * d, i * d:(i + 1) * d] = powers[t - i]
    return (mx.array(np.stack(powers[1:]).astype(np.float32)), mx.array(toeplitz.T.astype(np.float32)),
            powers[SUPERBLOCK])


def _block_recurrence(c, a, x0):
    """``x_b = a @ x_(b - 1) + c_b`` for the blocks ``b`` of ``c`` (``(R, B,
    d)``) from ``x_(-1) = x0`` (``(R, d)``), ``a`` a float64 ``(d, d)``
    array: within groups of :data:`SUPERBLOCK` blocks by a product with
    the Toeplitz matrix of the powers of ``a``, between the groups by the
    same recurrence with ``a^S``, recursively."""
    rows, n_blocks, d = c.shape
    s = SUPERBLOCK
    n_groups = -(-n_blocks // s)
    pad = n_groups * s - n_blocks
    if pad:
        c = mx.concatenate([c, mx.zeros((rows, pad, d), c.dtype)], axis=1)
    powers, toeplitz, a_s = _powers(np.ascontiguousarray(a).tobytes(), d)
    # the recurrence within each group from a zero state
    local = (c.reshape(rows, n_groups, s * d) @ toeplitz).reshape(rows, n_groups, s, d)
    # the states before each group
    if n_groups == 1:
        before = x0[:, None, :]
    else:
        ends = _block_recurrence(local[:, :, -1, :], a_s, x0)
        before = mx.concatenate([x0[:, None, :], ends[:, :-1, :]], axis=1)
    # x_(g, t) = a^(t + 1) x_before(g) + local(g, t)
    x = mx.einsum("tij,rgj->rgti", powers, before) + local
    return x.reshape(rows, n_groups * s, d)[:, :n_blocks]


def _uniform(kappa, width):
    """The moving average on a uniform grid (``width + 1`` weights of
    ``1 / (width + 1)``): JAX's ``uniform`` branch."""
    n_rows, length = kappa.shape
    half = width // 2
    end = length - width - 1
    n_blocks = -(-end // half)
    a, m = _uniform_matrices(width)
    # the sums of the original values that window id reads beyond its
    # half: kappa[id + half .. id + width]
    window = mx.ones((1, width - half + 1, 1), kappa.dtype)
    sums = mx.conv1d(kappa[:, half:, None], window)[:, :end, 0]
    sums = mx.concatenate([sums, mx.zeros((n_rows, n_blocks * half - end), kappa.dtype)], axis=1)
    c = sums.reshape(n_rows, n_blocks, half) @ mx.array(m.T.astype(np.float32))
    smoothed = _block_recurrence(c, a, kappa[:, :half]).reshape(n_rows, n_blocks * half)[:, :end]
    return mx.concatenate([kappa[:, :half], smoothed, kappa[:, half + end:]], axis=1)


def _equal_steps(frequencies):
    """Whether the steps of every row of ``frequencies`` are equal (to a few
    roundings of float32, as JAX's test)."""
    f = np.asarray(frequencies, np.float64)
    df = np.diff(f, axis=1)
    return bool(np.all(np.abs(df - df[:, :1]) <= 8.0 * np.finfo(np.float32).eps * np.abs(df[:, :1])))


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
    if length - width - 1 >= width and _equal_steps(mx.stop_gradient(frequencies)):
        return _uniform(kappa.astype(mx.float32), width)
    shared = frequencies.shape[0] == 1
    (out,) = _kernel()(
        inputs=[kappa.astype(mx.float32), mx.stop_gradient(frequencies).astype(mx.float32),
                mx.array([n_rows, length, 1 if shared else 0, width, rows_per_frequency], dtype=mx.uint32)],
        grid=(n_rows * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n_rows, length)], output_dtypes=[mx.float32])
    return out
