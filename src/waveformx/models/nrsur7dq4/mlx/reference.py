# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The converters of :mod:`..reference` in MLX (float32, one binary or a
batch, differentiable with ``mx.grad``): the dynamics integrated by the
kernel of :mod:`._kernel` from the old reference, the state at the new one
from the natural cubic spline through the nodes, corrected so that the
model's Euler step from it lands on the integration's state at the nearest
node, and the frame there."""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from ...._binary import Binary
from ...._options import REFERENCE_MF
from ..reference import ITERATIONS
from ._data import NRSurData, default_data, spline_system
from ._dynamics import _initialize_at_node, reference_omegas, reference_time, safe_norm
from ._kernel import integrate
from ._waveform import _FLIP, _shared_matmul, batched, labelled

__all__ = ["to_mass_scaled_reference", "to_physical_reference", "convert"]


@functools.lru_cache(maxsize=4)
def _second_derivatives(t_ds: bytes):
    """The operator ``(n_ds, n_ds)`` giving GSL's ``c`` coefficients of the
    natural spline on the nodes from the values there."""
    x = np.frombuffer(t_ds, dtype=np.float32).astype(np.float64)
    system, rhs = spline_system(x)
    c = np.zeros((x.size, x.size))
    c[1:-1] = np.linalg.solve(system, rhs)
    return mx.array(c.astype(np.float32))


def _states_at(data: NRSurData, states, c, t):
    """The spline through ``states`` (``(B, n_ds, 11)``, coefficients ``c``)
    at the times ``t`` (``(B,)``): ``(B, 11)``."""
    n = data.n_ds
    index = mx.clip(mx.sum(data.t_ds[None, :] <= mx.stop_gradient(t)[:, None], axis=1).astype(mx.int32) - 1, 0, n - 2)
    index = mx.stop_gradient(index)
    x_i = mx.take(data.t_ds, index)
    dx = mx.take(data.t_ds, index + 1) - x_i
    delx = (t - x_i)[:, None]
    dx = dx[:, None]

    def at(values, k):
        return mx.take_along_axis(values, (index + k)[:, None, None], axis=1)[:, 0]

    y_i, y_ip1, c_i, c_ip1 = at(states, 0), at(states, 1), at(c, 0), at(c, 1)
    b = (y_ip1 - y_i) / dx - dx * (c_ip1 + 2.0 * c_i) / 3.0
    d = (c_ip1 - c_i) / (3.0 * dx)
    return y_i + delx * (b + delx * (c_i + delx * d))


def _turn(v, angle):
    c, s = mx.cos(angle), mx.sin(angle)
    return mx.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=-1)


def _renormalize(chi, norm):
    square = mx.sum(chi * chi, axis=-1, keepdims=True)
    norm = norm[:, None]
    positive = (square > 0) & (norm > 0)
    return mx.where(positive, chi * (norm / mx.sqrt(mx.where(positive, square, 1.0))), chi)


def _frame(y):
    """The axes of the coorbital frame of the states ``y`` (``(B, 11)``) in
    the old frame, as the columns of ``(B, 3, 3)``."""
    quat = y[:, :4] / mx.sqrt(mx.sum(y[:, :4] * y[:, :4], axis=-1, keepdims=True))
    w, x, yy, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rotation = mx.stack([
        mx.stack([1 - 2 * (yy * yy + z * z), 2 * (x * yy - w * z), 2 * (x * z + w * yy)], axis=-1),
        mx.stack([2 * (x * yy + w * z), 1 - 2 * (x * x + z * z), 2 * (yy * z - w * x)], axis=-1),
        mx.stack([2 * (x * z - w * yy), 2 * (yy * z + w * x), 1 - 2 * (x * x + yy * yy)], axis=-1),
    ], axis=1)
    c, s = mx.cos(y[:, 4]), mx.sin(y[:, 4])
    zero, one = mx.zeros_like(c), mx.ones_like(c)
    turn = mx.stack([mx.stack([c, -s, zero], axis=-1), mx.stack([s, c, zero], axis=-1),
                     mx.stack([zero, zero, one], axis=-1)], axis=1)
    return rotation @ turn


def convert(binary, omega_from, omega_to, f_ref_out, data=None):
    """:func:`..reference.convert` in MLX for a batch (``batched``
    binaries, the omegas ``(B,)``): the converted binaries and ``psi``."""
    data = default_data() if data is None else data
    labels = labelled(binary)
    q, chi_a, chi_b = labels.q, labels.chi_a, labels.chi_b
    norm_a, norm_b = safe_norm(chi_a), safe_norm(chi_b)
    omegas = reference_omegas(data, q, chi_a, chi_b)
    states, valid = integrate(data, q, chi_a, chi_b, omega_from, omegas)
    c = _shared_matmul(_second_derivatives(np.asarray(data.t_ds, np.float32).tobytes()), states)

    def spins_at(y):
        return _renormalize(_turn(y[:, 5:8], y[:, 4]), norm_a), _renormalize(_turn(y[:, 8:11], y[:, 4]), norm_b)

    t, valid_to = reference_time(data, omegas, omega_to)
    y = _states_at(data, states, c, t)
    for _ in range(ITERATIONS):
        new_a, new_b = spins_at(y)
        t, valid_to = reference_time(data, reference_omegas(data, q, new_a, new_b), omega_to)
        # (the state whose Euler step to the nearest node lands on this
        # integration's state there, as in the JAX converter)
        node, stepped = _initialize_at_node(data, q, t, y, norm_a, norm_b)
        y = y - (stepped - mx.take_along_axis(states, mx.stop_gradient(node)[:, None, None], axis=1)[:, 0])
    new_a, new_b = spins_at(y)
    frame = _frame(y)
    flip = mx.where(labels.switched[:, None], _FLIP, mx.ones((1, 3)))
    inc, phi = binary.inclination, binary.phi_ref
    s = mx.sin(inc)
    n = mx.stack([s * mx.sin(phi), s * mx.cos(phi), mx.cos(inc)], axis=-1)
    rotated = flip * (mx.swapaxes(frame, 1, 2) @ (flip * n)[..., None])[..., 0]
    nx, ny, nz = rotated[:, 0], rotated[:, 1], rotated[:, 2]
    perp_sq = nx * nx + ny * ny
    has_perp = perp_sq > 0.0
    perp = mx.sqrt(mx.where(has_perp, perp_sq, 1.0))
    inclination = mx.arctan2(mx.where(has_perp, perp, 0.0), nz)
    azimuth = mx.arctan2(mx.where(has_perp, ny, 1.0), mx.where(has_perp, nx, 0.0))
    turn = mx.arctan2(frame[:, 1, 0], frame[:, 0, 0])
    phi_ref = mx.where(has_perp, 0.5 * np.pi - azimuth, phi + turn)
    phi_ref = phi_ref - 2.0 * np.pi * mx.floor(mx.stop_gradient(phi_ref) / (2.0 * np.pi))
    z_new = flip * frame[:, :, 2]
    z_old = mx.broadcast_to(mx.array([0.0, 0.0, 1.0]), n.shape)

    def projected(v):
        return v - mx.sum(v * n, axis=-1, keepdims=True) * n

    p_old, p_new = projected(z_old), projected(z_new)
    cross = mx.stack([p_old[:, 1] * p_new[:, 2] - p_old[:, 2] * p_new[:, 1],
                      p_old[:, 2] * p_new[:, 0] - p_old[:, 0] * p_new[:, 2],
                      p_old[:, 0] * p_new[:, 1] - p_old[:, 1] * p_new[:, 0]], axis=-1)
    psi = mx.arctan2(mx.sum(n * cross, axis=-1), mx.sum(p_old * p_new, axis=-1))
    spin_1 = mx.where(labels.switched[:, None], flip * new_b, new_a)
    spin_2 = mx.where(labels.switched[:, None], flip * new_a, new_b)
    ok = valid & valid_to
    out = Binary(mass_1=binary.mass_1, mass_2=binary.mass_2, spin_1=mx.where(ok[:, None], spin_1, np.nan),
                 spin_2=mx.where(ok[:, None], spin_2, np.nan), luminosity_distance=binary.luminosity_distance,
                 inclination=mx.where(ok, inclination, np.nan), phi_ref=mx.where(ok, phi_ref, np.nan),
                 f_ref=mx.broadcast_to(mx.array(f_ref_out, dtype=mx.float32), ok.shape))
    return out, mx.where(ok, psi, np.nan)


def _unbatched(out, psi, has_batch, return_angle):
    if not has_batch:
        out = Binary(*(value[0] for value in out))
        psi = psi[0]
    return (out, psi) if return_angle else out


def to_mass_scaled_reference(binary, reference_mf=REFERENCE_MF, f_min=None, return_angle=False, data=None):
    """:func:`..reference.to_mass_scaled_reference` in MLX (float32; one
    binary or a batch, the fields numbers or MLX arrays)."""
    binary, has_batch = batched(binary)
    f = binary.f_ref
    if f_min is not None:
        f = mx.where(f == 0, f_min, f)
    omega = np.pi * labelled(binary).m_tot_sec * f
    out, psi = convert(binary, omega, mx.full(omega.shape, np.pi * reference_mf), 0.0, data)
    return _unbatched(out, psi, has_batch, return_angle)


def to_physical_reference(binary, f_ref, reference_mf=REFERENCE_MF, return_angle=False, data=None):
    """:func:`..reference.to_physical_reference` in MLX (see
    :func:`to_mass_scaled_reference`)."""
    binary, has_batch = batched(binary)
    m_tot_sec = labelled(binary).m_tot_sec
    out, psi = convert(binary, mx.full(m_tot_sec.shape, np.pi * reference_mf), np.pi * m_tot_sec * f_ref,
                       f_ref, data)
    return _unbatched(out, psi, has_batch, return_angle)
