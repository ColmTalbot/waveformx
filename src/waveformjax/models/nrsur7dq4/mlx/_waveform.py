# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The NRSur7dq4 waveform as MLX functions of a batch of binaries (as
:mod:`.._waveform`, whose docstring gives the conventions; everything is
float32).

The functions take a :class:`~waveformjax.Binary` whose fields are batched
along a leading axis (``mass_1`` of shape ``(B,)``, ``spin_1`` of shape
``(B, 3)``; see :func:`batched`) and return arrays with that axis first.
"""

from __future__ import annotations

import functools
from math import factorial, sqrt
from typing import NamedTuple, Tuple

import mlx.core as mx
import numpy as np

from .._data import LMAX, MODES, PIECES
from .._waveform import C_PER_MPC, MTSUN_SI, STRAIN_SCALE, active_modes, output_ells
from ._data import NRSurData
from ._dynamics import index, reference_omegas, safe_norm
from ._fits import fit_parameters, fold_mass_ratio, q_fit, spin_monomials
from ._frames import coorbital_to_coprecessing
from ._kernel import integrate
from ._wigner import rotate

__all__ = [
    "batched",
    "Labelled",
    "labelled",
    "inertial_modes",
    "polarizations_at",
    "polarizations_at_dimensionless",
    "modes_at",
    "modes_at_dimensionless",
]


def _as_float32(value) -> mx.array:
    if isinstance(value, mx.array):
        return value if value.dtype == mx.float32 else value.astype(mx.float32)
    return mx.array(np.asarray(value, dtype=np.float32))


def batched(binary) -> Tuple[object, bool]:
    """``binary`` with float32 MLX fields broadcast to a leading batch axis
    (the scalars to ``(B,)``, the spins to ``(B, 3)``), and whether it had
    one (a scalar field of shape ``(B,)`` or a spin of shape ``(B, 3)``)."""
    fields = [_as_float32(value) for value in binary]
    spin = [name in ("spin_1", "spin_2") for name in binary._fields]
    sizes = {int(value.shape[0]) for value, s in zip(fields, spin) if value.ndim == (2 if s else 1)}
    if len(sizes) > 1:
        raise ValueError(f"the fields of the binary have different batch sizes {sorted(sizes)}")
    size = next(iter(sizes), 1)
    fields = [mx.broadcast_to(value, (size, 3) if s else (size,)) for value, s in zip(fields, spin)]
    return type(binary)(*fields), bool(sizes)


class Labelled(NamedTuple):
    """The binaries after ``PrecessingNRSur_switch_labels_if_needed``: black
    hole A is the heavier (each field ``(B,)``, the spins ``(B, 3)``)."""

    switched: mx.array
    q: mx.array
    chi_a: mx.array
    chi_b: mx.array
    m_tot_sec: mx.array


_FLIP = mx.array([-1.0, -1.0, 1.0])


def labelled(binary) -> Labelled:
    """Swap the labels where ``mass_1 < mass_2`` (rotating the in-plane
    spins by pi), with the masses in solar masses."""
    m1, m2 = binary.mass_1, binary.mass_2
    switched = m1 < m2
    mass_a = mx.where(switched, m2, m1)
    mass_b = mx.where(switched, m1, m2)
    chi_a = mx.where(switched[:, None], _FLIP * binary.spin_2, binary.spin_1)
    chi_b = mx.where(switched[:, None], _FLIP * binary.spin_1, binary.spin_2)
    return Labelled(switched, mass_a / mass_b, chi_a, chi_b, (mass_a + mass_b) * MTSUN_SI)


def _coorbital_modes(data: NRSurData, q, chi_a, chi_b):
    """The coorbital modes on ``t_coorb`` (real and imaginary parts, each
    ``(B, n_modes, n_coorb)``) from the coorbital-frame spins there
    (``(B, n_coorb, 3)``)."""
    times = data.ei_time
    coefs = fold_mass_ratio(data.ei_coefs, q_fit(q)[:, None, None])
    spins = spin_monomials(
        fit_parameters(q[:, None], mx.take(chi_a, times, axis=1), mx.take(chi_b, times, axis=1)), data.ei_spins
    )
    values = mx.sum(coefs * spins, axis=-1)
    bounds = data.ei_bounds
    pieces = mx.stack(
        [values[:, bounds[p] : bounds[p + 1]] @ data.ei_basis[bounds[p] : bounds[p + 1]] for p in range(len(PIECES))],
        axis=1,
    )
    to_re, to_im = _PIECES_TO_MODES
    return to_re @ pieces, to_im @ pieces


def _pieces_to_modes():
    """The real and imaginary parts of the coorbital modes as combinations
    of the data pieces (``(n_modes, n_pieces)`` each): ``h^{l,m} = X_+ +
    X_-``, ``h^{l,-m} = conj(X_+ - X_-)`` (``PrecessingNRSur_core``)."""
    to_re = np.zeros((len(MODES), len(PIECES)), dtype=np.float32)
    to_im = np.zeros_like(to_re)
    piece = {key: p for p, key in enumerate(PIECES)}
    row = {mode: i for i, mode in enumerate(MODES)}
    for ell in range(2, LMAX + 1):
        to_re[row[ell, 0], piece[ell, 0, "real"]] = 1.0
        to_im[row[ell, 0], piece[ell, 0, "imag"]] = 1.0
        for m in range(1, ell + 1):
            for sign, mode in ((1.0, (ell, m)), (-1.0, (ell, -m))):
                to_re[row[mode], piece[ell, m, "Re+"]] = 1.0
                to_re[row[mode], piece[ell, m, "Re-"]] = sign
                to_im[row[mode], piece[ell, m, "Im+"]] = sign
                to_im[row[mode], piece[ell, m, "Im-"]] = 1.0
    return mx.array(to_re), mx.array(to_im)


_PIECES_TO_MODES = _pieces_to_modes()


def _shared_matmul(matrix, x):
    """``matrix @ x[b]`` for every ``b`` of ``x`` (``(B, n, k)``), as one
    matrix product: broadcast over the batch, MLX reads the matrix again for
    every binary (25 times slower for a batch of 1024)."""
    size, n, k = x.shape
    out = matrix @ x.transpose(1, 0, 2).reshape(n, size * k)
    return out.reshape(matrix.shape[0], size, k).transpose(1, 0, 2)


def _coorbital_dynamics(data: NRSurData, states, chi_a0, chi_b0):
    """The quaternion (normalised), the orbital phase and the spins in the
    coorbital frame on the coorbital times (``(B, n_coorb, ...)``), from the
    states at the dynamics nodes (``(B, n_ds, 11)``) by one matrix product
    (the splines of :func:`.._waveform._coorbital_dynamics`)."""
    series = _shared_matmul(data.ds_to_coorb, states)
    quat = series[..., 0:4]
    phase = series[..., 4]
    norm_a = safe_norm(chi_a0)[:, None, None]
    norm_b = safe_norm(chi_b0)[:, None, None]

    def renormalize(chi, norm):
        square = mx.sum(chi * chi, axis=-1, keepdims=True)
        positive = (square > 0) & (norm > 0)
        n = mx.sqrt(mx.where(positive, square, 1.0))
        return mx.where(positive, chi * (norm / n), chi)

    chi_a = renormalize(series[..., 5:8], norm_a)
    chi_b = renormalize(series[..., 8:11], norm_b)
    quat = quat / mx.sqrt(mx.sum(quat * quat, axis=-1, keepdims=True))
    cp = mx.cos(phase)
    sp = mx.sin(phase)

    def rotate(chi):
        return mx.stack([chi[..., 0] * cp + chi[..., 1] * sp, -chi[..., 0] * sp + chi[..., 1] * cp, chi[..., 2]], axis=-1)

    return quat, phase, rotate(chi_a), rotate(chi_b)


def omega_ref(binary, labels: Labelled):
    """The dimensionless orbital reference frequency ``(M f_ref) pi``."""
    return (labels.m_tot_sec * binary.f_ref) * np.pi


def inertial_modes(data: NRSurData, binary, mode_array=None):
    """``PrecessingNRSur_core`` for a batch (see
    :func:`.._waveform.inertial_modes`): the inertial-frame modes on the
    coorbital times (real and imaginary parts, each ``(B, n_modes,
    n_coorb)``), the labelled binaries and whether each evaluation is
    valid."""
    labels = labelled(binary)
    omegas = reference_omegas(data, labels.q, labels.chi_a, labels.chi_b)
    states, valid = integrate(data, labels.q, labels.chi_a, labels.chi_b, omega_ref(binary, labels), omegas)
    quat, phase, chi_a, chi_b = _coorbital_dynamics(data, states, labels.chi_a, labels.chi_b)
    re, im = _coorbital_modes(data, labels.q, chi_a, chi_b)
    mask, _ = active_modes(mode_array)
    if not np.all(mask):
        mask = mx.array(mask.astype(np.float32))[:, None]
        re, im = re * mask, im * mask
    re, im = coorbital_to_coprecessing(re, im, phase)
    return tuple(rotate(re, im, quat)), labels, valid


_ODD = mx.array(np.array([m % 2 != 0 for _, m in MODES]))


def _mode_signs(labels: Labelled):
    """``-1`` for the odd-``m`` modes after a label switch, else ``1``
    (``(B, n_modes)``)."""
    return 1.0 - 2.0 * (labels.switched[:, None] & _ODD).astype(mx.float32)


def _amplitude(binary, labels: Labelled, scale):
    """``Mtot_sec C_PER_MPC scale / distance`` (``(B,)``)."""
    return labels.m_tot_sec * (C_PER_MPC * scale) / binary.luminosity_distance


def _small_d(ell: int, mp: int, m: int, beta):
    """Wigner's ``d^l_{m' m}(beta)`` in float64 (as
    :func:`waveformjax._harmonics.wigner_small_d`)."""
    cos_half = np.cos(beta / 2)
    sin_half = np.sin(beta / 2)
    total = 0.0
    prefactor = sqrt(factorial(ell + mp) * factorial(ell - mp) * factorial(ell + m) * factorial(ell - m))
    for k in range(max(0, m - mp), min(ell + m, ell - mp) + 1):
        coefficient = (-1) ** (k - m + mp) * prefactor / (
            factorial(ell + m - k) * factorial(k) * factorial(ell - k - mp) * factorial(k - m + mp)
        )
        total = total + coefficient * cos_half ** (2 * ell - 2 * k + m - mp) * sin_half ** (2 * k - m + mp)
    return total


@functools.lru_cache(maxsize=None)
def _harmonic_series(keep: Tuple[int, ...]):
    """``sqrt((2l + 1) / 4 pi) d^l_{m,2}(theta)`` of the modes ``keep``
    (indices into :data:`MODES`) as trigonometric polynomials (products of
    ``2 l`` powers of ``cos(theta / 2)`` and ``sin(theta / 2)``, of degree
    ``l`` in ``theta``): the coefficients of ``cos(j theta)`` and ``sin(j
    theta)``, ``j = 0 .. LMAX`` (``(LMAX + 1, n_modes)`` each), from their
    discrete Fourier transform in float64, and the ``m`` of the modes."""
    n = 8 * LMAX
    theta = 2 * np.pi * np.arange(n) / n
    values = np.stack(
        [np.sqrt((2 * MODES[i][0] + 1) / (4 * np.pi)) * _small_d(MODES[i][0], MODES[i][1], 2, theta) for i in keep],
        axis=1,
    )
    spectrum = np.fft.rfft(values, axis=0)[: LMAX + 1] / n
    cos_coefficients = 2 * spectrum.real
    cos_coefficients[0] /= 2
    sin_coefficients = -2 * spectrum.imag
    j = np.arange(LMAX + 1)[:, None]
    series = np.cos(j * theta).T @ cos_coefficients + np.sin(j * theta).T @ sin_coefficients
    assert np.max(np.abs(series - values)) < 1e-12
    orders = np.array([MODES[i][1] for i in keep], dtype=np.float32)
    return (mx.array(cos_coefficients.astype(np.float32)), mx.array(sin_coefficients.astype(np.float32)),
            mx.array(orders))


_HARMONIC_ORDERS = mx.array(np.arange(LMAX + 1, dtype=np.float32))


def _harmonics(theta, phi, keep: Tuple[int, ...]):
    """The real and imaginary parts of ``{}_{-2}Y_{lm}(theta, phi)`` for the
    modes ``keep`` (each ``(B, n_modes)``; as
    :func:`waveformjax._harmonics.spin_weighted_spherical_harmonic`), from
    their trigonometric series in ``theta``: a few operations on all the
    modes at once, whose derivatives MLX can compile (the powers of
    ``cos(theta / 2)`` and ``sin(theta / 2)`` fuse into a kernel with more
    arguments than Metal allows)."""
    cos_coefficients, sin_coefficients, orders = _harmonic_series(keep)
    angle = theta[:, None] * _HARMONIC_ORDERS
    d = mx.cos(angle) @ cos_coefficients + mx.sin(angle) @ sin_coefficients
    angle = phi[:, None] * orders
    return d * mx.cos(angle), d * mx.sin(angle)


def _resample(data: NRSurData, values, tau):
    """The natural splines through ``values`` (``(B, n_coorb, k)``) on
    ``t_coorb`` at the dimensionless times ``tau`` (``(B, m)``), zero
    outside the surrogate: ``(B, m, k)``. The second derivatives are the
    inverse of the spline system applied to its right-hand side."""
    x = data.coorb_x
    n = data.n_coorb
    dy = values[:, 1:] - values[:, :-1]
    h_inv = data.coorb_h_inv[:, None]
    rhs = 3.0 * (dy[:, 1:] * h_inv[1:] - dy[:, :-1] * h_inv[:-1])
    zeros = mx.zeros_like(values[:, :1])
    c = mx.concatenate([zeros, _shared_matmul(data.coorb_system_inverse, rhs), zeros], axis=1)
    interval = index(mx.clip(mx.searchsorted(x, tau, side="right").astype(mx.int32) - 1, 0, n - 2))
    shape = (*interval.shape, values.shape[-1])
    at = mx.broadcast_to(interval[..., None], shape)
    after = mx.broadcast_to((interval + 1)[..., None], shape)
    dx = mx.take(data.coorb_h, interval)[..., None]
    y_i = mx.take_along_axis(values, at, axis=1)
    c_i = mx.take_along_axis(c, at, axis=1)
    c_ip1 = mx.take_along_axis(c, after, axis=1)
    b = (mx.take_along_axis(values, after, axis=1) - y_i) / dx - dx * (c_ip1 + 2.0 * c_i) / 3.0
    d = (c_ip1 - c_i) / (3.0 * dx)
    delx = (tau - mx.take(x, interval))[..., None]
    out = y_i + delx * (b + delx * (c_i + delx * d))
    inside = (tau >= x[0]) & (tau <= x[-1])
    return mx.where(inside[..., None], out, 0.0)


def _times(times, labels: Labelled):
    """The times (s), shared ``(m,)`` or per binary ``(B, m)``, in units of
    the total mass: ``(B, m)``."""
    times = _as_float32(times)
    if times.ndim == 1:
        times = times[None]
    return times / labels.m_tot_sec[:, None]


def polarizations_from_modes(binary, h, labels: Labelled, mode_array=None, scale=STRAIN_SCALE):
    """``h_+`` and ``h_x`` on the coorbital times (``(B, n_coorb)``) from
    the inertial modes ``h = (re, im)``."""
    _, ells = active_modes(mode_array)
    theta = binary.inclination
    phi = 0.5 * np.pi - binary.phi_ref
    keep = tuple(i for i, (ell, _) in enumerate(MODES) if ell in ells)
    y_re, y_im = _harmonics(theta, phi, keep)
    rows = mx.array(keep)
    signs = mx.take(_mode_signs(labels), rows, axis=1)
    c_re = (y_re * signs)[..., None]
    c_im = (y_im * signs)[..., None]
    re, im = mx.take(h[0], rows, axis=1), mx.take(h[1], rows, axis=1)
    hp = mx.sum(c_re * re - c_im * im, axis=1)
    hc = -mx.sum(c_im * re + c_re * im, axis=1)
    a0 = _amplitude(binary, labels, scale)[:, None]
    return hp * a0, hc * a0


def polarizations_at_dimensionless(data: NRSurData, binary, tau, mode_array=None, scale=STRAIN_SCALE):
    """``(h_+, h_x)`` (each ``(B, m)``) of the batched ``binary`` at the
    times ``tau`` (``(B, m)``) in units of the total mass."""
    h, labels, valid = inertial_modes(data, binary, mode_array)
    hp, hc = polarizations_from_modes(binary, h, labels, mode_array, scale)
    out = _resample(data, mx.stack([hp, hc], axis=-1), tau)
    out = mx.where(valid[:, None, None], out, np.nan)
    return out[..., 0], out[..., 1]


def polarizations_at(data: NRSurData, binary, times, mode_array=None, scale=STRAIN_SCALE):
    """``(h_+, h_x)`` (each ``(B, m)``) of the batched ``binary`` at
    ``times`` (s, relative to the peak; ``(m,)`` or ``(B, m)``)."""
    return polarizations_at_dimensionless(data, binary, _times(times, labelled(binary)), mode_array, scale)


def modes_at_dimensionless(data: NRSurData, binary, tau, mode_array=None, scale=STRAIN_SCALE):
    """The inertial modes ``{(l, m): h_lm}`` (complex64, each ``(B, m)``)
    of the batched ``binary`` at the times ``tau`` in units of the total
    mass, for the ``ell`` of :func:`.._waveform.output_ells`."""
    (re, im), labels, valid = inertial_modes(data, binary, mode_array)
    ells = output_ells(mode_array)
    keep = [i for i, (ell, _) in enumerate(MODES) if ell in ells]
    rows = mx.array(keep)
    factor = mx.take(_mode_signs(labels) * _amplitude(binary, labels, scale)[:, None], rows, axis=1)[..., None]
    re, im = mx.take(re, rows, axis=1), mx.take(im, rows, axis=1)
    values = mx.concatenate([re * factor, im * factor], axis=1).transpose(0, 2, 1)
    out = _resample(data, values, tau)
    out = mx.where(valid[:, None, None], out, np.nan)
    n = len(keep)
    return {MODES[i]: out[..., k] + 1j * out[..., n + k] for k, i in enumerate(keep)}


def modes_at(data: NRSurData, binary, times, mode_array=None, scale=STRAIN_SCALE):
    """The inertial modes of the batched ``binary`` at ``times`` (s,
    relative to the peak)."""
    return modes_at_dimensionless(data, binary, _times(times, labelled(binary)), mode_array, scale)
