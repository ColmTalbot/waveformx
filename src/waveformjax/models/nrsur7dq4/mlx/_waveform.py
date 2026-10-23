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

from math import factorial, sqrt
from typing import NamedTuple, Tuple

import mlx.core as mx
import numpy as np

from .._data import LMAX, MODES, PIECES
from .._waveform import C_PER_MPC, MTSUN_SI, STRAIN_SCALE, active_modes, output_ells
from ._data import NRSurData
from ._dynamics import index, integrate, reference_omegas, safe_norm
from ._fits import fit_parameters, fold_mass_ratio, q_fit, spin_monomials
from ._frames import coorbital_to_coprecessing, coprecessing_to_inertial

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
    pieces = {
        piece: values[:, bounds[p] : bounds[p + 1]] @ data.ei_basis[bounds[p] : bounds[p + 1]]
        for p, piece in enumerate(PIECES)
    }
    re, im = {}, {}
    for ell in range(2, LMAX + 1):
        re[ell, 0] = pieces[ell, 0, "real"]
        im[ell, 0] = pieces[ell, 0, "imag"]
        for m in range(1, ell + 1):
            re_plus, re_minus = pieces[ell, m, "Re+"], pieces[ell, m, "Re-"]
            im_plus, im_minus = pieces[ell, m, "Im+"], pieces[ell, m, "Im-"]
            # h^{l,m} = X_+ + X_-, h^{l,-m} = conj(X_+ - X_-)
            re[ell, m] = re_plus + re_minus
            re[ell, -m] = re_plus - re_minus
            im[ell, m] = im_plus + im_minus
            im[ell, -m] = -im_plus + im_minus
    return mx.stack([re[mode] for mode in MODES], axis=1), mx.stack([im[mode] for mode in MODES], axis=1)


def _coorbital_dynamics(data: NRSurData, states, chi_a0, chi_b0):
    """The quaternion (normalised), the orbital phase and the spins in the
    coorbital frame on the coorbital times (``(B, n_coorb, ...)``), from the
    states at the dynamics nodes (``(B, n_ds, 11)``) by one matrix product
    (the splines of :func:`.._waveform._coorbital_dynamics`)."""
    series = data.ds_to_coorb @ states
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
    return coprecessing_to_inertial(re, im, quat), labels, valid


_ODD = mx.array(np.array([m % 2 != 0 for _, m in MODES]))


def _mode_signs(labels: Labelled):
    """``-1`` for the odd-``m`` modes after a label switch, else ``1``
    (``(B, n_modes)``)."""
    return 1.0 - 2.0 * (labels.switched[:, None] & _ODD).astype(mx.float32)


def _amplitude(binary, labels: Labelled, scale):
    """``Mtot_sec C_PER_MPC scale / distance`` (``(B,)``)."""
    return labels.m_tot_sec * (C_PER_MPC * scale) / binary.luminosity_distance


def _power(x, k: int):
    """``x^k`` by multiplication (``x ** 0`` has a NaN derivative at 0 in
    MLX)."""
    value = 1.0
    for _ in range(k):
        value = value * x
    return value


def _wigner_small_d(ell: int, mp: int, m: int, beta):
    """Wigner's ``d^l_{m' m}(beta)`` (as :func:`waveformjax._harmonics.wigner_small_d`)."""
    cos_half = mx.cos(beta / 2)
    sin_half = mx.sin(beta / 2)
    total = 0.0
    prefactor = sqrt(factorial(ell + mp) * factorial(ell - mp) * factorial(ell + m) * factorial(ell - m))
    for k in range(max(0, m - mp), min(ell + m, ell - mp) + 1):
        coefficient = (-1) ** (k - m + mp) * prefactor / (
            factorial(ell + m - k) * factorial(k) * factorial(ell - k - mp) * factorial(k - m + mp)
        )
        total = total + coefficient * _power(cos_half, 2 * ell - 2 * k + m - mp) * _power(sin_half, 2 * k - m + mp)
    return total


def _harmonic(theta, phi, ell: int, m: int):
    """The real and imaginary parts of ``{}_{-2}Y_{lm}(theta, phi)`` (as
    :func:`waveformjax._harmonics.spin_weighted_spherical_harmonic`)."""
    norm = float(np.sqrt((2 * ell + 1) / (4 * np.pi)))
    d = norm * _wigner_small_d(ell, m, 2, theta)
    return d * mx.cos(m * phi), d * mx.sin(m * phi)


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
    c = mx.concatenate([zeros, data.coorb_system_inverse @ rhs, zeros], axis=1)
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
    keep = [i for i, (ell, _) in enumerate(MODES) if ell in ells]
    harmonics = [_harmonic(theta, phi, *MODES[i]) for i in keep]
    rows = mx.array(keep)
    signs = mx.take(_mode_signs(labels), rows, axis=1)
    c_re = (mx.stack([y[0] for y in harmonics], axis=1) * signs)[..., None]
    c_im = (mx.stack([y[1] for y in harmonics], axis=1) * signs)[..., None]
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
