# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The NRSur7dq4 waveform as pure JAX functions of a binary
(``PrecessingNRSur_core``, ``XLALSimInspiralPrecessingNRSurPolarizations``
and ``XLALSimInspiralPrecessingNRSurModes``).

Conventions (as LAL's):

* The binary is set up where the orbital frequency of the coprecessing-frame
  waveform is ``pi f_ref``; there the inertial frame is the coorbital frame
  (``z`` along the principal axis of the radiation, ``x`` from the lighter
  to the heavier black hole) and the spins are given in it. ``f_ref = 0``
  selects the start of the surrogate (LAL's ``f_ref = f_min = 0``; the
  drivers of :mod:`._driver` replace ``f_ref = 0`` by ``f_min``).
* If ``mass_1 < mass_2`` the labels are swapped, the in-plane spins rotated
  by pi and the odd-``m`` modes negated.
* The modes are in units of the strain at the binary's luminosity distance,
  multiplied by ``scale`` (:data:`STRAIN_SCALE` unless the option
  ``scaled_strain`` is false); the polarizations are summed with
  ``-2Y_lm(inclination, pi/2 - phi_ref)``.
* Times are in seconds relative to the peak of the waveform (``t = 0`` at
  the surrogate's time 0); outside the surrogate (about ``-4300 M`` to
  ``100 M``) the waveform is zero.
* The waveform is evaluated in the precision of the binary's parameters
  (:func:`precision`): float32 if they are float32, else float64. The
  binary and the times are converted to it, and the data must be of it.
  Nothing passes through SI units: the masses stay in solar masses (LAL's
  ``m * MSUN_SI / MSUN_SI`` changes them by rounding only) and the
  amplitude is one constant times the total mass over the distance in
  Mpc, so that no intermediate, nor its derivatives, leaves the range of
  float32.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..._harmonics import spin_weighted_spherical_harmonic
from ._data import LMAX, MODES, PIECES, NRSurData
from ._data import precision as _precision
from ._dynamics import integrate, reference_omegas, safe_norm
from ._fits import fit_parameters, fold_mass_ratio, q_fit, spin_monomials
from ._frames import coorbital_to_coprecessing, coprecessing_to_inertial
from ._spline import evaluate, locate, second_derivatives

__all__ = [
    "MSUN_SI",
    "MTSUN_SI",
    "C_SI",
    "PC_SI",
    "C_PER_MPC",
    "STRAIN_SCALE",
    "strain_scale",
    "precision",
    "as_precision",
    "Labelled",
    "labelled",
    "inertial_modes",
    "polarizations_at",
    "modes_at",
    "active_modes",
    "output_ells",
]

# LAL's constants (lal 7.x, LALConstants.h)
MSUN_SI = 1.988409870698050731911960804878414216e30
MTSUN_SI = 4.925490947641266978197229498498379006e-6
C_SI = 299792458.0
PC_SI = 3.085677581491367278913937957796471611e16
#: The speed of light in Mpc/s: the amplitude is ``Mtot_sec C_PER_MPC /
#: distance`` with the distance in Mpc.
C_PER_MPC = C_SI / (1e6 * PC_SI)
#: The factor of the strain unless ``scaled_strain`` is false.
STRAIN_SCALE = 1e21


def strain_scale(options) -> float:
    """:data:`STRAIN_SCALE` if ``options.scaled_strain``, else 1."""
    return STRAIN_SCALE if options.scaled_strain else 1.0


def precision(binary):
    """The floating-point type of the evaluation of ``binary``: float32 if
    its parameters are float32 (Python numbers take the type of the
    others), else float64."""
    return _precision(jnp.result_type(*jax.tree_util.tree_leaves(binary)))


def as_precision(binary, dtype):
    """``binary`` with every parameter an array of ``dtype``."""
    return type(binary)(*(jnp.asarray(value, dtype) for value in binary))


class Labelled(NamedTuple):
    """The binary after ``PrecessingNRSur_switch_labels_if_needed``: black
    hole A is the heavier."""

    switched: jnp.ndarray
    q: jnp.ndarray
    chi_a: jnp.ndarray
    chi_b: jnp.ndarray
    m_tot_sec: jnp.ndarray


def labelled(binary) -> Labelled:
    """Swap the labels if ``mass_1 < mass_2`` (rotating the in-plane spins
    by pi), with the masses in solar masses."""
    m1 = jnp.asarray(binary.mass_1)
    m2 = jnp.asarray(binary.mass_2)
    s1 = jnp.asarray(binary.spin_1)
    s2 = jnp.asarray(binary.spin_2)
    switched = m1 < m2
    flip = jnp.array([-1.0, -1.0, 1.0], dtype=s1.dtype)
    mass_a = jnp.where(switched, m2, m1)
    mass_b = jnp.where(switched, m1, m2)
    chi_a = jnp.where(switched, flip * s2, s1)
    chi_b = jnp.where(switched, flip * s1, s2)
    m_tot_sec = (mass_a + mass_b) * MTSUN_SI
    return Labelled(switched, mass_a / mass_b, chi_a, chi_b, m_tot_sec)


def omega_ref(binary, labels: Labelled):
    """The dimensionless orbital reference frequency ``(M f_ref) pi``
    (``get_dimless_omega``)."""
    return (labels.m_tot_sec * jnp.asarray(binary.f_ref)) * jnp.pi


def active_modes(mode_array) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """The coorbital modes LAL evaluates for a ``ModeArray`` (``None`` for
    all): ``(l, m)`` and ``(l, -m)`` together if either is listed. Returns a
    0/1 mask over :data:`MODES` and the ``ell`` whose inertial modes are
    computed (those with a listed coorbital mode)."""
    if mode_array is None:
        return np.ones(len(MODES)), tuple(range(2, LMAX + 1))
    listed = set(tuple(mode) for mode in mode_array)
    mask = np.array([1.0 if (ell, m) in listed or (ell, -m) in listed else 0.0 for ell, m in MODES])
    ells = tuple(ell for ell in range(2, LMAX + 1) if any((ell, m) in listed for m in range(-ell, ell + 1)))
    return mask, ells


def output_ells(mode_array) -> Tuple[int, ...]:
    """The ``ell`` of the inertial modes returned for a ``ModeArray``."""
    return active_modes(mode_array)[1]


def _coorbital_modes(data: NRSurData, q, chi_a, chi_b):
    """The coorbital modes on ``t_coorb`` (real and imaginary parts, each
    ``(n_modes, n_coorb)``) from the coorbital-frame spins there
    (``(n_coorb, 3)``): the fits at the empirical nodes times the
    empirical-interpolation bases (``PrecessingNRSur_eval_data_piece``),
    combined into the modes as ``PrecessingNRSur_core``."""
    times = data.ei_time
    coefs = fold_mass_ratio(data.ei_coefs, q_fit(q))
    spins = spin_monomials(fit_parameters(q, chi_a[times], chi_b[times]), data.ei_spins)
    values = jnp.sum(coefs * spins, axis=1)
    bounds = data.ei_bounds
    pieces = {
        piece: values[bounds[p] : bounds[p + 1]] @ data.ei_basis[bounds[p] : bounds[p + 1]]
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
    return jnp.stack([re[mode] for mode in MODES]), jnp.stack([im[mode] for mode in MODES])


def _coorbital_dynamics(data: NRSurData, states, chi_a0, chi_b0):
    """The dynamics on the coorbital times: the quaternion (normalised), the
    orbital phase and the spins rotated to the coorbital frame
    (``spline_array_interp``, ``PrecessingNRSur_normalize_results``,
    ``PrecessingNRSur_rotate_spins``)."""
    c = second_derivatives(data.ds_knots, states)
    series = evaluate(data.ds_knots, states, c, data.coorb_index, data.coorb_delx)
    quat = series[:, 0:4]
    phase = series[:, 4]
    norm_a = safe_norm(chi_a0)
    norm_b = safe_norm(chi_b0)

    def renormalize(chi, norm):
        square = jnp.sum(chi * chi, axis=-1, keepdims=True)
        positive = (square > 0) & (norm > 0)
        n = jnp.sqrt(jnp.where(positive, square, 1.0))
        return jnp.where(positive, chi * (norm / n), chi)

    chi_a = renormalize(series[:, 5:8], norm_a)
    chi_b = renormalize(series[:, 8:11], norm_b)
    quat = quat / jnp.sqrt(jnp.sum(quat * quat, axis=-1, keepdims=True))
    cp = jnp.cos(phase)
    sp = jnp.sin(phase)

    def rotate(chi):
        return jnp.stack([chi[:, 0] * cp + chi[:, 1] * sp, -chi[:, 0] * sp + chi[:, 1] * cp, chi[:, 2]], axis=-1)

    return quat, phase, rotate(chi_a), rotate(chi_b)


def inertial_modes(data: NRSurData, binary, mode_array=None):
    """``PrecessingNRSur_core``: the inertial-frame modes on the coorbital
    times (real and imaginary parts, each ``(n_modes, n_coorb)``,
    dimensionless, in the labelled
    frame: without the sign of the odd modes of a label switch), the
    labelled binary and whether the evaluation is valid (false where LAL
    would fail to find the reference time). ``binary.f_ref = 0`` starts the
    dynamics at the beginning of the surrogate."""
    binary = as_precision(binary, data.dtype)
    labels = labelled(binary)
    omegas = reference_omegas(data, labels.q, labels.chi_a, labels.chi_b)
    states, valid = integrate(data, labels.q, labels.chi_a, labels.chi_b, omega_ref(binary, labels), omegas)
    quat, phase, chi_a, chi_b = _coorbital_dynamics(data, states, labels.chi_a, labels.chi_b)
    re, im = _coorbital_modes(data, labels.q, chi_a, chi_b)
    mask, _ = active_modes(mode_array)
    if not np.all(mask):
        mask = jnp.asarray(mask, re.dtype)[:, None]
        re, im = re * mask, im * mask
    re, im = coorbital_to_coprecessing(re, im, phase)
    return coprecessing_to_inertial(re, im, quat.T), labels, valid


def _mode_signs(labels: Labelled):
    """``-1`` for the odd-``m`` modes after a label switch, else ``1``."""
    odd = jnp.asarray(np.array([m % 2 != 0 for _, m in MODES]))
    return 1.0 - 2.0 * (labels.switched & odd).astype(labels.q.dtype)


def _amplitude(binary, labels: Labelled, scale):
    """``Mtot_sec C_SI / distance`` times ``scale``, with the distance in
    Mpc (``Mtot_sec C_PER_MPC scale / distance``)."""
    return labels.m_tot_sec * (C_PER_MPC * scale) / jnp.asarray(binary.luminosity_distance)


def _resample(data: NRSurData, values, tau):
    """The natural splines through ``values`` (``(n_coorb, k)``) on
    ``t_coorb`` at the dimensionless times ``tau``, zero outside the
    surrogate."""
    knots = data.coorb_knots
    index = locate(knots.x, tau)
    c = second_derivatives(knots, values)
    out = evaluate(knots, values, c, index, tau - knots.x[index])
    inside = (tau >= knots.x[0]) & (tau <= knots.x[-1])
    return jnp.where(inside[:, None], out, 0.0)


def polarizations_from_modes(binary, h, labels: Labelled, mode_array=None, scale=STRAIN_SCALE):
    """``h_+`` and ``h_x`` on the coorbital times from the inertial modes
    ``h = (re, im)`` (dimensionless) summed with ``-2Y_lm(inclination, pi/2
    - phi_ref)``, scaled to the distance and multiplied by ``scale``."""
    _, ells = active_modes(mode_array)
    theta = jnp.asarray(binary.inclination)
    phi = 0.5 * jnp.pi - jnp.asarray(binary.phi_ref)
    keep = np.array([i for i, (ell, _) in enumerate(MODES) if ell in ells], dtype=int)
    harmonics = jnp.stack([spin_weighted_spherical_harmonic(theta, phi, *MODES[i]) for i in keep])
    harmonics = harmonics * _mode_signs(labels)[keep]
    c_re, c_im = jnp.real(harmonics)[:, None], jnp.imag(harmonics)[:, None]
    re, im = h[0][keep], h[1][keep]
    hp = jnp.sum(c_re * re - c_im * im, axis=0)
    hc = -jnp.sum(c_im * re + c_re * im, axis=0)
    a0 = _amplitude(binary, labels, scale)
    return hp * a0, hc * a0


def polarizations_at_dimensionless(data: NRSurData, binary, tau, mode_array=None, scale=STRAIN_SCALE):
    """``(h_+, h_x)`` at the times ``tau`` in units of the total mass,
    multiplied by ``scale``."""
    binary = as_precision(binary, data.dtype)
    h, labels, valid = inertial_modes(data, binary, mode_array)
    hp, hc = polarizations_from_modes(binary, h, labels, mode_array, scale)
    out = _resample(data, jnp.stack([hp, hc], axis=-1), jnp.asarray(tau, data.dtype))
    out = jnp.where(valid, out, jnp.nan)
    return out[:, 0], out[:, 1]


def polarizations_at(data: NRSurData, binary, times, mode_array=None, scale=STRAIN_SCALE):
    """``(h_+, h_x)`` at ``times`` (s, relative to the peak), multiplied by
    ``scale``."""
    binary = as_precision(binary, data.dtype)
    tau = jnp.asarray(times, data.dtype) / labelled(binary).m_tot_sec
    return polarizations_at_dimensionless(data, binary, tau, mode_array, scale)


def modes_at_dimensionless(data: NRSurData, binary, tau, mode_array=None, scale=STRAIN_SCALE):
    """The inertial modes ``{(l, m): h_lm}`` (complex, scaled to the
    distance and multiplied by ``scale``, with LAL's signs) at the times
    ``tau`` in units of the total mass, for the ``ell`` of
    :func:`output_ells`."""
    binary = as_precision(binary, data.dtype)
    (re, im), labels, valid = inertial_modes(data, binary, mode_array)
    ells = output_ells(mode_array)
    keep = [i for i, (ell, _) in enumerate(MODES) if ell in ells]
    factor = (_mode_signs(labels) * _amplitude(binary, labels, scale))[np.array(keep, dtype=int), None]
    values = jnp.concatenate([re[np.array(keep, dtype=int)] * factor, im[np.array(keep, dtype=int)] * factor], axis=0).T
    out = _resample(data, values, jnp.asarray(tau, data.dtype))
    out = jnp.where(valid, out, jnp.nan)
    n = len(keep)
    return {MODES[i]: out[:, k] + 1j * out[:, n + k] for k, i in enumerate(keep)}


def modes_at(data: NRSurData, binary, times, mode_array=None, scale=STRAIN_SCALE):
    """The inertial modes at ``times`` (s, relative to the peak), multiplied
    by ``scale``."""
    binary = as_precision(binary, data.dtype)
    tau = jnp.asarray(times, data.dtype) / labelled(binary).m_tot_sec
    return modes_at_dimensionless(data, binary, tau, mode_array, scale)
