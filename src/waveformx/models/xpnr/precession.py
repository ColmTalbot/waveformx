# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The precession set-up of IMRPhenomXP/XPHM/XPNR
(``IMRPhenomXGetAndSetPrecessionVariables`` of
``LALSimIMRPhenomX_precession.c``) for the NNLO (``PhenomXPrecVersion`` 102)
and MSA (223) Euler angles, after waveformrs's ``imrphenomx/precession.rs``
and ``imrphenomx/pnr.rs``:

* the PN orbital angular momentum (:func:`orbital_angular_momentum_3pn`,
  :func:`orbital_angular_momentum_3pn_msa`, :func:`l_ansatz`);
* the effective precessing spin :func:`chi_p` and in-plane spins;
* the J frame (:func:`j_frame`), the line of sight (:func:`line_of_sight`),
  ``kappa`` and ``alpha0`` (:func:`kappa_alpha0`), ``thetaJN``
  (:func:`theta_jn`) and the polarization angle ``zeta``
  (:func:`polarization_angle`), for ``PhenomXPConvention`` 1;
* the precessing final spin (:func:`precessing_final_spin_2017`);
* :func:`precession_struct`, the set-up of versions 102 and 223 as LAL does it
  for IMRPhenomXP/XPHM (with the fallback of 223 to 102 when the MSA system
  fails as a traced boolean), and the reference offsets of the angles;
* :func:`msa_angles` and :func:`pn_angles`, ``XLALSimIMRPhenomXPMSAAngles`` and
  ``XLALSimIMRPhenomXPPNAngles``.

Masses are in units of the total mass with the body labels of
``XLALIMRPhenomXPCheckMassesAndSpins`` (body 1 strictly heavier: equal masses
are swapped). Spins are 3-sequences ``(x, y, z)`` of arrays in the L frame at
the reference frequency. Where LAL's expressions are ill-conditioned (``acos``
of a ratio close to one) the values are LAL's and the derivatives those of an
equivalent well-conditioned expression, so that gradients stay finite and
accurate for (nearly) aligned spins and face-on lines of sight.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._generated.utilities import final_spin_2017
from ._math import (
    C_SI, FRAC_PI_2, PC_SI, PI, POWERS_OF_LALPI, Struct, g_si, hz_to_mf, mass_unit, real, rotate_y, rotate_z,
)
from .internals import set_ringdown_frequencies, waveform_struct
from .msa import acos_safe, msa_system, phi_zeta_costhetal
from .msa import final_spin as msa_final_spin
from .nnlo import nnlo_coefficients, nnlo_cos_beta, nnlo_evaluate

__all__ = [
    "MAX_TOL_ATAN", "atan2_tol", "orbital_angular_momentum_3pn", "orbital_angular_momentum_3pn_msa",
    "three_pn", "three_pn_msa", "chirp_time_bound", "delta_mf_from_wfstruct",
    "select_orbital_angular_momentum", "l_ansatz", "l_ansatz_3pn", "hypot_safe", "chi_in_plane", "chi_p",
    "precessing_final_spin_2017", "line_of_sight", "j_frame", "kappa_alpha0", "theta_jn",
    "polarization_angle", "angle_offsets", "pi_gm", "precession_struct", "check_masses_and_spins",
    "msa_angles", "pn_angles",
]

MAX_TOL_ATAN = 1.0e-15


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def hypot_safe(*components):
    """``sqrt(x^2 + y^2 + ...)`` (summed in order), zero with a zero
    derivative where all components vanish."""
    r2 = components[0] * components[0]
    for c in components[1:]:
        r2 = r2 + c * c
    positive = r2 > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, r2, 1.0)), 0.0)


def _atan2_safe(y, x):
    """``atan2(y, x)``, zero with a zero derivative at ``(0, 0)``."""
    zero = (x == 0.0) & (y == 0.0)
    return jnp.where(zero, 0.0, jnp.arctan2(jnp.where(zero, 0.0, y), jnp.where(zero, 1.0, x)))


def atan2_tol(a, b, tol=MAX_TOL_ATAN):
    """``XLALSimIMRPhenomXatan2tol``: ``atan2(a, b)``, zero if both arguments
    are below ``tol`` in magnitude."""
    small = (jnp.abs(a) < tol) & (jnp.abs(b) < tol)
    return jnp.where(small, 0.0, jnp.arctan2(jnp.where(small, 0.0, a), jnp.where(small, 1.0, b)))


def _value_with_gradient_of(value, smooth):
    """``value`` with the derivative of ``smooth`` (an equivalent,
    better-conditioned expression)."""
    return smooth + jax.lax.stop_gradient(value - smooth)


def _angle_between(a, b, a_dot_b, norms):
    """LAL's ``acos(a.b / (|a| |b|))`` with the derivative of
    ``atan2(|a x b|, a.b)``."""
    value = acos_safe(a_dot_b / norms)
    cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    smooth = _atan2_safe(hypot_safe(*cross), a_dot_b)
    return _value_with_gradient_of(value, smooth)


# ---------------------------------------------------------------------------
# orbital angular momentum
# ---------------------------------------------------------------------------


def orbital_angular_momentum_3pn(eta, delta, chi1l, chi2l):
    """The coefficients ``pPrec->L0 ... L8L`` of the 3PN orbital angular
    momentum with linear-in-spin terms (versions 102, 220, 221, 224, 3xx)."""
    zero = jnp.zeros_like(eta * chi1l * chi2l)
    return Struct(
        l0=zero + 1.0,
        l1=zero,
        l2=3.0 / 2. + eta / 6.0 + zero,
        l3=(5 * (chi1l * (-2 - 2 * delta + eta) + chi2l * (-2 + 2 * delta + eta))) / 6.,
        l4=(81 + (-57 + eta) * eta) / 24. + zero,
        l5=(-7 * (chi1l * (72 + delta * (72 - 31 * eta) + eta * (-121 + 2 * eta))
                  + chi2l * (72 + eta * (-121 + 2 * eta) + delta * (-72 + 31 * eta)))) / 144.,
        l6=(10935 + eta * (-62001 + eta * (1674 + 7 * eta) + 2214 * POWERS_OF_LALPI.two)) / 1296. + zero,
        l7=zero,
        l8=zero,
        l8l=zero,
    )


def orbital_angular_momentum_3pn_msa(eta, delta, chi1l, chi2l):
    """The 3PN orbital angular momentum of versions 222 and 223, with the
    spin-orbit terms of ``LALSimInspiralFDPrecAngles``."""
    eta2 = eta * eta
    l = orbital_angular_momentum_3pn(eta, delta, chi1l, chi2l)
    zero = l.l1
    l.l3 = (-7 * (chi1l + chi2l + chi1l * delta - chi2l * delta) + 5 * (chi1l + chi2l) * eta) / 6.
    l.l5 = (-1650 * (chi1l + chi2l + chi1l * delta - chi2l * delta) + 1336 * (chi1l + chi2l) * eta
            + 511 * (chi1l - chi2l) * delta * eta + 28 * (chi1l + chi2l) * eta2) / 600.
    l.l6 = (10935 + eta * (-62001 + 1674 * eta + 7 * eta2 + 2214 * POWERS_OF_LALPI.two)) / 1296. + zero
    return l


three_pn = orbital_angular_momentum_3pn
three_pn_msa = orbital_angular_momentum_3pn_msa


def select_orbital_angular_momentum(condition, l_true, l_false):
    """``l_true`` where ``condition``, else ``l_false``."""
    return jax.tree_util.tree_map(lambda a, b: jnp.where(condition, a, b), l_true, l_false)


def l_ansatz(l, v, l_norm):
    """``XLALSimIMRPhenomXLPNAnsatz``: ``LNorm sum_a L_a x^(a/2)`` with
    ``x = v^2`` and the coefficients ``l``."""
    x = v * v
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    sqx = jnp.sqrt(x)
    return l_norm * (l.l0 + l.l1 * sqx + l.l2 * x + l.l3 * (x * sqx) + l.l4 * x2 + l.l5 * (x2 * sqx) + l.l6 * x3
                     + l.l7 * (x3 * sqx) + l.l8 * x4 + l.l8l * x4 * jnp.log(x))


def l_ansatz_3pn(l, v, l_norm):
    """:func:`l_ansatz` without the terms that vanish for the 3PN
    coefficients (``L1``, ``L7``, ``L8``, ``L8L``)."""
    x = v * v
    x2 = x * x
    x3 = x * x2
    sqx = jnp.sqrt(x)
    return l_norm * (l.l0 + l.l2 * x + l.l3 * (x * sqx) + l.l4 * x2 + l.l5 * (x2 * sqx) + l.l6 * x3)


# ---------------------------------------------------------------------------
# spins
# ---------------------------------------------------------------------------


def chi_in_plane(chi1, chi2):
    """``sqrt(chi1x^2 + chi1y^2 + chi2x^2 + chi2y^2)``, which selects the
    NNLO angles for negligible in-plane spins."""
    return hypot_safe(chi1[0], chi1[1], chi2[0], chi2[1])


def chi_p(m1, m2, chi1, chi2):
    """The in-plane spin quantities of
    ``IMRPhenomXGetAndSetPrecessionVariables`` for the masses ``m1 >= m2``
    (units of the total mass, ``m1_SI / Mtot_SI``): a :class:`~._math.Struct`
    with ``chi_p`` (Schmidt et al. 2015), ``chi1_perp``, ``chi2_perp``,
    ``s1_perp``, ``s2_perp`` (``m_i^2 chi_i_perp``), ``s_tot_perp`` and
    ``chi_tot_perp`` (``|S1_perp + S2_perp| M^2 / m1^2``)."""
    m = m1 + m2
    m1_2 = m1 * m1
    m2_2 = m2 * m2
    chi1_perp = hypot_safe(chi1[0], chi1[1])
    chi2_perp = hypot_safe(chi2[0], chi2[1])
    s1_perp = m1_2 * chi1_perp
    s2_perp = m2_2 * chi2_perp
    s1x = chi1[0] * m1_2
    s1y = chi1[1] * m1_2
    s2x = chi2[0] * m2_2
    s2y = chi2[1] * m2_2
    s_tot_perp = hypot_safe(s1x + s2x, s1y + s2y)
    a1 = 2.0 + (3.0 * m2) / (2.0 * m1)
    a2 = 2.0 + (3.0 * m1) / (2.0 * m2)
    asp1 = a1 * s1_perp
    asp2 = a2 * s2_perp
    num = jnp.where(asp2 > asp1, asp2, asp1)
    den = jnp.where(m2 > m1, a2 * m2_2, a1 * m1_2)
    return Struct(chi_p=num / den, chi1_perp=chi1_perp, chi2_perp=chi2_perp, s1_perp=s1_perp, s2_perp=s2_perp,
                  s_tot_perp=s_tot_perp, chi_tot_perp=s_tot_perp * (m * m) / m1_2)


def precessing_final_spin_2017(eta, chi1l, chi2l, chi_perp):
    """``XLALSimIMRPhenomXPrecessingFinalSpin2017``: the final spin from the
    aligned-spin fit and the in-plane spin ``chi_perp`` (weighted by
    ``(m1 / M)^2``), with the sign of the aligned-spin fit."""
    x = 1.0 - 4.0 * eta
    positive = x > 0.0
    sqrt_x = jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)
    m1 = 0.5 * (1.0 + sqrt_x)
    m2 = 0.5 * (1.0 - sqrt_x)
    m = m1 + m2
    larger = m1 >= m2
    q_factor = jnp.where(larger, m1, m2) / m
    af_parallel = jnp.where(larger, final_spin_2017(eta, chi1l, chi2l), final_spin_2017(eta, chi2l, chi1l))
    sperp = chi_perp * q_factor * q_factor
    return jnp.copysign(1.0, af_parallel) * hypot_safe(sperp, af_parallel)


def chirp_time_bound(f_start, m1, m2, s1, s2):
    """``XLALSimInspiralChirpTimeBound``: an upper bound on the chirp time (s)
    from ``f_start`` (Hz) for the masses ``m1``, ``m2`` in kg and the aligned
    spins ``s1``, ``s2``."""
    m = m1 + m2
    mu = m1 * m2 / m
    eta = mu / m
    chi = jnp.abs(jnp.where(jnp.abs(s1) > jnp.abs(s2), s1, s2))
    # XLALSimInspiralTaylorT2Timing_0PNCoeff, _2PNCoeff, _4PNCoeff
    total_mass = m * (g_si() / C_SI ** 3.0)
    c0 = jnp.abs(-5. * total_mass / (256. * eta))
    c2 = 7.43 / 2.52 + 11. / 3. * eta
    c3 = (226.0 / 15.0) * chi
    c4 = 30.58673 / 5.08032 + 54.29 / 5.04 * eta + 61.7 / 7.2 * eta * eta
    # (LAL: cbrt(pi G m f) / c, which XLA turns into a power of c beyond
    # the range of float32)
    v = jnp.cbrt(PI * total_mass * f_start)
    return c0 * v ** -8 * (1.0 + (c2 + (c3 + c4 * v) * v) * v * v)


def delta_mf_from_wfstruct(wf):
    """``get_deltaF_from_wfstruct``: the frequency spacing (units of the total
    mass) of the default uniform grid of the binary, ``min(1 / max(4,
    2^ceil(log2(T))), 0.1)`` Hz for the chirp time bound ``T`` from ``f_ref``
    (used by the SpinTaylor angles on non-uniform grids)."""
    seglen = chirp_time_bound(wf.f_ref, wf.m1_si, wf.m2_si, wf.chi1l, wf.chi2l)
    delta_f_v1 = 1. / jnp.maximum(4., 2.0 ** jnp.ceil(jnp.log(seglen) / jnp.log(2.0)))
    delta_f = jnp.minimum(delta_f_v1, 0.1)
    return hz_to_mf(delta_f, wf.m_tot)


# ---------------------------------------------------------------------------
# J frame and line of sight (PhenomXPConvention 1)
# ---------------------------------------------------------------------------


def line_of_sight(inclination, phi_ref):
    """The line of sight ``N`` in the source (L) frame,
    ``(sin(i) cos(pi/2 - phiRef), sin(i) sin(pi/2 - phiRef), cos(i))``."""
    s = jnp.sin(inclination)
    return [s * jnp.cos(FRAC_PI_2 - phi_ref), s * jnp.sin(FRAC_PI_2 - phi_ref), jnp.cos(inclination)]


def j_frame(m1_2, m2_2, chi1, chi2, l_ref):
    """The total angular momentum ``J = L + S1 + S2`` at the reference
    frequency in the source frame (``L = l_ref`` along ``z``, units of
    ``M^2``) and its polar and azimuthal angles (convention 1: ``phiJ_Sf = 0``
    where both in-plane components are below ``MAX_TOL_ATAN``).

    Returns a :class:`~._math.Struct` with ``j0`` (a list), ``j0_norm``,
    ``theta_j_sf`` and ``phi_j_sf``."""
    j0 = [
        m1_2 * chi1[0] + m2_2 * chi2[0],
        m1_2 * chi1[1] + m2_2 * chi2[1],
        m1_2 * chi1[2] + m2_2 * chi2[2] + l_ref,
    ]
    j0_norm = jnp.sqrt(j0[0] * j0[0] + j0[1] * j0[1] + j0[2] * j0[2])
    # thetaJ_Sf = acos(J0z / J0) (0 for |J0| < 1e-10), with the derivative of
    # atan2(|J0_perp|, J0z)
    value = acos_safe(j0[2] / jnp.where(j0_norm < 1e-10, 1.0, j0_norm))
    smooth = _atan2_safe(hypot_safe(j0[0], j0[1]), j0[2])
    theta_j_sf = jnp.where(j0_norm < 1e-10, 0.0, _value_with_gradient_of(value, smooth))
    phi_j_sf = atan2_tol(j0[1], j0[0], MAX_TOL_ATAN)
    return Struct(j0=j0, j0_norm=j0_norm, theta_j_sf=theta_j_sf, phi_j_sf=phi_j_sf)


def kappa_alpha0(theta_j_sf, phi_j_sf, n_sf):
    """``kappa``, the third rotation from the source frame to the J frame
    (``N`` rotated with ``RY[-thetaJ] RZ[-phiJ]`` lies in the ``xz`` plane of
    the J frame at azimuth ``kappa``), and ``alpha0 = pi - kappa``
    (convention 1). Also returns ``N`` rotated by ``RY[-thetaJ] RZ[-phiJ]``
    (before the rotation by ``-kappa``)."""
    tmp = rotate_y(-theta_j_sf, rotate_z(-phi_j_sf, n_sf))
    kappa = atan2_tol(tmp[1], tmp[0], MAX_TOL_ATAN)
    return kappa, PI - kappa, tmp


def theta_jn(j0, j0_norm, n_sf):
    """``thetaJN = acos(J0 . N / |J0|)`` (convention 1), with the derivative
    of ``atan2(|J0 x N|, J0 . N)`` (finite for ``N`` along ``J``)."""
    j0_dot_n = j0[0] * n_sf[0] + j0[1] * n_sf[1] + j0[2] * n_sf[2]
    return _angle_between(j0, n_sf, j0_dot_n, j0_norm)


def polarization_angle(inclination, phi_ref, theta_j_sf, phi_j_sf, kappa, nx_jf, nz_jf):
    """The polarization angle ``zeta_polarization`` (convention 1): the angle
    of the ``X`` axis of the LAL wave frame, rotated into the J frame, in the
    ``P``, ``Q`` basis of Arun et al. (``P = (Nz, 0, -Nx)``, ``Q = (0, 1, 0)``
    for the components ``(Nx, 0, Nz)`` of the line of sight in the J frame)."""
    ci = jnp.cos(inclination)
    x_sf = [-ci * jnp.sin(phi_ref), -ci * jnp.cos(phi_ref), jnp.sin(inclination)]
    tmp = rotate_z(-kappa, rotate_y(-theta_j_sf, rotate_z(-phi_j_sf, x_sf)))
    p_arun = (nz_jf, 0.0, -nx_jf)
    q_arun = (0.0, 1.0, 0.0)
    x_dot_p = (tmp[0] * p_arun[0]) + (tmp[1] * p_arun[1]) + (tmp[2] * p_arun[2])
    x_dot_q = (tmp[0] * q_arun[0]) + (tmp[1] * q_arun[1]) + (tmp[2] * q_arun[2])
    return jnp.arctan2(x_dot_q, x_dot_p)


# ---------------------------------------------------------------------------
# the precession struct for versions 102 and 223
# ---------------------------------------------------------------------------


def pi_gm(m1_si, m2_si):
    """``pPrec->piGM = pi G M / c^3`` (s) for the masses in kg."""
    return PI * (m1_si + m2_si) * (g_si() / C_SI) / (C_SI * C_SI)


def angle_offsets(wf, use_msa, msa, nnlo, alpha0, epsilon0, mprime=2):
    """``Get_alphaepsilon_atfref``: the offsets ``(alpha_offset,
    epsilon_offset)`` of the Euler angles, their values at ``omega = pi M
    f_ref 2 / m'`` minus ``alpha0`` and ``epsilon0``, from the MSA system
    where ``use_msa`` (a possibly traced boolean; ``msa`` may be ``None`` if
    it is ``False``) and from the NNLO coefficients ``nnlo`` elsewhere."""
    omega_ref = wf.pi_m * wf.f_ref * 2. / mprime
    alpha, epsilon = nnlo_evaluate(nnlo, omega_ref)
    alpha_offset = alpha - alpha0
    epsilon_offset = epsilon - epsilon0
    if msa is not None:
        angles = phi_zeta_costhetal(msa, jnp.cbrt(omega_ref))
        alpha_offset = jnp.where(use_msa, angles.phiz - alpha0, alpha_offset)
        epsilon_offset = jnp.where(use_msa, angles.zeta - epsilon0, epsilon_offset)
    return alpha_offset, epsilon_offset


def precession_struct(wf, m1_si, m2_si, chi1, chi2, version=223):
    """``IMRPhenomXGetAndSetPrecessionVariables`` for ``PhenomXPrecVersion``
    102 or 223 (a static Python int), ``PhenomXPConvention`` 1 and the
    default ``PhenomXPFinalSpinMod`` 4 (which becomes 3 for these versions),
    as LAL does it for IMRPhenomXP/XPHM.

    ``wf`` is the waveform struct of :func:`.internals.waveform_struct`,
    ``m1_si >= m2_si`` (kg) and the spins ordered as by
    :func:`check_masses_and_spins`.

    Returns ``(prec, wf)``: the precession :class:`~._math.Struct` and the
    waveform struct modified as LAL modifies it (``phi0 = 0``, the precessing
    final spin and the (2,2) ringdown frequencies).

    ``prec.use_msa`` is true where the MSA angles are used (version 223 and
    ``MSA_ERROR`` unset), ``prec.msa_error`` the failure of the MSA system
    (LAL then continues as version 102: NNLO angles, 3PN orbital angular
    momentum and the final spin of ``chi_p``). Fields: ``m1``, ``m2``,
    ``m1_2``, ``m2_2``, ``pi_gm``, ``chi1``, ``chi2``, ``chi1_norm``,
    ``chi2_norm``, ``error_spin`` (a spin beyond the Kerr bound), the
    quantities of :func:`chi_p`, ``sl``, ``sperp``, ``l`` (orbital angular
    momentum coefficients), ``l_ref``, ``j0``, ``j0_norm``, ``theta_j_sf``,
    ``phi_j_sf``, ``kappa``, ``alpha0``, ``epsilon0``, ``theta_jn``,
    ``zeta_polarization``, ``nnlo`` (the NNLO coefficients, evaluated for
    every version), ``msa`` (the MSA system or ``None``), ``alpha_offset``,
    ``epsilon_offset``.
    """
    if version not in (102, 223):
        raise ValueError("only PhenomXPrecVersion 102 and 223 are supported")
    chi1 = [jnp.asarray(c, dtype=real()) for c in chi1]
    chi2 = [jnp.asarray(c, dtype=real()) for c in chi2]
    m1 = m1_si / wf.m_tot_si
    m2 = m2_si / wf.m_tot_si
    m = m1 + m2
    m1_2 = m1 * m1
    m2_2 = m2 * m2
    q = m1 / m2
    eta = wf.eta
    delta = wf.delta
    pgm = pi_gm(m1_si, m2_si)

    chi1_norm = jnp.sqrt(chi1[0] * chi1[0] + chi1[1] * chi1[1] + chi1[2] * chi1[2])
    chi2_norm = jnp.sqrt(chi2[0] * chi2[0] + chi2[1] * chi2[1] + chi2[2] * chi2[2])
    error_spin = ~(jnp.abs(chi1_norm) <= 1.0) | ~(jnp.abs(chi2_norm) <= 1.0)

    perp = chi_p(m1, m2, chi1, chi2)
    chip = perp.chi_p
    chi1l = chi1[2]
    chi2l = chi2[2]
    sl = chi1l * m1_2 + chi2l * m2_2
    sperp = chip * m1_2

    if version == 223:
        msa = msa_system(eta, wf.m1, wf.m2, m1_2, m2_2, chi1, chi2, pgm, wf.f_ref)
        msa_error = msa.msa_error
        use_msa = ~msa_error
    else:
        msa = None
        msa_error = jnp.zeros(jnp.shape(eta), dtype=bool)
        use_msa = jnp.zeros(jnp.shape(eta), dtype=bool)

    # IMRPhenomX_SetPrecessingRemnantParams: PhenomXPFinalSpinMod 4 -> 3, the
    # precession-averaged final spin of the MSA system, which defaults to
    # version 0 (chi_p) for the other versions and a failed MSA system
    a_final_prec = precessing_final_spin_2017(eta, chi1l, chi2l, chip)
    if msa is not None:
        af_parallel = final_spin_2017(eta, chi1l, chi2l)
        a_final_prec = jnp.where(use_msa, msa_final_spin(msa, af_parallel, m, m1_2, m2_2, chi1l, chi2l),
                                 a_final_prec)
    a_final_prec = jnp.where(jnp.abs(a_final_prec) > 1.0, jnp.copysign(1.0, a_final_prec), a_final_prec)
    wf = wf.replace(a_final=a_final_prec, a_final_prec=a_final_prec, phi0=0.0)
    wf = set_ringdown_frequencies(wf)

    l = orbital_angular_momentum_3pn(eta, delta, chi1l, chi2l)
    if msa is not None:
        l = select_orbital_angular_momentum(use_msa, orbital_angular_momentum_3pn_msa(eta, delta, chi1l, chi2l), l)
    l_ref = m * m * l_ansatz(l, wf.v_ref, wf.eta / wf.v_ref)

    frame = j_frame(m1_2, m2_2, chi1, chi2, l_ref)
    n_sf = line_of_sight(wf.inclination, wf.phi_ref_in)
    kappa, alpha0, _ = kappa_alpha0(frame.theta_j_sf, frame.phi_j_sf, n_sf)
    thetajn = theta_jn(frame.j0, frame.j0_norm, n_sf)
    zeta_polarization = polarization_angle(wf.inclination, wf.phi_ref_in, frame.theta_j_sf, frame.phi_j_sf, kappa,
                                           jnp.sin(thetajn), jnp.cos(thetajn))

    nnlo = nnlo_coefficients(eta, delta, q, wf.chi_eff, chip, m1)
    epsilon0 = frame.phi_j_sf - PI
    alpha_offset, epsilon_offset = angle_offsets(wf, use_msa, msa, nnlo, alpha0, epsilon0)

    prec = Struct(
        version=version, use_msa=use_msa, msa_error=msa_error, m1=m1, m2=m2, m1_2=m1_2, m2_2=m2_2, pi_gm=pgm,
        chi1=chi1, chi2=chi2, chi1_norm=chi1_norm, chi2_norm=chi2_norm, error_spin=error_spin,
        chi_p=chip, chi1_perp=perp.chi1_perp, chi2_perp=perp.chi2_perp, s1_perp=perp.s1_perp,
        s2_perp=perp.s2_perp, s_tot_perp=perp.s_tot_perp, chi_tot_perp=perp.chi_tot_perp, sl=sl, sperp=sperp,
        l=l, l_ref=l_ref, j0=frame.j0, j0_norm=frame.j0_norm, theta_j_sf=frame.theta_j_sf,
        phi_j_sf=frame.phi_j_sf, kappa=kappa, alpha0=alpha0, epsilon0=epsilon0, theta_jn=thetajn,
        zeta_polarization=zeta_polarization, nnlo=nnlo, msa=msa, alpha_offset=alpha_offset,
        epsilon_offset=epsilon_offset,
    )
    return prec, wf


# ---------------------------------------------------------------------------
# XLALSimIMRPhenomXPMSAAngles and XLALSimIMRPhenomXPPNAngles
# ---------------------------------------------------------------------------


def check_masses_and_spins(m1, m2, chi1, chi2):
    """``XLALIMRPhenomXPCheckMassesAndSpins``: the bodies ordered with the
    first strictly heavier (equal masses are swapped)."""
    keep = m1 > m2
    chi1 = [jnp.asarray(c, dtype=real()) for c in chi1]
    chi2 = [jnp.asarray(c, dtype=real()) for c in chi2]
    return (jnp.where(keep, m1, m2), jnp.where(keep, m2, m1), [jnp.where(keep, a, b) for a, b in zip(chi1, chi2)],
            [jnp.where(keep, b, a) for a, b in zip(chi1, chi2)])


def _angles_setup(frequencies, m1, m2, chi1, chi2, inclination, f_ref, version):
    frequencies = jnp.asarray(frequencies, dtype=real())
    m1_si, m2_si, chi1, chi2 = check_masses_and_spins(m1 * mass_unit(), m2 * mass_unit(), chi1, chi2)
    f_ref = jnp.where(f_ref == 0.0, frequencies[0], f_ref)
    # (the distance does not matter: 1 m, in Mpc)
    wf = waveform_struct(m1_si, m2_si, chi1[2], chi2[2], 0.0, f_ref, 0.0, frequencies[0], frequencies[-1],
                         1.0e-6 / PC_SI,
                         inclination)
    prec, wf = precession_struct(wf, m1_si, m2_si, chi1, chi2, version)
    return frequencies, prec, wf


def msa_angles(frequencies, m1, m2, chi1, chi2, inclination, f_ref, mprime=2):
    """``XLALSimIMRPhenomXPMSAAngles`` with ``PhenomXPrecVersion`` 223 and the
    masses in solar masses: ``(alpha, gamma, cos(beta))`` of the MSA angles of
    the co-precessing mode ``m'`` at the gravitational-wave frequencies
    ``frequencies`` (Hz), ``v = (pi G M f 2 / m')^(1/3)``.

    Where the MSA system fails, LAL's set-up falls back to version 102 but
    this function still evaluates the MSA angles, with the NNLO offsets, the
    MSA corrections of the versions other than 222/223 and the 3PN orbital
    angular momentum of version 102 in ``cos(beta)``; so does this port.
    Returns ``(alpha, gamma, cos_beta, prec)``."""
    frequencies, prec, wf = _angles_setup(frequencies, m1, m2, chi1, chi2, inclination, f_ref, 223)
    v = jnp.cbrt(frequencies * prec.pi_gm * (2.0 / mprime))
    angles = phi_zeta_costhetal(prec.msa, v, paper_corrections=prec.msa_error, l=prec.l)
    return angles.phiz - prec.alpha_offset, -(angles.zeta - prec.epsilon_offset), angles.cos_theta_l, prec


def pn_angles(frequencies, m1, m2, chi1, chi2, inclination, f_ref, mprime=2):
    """``XLALSimIMRPhenomXPPNAngles`` (the NNLO angles of version 102) with
    the masses in solar masses: ``(alpha, gamma, cos(beta), prec)`` of the
    co-precessing mode ``m'`` at the gravitational-wave frequencies
    ``frequencies`` (Hz)."""
    frequencies, prec, wf = _angles_setup(frequencies, m1, m2, chi1, chi2, inclination, f_ref, 102)
    omega = frequencies * prec.pi_gm * (2.0 / mprime)
    v = jnp.cbrt(omega)
    l = l_ansatz(prec.l, v, wf.eta / v)
    alpha, epsilon = nnlo_evaluate(prec.nnlo, omega)
    cos_beta = nnlo_cos_beta(l, prec.sperp, prec.sl)
    return alpha - prec.alpha_offset, -(epsilon - prec.epsilon_offset), cos_beta, prec
