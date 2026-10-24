# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The phase of the IMRPhenomXHM higher modes (``PhenomXHMReleaseVersion``
122019 and 122022 share it): ``IMRPhenomXHM_GetPhaseCoefficients``,
``GetSpheroidalCoefficients``, ``SpheroidalToSpherical`` and the phase
ansatzes of ``LALSimIMRPhenomXHM_internals.c``, after waveformrs's
``imrphenomx/hm_internals.rs``.

The inspiral is the 22 phase rescaled to the mode, the intermediate region
a fit to collocation points of the phase derivative, and the ringdown the
rescaled 22 ringdown or, for the (3,2) mode, the spheroidal (3,2) ringdown
rotated to the spherical harmonics with the 22 ringdown
(spheroidal-to-spherical mode mixing). The intermediate phase of the (3,2)
is fixed by the first and second derivatives of the rotated ringdown phase,
taken by finite differences with a step of 1e-7
(:mod:`.finite_differences`): the rotated phase contains the 22 phase
(scaled by ``1 / eta``, up to ~1e4 rad for extreme mass ratios), whose
rounding the second difference amplifies by 1e14. This mode therefore agrees
with LAL to ~1e-4 for comparable masses and ~1e-2 at mass ratio 100, as
different builds of LAL do.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from .. import phase22 as phase22_module
from .._generated import hm_amp_122019, hm_inspiral, hm_intermediate, hm_ringdown
from .._generated import ringdown as ringdown22
from .._math import FRAC_PI_4, PI, Struct, fma, powers
from . import inspiral
from .finite_differences import phase_derivatives
from .variables import both, is_mode, where_mode

__all__ = [
    "spheroidal_coefficients",
    "spheroidal_to_spherical",
    "phase_coefficients",
    "phase_fits",
    "phase",
]

def _solve(rows, values):
    """``gsl_linalg_LU_solve`` of a small collocation system."""
    matrix = jnp.stack([jnp.stack([jnp.asarray(x, dtype=jnp.float64) for x in row]) for row in rows])
    rhs = jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in values])
    solution = jnp.linalg.solve(matrix, rhs)
    return [solution[i] for i in range(len(values))]


# ------------------------------------------------------------------------
# ansatzes
# ------------------------------------------------------------------------

def ringdown_phase_ansatz(ff, p, hm, ph):
    """``IMRPhenomXHM_RD_Phase_Ansatz``: the ringdown phase derivative (the
    rescaled 22 ringdown, or the spheroidal ansatz for the (3,2))."""
    frd = hm.f_ring
    fda = hm.f_damp
    den = fma(fda, fda, (ff - frd) * (ff - frd))
    if not hm.mixing_on:
        return fma(frd * frd * ph.alpha2, p.m_two, ph.alpha0) + (ph.alpha_l * fda) / den
    d = fma(ph.alpha2_s, p.m_two, ph.alpha0_s)
    return fma(ph.alpha4_s, p.m_four, d) + (ph.alpha_l_s * fda) / den


def ringdown_phase_ansatz_int(ff, p, hm, ph):
    """``IMRPhenomXHM_RD_Phase_AnsatzInt``: the ringdown phase."""
    invf = p.m_one
    frd = hm.f_ring
    fda = hm.f_damp
    arctan = jnp.arctan((ff - frd) / fda)
    if not hm.mixing_on:
        phase_rd = fma(ph.alpha0, ff, -(frd * frd * ph.alpha2 * invf))
        return fma(ph.alpha_l, arctan, phase_rd)
    phase_rd = fma(ph.alpha0_s, ff, ph.phi0_s)
    phase_rd = fma(-ph.alpha2_s, invf, phase_rd)
    phase_rd = fma(-(1.0 / 3.0 * ph.alpha4_s), p.m_three, phase_rd)
    return fma(ph.alpha_l_s, arctan, phase_rd)


def intermediate_phase_ansatz(ff, p, hm, ph):
    """``IMRPhenomXHM_Inter_Phase_Ansatz``: the intermediate phase
    derivative."""
    fda = hm.f_damp
    frd = hm.f_ring
    lorentzian = (ph.c_l * fda) / fma(fda, fda, (ff - frd) * (ff - frd))
    dphase = fma(ph.c1, p.m_one, ph.c0)
    dphase = fma(ph.c2, p.m_two, dphase)
    dphase = fma(ph.c4, p.m_four, dphase)
    if is_mode(hm, 32) is not False:
        # (c3 = 0 for the other modes computed with the (3,2))
        dphase = fma(ph.c3, p.m_three, dphase)
    return dphase + lorentzian


def intermediate_phase_ansatz_int(ff, p, hm, ph):
    """``IMRPhenomXHM_Inter_Phase_AnsatzInt``: the intermediate phase."""
    fda = hm.f_damp
    frd = hm.f_ring
    phase_ir = fma(ph.c0, ff, ph.c1 * p.log)
    phase_ir = fma(-ph.c2, p.m_one, phase_ir)
    phase_ir = fma(-(1.0 / 3.0 * ph.c4), p.m_three, phase_ir)
    if is_mode(hm, 32) is not False:
        phase_ir = fma(-(0.5 * ph.c3), p.m_two, phase_ir)
    return fma(ph.c_l, jnp.arctan((ff - frd) / fda), phase_ir)


# ------------------------------------------------------------------------
# the (3,2) ringdown
# ------------------------------------------------------------------------

def _phi_ref_22(wf, phase_22_at_mf_ref, timeshift, phaseshift):
    """``phiref22``: the phase offset ``phifRef`` of the 22 mode as computed
    in ``IMRPhenomXHM_GetPhaseCoefficients`` and
    ``GetSpheroidalCoefficients``, from the 22 phase at ``Mf_ref``."""
    return fma(-1.0 / wf.eta, phase_22_at_mf_ref, -(timeshift * wf.mf_ref)) - phaseshift + 2.0 * wf.phi0 + FRAC_PI_4


def _phase_22_at(phase22, *mf):
    """``IMRPhenomX_Phase_22`` at the frequencies ``mf`` (one evaluation)."""
    x = jnp.stack(mf)
    values = phase22_module.phase_22(x, powers(x), phase22)
    return [values[i] for i in range(len(mf))]


def spheroidal_coefficients(wf, hm, phase22):
    """``GetSpheroidalCoefficients`` (ringdown phase version 122019): the
    spheroidal ringdown phase of the (3,2) mode (``alpha0_s``, ``alphaL_s``,
    ``alpha2_s``, ``alpha4_s``, ``phi0_s``) aligned with the 22 ringdown,
    and the 22 time shift and phase offset (``timeshift``, ``phiref22``).
    ``phase22`` carries the 22 connection coefficients."""
    fringlm = hm.f_ring
    fdamplm = hm.f_damp
    freqs = [wf.f_ring, fma(-1.5, fdamplm, fringlm), fma(-0.5, fdamplm, fringlm), fma(0.5, fdamplm, fringlm)]
    rows = []
    values = []
    for i, ff in enumerate(freqs):
        values.append(hm_ringdown.phase_32_fit(i, wf))
        ffm1 = 1.0 / ff
        ffm2 = ffm1 * ffm1
        lorentzian = fdamplm / (fdamplm * fdamplm + (ff - fringlm) * (ff - fringlm))
        rows.append([1.0, lorentzian, ffm2, ffm2 * ffm2])
    alpha0_s, alpha_l_s, alpha2_s, alpha4_s = _solve(rows, values)
    ph = Struct(alpha0_s=alpha0_s, alpha_l_s=alpha_l_s, alpha2_s=alpha2_s, alpha4_s=alpha4_s, phi0_s=0.0)
    hm = Struct(f_ring=fringlm, f_damp=fdamplm, mixing_on=True)

    # time shift of the spheroidal ansatz
    fref_rd = wf.f_ring + wf.f_damp
    p_ref = powers(fref_rd)
    tshift = hm_ringdown.phase_32_spheroidal_time_shift(wf)
    timeshift = phase22_module.time_shift_22(phase22, wf)
    dphi22_ref = fma(1.0 / wf.eta, phase22_module.dphase_22(fref_rd, p_ref, phase22), timeshift)
    ph.alpha0_s = ph.alpha0_s + dphi22_ref + tshift - ringdown_phase_ansatz(fref_rd, p_ref, hm, ph)

    # phase shift of the spheroidal ansatz
    phaseshift = 0.0
    fref_rd = wf.f_ring
    p_ref = powers(fref_rd)
    phase_22_mf_ref, phase_22_fref = _phase_22_at(phase22, wf.mf_ref, fref_rd)
    phiref22 = _phi_ref_22(wf, phase_22_mf_ref, timeshift, phaseshift)
    phi22_ref = fma(1.0 / wf.eta, phase_22_fref, timeshift * fref_rd) + phaseshift + phiref22
    phishift = hm_ringdown.phase_32_spheroidal_phase_shift(wf)
    ph.phi0_s = phi22_ref - ringdown_phase_ansatz_int(fref_rd, p_ref, hm, ph) + phishift
    return ph, timeshift, phiref22


def spheroidal_to_spherical(mf, p, mix, rd):
    """``SpheroidalToSpherical`` (release 122019): the spherical (3,2) mode
    in the ringdown, ``conj(a322) h22 + conj(a323) h32_spheroidal``, from the
    22 ringdown and the spheroidal (3,2) ansatz, both rescaled by the
    leading-order 22 amplitude (``(re, im)``, without ``amp0``).

    ``rd`` are the ringdown amplitude coefficients of the (3,2); ``mix``
    holds the 22 amplitude and phase coefficients (``amp22``, ``phase22``,
    ``f_ring22``, ``f_damp22``, ``inveta``), the 22 alignment
    (``timeshift``, ``phaseshift``, ``phiref22``), the spheroidal phase
    (``spheroidal``), the mixing coefficients and the ringdown and damping
    frequencies of the (3,2) (``f_ring``, ``f_damp``)."""
    amp22 = ringdown22.amp_22_ansatz_exact(mf, mix.f_ring22, mix.f_damp22, mix.amp22)
    phi22 = fma(mix.inveta, phase22_module.phase_22(mf, p, mix.phase22), mix.timeshift * mf) + mix.phaseshift + mix.phiref22
    wf22r_re = amp22 * jnp.cos(phi22)
    wf22r_im = amp22 * jnp.sin(phi22)
    # (the 122019 ringdown amplitude keeps the rescaling by the leading order)
    amplm = hm_amp_122019.rd_amp_ansatz0(mf, mix.f_ring, mix.f_damp, rd[0], rd[1], rd[2], rd[3])
    view = Struct(f_ring=mix.f_ring, f_damp=mix.f_damp, mixing_on=True, mode_tag=32)
    philm = ringdown_phase_ansatz_int(mf, p, view, mix.spheroidal)
    (mix2_re, mix2_im), (mix3_re, mix3_im) = mix.mixing_coeffs[2], mix.mixing_coeffs[3]
    # conj(mix2) wf22r
    a_re = wf22r_re * mix2_re + wf22r_im * mix2_im
    a_im = wf22r_im * mix2_re - wf22r_re * mix2_im
    # conj(mix3) amplm exp(i philm)
    b_re = amplm * mix3_re
    b_im = -(mix3_im * amplm)
    c, s = jnp.cos(philm), jnp.sin(philm)
    c_re = b_re * c - b_im * s
    c_im = b_re * s + b_im * c
    return a_re + c_re, a_im + c_im


# ------------------------------------------------------------------------
# coefficients
# ------------------------------------------------------------------------

def get_fcut_insp(wf, hm):
    """``GetfcutInsp``: the end of the inspiral phase region."""
    eta_factor = fma(0.001, 0.25 / wf.eta - 1.0, 1.0)
    return eta_factor * (hm.emm * 1.0) * 0.5 * wf.f_meco


def pn21_amp_sign(ff, wf):
    """``IMRPhenomXHM_PN21AmpSign``: the sign of the real part of the (2,1)
    PN amplitude at ``ff`` (a Python float), positive as ``True``."""
    eta = wf.eta
    chi1 = wf.chi1l
    chi2 = wf.chi2l
    delta = jnp.sqrt(fma(-4.0, eta, 1.0))
    sqrt5 = math.sqrt(5.0)
    output = (-16.0 * delta * eta * ff * PI ** 1.5) / (3.0 * sqrt5)
    output = output + (4.0 * 2.0 ** 0.3333333333333333 * (chi1 - chi2 + delta * (chi1 + chi2)) * eta
                       * ff ** 1.3333333333333333 * PI ** 1.8333333333333333) / sqrt5
    output = output + (2.0 * 2.0 ** 0.6666666666666666 * eta * (306.0 * delta - 360.0 * delta * eta)
                       * ff ** 1.6666666666666667 * PI ** 2.1666666666666665) / (189.0 * sqrt5)
    return output >= 0.0


def phase_fits(wf, hm):
    """The fits of the phase of the mode of ``hm`` (a static mode): the
    intermediate collocation values and ``lambda`` of the inspiral
    (:func:`phase_coefficients` reads them from ``hm.fits``)."""
    return Struct(
        phase_values=[hm_intermediate.phase_fit(hm.mode_int, i, wf) for i in range(6)],
        lambda_pn=jnp.where(wf.eta > 0.01, hm_inspiral.lambda_pn(wf.eta, hm.mode_tag),
                            hm_inspiral.phase_lambda_fit(hm.mode_int, wf)),
    )


def _power(x, exponent):
    """``x ** exponent`` (Python floats while tracing for a static mode)."""
    return x ** exponent if isinstance(x, float) else jnp.power(x, exponent)


def _log(x):
    return math.log(x) if isinstance(x, float) else jnp.log(x)


def phase_coefficients(wf, hm, phase22, mix=None):
    """``IMRPhenomXHM_GetPhaseCoefficients`` (version 122019): the
    coefficients of the phase of the mode and the 22 time shift and phase
    offset used to align it (``timeshift``, ``phiref22``).

    ``phase22`` is :func:`~..phase22.phase_coefficients` (with the 22
    connection coefficients); ``hm`` holds the mode's fits (``hm.fits``,
    :func:`phase_fits`), and its mode may be one of several computed together
    (see :func:`~.variables.is_mode`). For the (3,2) mode ``mix`` holds the
    spheroidal coefficients, the 22 data and the ringdown amplitude
    coefficients ``mix.rd`` (:func:`spheroidal_to_spherical`)."""
    ell = hm.ell
    emm = hm.emm
    is21, is32, is33 = is_mode(hm, 21), is_mode(hm, 32), is_mode(hm, 33)
    diagonal = ell == emm
    eta = wf.eta
    eta_m1 = 1.0 / eta

    # IMRPhenomXHM_Intermediate_CollocPtsFreqs (version 122019)
    fring = hm.f_ring
    fdamp = hm.f_damp
    fcut = get_fcut_insp(wf, hm)
    sqrt3 = math.sqrt(3.0)
    if is32 is not False:
        f_end = fma(-0.5, wf.f_damp, wf.f_ring)
        freqs32 = [
            fcut,
            fma(sqrt3, fcut - f_end, 2.0 * (fcut + f_end)) / 4.0,
            fma(3.0, fcut, f_end) / 4.0,
            (fcut + f_end) / 2.0,
            f_end,
            f_end,
        ]
        # correct the cutting frequency for EMR with negative spins
        match32 = jnp.where((eta < 0.01) & (wf.chi1l < 0.0), f_end * fma(-0.25, wf.chi1l, 1.2), f_end)
    if is32 is not True:
        freqs = [
            fcut,
            fma(sqrt3, fcut - fring, 2.0 * (fcut + fring)) / 4.0,
            fma(3.0, fcut, fring) / 4.0,
            (fcut + fring) / 2.0,
            fma(3.0, fring, fcut) / 4.0,
            fma(7.0, fring, fcut) / 8.0,
        ]
        f_phase_match_im = fring - fdamp
    if is32 is True:
        freqs, f_phase_match_im = freqs32, match32
    elif is32 is not False:
        freqs = [jnp.where(is32, a, b) for a, b in zip(freqs32, freqs)]
        f_phase_match_im = jnp.where(is32, match32, f_phase_match_im)
    f_phase_match_in = hm.f_meco_lm
    values = [value + hm.delta_t for value in hm.fits.phase_values]
    fcut_rd = f_phase_match_im
    fcut_insp = f_phase_match_in

    # inspiral: rescale the 22 coefficients, phi_lm(f) ~ m/2 phi_22(2 f / m)
    zero = 0.0
    non_log = [phase22.phi0, phase22.phi1, phase22.phi2, phase22.phi3, phase22.phi4, phase22.phi5,
               phase22.phi6, phase22.phi7, phase22.phi8, phase22.phi9, zero, zero, zero]
    log_terms = [zero, zero, zero, zero, zero, phase22.phi5l, phase22.phi6l, zero, phase22.phi8l,
                 phase22.phi9l, zero, zero, zero]
    pseudo_pn = [zero] * 8 + [phase22.sigma1, phase22.sigma2, phase22.sigma3, phase22.sigma4, phase22.sigma5]
    m_over_2 = emm * 0.5
    two_over_m = 1.0 / m_over_2
    fact = phase22.phi_norm / eta
    log_m_over_2 = _log(m_over_2)
    phi = []
    phi_l = []
    for i in range(13):
        nl = non_log[i] * fact
        lg = log_terms[i] * fact
        ps = pseudo_pn[i] * fact
        scale = _power(m_over_2, (8 - i) / 3.0)
        phi.append(fma(-lg, log_m_over_2, nl + ps) * scale)
        phi_l.append(lg * scale)
    lambda_pn = hm.fits.lambda_pn
    phi[8] = phi[8] + lambda_pn
    ph = Struct(phi=phi, phi_l=phi_l, lambda_pn=lambda_pn, alpha0=0.0, alpha2=0.0, alpha_l=0.0,
                c0=0.0, c1=0.0, c2=0.0, c3=0.0, c4=0.0, c_l=0.0)

    # intermediate-ringdown: for high spins extend the inspiral of the 21
    # into the intermediate region (with the 22 phase derivative before the
    # 22 connection coefficients are computed, as LAL does)
    if is21 is not False:
        unconnected = phase22.replace(c1_int=0.0, c2_int=0.0, c1_mrd=0.0, c2_mrd=0.0)
        ff = two_over_m * jnp.stack(freqs[:3])
        insp_vals = 1.0 / eta * phase22_module.dphase_22(ff, powers(ff), unconnected)
        diff12 = insp_vals[0] - insp_vals[1]
        diff23 = insp_vals[1] - insp_vals[2]
        high = both(is21, wf.s_tot_r >= 0.8)
        v1 = values[2] + diff23
        v0 = v1 + diff12
        values[1] = jnp.where(high, v1, values[1])
        values[0] = jnp.where(high, v0, values[0])

    def row(ff):
        ffm1 = 1.0 / ff
        ffm2 = ffm1 * ffm1
        lorentzian = fdamp / (fdamp * fdamp + (ff - fring) * (ff - fring))
        return [1.0, lorentzian, ffm1, ffm2, ffm2 * ffm2, ffm1 * ffm2]

    # (the mode's own ringdown ansatz: the rescaled 22 one, or the
    # spheroidal one with mode mixing, both where the mode is one of several)
    plain = Struct(f_ring=fring, f_damp=fdamp, mixing_on=False)
    if is32 is not True:
        # the collocation points according to spin and mass ratio
        cond_a = eta < hm.eta_emr
        if diagonal is not False:
            cond_a = cond_a | both(diagonal, wf.s_tot_r >= 0.8)
        if is33 is not False:
            cond_a = cond_a | both(is33, wf.s_tot_r < 0.0)
        cond_b = jnp.asarray(both(is21, wf.s_tot_r >= 0.8))
        idx2 = jnp.where(cond_a, 3, 2)
        idx3 = jnp.where(cond_a, 4, jnp.where(cond_b, 4, 3))
        freqs_arr = jnp.stack(freqs)
        values_arr = jnp.stack(values)
        chosen = [(freqs[0], values[0]), (freqs[1], values[1]),
                  (freqs_arr[idx2], values_arr[idx2]), (freqs_arr[idx3], values_arr[idx3]),
                  (freqs[5], values[5])]
        # rescale the 22 ringdown: alpha0 + fRD^2 alpha2 / f^2 + alphaL fdamp / (fdamp^2 + (f - fRD)^2)
        wlm = where_mode(diagonal, 2.0, emm / 3.0)
        ph.alpha2 = 1.0 / (fring * fring) * wlm * hm_ringdown.phase_22_alpha2(wf)
        ph.alpha_l = eta_m1 * hm_ringdown.phase_22_alpha_l(wf)
        p_rd = powers(fcut_rd)
        phi0_rd = ringdown_phase_ansatz_int(fcut_rd, p_rd, plain, ph)
        dphi0_rd = ringdown_phase_ansatz(fcut_rd, p_rd, plain, ph)
        x = _solve([row(ff)[:5] for ff, _ in chosen], [v for _, v in chosen]) + [0.0]
    if is32 is not False:
        # derivatives of the rotated spheroidal ringdown phase at fcut_rd by
        # finite differences
        def rotated_phase(ff):
            re, im = spheroidal_to_spherical(ff, powers(ff), mix, mix.rd)
            return jnp.arctan2(im, re)

        phi0_mix, dphi0_mix, d2phi0_rd = phase_derivatives(fcut_rd, rotated_phase, "exact_derivatives" in wf)
        comparable = eta > hm.eta_emr
        freqs = list(freqs)
        values = list(values)
        # (six collocation points with mode mixing)
        n_inter = 6
        freqs[n_inter - 2] = jnp.where(comparable, fcut_rd, freqs[n_inter - 2])
        values[n_inter - 2] = jnp.where(comparable, dphi0_mix, values[n_inter - 2])
        freqs[n_inter - 1] = fcut_rd
        values[n_inter - 1] = d2phi0_rd
        rows = [row(freqs[i]) for i in range(6)]
        # the last point is the second derivative of the rotated ansatz
        ff = freqs[5]
        ffm1 = 1.0 / ff
        ffm2 = ffm1 * ffm1
        ffm3 = ffm2 * ffm1
        ffm4 = ffm2 * ffm2
        ffm5 = ffm3 * ffm2
        den = fdamp * fdamp + (ff - fring) * (ff - fring)
        rows[5] = [0.0, -2.0 * fdamp * (ff - fring) / (den * den), -ffm2, -2.0 * ffm3, -4.0 * ffm5, -3.0 * ffm4]
        x_mix = _solve(rows, values)
        if is32 is True:
            x, phi0_rd, dphi0_rd = x_mix, phi0_mix, dphi0_mix
        else:
            x = [jnp.where(is32, a, b) for a, b in zip(x_mix, x)]
            phi0_rd = jnp.where(is32, phi0_mix, phi0_rd)
            dphi0_rd = jnp.where(is32, dphi0_mix, dphi0_rd)

    ph.c0 = x[0]
    ph.c_l = x[1]
    ph.c1 = x[2]
    ph.c2 = x[3]
    ph.c4 = x[4]
    if is32 is not False:
        ph.c3 = where_mode(is32, x[5], 0.0)
        # for extreme mass ratios glue the intermediate and ringdown regions
        p_rd = powers(fcut_rd)
        glued = ph.c0 + dphi0_rd - intermediate_phase_ansatz(fcut_rd, p_rd, hm, ph)
        ph.c0 = jnp.where(both(is32, eta < hm.eta_emr), glued, ph.c0)

    # glue inspiral and intermediate
    p_in = powers(fcut_insp)
    ph.c1_insp = intermediate_phase_ansatz(fcut_insp, p_in, hm, ph) - inspiral.phase_ansatz(fcut_insp, p_in, ph)
    ph.c_insp = fma(-ph.c1_insp, fcut_insp, intermediate_phase_ansatz_int(fcut_insp, p_in, hm, ph)) - \
        inspiral.phase_ansatz_int(fcut_insp, p_in, ph)
    # glue intermediate and ringdown
    p_rd = powers(fcut_rd)
    ph.c1_rd = intermediate_phase_ansatz(fcut_rd, p_rd, hm, ph) - dphi0_rd
    ph.c_rd = fma(-ph.c1_rd, fcut_rd, intermediate_phase_ansatz_int(fcut_rd, p_rd, hm, ph)) - phi0_rd

    # align the mode with the 22 at low frequency (Eq. 4.13 of arXiv:2001.10914)
    falign = jnp.where(eta > hm.eta_emr, 0.6 * m_over_2 * wf.f_meco, m_over_2 * wf.f_meco)
    p_falign = powers(falign)
    phaseshift = 0.0
    if is32 is True:
        timeshift = mix.timeshift
        phiref22 = mix.phiref22
        (phase_22_falign,) = _phase_22_at(phase22, two_over_m * falign)
    else:
        timeshift = phase22_module.time_shift_22(phase22, wf)
        phase_22_mf_ref, phase_22_falign = _phase_22_at(phase22, wf.mf_ref, two_over_m * falign)
        phiref22 = _phi_ref_22(wf, phase_22_mf_ref, timeshift, phaseshift)
        if is32 is not False:
            timeshift = jnp.where(is32, mix.timeshift, timeshift)
            phiref22 = jnp.where(is32, mix.phiref22, phiref22)
    phi22 = fma(1.0 / eta, phase_22_falign, phaseshift) + phiref22
    phi_lm = fma(ph.c1_insp, falign, inspiral.phase_ansatz_int(falign, p_falign, ph)) + ph.c_insp
    deltaphi_lm = fma(-(3.0 * FRAC_PI_4), 1.0 - m_over_2, fma(m_over_2, phi22, timeshift * falign)) - phi_lm
    deltaphi_lm = jnp.fmod(deltaphi_lm, 2.0 * PI)
    if is21 is not False:
        # the 21 amplitude changes sign across the parameter space
        deltaphi_lm = deltaphi_lm + jnp.where(both(is21, pn21_amp_sign(0.008, wf)), PI, 0.0)
    ph.deltaphi_lm = deltaphi_lm
    ph.f_phase_match_in = f_phase_match_in
    ph.f_phase_match_im = f_phase_match_im
    ph.phi0_rd = phi0_rd
    ph.dphi0_rd = dphi0_rd
    ph.timeshift = timeshift
    ph.phiref22 = phiref22
    return ph


def phase(mf, p, ph, hm, rotation=None):
    """``IMRPhenomXHM_Phase_noModeMixing`` / ``_ModeMixing``: the phase of
    the mode (``rotation = (re, im)`` is the spherical (3,2) ringdown for the
    mode with mixing)."""
    ins = fma(ph.c1_insp, mf, inspiral.phase_ansatz_int(mf, p, ph)) + ph.c_insp + ph.deltaphi_lm
    if hm.mixing_on:
        phi_mrd = jnp.arctan2(rotation[1], rotation[0])
    else:
        phi_mrd = ringdown_phase_ansatz_int(mf, p, hm, ph)
    rd = fma(ph.c1_rd, mf, phi_mrd) + ph.c_rd + ph.deltaphi_lm
    int_ = intermediate_phase_ansatz_int(mf, p, hm, ph) + ph.deltaphi_lm
    return jnp.where(mf < ph.f_phase_match_in, ins, jnp.where(mf >= ph.f_phase_match_im, rd, int_))
