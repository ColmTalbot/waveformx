# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXHM as LALSimulation evaluates it, in ``Mf``: the reference of
the tables of :mod:`.core`.

One code path, LAL's defaults: ``PhenomXHMReleaseVersion = 122022`` (the
recalibrated amplitudes), the default versions of every region, no
multibanding, and the evaluation of ``IMRPhenomXHM_MultiMode2`` (the
polarizations), in which the ringdown of the (3,2) mode recycles the (2,2)
mode. The modes are those whose sum makes the polarizations. After
waveformrs's ``imrphenomx/xhm.rs``, ``hm_internals.rs`` and
``hm_{inspiral,intermediate,ringdown}.rs``.

Where LAL takes derivatives of the ansatzes by finite differences (the
amplitude at the boundaries of the intermediate region with a step of
``1e-9``, the rotated phase of the (3,2) mode with ``1e-7``, whose rounding
the second difference amplifies by ``1e14``) the derivatives here are exact,
as in waveformrs: the same in float32 and float64.

The waveform variables are those of :mod:`..xas.core` (masses in solar
masses, the QNM fits of the (2,2) in powers of ``1 - a_f``).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..xas import core as xas_core
from ..xpnr import phase22 as phase22_module
from ..xpnr._generated import hm_amp_fits_inspiral, hm_amp_fits_intermediate, hm_amp_fits_ringdown
from ..xpnr._generated import hm_qnm
from ..xpnr._generated import ringdown as ringdown22
from ..xpnr._math import FRAC_PI_4, PI, Struct, fma, lean, powers, real
from ..xpnr.amp22 import amp_coefficients as amp22_coefficients
from ..xpnr.hm.inspiral import phase_ansatz as inspiral_phase_ansatz
from ..xpnr.hm.inspiral import phase_ansatz_int as inspiral_phase_ansatz_int
from ..xpnr.hm.inspiral import pn_amplitude_coefficients
from ..xpnr.hm.phase import (
    get_fcut_insp, intermediate_phase_ansatz, intermediate_phase_ansatz_int, phase_fits, pn21_amp_sign,
    ringdown_phase_ansatz, ringdown_phase_ansatz_int, spheroidal_coefficients,
)
from ..xpnr.hm.variables import hm_waveform_struct
from ..xpnr.internals import waveform_struct
from ..xpnr.phase22 import phase_coefficients as phase22_coefficients

__all__ = ["HIGHER_MODES", "MODES", "setup", "mode_setup", "mode_amplitude_phase", "evaluate", "mode_factors"]

#: The higher modes ``(l, m)``, ``m > 0``, in the order of LAL's mode loop.
HIGHER_MODES = ((2, 1), (3, 3), (3, 2), (4, 4))
#: All modes, the (2,2) first.
MODES = ((2, 2),) + HIGHER_MODES
#: ``IMRPhenomXHMIntermediateAmpVersion``: per collocation point of the
#: intermediate amplitude, 0 skipped, 1 value, 2 value and derivative.
INTERMEDIATE_AMP_VERSION = {21: (1, 1, 0, 1, 0, 2), 33: (2, 1, 1, 1, 1, 2), 32: (2, 1, 1, 1, 1, 2),
                            44: (2, 1, 1, 1, 1, 2)}
#: ``FALSE_ZERO``: what replaces a negative amplitude.
FALSE_ZERO = 1.0e-15


def _solve(rows, values):
    matrix = jnp.stack([jnp.stack([jnp.asarray(x, real()) for x in row]) for row in rows])
    rhs = jnp.stack([jnp.asarray(v, real()) for v in values])
    solution = jnp.linalg.solve(matrix, rhs)
    return [solution[i] for i in range(len(values))]


def _derivative(function, x):
    """``function(x)`` and its derivative (exact)."""
    return jax.jvp(function, (x,), (jnp.ones_like(x),))


# ---------------------------------------------------------------------------
# the waveform variables
# ---------------------------------------------------------------------------


def waveform_variables(params, f_min, amp_scale):
    """``IMRPhenomXSetWaveformVariables`` as :func:`..xas.core.table` makes
    it (``params`` as there; ``f_max = 0``: the default cutoff)."""
    m1, m2, chi1, chi2, distance, inclination, phi_ref, f_ref = params
    wf = waveform_struct(m1, m2, chi1, chi2, 0.0, f_ref, phi_ref, f_min, 0.0, distance, inclination, amp_scale)
    f_ring, f_damp = xas_core.qnm_22(wf.a_final)
    wf.f_ring = f_ring / wf.m_final
    wf.f_damp = f_damp / wf.m_final
    wf.hm_release = 122022
    return wf


#: The QNM fits of the higher modes and the mixing coefficients in powers of
#: ``1 - a_f`` (:func:`..xas.core.about_one`): in powers of ``a_f`` they
#: lose up to a part in 100 near ``a_f = 1`` in float32.
QNM = {name: xas_core.about_one(getattr(hm_qnm, name)) for name in dir(hm_qnm)
       if name.startswith(("fring", "fdamp", "re_l", "im_l"))}


def mode_struct(wf, ell, emm):
    """``IMRPhenomXHM_SetHMWaveformVariables`` for the 122022 release (the
    QNM fits of :data:`QNM`)."""
    hm = hm_waveform_struct(wf, ell, emm)
    tag = hm.mode_tag
    hm.f_ring = QNM[f"fring{tag}"](wf.a_final) / wf.m_final
    hm.f_damp = QNM[f"fdamp{tag}"](wf.a_final) / wf.m_final
    hm.intermediate_amp_version = INTERMEDIATE_AMP_VERSION[tag]
    hm.ringdown_amp_version = 1 if tag == 32 else 2
    hm.f_amp_rd_falloff = fma(2.0, hm.f_damp, hm.f_ring)
    if hm.mixing_on:
        # (in the working precision: LAL stores them as COMPLEX8, whose
        # rounding would make the waveform a staircase in the parameters;
        # as waveformrs, which puts it at ~1e-7 in the (3,2) ringdown)
        af = wf.a_final
        hm.mixing_coeffs = (
            (QNM["re_l2m2lp2"](af), QNM["im_l2m2lp2"](af)), (QNM["re_l2m2lp3"](af), QNM["im_l2m2lp3"](af)),
            (-QNM["re_l3m2lp2"](af), -QNM["im_l3m2lp2"](af)), (-QNM["re_l3m2lp3"](af), -QNM["im_l3m2lp3"](af)),
        )
    return hm


# ---------------------------------------------------------------------------
# amplitude (122022)
# ---------------------------------------------------------------------------


def amplitude_fcut_insp(hm, wf):
    """``IMRPhenomXHM_Amplitude_fcutInsp`` (version 122022)."""
    emm = float(hm.emm)
    eta = wf.eta
    chi1 = wf.chi1l
    fcut_emr = 1.25 * emm * (
        fma(0.000316707064291237, chi1 * chi1, fma(-0.0000858396080377194, chi1, 0.011671068725758493))
        * fma(6.2873167352395125, eta, 0.8447212540381764)
    ) / fma(-0.9977728883419751, chi1, 1.2857082764038923)
    transition_eta = 0.0192234  # q = 50
    sharpness = 0.004
    funcs = fma(0.5, jnp.tanh((eta - transition_eta) / sharpness), 0.5)
    return jnp.where(wf.q < 20.0, hm.f_meco_lm, fma(funcs, hm.f_meco_lm, (1.0 - funcs) * fcut_emr))


def amplitude_fcut_rd(hm, wf):
    """``IMRPhenomXHM_Amplitude_fcutRD`` (version 122022)."""
    if hm.mixing_on:
        return fma(-0.5, wf.f_damp, wf.f_ring)
    return hm.f_ring - hm.f_damp


def pn_amp_ansatz(p, amp):
    """``IMRPhenomXHM_Inspiral_PNAmp_Ansatz``: the modulus of the PN series
    times ``pnGlobalFactor ampNorm f^(-7/6)``."""
    terms = [p.one_third, p.two_thirds, p.itself, p.four_thirds, p.five_thirds, p.two]
    re = amp.pn.pn_re[0]
    im = amp.pn.pn_im[0]
    for k in range(6):
        re = re + terms[k] * amp.pn.pn_re[k + 1]
        im = im + terms[k] * amp.pn.pn_im[k + 1]
    return jnp.hypot(re, im) * (amp.pn.pn_global_factor * p.m_seven_sixths * amp.amp_norm)


def inspiral_amp_ansatz(p, amp):
    """``IMRPhenomXHM_Inspiral_Amp_Ansatz`` (122022): PN plus pseudo-PN
    terms."""
    c = amp.inspiral_coefficients
    pseudo = fma(p.seven_thirds / amp.fcut_insp_seven_thirds, c[0], p.eight_thirds / amp.fcut_insp_eight_thirds * c[1])
    pseudo = fma(p.three / amp.fcut_insp_three, c[2], pseudo)
    return pn_amp_ansatz(p, amp) + pseudo * p.m_seven_sixths * amp.pn_dominant


def inspiral_amp_coefficients(amp, hm, wf):
    """``IMRPhenomXHM_Get_Inspiral_Amp_Coefficients`` (122022, three
    collocation points)."""
    fcut = amp.f_amp_match_in
    freqs = [0.5 * fcut, 0.75 * fcut, fcut]
    values = [jnp.abs(hm_amp_fits_inspiral.amp_fit(hm.mode_int, i, wf)) for i in range(3)]
    f1, f2, f3 = (powers(f) for f in freqs)
    finsp = powers(fcut)
    amp.fcut_insp_seven_thirds = finsp.seven_thirds
    amp.fcut_insp_eight_thirds = finsp.eight_thirds
    amp.fcut_insp_three = finsp.three
    pn = [pn_amp_ansatz(f, amp) for f in (f1, f2, f3)]
    v = [(values[i] - pn[i]) / amp.pn_dominant / f.m_seven_sixths for i, f in enumerate((f1, f2, f3))]
    # The pseudo-PN terms are c_j (f / fcut)^p_j (p = 7/3, 8/3, 3) through the
    # collocation points 0.5, 0.75 and 1 times fcut: a constant system, whose
    # inverse (float64) gives LAL's closed form without its products of small
    # powers of Mf (which underflow in the derivatives in float32).
    c0, c1, c2 = (sum(float(_INSPIRAL_INVERSE[j, i]) * v[i] for i in range(3)) for j in range(3))
    amp.inspiral_coefficients = [c0, c1, c2]


_INSPIRAL_INVERSE = np.linalg.inv(np.array([[x ** p for p in (7.0 / 3.0, 8.0 / 3.0, 3.0)] for x in (0.5, 0.75, 1.0)]))


def _lorentzian(mf, hm, rd):
    fda = hm.f_damp
    dfr = mf - hm.f_ring
    dfd = fda * rd[2]
    return rd[0] * fda / (jnp.exp(rd[1] / dfd * dfr) * fma(dfr, dfr, dfd * dfd))


def rd_amp_dansatz(mf, hm, rd):
    """``IMRPhenomXHM_RD_Amp_DAnsatz`` (versions 1, 2): the derivative of the
    Lorentzian."""
    fda = hm.f_damp
    dfr = mf - hm.f_ring
    numerator = rd[0] * fma(fda * fda * rd[1] * rd[2], rd[2], fma(dfr * dfr, rd[1], 2.0 * fda * dfr * rd[2]))
    denom = fma(dfr, dfr, fda * fda * rd[2] * rd[2])
    denom = rd[2] * denom * denom * jnp.exp(dfr * rd[1] / (fda * rd[2]))
    return -numerator / denom


def rd_amp_ansatz(mf, hm, amp):
    """``IMRPhenomXHM_RD_Amp_Ansatz`` (versions 1, 2): the Lorentzian, the
    cubic of the auxiliary region of the (3,2) below ``fRDAux`` and the
    exponential fall-off above ``fAmpRDfalloff`` (the spheroidal amplitude
    for the (3,2))."""
    rd = amp.rd
    out = jnp.where(mf >= hm.f_amp_rd_falloff, rd[3] * jnp.exp(-rd[4] * (mf - hm.f_amp_rd_falloff)),
                    _lorentzian(mf, hm, rd))
    if hm.mixing_on:
        a = amp.rd_aux
        aux = fma(fma(fma(a[3], mf, a[2]), mf, a[1]), mf, a[0])
        out = jnp.where(mf < amp.f_rd_aux, aux, out)
    return out


def rd_amp_coefficients(amp, hm, wf):
    """``IMRPhenomXHM_RD_Amp_Coefficients`` with the exponential fall-off and,
    for the (3,2), ``IMRPhenomXHM_RDAux_Amp_Coefficients``."""
    fdamp = hm.f_damp
    if hm.ringdown_amp_version == 1:
        rd = [jnp.abs(hm_amp_fits_ringdown.amp_32_alambda(wf)), hm_amp_fits_ringdown.amp_32_lambda(wf),
              hm_amp_fits_ringdown.amp_32_sigma(wf)]
    else:
        tag = hm.mode_tag
        rdcp1, rdcp2, rdcp3 = (jnp.abs(getattr(hm_amp_fits_ringdown, f"amp_{tag}_rdcp{i}")(wf)) for i in (1, 2, 3))
        # vetos
        rdcp3 = jnp.where(rdcp3 >= rdcp2 * rdcp2 / rdcp1, 0.5 * rdcp2 * rdcp2 / rdcp1, rdcp3)
        rdcp3 = jnp.where(rdcp3 > rdcp2, 0.5 * rdcp2, rdcp3)
        rdcp3 = jnp.where((rdcp1 < rdcp2) & (rdcp3 > rdcp1), rdcp1, rdcp3)
        deno = jnp.sqrt(rdcp1 / rdcp3) - rdcp1 / rdcp2
        deno = jnp.where(deno <= 0.0, 1e-16, deno)
        a_lambda = rdcp1 * fdamp / deno
        sigma = jnp.sqrt(a_lambda / (rdcp2 * fdamp))
        rd = [a_lambda, 0.5 * sigma * jnp.log(rdcp1 / rdcp3), sigma]
    falloff = hm.f_amp_rd_falloff
    value = _lorentzian(falloff, hm, rd)
    rd = rd + [value, -1.0 * rd_amp_dansatz(falloff, hm, rd) / value]
    amp.rd = rd
    if hm.mixing_on:
        values = [jnp.abs(hm_amp_fits_ringdown.amp_32_rdaux1(wf)), jnp.abs(hm_amp_fits_ringdown.amp_32_rdaux2(wf)),
                  _lorentzian(amp.f_rd_aux, hm, rd), rd_amp_dansatz(amp.f_rd_aux, hm, rd)]
        freqs = [amp.f_amp_match_im, 0.5 * (amp.f_amp_match_im + amp.f_rd_aux), amp.f_rd_aux]
        rows = [[1.0, f, f * f, f * f * f] for f in freqs]
        f = amp.f_rd_aux
        rows.append([0.0, 1.0, 2.0 * f, 3.0 * f * f])
        amp.rd_aux = _solve(rows, values)


def intermediate_amp_coefficients(amp, hm, wf, rotated):
    """``IMRPhenomXHM_Intermediate_Amp_CollocationPoints`` and
    ``_Coefficients`` (122022): a polynomial times ``f^(-7/6)`` through
    equispaced collocation points, with the values and derivatives of the
    inspiral and ringdown at the boundaries. ``rotated(mf)`` is the
    spherical (3,2) ringdown ``(re, im)`` (not recycling the (2,2)).

    The polynomial (of up to eight coefficients) is in ``u = (f - f_c) /
    f_h``, which maps the region to ``[-1, 1]``, rather than LAL's ``f``:
    the same polynomial, whose collocation system and evaluation are then
    well conditioned (in powers of ``f`` float32 loses a few per cent of
    the amplitude of the (3,2))."""
    version = hm.intermediate_amp_version
    n_colloc = len(version)
    deltaf = (amp.f_amp_match_im - amp.f_amp_match_in) / (n_colloc - 1)
    rows, values = [], []
    seven_sixths = 7.0 / 6.0
    n_coefficients = sum(version)

    center = 0.5 * (amp.f_amp_match_im + amp.f_amp_match_in)
    half = 0.5 * (amp.f_amp_match_im - amp.f_amp_match_in)
    amp.inter_center = center
    amp.inter_half = half

    def value_row(f):
        m76 = jnp.power(f, -seven_sixths)
        u = (f - center) / half
        return [m76 * u ** j for j in range(n_coefficients)]

    def derivative_row(f):
        m76 = jnp.power(f, -seven_sixths)
        u = (f - center) / half
        return [m76 * (-seven_sixths / f * u ** j + (j * u ** (j - 1) / half if j else 0.0))
                for j in range(n_coefficients)]

    f_in = amp.f_amp_match_in
    ins_value, ins_derivative = _derivative(lambda f: inspiral_amp_ansatz(powers(f), amp), f_in)
    f_rd = amp.f_amp_match_im
    if hm.mixing_on:
        rd_value, rd_derivative = _derivative(lambda f: jnp.hypot(*rotated(f)), f_rd)
    else:
        rd_value = rd_amp_ansatz(f_rd, hm, amp)
        rd_derivative = rd_amp_dansatz(f_rd, hm, amp.rd)
    for i, digit in enumerate(version):
        f = fma(deltaf, float(i), f_in)
        if digit == 0:
            continue
        if i == 0:
            rows.append(value_row(f))
            values.append(ins_value)
            if digit == 2:
                rows.append(derivative_row(f))
                values.append(ins_derivative)
        elif i == n_colloc - 1:
            rows.append(value_row(f))
            values.append(rd_value)
            if digit == 2:
                rows.append(derivative_row(f))
                values.append(rd_derivative)
        else:
            rows.append(value_row(f))
            values.append(jnp.abs(hm_amp_fits_intermediate.amp_fit(hm.mode_int, i - 1, wf)))
    amp.inter = _solve(rows, values)


def inter_amp_ansatz(mf, p, amp):
    """``IMRPhenomXHM_Intermediate_Amp_Ansatz`` (122022), in ``u``
    (:func:`intermediate_amp_coefficients`)."""
    u = (mf - amp.inter_center) / amp.inter_half
    result = amp.inter[-1]
    for c in amp.inter[-2::-1]:
        result = fma(result, u, c)
    return result * p.m_seven_sixths


def amplitude_coefficients(wf, hm, rotated):
    """``IMRPhenomXHM_GetAmplitudeCoefficients`` (122022)."""
    amp = Struct(amp_norm=wf.amp_norm, pn_dominant=wf.amp_norm * (2.0 / hm.emm) ** (-7.0 / 6.0))
    if hm.mixing_on:
        amp.f_rd_aux = hm.f_ring - hm.f_damp
    amp.f_amp_match_in = amplitude_fcut_insp(hm, wf)
    amp.f_amp_match_im = amplitude_fcut_rd(hm, wf)
    amp.pn = pn_amplitude_coefficients(wf, hm)
    inspiral_amp_coefficients(amp, hm, wf)
    rd_amp_coefficients(amp, hm, wf)
    intermediate_amp_coefficients(amp, hm, wf, lambda f: rotated(f, amp))
    return amp


# ---------------------------------------------------------------------------
# the (3,2) ringdown
# ---------------------------------------------------------------------------


def rotate(mf, p, mix, amp, wf22r):
    """``conj(a322) h22 + conj(a323) h32_spheroidal`` (``(re, im)``) for the
    (2,2) ringdown ``wf22r`` (``(re, im)``, without ``amp0``)."""
    amplm = rd_amp_ansatz(mf, mix.hm, amp)
    philm = ringdown_phase_ansatz_int(mf, p, mix.hm, mix.spheroidal)
    (mix2_re, mix2_im), (mix3_re, mix3_im) = mix.hm.mixing_coeffs[2], mix.hm.mixing_coeffs[3]
    a_re = wf22r[0] * mix2_re + wf22r[1] * mix2_im
    a_im = wf22r[1] * mix2_re - wf22r[0] * mix2_im
    b_re = amplm * mix3_re
    b_im = -(mix3_im * amplm)
    c, s = jnp.cos(philm), jnp.sin(philm)
    return a_re + (b_re * c - b_im * s), a_im + (b_re * s + b_im * c)


def spheroidal_to_spherical(mf, mix, amp):
    """``SpheroidalToSpherical`` (122022): the rotation with the (2,2)
    ringdown ansatz, rescaled by its leading order."""
    p = powers(mf)
    amp22 = ringdown22.amp_22_ansatz_exact(mf, mix.f_ring22, mix.f_damp22, mix.amp22) * mix.amp_norm * p.m_seven_sixths
    phi22 = fma(mix.inveta, phase22_module.phase_22(mf, p, mix.phase22), mix.timeshift * mf) + mix.phiref22
    return rotate(mf, p, mix, amp, (amp22 * jnp.cos(phi22), amp22 * jnp.sin(phi22)))


# ---------------------------------------------------------------------------
# phase
# ---------------------------------------------------------------------------


def _phi_ref_22(wf, phase_22_at_mf_ref, timeshift):
    return fma(-1.0 / wf.eta, phase_22_at_mf_ref, -(timeshift * wf.mf_ref)) + 2.0 * wf.phi0 + FRAC_PI_4


def phase_coefficients(wf, hm, phase22, mix, amp, rotated=None):
    """``IMRPhenomXHM_GetPhaseCoefficients`` (shared by the releases), for
    the 122022 amplitude of the (3,2) and exact derivatives of its rotated
    ringdown phase; ``rotated(mf)`` (default :func:`spheroidal_to_spherical`)
    is that rotation, ``(re, im)``."""
    if rotated is None:
        def rotated(ff):
            return spheroidal_to_spherical(ff, mix, amp)

    tag = hm.mode_tag
    emm = hm.emm
    eta = wf.eta
    eta_m1 = 1.0 / eta
    fring = hm.f_ring
    fdamp = hm.f_damp
    fcut = get_fcut_insp(wf, hm)
    sqrt3 = math.sqrt(3.0)
    if tag == 32:
        f_end = fma(-0.5, wf.f_damp, wf.f_ring)
        freqs = [fcut, fma(sqrt3, fcut - f_end, 2.0 * (fcut + f_end)) / 4.0, fma(3.0, fcut, f_end) / 4.0,
                 (fcut + f_end) / 2.0, f_end, f_end]
        f_phase_match_im = jnp.where((eta < 0.01) & (wf.chi1l < 0.0), f_end * fma(-0.25, wf.chi1l, 1.2), f_end)
    else:
        freqs = [fcut, fma(sqrt3, fcut - fring, 2.0 * (fcut + fring)) / 4.0, fma(3.0, fcut, fring) / 4.0,
                 (fcut + fring) / 2.0, fma(3.0, fring, fcut) / 4.0, fma(7.0, fring, fcut) / 8.0]
        f_phase_match_im = fring - fdamp
    f_phase_match_in = hm.f_meco_lm
    fits = phase_fits(wf, hm)
    values = [value + hm.delta_t for value in fits.phase_values]
    fcut_rd = f_phase_match_im
    fcut_insp = f_phase_match_in

    # inspiral: the 22 coefficients rescaled, phi_lm(f) ~ m/2 phi_22(2 f / m)
    zero = 0.0
    non_log = [phase22.phi0, phase22.phi1, phase22.phi2, phase22.phi3, phase22.phi4, phase22.phi5,
               phase22.phi6, phase22.phi7, phase22.phi8, phase22.phi9, zero, zero, zero]
    log_terms = [zero, zero, zero, zero, zero, phase22.phi5l, phase22.phi6l, zero, phase22.phi8l,
                 phase22.phi9l, zero, zero, zero]
    pseudo_pn = [zero] * 8 + [phase22.sigma1, phase22.sigma2, phase22.sigma3, phase22.sigma4, phase22.sigma5]
    m_over_2 = emm * 0.5
    two_over_m = 1.0 / m_over_2
    fact = phase22.phi_norm / eta
    log_m_over_2 = math.log(m_over_2)
    phi, phi_l = [], []
    for i in range(13):
        nl = non_log[i] * fact
        lg = log_terms[i] * fact
        ps = pseudo_pn[i] * fact
        scale = m_over_2 ** ((8 - i) / 3.0)
        phi.append(fma(-lg, log_m_over_2, nl + ps) * scale)
        phi_l.append(lg * scale)
    phi[8] = phi[8] + fits.lambda_pn
    ph = Struct(phi=phi, phi_l=phi_l, alpha0=0.0, alpha2=0.0, alpha_l=0.0, c0=0.0, c1=0.0, c2=0.0, c3=0.0,
                c4=0.0, c_l=0.0)

    # for high spins the 21 inspiral extends into the intermediate region
    if tag == 21:
        unconnected = phase22.replace(c1_int=0.0, c2_int=0.0, c1_mrd=0.0, c2_mrd=0.0)
        ff = two_over_m * jnp.stack(freqs[:3])
        insp_vals = 1.0 / eta * phase22_module.dphase_22(ff, powers(ff), unconnected)
        diff12 = insp_vals[0] - insp_vals[1]
        diff23 = insp_vals[1] - insp_vals[2]
        high = wf.s_tot_r >= 0.8
        v1 = values[2] + diff23
        values[1] = jnp.where(high, v1, values[1])
        values[0] = jnp.where(high, v1 + diff12, values[0])

    def row(ff):
        ffm1 = 1.0 / ff
        ffm2 = ffm1 * ffm1
        lorentzian = fdamp / (fdamp * fdamp + (ff - fring) * (ff - fring))
        return [1.0, lorentzian, ffm1, ffm2, ffm2 * ffm2, ffm1 * ffm2]

    if tag != 32:
        cond_a = eta < hm.eta_emr
        if emm == hm.ell:
            cond_a = cond_a | (wf.s_tot_r >= 0.8)
        if tag == 33:
            cond_a = cond_a | (wf.s_tot_r < 0.0)
        cond_b = (wf.s_tot_r >= 0.8) if tag == 21 else False
        idx2 = jnp.where(cond_a, 3, 2)
        idx3 = jnp.where(cond_a, 4, jnp.where(cond_b, 4, 3))
        freqs_arr = jnp.stack(freqs)
        values_arr = jnp.stack(values)
        chosen = [(freqs[0], values[0]), (freqs[1], values[1]), (freqs_arr[idx2], values_arr[idx2]),
                  (freqs_arr[idx3], values_arr[idx3]), (freqs[5], values[5])]
        wlm = 2.0 if emm == hm.ell else emm / 3.0
        ph.alpha2 = 1.0 / (fring * fring) * wlm * mix.alpha2_fit
        ph.alpha_l = eta_m1 * mix.alpha_l_fit
        p_rd = powers(fcut_rd)
        phi0_rd = ringdown_phase_ansatz_int(fcut_rd, p_rd, hm, ph)
        dphi0_rd = ringdown_phase_ansatz(fcut_rd, p_rd, hm, ph)
        x = _solve([row(ff)[:5] for ff, _ in chosen], [v for _, v in chosen]) + [0.0]
    else:
        def rotated_phase(ff):
            re, im = rotated(ff)
            return jnp.arctan2(im, re)

        def first(ff):
            return _derivative(rotated_phase, ff)[1]

        phi0_rd, dphi0_rd = _derivative(rotated_phase, fcut_rd)
        d2phi0_rd = _derivative(first, fcut_rd)[1]
        comparable = eta > hm.eta_emr
        freqs = list(freqs)
        values = list(values)
        freqs[4] = jnp.where(comparable, fcut_rd, freqs[4])
        values[4] = jnp.where(comparable, dphi0_rd, values[4])
        freqs[5] = fcut_rd
        values[5] = d2phi0_rd
        # The intermediate phase derivative of the (3,2) is a quartic in g =
        # 1/f and a Lorentzian. LAL solves for its coefficients in powers of
        # g, nearly collinear over the region (float32 loses a part in 10^3
        # of the coefficients, 10^-2 rad of the phase); here the quartic is
        # in v = (g - g_c) / g_h (g_h the half range of g), converted to
        # LAL's coefficients afterwards: the same polynomial.
        g_c = 0.5 * (1.0 / fcut_insp + 1.0 / fcut_rd)
        g_h = 0.5 * (1.0 / fcut_insp - 1.0 / fcut_rd)

        def lorentzian(ff):
            return fdamp / (fdamp * fdamp + (ff - fring) * (ff - fring))

        rows = []
        for i in range(5):
            v = (1.0 / freqs[i] - g_c) / g_h
            rows.append([1.0, lorentzian(freqs[i]), v, v * v, v * v * v, v * v * v * v])
        # the last point is the second derivative of the rotated ansatz
        ff = freqs[5]
        v = (1.0 / ff - g_c) / g_h
        dv = -1.0 / (ff * ff * g_h)
        den = fdamp * fdamp + (ff - fring) * (ff - fring)
        rows.append([0.0, -2.0 * fdamp * (ff - fring) / (den * den), dv, 2.0 * v * dv, 3.0 * v * v * dv,
                     4.0 * v * v * v * dv])
        b = _solve(rows, values)
        quartic = [b[0], b[2], b[3], b[4], b[5]]
        # sum_k b_k ((g - g_c) / g_h)^k = sum_j p_j g^j
        p = [0.0] * 5
        for k in range(5):
            scale = quartic[k] / g_h ** k
            for j in range(k + 1):
                p[j] = p[j] + scale * math.comb(k, j) * (-g_c) ** (k - j)
        x = [p[0], b[1], p[1], p[2], p[4], p[3]]
    ph.c0, ph.c_l, ph.c1, ph.c2, ph.c4 = x[0], x[1], x[2], x[3], x[4]
    if tag == 32:
        ph.c3 = x[5]
        p_rd = powers(fcut_rd)
        glued = ph.c0 + dphi0_rd - intermediate_phase_ansatz(fcut_rd, p_rd, hm, ph)
        ph.c0 = jnp.where(eta < hm.eta_emr, glued, ph.c0)

    p_in = powers(fcut_insp)
    ph.c1_insp = intermediate_phase_ansatz(fcut_insp, p_in, hm, ph) - inspiral_phase_ansatz(fcut_insp, p_in, ph)
    ph.c_insp = fma(-ph.c1_insp, fcut_insp, intermediate_phase_ansatz_int(fcut_insp, p_in, hm, ph)) - \
        inspiral_phase_ansatz_int(fcut_insp, p_in, ph)
    p_rd = powers(fcut_rd)
    ph.c1_rd = intermediate_phase_ansatz(fcut_rd, p_rd, hm, ph) - dphi0_rd
    ph.c_rd = fma(-ph.c1_rd, fcut_rd, intermediate_phase_ansatz_int(fcut_rd, p_rd, hm, ph)) - phi0_rd

    # align the mode with the 22 at low frequency (Eq. 4.13 of arXiv:2001.10914)
    falign = jnp.where(eta > hm.eta_emr, 0.6 * m_over_2 * wf.f_meco, m_over_2 * wf.f_meco)
    p_falign = powers(falign)
    x22 = two_over_m * falign
    phase_22_falign = phase22_module.phase_22(x22, powers(x22), phase22)
    phi22 = phase_22_falign / eta + mix.phiref22
    phi_lm = fma(ph.c1_insp, falign, inspiral_phase_ansatz_int(falign, p_falign, ph)) + ph.c_insp
    deltaphi_lm = fma(-(3.0 * FRAC_PI_4), 1.0 - m_over_2, fma(m_over_2, phi22, mix.timeshift * falign)) - phi_lm
    if tag == 21:
        deltaphi_lm = deltaphi_lm + jnp.where(pn21_amp_sign(0.008, wf), PI, 0.0)
    ph.deltaphi_lm = deltaphi_lm
    ph.f_phase_match_in = f_phase_match_in
    ph.f_phase_match_im = f_phase_match_im
    return ph


# ---------------------------------------------------------------------------
# set-up and evaluation
# ---------------------------------------------------------------------------


def mode_setup(wf, phase22, ell, emm):
    """The coefficients of the higher mode ``(ell, emm)``: ``hm`` (its
    variables), ``amp``, ``phase`` and, for the (3,2), ``mix`` (the data of
    the rotation)."""
    from ..xpnr._generated import hm_ringdown

    hm = mode_struct(wf, ell, emm)
    timeshift = phase22_module.time_shift_22(phase22, wf)
    mf_ref = wf.mf_ref
    phiref22 = _phi_ref_22(wf, phase22_module.phase_22(mf_ref, powers(mf_ref), phase22), timeshift)
    mix = Struct(timeshift=timeshift, phiref22=phiref22, alpha2_fit=hm_ringdown.phase_22_alpha2(wf),
                 alpha_l_fit=hm_ringdown.phase_22_alpha_l(wf))
    if hm.mixing_on:
        spheroidal, timeshift32, phiref22_32 = spheroidal_coefficients(wf, hm, phase22)
        view = Struct(f_ring=hm.f_ring, f_damp=hm.f_damp, mixing_on=True, mode_tag=32, ell=3, emm=2,
                      f_amp_rd_falloff=hm.f_amp_rd_falloff, mixing_coeffs=hm.mixing_coeffs)
        mix = mix.replace(spheroidal=spheroidal, hm=view, amp22=amp22_coefficients(wf), phase22=phase22,
                          f_ring22=wf.f_ring, f_damp22=wf.f_damp, inveta=1.0 / wf.eta, amp_norm=wf.amp_norm,
                          timeshift=timeshift32, phiref22=phiref22_32)
    amp = amplitude_coefficients(wf, hm, lambda f, a: spheroidal_to_spherical(f, mix, a))
    ph = phase_coefficients(wf, hm, phase22, mix, amp)
    return Struct(hm=hm, amp=amp, phase=ph, mix=mix, amp0=(-1.0 if ell % 2 else 1.0) * wf.amp0)


def setup(params, f_min, amp_scale=1.0):
    """The waveform variables, the (2,2) coefficients (``xas``, ``phif_ref``)
    and the coefficients of the higher modes."""
    with lean(True):
        wf = waveform_variables(params, f_min, amp_scale)
        phase22 = phase22_coefficients(wf)
        from ..xpnr import xas

        x = xas.xas_setup(wf)
        modes = {mode: mode_setup(wf, phase22, *mode) for mode in HIGHER_MODES}
        return Struct(wf=wf, xas=x, phif_ref=xas.phif_ref(wf, x), modes=modes)


def mode_amplitude_phase(mf, c, h22=None):
    """Amplitude (without ``(-1)^l amp0``) and phase of the higher mode of
    ``c`` (:func:`mode_setup`) at ``mf``; ``h22`` the (2,2) mode (strain,
    ``(re, im)``) recycled in the ringdown of the (3,2)."""
    hm, amp, ph = c.hm, c.amp, c.phase
    p = powers(mf)
    if hm.mixing_on:
        if h22 is None:
            rotation = spheroidal_to_spherical(mf, c.mix, amp)
        else:
            amp0 = jnp.abs(c.amp0)
            rotation = rotate(mf, p, c.mix, amp, (h22[0] / amp0, h22[1] / amp0))
    ins = inspiral_amp_ansatz(p, amp)
    inter = inter_amp_ansatz(mf, p, amp)
    rd = jnp.hypot(*rotation) if hm.mixing_on else rd_amp_ansatz(mf, hm, amp)
    a = jnp.where(mf < amp.f_amp_match_in, ins, jnp.where(mf >= amp.f_amp_match_im, rd, inter))
    a = jnp.where(a < 0.0, FALSE_ZERO, a)
    phi_ins = fma(ph.c1_insp, mf, inspiral_phase_ansatz_int(mf, p, ph)) + ph.c_insp
    phi_mrd = jnp.arctan2(rotation[1], rotation[0]) if hm.mixing_on else ringdown_phase_ansatz_int(mf, p, hm, ph)
    phi_rd = fma(ph.c1_rd, mf, phi_mrd) + ph.c_rd
    phi_int = intermediate_phase_ansatz_int(mf, p, hm, ph)
    phi = jnp.where(mf < ph.f_phase_match_in, phi_ins, jnp.where(mf >= ph.f_phase_match_im, phi_rd, phi_int))
    return a, phi + ph.deltaphi_lm


def mode_factors(inclination, ell, emm):
    """The factors ``(F+, Fx)`` of ``h_{l,-m}`` in the polarizations with
    both signs of ``m`` (``IMRPhenomXHMFDAddMode``)."""
    from ..._harmonics import spin_weighted_spherical_harmonic

    sign = -1.0 if ell % 2 else 1.0
    ym = spin_weighted_spherical_harmonic(inclination, PI / 2.0, ell, -emm)
    ystar = jnp.conj(spin_weighted_spherical_harmonic(inclination, PI / 2.0, ell, emm))
    return 0.5 * (ym + sign * ystar), 0.5j * (ym - sign * ystar)


def evaluate(s, frequencies, inclination):
    """The modes ``{(l, m): h_{l,-m}(f)}`` and the polarizations at the
    frequencies (Hz) for the set-up ``s``: the (2,2) zero above the cutoff,
    the higher modes at every frequency (the (3,2) recycling the (2,2))."""
    from ..xpnr import xas

    wf = s.wf
    mf = wf.m_sec * jnp.asarray(frequencies, real())
    h22 = xas.h22(mf, wf, s.xas, s.phif_ref)
    modes = {(2, 2): h22}
    for mode, c in s.modes.items():
        a, phi = mode_amplitude_phase(mf, c, (h22.real, h22.imag))
        r = c.amp0 * a
        h = jax.lax.complex(r * jnp.cos(phi), r * jnp.sin(phi))
        modes[mode] = jnp.where(c.hm.amp_zero, jnp.zeros_like(h), h)
    hp = 0.0
    hc = 0.0
    for (ell, emm), h in modes.items():
        fp, fc = mode_factors(inclination, ell, emm)
        hp = hp + fp * h
        hc = hc + fc * h
    return modes, (hp, hc)
