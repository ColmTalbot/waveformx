# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The amplitude of the (2,2) mode: ``IMRPhenomXGetAmplitudeCoefficients``
and ``IMRPhenomX_Amplitude_22`` of ``LALSimIMRPhenomX_internals.c`` for the
default versions (InsAmpVersion 103, IntAmpVersion 104, RDAmpVersion 103),
after waveformrs's ``imrphenomx/amplitude.rs``."""

from __future__ import annotations

import jax.numpy as jnp

from ._generated import inspiral as _inspiral
from ._generated import intermediate as _intermediate
from ._generated import ringdown as _ringdown
from ._math import FencedModule, PI, POWERS_OF_LALPI, Struct, fenced, fma, powers

inspiral = FencedModule(_inspiral)
intermediate = FencedModule(_intermediate)
ringdown = FencedModule(_ringdown)

__all__ = ["amp_coefficients", "amp_ansatz", "amplitude_22"]


@fenced
def amp_coefficients(wf):
    """``IMRPhenomXGetAmplitudeCoefficients`` (with the co-precessing
    deviations of IMRPhenomXPNR in ``wf.coprec``)."""
    pi = POWERS_OF_LALPI
    chi1l = wf.chi1l
    chi2l = wf.chi2l
    chi1l2 = wf.chi1l2
    chi1l3 = wf.chi1l3
    chi2l2 = wf.chi2l2
    delta = wf.delta
    eta = wf.eta
    eta2 = wf.eta2
    eta3 = wf.eta3
    f_ring = wf.f_ring
    f_damp = wf.f_damp

    gamma2 = ringdown.amp_22_gamma2(eta, wf.s_tot_r, wf.dchi, delta)
    gamma3 = ringdown.amp_22_gamma3(eta, wf.s_tot_r, wf.dchi, delta)
    gamma3 = fma(wf.coprec.dev, wf.coprec.mu2, gamma3)
    f_amp_rd_min = ringdown.amp_22_peak_frequency(gamma2, gamma3, f_ring, f_damp)
    v1_rd = ringdown.amp_22_v1(eta, wf.s_tot_r, wf.dchi, delta)
    v1_rd = fma(wf.coprec.dev, wf.coprec.mu1, v1_rd)
    f1 = f_amp_rd_min
    gamma1 = (
        (v1_rd / (f_damp * gamma3))
        * fma(f_damp * f_damp * gamma3, gamma3, fma(f_ring, f_ring, fma(f1, f1, -(2.0 * f1 * f_ring))))
        * jnp.exp(((f1 - f_ring) * gamma2) / (f_damp * gamma3))
    )
    gamma_r = gamma2 / (f_damp * gamma3)
    gamma_d2 = (gamma3 * f_damp) * (gamma3 * f_damp)
    gamma_d13 = f_damp * gamma1 * gamma3

    f_amp_ins_min = 0.0026
    f_amp_ins_max = wf.f_meco + 0.25 * (wf.f_isco - wf.f_meco)
    f_amp_match_in = f_amp_ins_max
    f_amp_int_max = f_amp_rd_min

    pn_initial = 1.0
    pn_one_third = 0.0
    pn_two_thirds = fma(1804.0, eta, -969.0) / 672.0 * pi.two_thirds
    pn_three_thirds = fma(-(44.0 * (chi1l + chi2l)), eta, fma(-(81.0 * chi2l), delta, fma(81.0, chi1l + chi2l, 81.0 * chi1l * delta))) / 48.0 * pi.itself
    pn_four_thirds = fma(24.0, fma(1473794.0, eta2, fma(428652.0 * chi2l2, -1.0 + delta, fma(10584.0, fma(81.0, chi2l2, fma(81.0, chi1l2, -(94.0 * chi1l * chi2l))), -1975055.0) * eta)), fma(-(10287648.0 * chi1l2), 1.0 + delta, -27312085.0)) / 8.128512e6 * pi.one_third * pi.itself
    pn_five_thirds = fma(42840.0 * fma(4.0, eta, -1.0), PI, fma(chi1l, fma(287213.0, 1.0 + delta, -(4.0 * eta * fma(8908.0, eta, fma(2083.0, delta, 93414.0)))), fma(-6048.0 * chi1l3, fma(3.0 + delta, eta, -1.0 - delta), chi2l * fma(-35632.0, eta2, fma(-fma(6048.0, chi2l2, 287213.0), -1.0 + delta, 4.0 * fma(2083.0, delta, fma(1512.0 * chi2l2, -3.0 + delta, -93414.0)) * eta))))) / 32256.0 * pi.five_thirds
    pn_six_thirds = fma(12.0, fma(eta, fma(-77616.0, fma(275520.0, pi.two, fma(-(107520.0 * (chi1l + chi2l)), PI, fma(-chi2l2, fma(282718.0, delta, 158633.0), fma(638642.0 * chi1l, chi2l, chi1l2 * fma(282718.0, delta, -158633.0))))), 545384828789.0), fma(-116424.0, fma(147117.0, fma(-chi2l2, -1.0 + delta, chi1l2 * (1.0 + delta)), 60928.0 * fma(-chi2l, delta, fma(chi1l, delta, chi1l + chi2l)) * PI), fma(28.0 * fma(11088.0, fma(163199.0, chi2l2, fma(163199.0, chi1l2, -(266498.0 * chi1l * chi2l))), -3248849057.0), eta2, 27026893936.0 * eta3))), -1242641879927.0) / 6.0085960704e10 * pi.two

    f_ins = [0.25 * f_amp_match_in, 0.50 * f_amp_match_in, 0.75 * f_amp_match_in, 1.00 * f_amp_match_in]
    v_ins = [
        0.0,
        inspiral.amp_22_v2(eta, wf.chi_pn_hat, wf.dchi, delta),
        inspiral.amp_22_v3(eta, wf.chi_pn_hat, wf.dchi, delta),
        inspiral.amp_22_v4(eta, wf.chi_pn_hat, wf.dchi, delta),
    ]
    rho1, rho2, rho3 = inspiral.amp_22_rho([v_ins[1], v_ins[2], v_ins[3]], [f_ins[1], f_ins[2], f_ins[3]])

    amp = Struct(
        f_amp_ins_min=f_amp_ins_min,
        f_amp_ins_max=f_amp_ins_max,
        f_amp_match_in=f_amp_match_in,
        f_amp_int_max=f_amp_int_max,
        f_amp_rd_min=f_amp_rd_min,
        pn_initial=pn_initial,
        pn_one_third=pn_one_third,
        pn_two_thirds=pn_two_thirds,
        pn_three_thirds=pn_three_thirds,
        pn_four_thirds=pn_four_thirds,
        pn_five_thirds=pn_five_thirds,
        pn_six_thirds=pn_six_thirds,
        pn_seven_thirds=rho1,
        pn_eight_thirds=rho2,
        pn_nine_thirds=rho3,
        rho1=rho1,
        rho2=rho2,
        rho3=rho3,
        v1_rd=v1_rd,
        gamma1=gamma1,
        gamma2=gamma2,
        gamma3=gamma3,
        gamma_r=gamma_r,
        gamma_d2=gamma_d2,
        gamma_d13=gamma_d13,
        f_ring=f_ring,
        f_damp=f_damp,
    )

    f1 = amp.f_amp_match_in
    f4 = amp.f_amp_rd_min
    p1 = powers(f1)
    p4 = powers(f4)
    d1 = inspiral.amp_22_dansatz(f1, wf, amp)
    d4 = ringdown.amp_22_dansatz(f4, f_ring, f_damp, amp)
    insp_f1 = inspiral.amp_22_ansatz_exact(f1, p1, amp)
    rd_f4 = ringdown.amp_22_ansatz_exact(f4, f_ring, f_damp, amp)
    d1 = ((7.0 / 6.0) * p1.one_sixth / insp_f1) - (p1.seven_sixths * d1 / (insp_f1 * insp_f1))
    d4 = ((7.0 / 6.0) * p4.one_sixth / rd_f4) - (p4.seven_sixths * d4 / (rd_f4 * rd_f4))
    f2 = f1 + (1.0 / 2.0) * (f4 - f1)
    v1 = 1.0 / (p1.m_seven_sixths * insp_f1)
    v2 = 1.0 / intermediate.amp_22_va(eta, wf.s_tot_r, wf.dchi, delta)
    v2 = fma(wf.coprec.dev, wf.coprec.mu3, v2)
    v4 = 1.0 / (p4.m_seven_sixths * rd_f4)
    deltas = intermediate.amp_22_deltas(d1, d4, v1, v2, v4, f1, f2, f4)
    amp.delta0, amp.delta1, amp.delta2, amp.delta3, amp.delta4, amp.delta5 = deltas
    amp.d1 = d1
    amp.d4 = d4
    return amp


def amp_ansatz(mf, p, amp):
    """The amplitude ansatz without the ``f^(-7/6)`` prefactor, with the
    region boundaries of ``IMRPhenomXASGenerateFD`` (``Mf > fAmpRDMin`` is
    the ringdown)."""
    ins = inspiral.amp_22_ansatz(mf, p, amp)
    rd = ringdown.amp_22_ansatz(mf, amp)
    int_ = intermediate.amp_22_ansatz(mf, p, amp)
    return jnp.where(mf < amp.f_amp_match_in, ins, jnp.where(mf > amp.f_amp_rd_min, rd, int_))


def amplitude_22(mf, p, amp, wf):
    """``IMRPhenomX_Amplitude_22``: the amplitude including ``ampNorm
    f^(-7/6)`` (multiply by ``amp0`` for strain); ``Mf >= fAmpRDMin`` is the
    ringdown."""
    prefactor = wf.amp_norm / p.seven_sixths
    ins = inspiral.amp_22_ansatz(mf, p, amp)
    rd = ringdown.amp_22_ansatz(mf, amp)
    int_ = intermediate.amp_22_ansatz(mf, p, amp)
    return prefactor * jnp.where(mf < amp.f_amp_match_in, ins, jnp.where(mf >= amp.f_amp_rd_min, rd, int_))
