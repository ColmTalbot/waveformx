# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The phase of the (2,2) mode: ``IMRPhenomXGetPhaseCoefficients``,
``IMRPhenomX_Phase_22_ConnectionCoefficients``, ``IMRPhenomX_TimeShift_22``,
``IMRPhenomX_Phase_22`` and ``IMRPhenomX_dPhase_22`` of
``LALSimIMRPhenomX_internals.c`` for the default versions (InsPhaseVersion
104, IntPhaseVersion 105, RDPhaseVersion 105), after waveformrs's
``imrphenomx/phase.rs``.

The phase is evaluated in the three regions on every frequency and the
region is chosen with ``jnp.where``: all three ansatzes are finite for
``Mf > 0``.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._generated import inspiral as _inspiral
from ._generated import intermediate as _intermediate
from ._generated import ringdown as _ringdown
from ._generated import utilities as _utilities
from ._math import EGAMMA, FencedModule, LN_2, PI, POWERS_OF_LALPI, Struct, fenced, fma, powers

inspiral = FencedModule(_inspiral)
intermediate = FencedModule(_intermediate)
ringdown = FencedModule(_ringdown)
utilities = FencedModule(_utilities)

__all__ = [
    "phase_coefficients",
    "phase_xas",
    "phase_22",
    "dphase_22",
    "time_shift_22",
]

GPOINTS4 = (0.0, 1.0 / 4.0, 3.0 / 4.0, 1.0)
_S = 1.0 / (2.0 * 2.0 ** 0.5)
GPOINTS5 = (0.0, 0.5 - _S, 0.5, 0.5 + _S, 1.0)


def _solve(rows, values):
    """``gsl_linalg_LU_solve`` of the small collocation systems."""
    matrix = jnp.stack([jnp.stack([jnp.asarray(x, dtype=jnp.float64) for x in row]) for row in rows])
    rhs = jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in values])
    solution = jnp.linalg.solve(matrix, rhs)
    return [solution[i] for i in range(len(values))]


@fenced
def phase_coefficients(wf):
    """``IMRPhenomXGetPhaseCoefficients`` followed by
    ``IMRPhenomX_Phase_22_ConnectionCoefficients`` (the connection
    coefficients ``c1_int``, ``c2_int``, ``c1_mrd``, ``c2_mrd`` are set),
    including the co-precessing deviations of IMRPhenomXPNR in
    ``wf.coprec`` (zero otherwise)."""
    pi = POWERS_OF_LALPI
    f_ring = wf.f_ring
    f_damp = wf.f_damp
    dphase0 = wf.dphase0

    # matching regions, Eqs. 5.10, 5.11 and 7.7 of arXiv:2001.11412
    f_im_match = 0.6 * (0.5 * f_ring + wf.f_isco)
    f_in_match = wf.f_meco
    deltaf = (f_im_match - f_in_match) * 0.03
    f_phase_match_in = f_in_match - 1.0 * deltaf
    f_phase_match_im = f_im_match + 0.5 * deltaf
    f_phase_ins_min = 0.0026
    f_phase_ins_max = 1.020 * wf.f_meco
    f_phase_rd_min = f_im_match
    f_phase_rd_max = fma(1.25, f_damp, f_ring)
    phi_norm = -(3.0 * pi.m_five_thirds) / 128.0

    chi1l = wf.chi1l
    chi2l = wf.chi2l
    chi1l2l = chi1l * chi2l
    chi1l2 = chi1l * chi1l
    chi1l3 = chi1l * chi1l2
    chi2l2 = chi2l * chi2l
    chi2l3 = chi2l * chi2l2
    eta = wf.eta
    eta2 = eta * eta
    eta3 = eta * eta2
    delta = wf.delta
    s, dchi = wf.s_tot_r, wf.dchi

    # ---------------------------- ringdown ----------------------------
    deltax = f_phase_rd_max - f_phase_rd_min
    points_rd = [fma(g, deltax, f_phase_rd_min) for g in GPOINTS5]
    # collocation point 4 at the ringdown frequency ~ dip in the Lorentzian
    points_rd[3] = f_ring
    rd_v4 = ringdown.phase_22_v4(eta, s, dchi, delta)
    values_rd = [
        ringdown.phase_22_d12(eta, s, dchi, delta),
        ringdown.phase_22_d24(eta, s, dchi, delta),
        ringdown.phase_22_d34(eta, s, dchi, delta),
        rd_v4,
        ringdown.phase_22_d54(eta, s, dchi, delta),
    ]
    values_rd[4] = values_rd[4] + values_rd[3]
    values_rd[2] = values_rd[2] + values_rd[3]
    values_rd[1] = values_rd[1] + values_rd[3]
    values_rd[0] = values_rd[0] + values_rd[1]
    phase_rd = values_rd[0]
    rows = []
    for ff in points_rd:
        invff = 1.0 / ff
        ff1 = jnp.cbrt(invff)
        ff2 = invff * invff
        ff3 = ff2 * ff2
        ff4 = -dphase0 / fma(f_damp, f_damp, (ff - f_ring) * (ff - f_ring))
        rows.append([1.0, ff1, ff2, ff3, ff4])
    c0, c1, c2, c4, c_rd = _solve(rows, values_rd)
    c_l = -(dphase0 * c_rd)
    # IMRPhenomXPNR co-precessing deviation (zero otherwise)
    c_l = fma(wf.coprec.dev, wf.coprec.nu4, c_l)

    # ---------------------------- inspiral ----------------------------
    deltax = f_phase_ins_max - f_phase_ins_min
    points_ins = [fma(g, deltax, f_phase_ins_min) for g in GPOINTS4]
    s_ins = wf.chi_pn_hat
    v3 = inspiral.phase_22_v3(eta, s_ins, dchi, delta)
    values_ins = [
        inspiral.phase_22_d13(eta, s_ins, dchi, delta) + v3,
        inspiral.phase_22_d23(eta, s_ins, dchi, delta) + v3,
        v3,
        inspiral.phase_22_d43(eta, s_ins, dchi, delta) + v3,
    ]
    rows = []
    for ff in points_ins:
        ff1 = jnp.cbrt(ff)
        ff2 = ff1 * ff1
        rows.append([1.0, ff1, ff2, ff])
    a0, a1, a2, a3 = _solve(rows, values_ins)
    a4 = 0.0
    sigma1 = (-5.0 / 3.0) * a0
    sigma2 = (-5.0 / 4.0) * a1
    sigma3 = (-5.0 / 5.0) * a2
    sigma4 = (-5.0 / 6.0) * a3

    # ---------------------- TaylorF2 PN coefficients ----------------------
    phi0_ns = 1.0
    phi2_ns = (3715.0 / 756.0 + 55.0 * eta / 9.0) * pi.two_thirds
    phi3_ns = -16.0 * pi.two
    phi3_s = fma(113.0, fma(-chi2l, delta, fma(chi1l, delta, chi1l + chi2l)), -(76.0 * (chi1l + chi2l) * eta)) / 6.0 * pi.itself
    phi4_ns = (15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta2 / 72.0) * pi.four_thirds
    phi4_s = -5.0 * fma(-(81.0 * chi2l2), fma(2.0, eta, -1.0 + delta), fma(81.0 * chi1l2, fma(-2.0, eta, 1.0 + delta), 316.0 * chi1l2l * eta)) / 16.0 * pi.four_thirds
    phi5l_ns = 5.0 * fma(-6552.0, eta, 46374.0) * PI / 4536.0 * pi.five_thirds
    phi5l_s = fma(85680.0 * (chi1l + chi2l), eta2, fma(-732985.0, fma(-chi2l, delta, fma(chi1l, delta, chi1l + chi2l)), -(560.0 * fma(-1213.0, chi1l + chi2l, 63.0 * (chi1l - chi2l) * delta) * eta))) / 4536.0 * pi.five_thirds
    phi6_ns = (11583231236531.0 / 4.69421568e9 - 5.0 * eta * fma(588.0 * eta, fma(102260.0, eta, -45633.0), 3147553127.0) / 3.048192e6 - 6848.0 * EGAMMA / 21.0 - 640.0 * pi.two / 3.0 + 2255.0 * eta * pi.two / 12.0 - 13696.0 * LN_2 / 21.0 - 6848.0 * pi.log / 63.0) * pi.two
    phi6_s = 5.0 * fma(227.0, fma(-chi2l, delta, fma(chi1l, delta, chi1l + chi2l)), -(156.0 * (chi1l + chi2l) * eta)) * PI / 3.0 * pi.two
    phi6_s = fma(5.0 * fma(-(7.0 * chi1l2), fma(9808.0, eta2, fma(-15103.0, 1.0 + delta, 2.0 * fma(6580.0, delta, 21683.0) * eta)), fma(20.0 * chi1l2l * eta, fma(12488.0, eta, 11763.0), 7.0 * chi2l2 * fma(-9808.0, eta2, fma(-15103.0, -1.0 + delta, 2.0 * fma(6580.0, delta, -21683.0) * eta)))) / 4032.0, pi.two, phi6_s)
    phi6l_ns = -6848.0 / 63.0 * pi.two
    phi7_ns = 5.0 * fma(168.0 * fma(-29618.0, eta, 75703.0), eta, 15419335.0) * PI / 254016.0 * pi.seven_thirds
    phi7_s = 5.0 * fma(90514368.0 * (chi1l + chi2l), eta3, fma(-(1008.0 * fma(208433.0, chi1l + chi2l, 25011.0 * (chi1l - chi2l) * delta)), eta2, fma(-5030016755.0, fma(-chi2l, delta, fma(chi1l, delta, chi1l + chi2l)), 4.0 * fma(2113331119.0, chi1l + chi2l, 675484362.0 * (chi1l - chi2l) * delta) * eta))) / 6.096384e6 * pi.seven_thirds
    phi7_s = fma(-5.0 * fma(-(57.0 * chi2l2), fma(2.0, eta, -1.0 + delta), fma(57.0 * chi1l2, fma(-2.0, eta, 1.0 + delta), 220.0 * chi1l2l * eta)) * PI, pi.seven_thirds, phi7_s)
    phi7_s = fma(fma(40.0 * (chi1l + chi2l) * fma(17.0, chi2l2, fma(17.0, chi1l2, -(14.0 * chi1l2l))), eta2, fma(14585.0, fma(-chi2l3, -1.0 + delta, chi1l3 * (1.0 + delta)), -(5.0 * fma(chi1l3, fma(2985.0, delta, 8819.0), fma(-(8439.0 * chi1l2 * chi2l), 1.0 + delta, fma(chi2l3, fma(-2985.0, delta, 8819.0), 8439.0 * chi1l * chi2l2 * (-1.0 + delta)))) * eta))) / 48.0, pi.seven_thirds, phi7_s)
    phi8_common = fma(94344.0 * (chi1l + chi2l), eta2, fma(1263141.0, fma(-chi2l, delta, fma(chi1l, delta, chi1l + chi2l)), -(2.0 * fma(794075.0, chi1l + chi2l, 178533.0 * (chi1l - chi2l) * delta) * eta)))
    phi8_s = -5.0 * phi8_common * PI * (-1.0 + pi.log) / 9072.0 * pi.eight_thirds
    phi8l_s = -5.0 * phi8_common * PI / 9072.0 * pi.eight_thirds

    phi0 = phi0_ns
    phi2 = phi2_ns
    phi3 = phi3_ns + phi3_s
    phi4 = phi4_ns + phi4_s
    phi5l = phi5l_ns + phi5l_s
    phi6 = phi6_ns + phi6_s
    phi6l = phi6l_ns
    phi7 = phi7_ns + phi7_s
    phi8 = phi8_s
    phi8l = phi8l_s

    dphi0 = phi0
    dphi2 = 3.0 / 5.0 * phi2
    dphi3 = 2.0 / 5.0 * phi3
    dphi4 = 1.0 / 5.0 * phi4
    dphi5 = -3.0 / 5.0 * phi5l
    dphi6 = fma(-(1.0 / 5.0), phi6, -(3.0 / 5.0 * phi6l))
    dphi6l = -1.0 / 5.0 * phi6l
    dphi7 = -2.0 / 5.0 * phi7
    dphi8 = fma(-(3.0 / 5.0), phi8, -(3.0 / 5.0 * phi8l))
    dphi8l = -3.0 / 5.0 * phi8l

    # the phase derivative at the inspiral-intermediate transition
    p = powers(f_phase_match_in)
    phase_in = dphi0
    phase_in = fma(dphi2, p.two_thirds, phase_in)
    phase_in = fma(dphi3, p.itself, phase_in)
    phase_in = fma(dphi4, p.four_thirds, phase_in)
    phase_in = fma(dphi5, p.five_thirds, phase_in)
    phase_in = fma(dphi6, p.two, phase_in)
    phase_in = fma(dphi6l * p.two, p.log, phase_in)
    phase_in = fma(dphi7, p.seven_thirds, phase_in)
    phase_in = fma(dphi8, p.eight_thirds, phase_in)
    phase_in = fma(dphi8l * p.eight_thirds, p.log, phase_in)
    pseudo = fma(a0, p.eight_thirds, a1 * p.three)
    pseudo = fma(a2 * p.eight_thirds, p.two_thirds, pseudo)
    pseudo = fma(a3 * p.eight_thirds, p.itself, pseudo)
    phase_in = phase_in + pseudo
    phase_in = phase_in * p.m_eight_thirds * dphase0

    # -------------------------- intermediate --------------------------
    deltax = f_phase_match_im - f_phase_match_in
    points_int = [fma(g, deltax, f_phase_match_in) for g in GPOINTS5]
    v2_im_m_rd_v4 = intermediate.phase_22_v2m_rdv4(eta, s, dchi, delta)
    v3_im_m_rd_v4 = intermediate.phase_22_v3m_rdv4(eta, s, dchi, delta)
    v2_im = intermediate.phase_22_v2(eta, s, dchi, delta)
    values_int = [
        phase_in,
        fma(0.75, v2_im_m_rd_v4 + rd_v4, 0.25 * v2_im),
        v3_im_m_rd_v4 + rd_v4,
        intermediate.phase_22_d43(eta, s, dchi, delta),
        phase_rd,
    ]
    values_int[3] = values_int[3] + values_int[2]
    rows = []
    rhs = []
    for i, ff in enumerate(points_int):
        ff1 = f_ring / ff
        ff2 = ff1 * ff1
        ff3 = ff1 * ff2
        ff4 = ff2 * ff2
        if i == 0:
            ff0 = (4.0 * c_l) / fma(2.0 * f_damp, 2.0 * f_damp, (ff - f_ring) * (ff - f_ring))
        else:
            ff0 = (4.0 * c_l) / fma(4.0 * f_damp, f_damp, (ff - f_ring) * (ff - f_ring))
        rows.append([1.0, ff1, ff2, ff3, ff4])
        rhs.append(values_int[i] - ff0)
    x = _solve(rows, rhs)
    b0 = x[0]
    b1 = x[1] * f_ring
    b2 = x[2] * f_ring * f_ring
    b3 = x[3] * f_ring * f_ring * f_ring
    b4 = x[4] * f_ring * f_ring * f_ring * f_ring
    # IMRPhenomXPNR co-precessing deviations (zero otherwise)
    b1 = fma(wf.coprec.dev, wf.coprec.zeta2, b1)
    b4 = fma(wf.coprec.dev, wf.coprec.zeta1, b4)

    phase = Struct(
        f_phase_ins_min=f_phase_ins_min,
        f_phase_ins_max=f_phase_ins_max,
        f_phase_rd_min=f_phase_rd_min,
        f_phase_rd_max=f_phase_rd_max,
        f_phase_match_in=f_phase_match_in,
        f_phase_match_im=f_phase_match_im,
        c0=c0, c1=c1, c2=c2, c4=c4, c_rd=c_rd, c_l=c_l, c_l_gr=c_l,
        c4_ov3=c4 / 3.0, c_l_ov_fda=c_l / f_damp,
        b0=b0, b1=b1, b2=b2, b3=b3, b4=b4,
        a0=a0, a1=a1, a2=a2, a3=a3, a4=a4,
        sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, sigma4=sigma4, sigma5=0.0,
        phi0=phi0, phi1=0.0, phi2=phi2, phi3=phi3, phi4=phi4, phi5=0.0, phi5l=phi5l, phi6=phi6,
        phi6l=phi6l, phi7=phi7, phi8=phi8, phi8l=phi8l, phi9=0.0, phi9l=0.0,
        phi_initial=-PI / 4.0, phi_norm=phi_norm,
        dphi0=dphi0, dphi1=0.0, dphi2=dphi2, dphi3=dphi3, dphi4=dphi4, dphi5=dphi5, dphi6=dphi6,
        dphi6l=dphi6l, dphi7=dphi7, dphi8=dphi8, dphi8l=dphi8l, dphi9=0.0, dphi9l=0.0,
        f_ring=f_ring, f_damp=f_damp,
        dphase_norm=inspiral.inspiral_dphase_norm(),
        c1_int=0.0, c2_int=0.0, c1_mrd=0.0, c2_mrd=0.0,
    )
    _connection_coefficients(phase)
    return phase


def _connection_coefficients(phase):
    """``IMRPhenomX_Phase_22_ConnectionCoefficients``: ``C1``, ``C2`` making
    the phase and its derivative continuous (in place)."""
    f_ins = phase.f_phase_match_in
    f_int = phase.f_phase_match_im
    p_ins = powers(f_ins)
    dphi_ins = inspiral.phase_22_ansatz_exact(f_ins, p_ins, phase)
    dphi_int = intermediate.phase_22_ansatz_exact(f_ins, p_ins, phase)
    phase.c2_int = dphi_ins - dphi_int
    phi_in = inspiral.phase_22_ansatz_int_exact(f_ins, p_ins, phase)
    phi_im = intermediate.phase_22_ansatz_int_exact(f_ins, p_ins, phase)
    phase.c1_int = fma(-phase.c2_int, f_ins, phi_in - phi_im)

    p_int = powers(f_int)
    phi_imc = fma(phase.c2_int, f_int, intermediate.phase_22_ansatz_int_exact(f_int, p_int, phase) + phase.c1_int)
    phi_rd = ringdown.phase_22_ansatz_int_exact(f_int, p_int, phase)
    dphi_intc = intermediate.phase_22_ansatz_exact(f_int, p_int, phase) + phase.c2_int
    dphi_rd = ringdown.phase_22_ansatz_exact(f_int, p_int, phase)
    phase.c2_mrd = dphi_intc - dphi_rd
    phase.c1_mrd = fma(-phase.c2_mrd, f_int, phi_imc - phi_rd)
    return phase


def _regions(mf, p, phase):
    inspiral_phase = inspiral.phase_22_ansatz_int_exact(mf, p, phase)
    ringdown_phase = fma(phase.c2_mrd, mf, ringdown.phase_22_ansatz_int_exact(mf, p, phase) + phase.c1_mrd)
    intermediate_phase = fma(phase.c2_int, mf, intermediate.phase_22_ansatz_int_exact(mf, p, phase) + phase.c1_int)
    return inspiral_phase, intermediate_phase, ringdown_phase


def phase_xas(mf, p, phase):
    """The phase before the ``1/eta`` scaling and the linear terms, with the
    region boundaries of ``IMRPhenomXASGenerateFD`` (``Mf > fPhaseMatchIM``
    is the ringdown)."""
    ins, int_, rd = _regions(mf, p, phase)
    return jnp.where(mf < phase.f_phase_match_in, ins, jnp.where(mf > phase.f_phase_match_im, rd, int_))


def phase_22(mf, p, phase):
    """``IMRPhenomX_Phase_22`` (``Mf >= fPhaseMatchIM`` is the ringdown)."""
    ins, int_, rd = _regions(mf, p, phase)
    return jnp.where(mf < phase.f_phase_match_in, ins, jnp.where(mf >= phase.f_phase_match_im, rd, int_))


def dphase_22(mf, p, phase):
    """``IMRPhenomX_dPhase_22``: the phase derivative before the ``1/eta``
    scaling."""
    ins = inspiral.phase_22_ansatz_exact(mf, p, phase)
    rd = ringdown.phase_22_ansatz_exact(mf, p, phase) + phase.c2_mrd
    int_ = intermediate.phase_22_ansatz_exact(mf, p, phase) + phase.c2_int
    return jnp.where(mf < phase.f_phase_match_in, ins, jnp.where(mf >= phase.f_phase_match_im, rd, int_))


def time_shift_22(phase, wf):
    """``IMRPhenomX_TimeShift_22``: the linear phase coefficient ``linb``
    aligning the peak of the strain near ``t = 0``."""
    linb = utilities.linb(wf.eta, wf.s_tot_r, wf.dchi, wf.delta)
    f_ref_fit = wf.f_ring - wf.f_damp
    dphi22_ref = 1.0 / wf.eta * dphase_22(f_ref_fit, powers(f_ref_fit), phase)
    psi4_to_strain = utilities.psi4_to_strain(wf.eta, wf.s_tot_r, wf.dchi)
    return fma(-(2.0 * PI), 500.0 + psi4_to_strain, linb - dphi22_ref)
