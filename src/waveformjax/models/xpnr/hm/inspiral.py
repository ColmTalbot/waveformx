# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The inspiral of the IMRPhenomXHM higher modes (``LALSimIMRPhenomXHM_inspiral.c``):
the re-expanded PN amplitude (``IMRPhenomXHM_GetPNAmplitudeCoefficients``),
the TaylorT4 based amplitude of the (2,1) mode of release 122019
(``Get21PNAmplitudeCoefficients``, ``IMRPhenomXHM_Inspiral_PNAmp_21Ansatz``)
and the inspiral phase (the rescaled 22 phase), after waveformrs's
``imrphenomx/hm_inspiral.rs`` and ``imrphenomx/hm_amp_122019.rs``.

LAL evaluates the complex PN coefficients in C complex arithmetic; they are
kept here as pairs ``(re, im)`` following the same order of operations.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .._generated import hm_amp_122019 as gen
from .._math import LN_2, POWERS_OF_LALPI, Struct, fma, powers

__all__ = [
    "pn_amplitude_coefficients",
    "pn21_coefficients",
    "pn_amp_rescaled",
    "phase_ansatz_int",
    "phase_ansatz",
]


def constant_powers(x):
    """``IMRPhenomX_UsefulPowers`` of a constant, as Python floats."""
    with jax.ensure_compile_time_eval():
        p = powers(np.float64(x))
        return Struct(**{name: float(value) for name, value in p.items()})


_P2 = {}


def _p2(x):
    """The powers of the constants 2, 2/3 and 1/2 of the PN coefficients."""
    if x not in _P2:
        _P2[x] = constant_powers(x)
    return _P2[x]


def pn_amplitude_coefficients(wf, hm):
    """``IMRPhenomXHM_GetPNAmplitudeCoefficients``: the coefficients
    ``pnInitial`` ... ``pnSixThirds`` of the re-expanded Fourier-domain PN
    amplitude of the mode (a power series in ``f^(1/3)``), as lists ``re``,
    ``im``, and ``pnGlobalFactor``, the leading PN factor relative to the
    22."""
    chi_a = hm.chi_a
    chi_s = hm.chi_s
    eta = wf.eta
    delta = wf.delta
    lalpi = POWERS_OF_LALPI
    pi = lalpi.itself
    sqrt_5_7 = math.sqrt(5.0 / 7.0)
    prefactors = [
        math.sqrt(2.0) / 3.0,
        0.75 * sqrt_5_7,
        sqrt_5_7 / 3.0,
        4.0 * math.sqrt(2.0) / 9.0 * sqrt_5_7,
    ]
    pn_global_factor = (2.0 / hm.emm) ** (-7.0 / 6.0) * prefactors[hm.mode_int]
    zero = 0.0 * eta
    tag = hm.mode_tag
    if tag == 21:
        p2 = _p2(2.0)
        re = 3427.0 * chi_a
        re = re + 3427.0 * chi_s * delta
        re = re - 8404.0 * chi_a * eta
        re = re - 3860.0 * chi_s * delta * eta
        re = re - 1344.0 * delta * pi
        im = -(672.0 * delta)
        im = im - 672.0 * delta * math.log(16.0)
        scale = lalpi.four_thirds * p2.four_thirds
        four_thirds = (re / 1344.0 * scale, im / 1344.0 * scale)

        re = 143063173.0 * chi_a
        re = re + 143063173.0 * chi_s * delta
        re = re - 546199608.0 * chi_a * eta
        re = re - 169191096.0 * chi_s * delta * eta
        re = re - 9898560.0 * delta * pi
        re = re + 20176128.0 * delta * eta * pi
        re = re + 61725888.0 * chi_s * delta * chi_a * chi_a
        re = re - 81285120.0 * chi_s * delta * eta * chi_a * chi_a
        re = re + 20575296.0 * jnp.power(chi_a, 3.0)
        re = re - 81285120.0 * eta * jnp.power(chi_a, 3.0)
        re = re + 61725888.0 * chi_a * chi_s * chi_s
        re = re - 165618432.0 * chi_a * eta * chi_s * chi_s
        re = re + 20575296.0 * delta * jnp.power(chi_s, 3.0)
        re = re - 1016064.0 * delta * eta * chi_s * chi_s * chi_s
        re = re + 128873808.0 * chi_a * eta * eta
        re = re - 3859632.0 * chi_s * delta * eta * eta
        im = -(1350720.0 * delta)
        im = im - 72043776.0 * delta * eta
        im = im - 5402880.0 * delta * LN_2
        im = im - 17224704.0 * delta * eta * LN_2
        scale = lalpi.two * p2.two
        six_thirds = (re / 5.419008e6 * scale, im / 5.419008e6 * scale)

        five_thirds = fma(21535920.0 * delta * eta, eta, fma(39626496.0 * delta * eta * chi_s, chi_s, fma(
            -(77982912.0 * delta * chi_s), chi_s, fma(81285120.0 * delta * eta * chi_a, chi_a, fma(
                -(77982912.0 * delta * chi_a), chi_a, fma(24385536.0 * chi_s * delta, pi, fma(
                    24385536.0 * chi_a, pi, fma(-(23670792.0 * delta), eta, fma(
                        432843264.0 * chi_a * chi_s, eta, fma(-155965824.0 * chi_a, chi_s, -(964357.0 * delta)))))))))))
        pn = [
            (zero, zero),
            (delta * lalpi.one_third * p2.one_third, zero),
            (-3.0 * fma(chi_s, delta, chi_a) / 2.0 * lalpi.two_thirds * p2.two_thirds, zero),
            (fma(335.0, delta, 1404.0 * delta * eta) / 672.0 * lalpi.itself * p2.itself, zero),
            four_thirds,
            (five_thirds / 8.128512e6 * lalpi.five_thirds * p2.five_thirds, zero),
            six_thirds,
        ]
    elif tag == 33:
        p2 = _p2(2.0 / 3.0)
        ln15 = math.log(1.5)
        re = 325.0 * chi_a
        re = re + 325.0 * chi_s * delta
        re = re - 1120.0 * chi_a * eta
        re = re - 80.0 * chi_s * delta * eta
        re = re + 120.0 * delta * pi
        im = -(504.0 * delta)
        im = im + 720.0 * delta * ln15
        scale = lalpi.four_thirds * p2.four_thirds
        four_thirds = (re / 120.0 * scale, im / 120.0 * scale)

        re = 22007835.0 * chi_a
        re = re + 22007835.0 * chi_s * delta
        re = re - 80190540.0 * chi_a * eta
        re = re - 31722300.0 * chi_s * delta * eta
        re = re - 9193500.0 * delta * pi
        re = re + 17826480.0 * delta * eta * pi
        re = re - 12428640.0 * chi_a * eta * eta
        re = re - 6078240.0 * chi_s * delta * eta * eta
        im = 26467560.0 * delta
        im = im - 98774368.0 * delta * eta
        im = im - 37810800.0 * delta * ln15
        im = im + 37558080.0 * delta * eta * ln15
        scale = lalpi.two * p2.two
        six_thirds = (re / 2.17728e6 * scale, im / 2.17728e6 * scale)

        five_thirds = fma(2966264784.0 * delta * eta, eta, fma(55883520.0 * delta * eta * chi_s, chi_s, fma(
            -(1131641280.0 * delta * chi_s), chi_s, fma(4470681600.0 * delta * eta * chi_a, chi_a, fma(
                -(1131641280.0 * delta * chi_a), chi_a, fma(-(5926068792.0 * delta), eta, fma(
                    9053130240.0 * chi_a * chi_s, eta, fma(-2263282560.0 * chi_a, chi_s, -(1077664867.0 * delta)))))))))
        pn = [
            (zero, zero),
            (delta * lalpi.one_third * p2.one_third, zero),
            (zero, zero),
            (fma(-1945.0, delta, 2268.0 * delta * eta) / 672.0 * lalpi.itself * p2.itself, zero),
            four_thirds,
            (five_thirds / 4.4706816e8 * lalpi.five_thirds * p2.five_thirds, zero),
            six_thirds,
        ]
    elif tag == 32:
        re = 0.0 - 3955.0 * chi_s
        re = re - 3955.0 * chi_a * delta
        re = re + 10810.0 * chi_s * eta
        re = re + 11865.0 * chi_a * delta * eta
        re = re - 12600.0 * chi_s * eta * eta
        im = 2520.0 + zero
        im = im - 11088.0 * eta
        scale = lalpi.five_thirds
        five_thirds = (re / 840.0 * scale, im / 840.0 * scale)
        six_thirds = fma(5907445488.0, jnp.power(eta, 3.0), fma(5830513920.0 * chi_s * chi_s * eta, eta, fma(
            13412044800.0 * chi_a * chi_a * eta, eta, fma(-(13202119896.0 * eta), eta, fma(
                -(11870591040.0 * eta * chi_s), chi_s, fma(1131641280.0 * chi_s, chi_s, fma(
                    -(7865605440.0 * eta * chi_a), chi_a, fma(1131641280.0 * chi_a, chi_a, fma(
                        3576545280.0 * chi_s * eta, pi, fma(-(15209631360.0 * chi_a * chi_s * delta), eta, fma(
                            -26069649.0, eta, fma(2263282560.0 * chi_a * chi_s, delta, 824173699.0))))))))))))
        pn = [
            (zero, zero),
            (zero, zero),
            (fma(3.0, eta, -1.0) * lalpi.two_thirds, zero),
            (-4.0 * chi_s * eta * lalpi.itself, zero),
            (fma(82460.0 * eta, eta, fma(-61625.0, eta, 10471.0)) / 10080.0 * lalpi.four_thirds, zero),
            five_thirds,
            (six_thirds / 4.4706816e8 * lalpi.two, zero),
        ]
    else:
        p2 = _p2(0.5)
        re = -0.0 + 565.0 * chi_s
        re = re + 565.0 * chi_a * delta
        re = re - 2075.0 * chi_s * eta
        re = re - 1695.0 * chi_a * delta * eta
        re = re + 240.0 * pi
        re = re - 720.0 * eta * pi
        re = re + 1140.0 * chi_s * eta * eta
        im = -1008.0 + zero
        im = im + 3579.0 * eta
        im = im + 960.0 * LN_2
        im = im - 2880.0 * eta * LN_2
        scale = lalpi.five_thirds * p2.five_thirds
        five_thirds = (re / 120.0 * scale, im / 120.0 * scale)
        six_thirds = fma(-805075876080.0, jnp.power(eta, 3.0), fma(-(10897286400.0 * chi_s * chi_s * eta), eta, fma(
            -(871782912000.0 * chi_a * chi_a * eta), eta, fma(2271682065240.0 * eta, eta, fma(
                224302478400.0 * eta * chi_s, chi_s, fma(-(73556683200.0 * chi_s), chi_s, fma(
                    511264353600.0 * eta * chi_a, chi_a, fma(-(73556683200.0 * chi_a), chi_a, fma(
                        441340099200.0 * chi_a * chi_s * delta, eta, fma(-745140957231.0, eta, fma(
                            -(147113366400.0 * chi_a * chi_s), delta, 7888301437.0)))))))))))
        pn = [
            (zero, zero),
            (zero, zero),
            (fma(-3.0, eta, 1.0) * lalpi.two_thirds * p2.two_thirds, zero),
            (zero, zero),
            (fma(-(446460.0 * eta), eta, fma(641105.0, eta, -158383.0)) / 36960.0 * lalpi.four_thirds * p2.four_thirds, zero),
            five_thirds,
            (six_thirds / 2.90594304e10 * lalpi.two * p2.two, zero),
        ]
    return Struct(
        pn_re=[c[0] for c in pn],
        pn_im=[c[1] for c in pn],
        pn_global_factor=pn_global_factor,
    )


def pn21_coefficients(wf):
    """``Get21PNAmplitudeCoefficients``: the coefficients of the TaylorT4
    based inspiral amplitude of the (2,1) mode (release 122019), with C's
    complex arithmetic (``I * x`` is ``(0 x, x)``; the ``I / 3`` of LAL is
    evaluated in single precision)."""
    lalpi = POWERS_OF_LALPI
    p2 = _p2(2.0)
    logof2 = p2.log
    delta = wf.delta
    eta = wf.eta
    m1 = wf.m1
    m2 = wf.m2
    chi1 = wf.chi1l
    chi2 = wf.chi2l
    chi_s = (chi1 + chi2) * 0.5
    chi_a = (chi1 - chi2) * 0.5
    sc = fma(m1 * m1, chi1, m2 * m2 * chi2)
    sigmac = fma(m2, chi2, -(m1 * chi1))

    def scale(z, a):
        return (z[0] * a, z[1] * a)

    def divide(z, a):
        return (z[0] / a, z[1] / a)

    def mul(a, b):
        return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])

    i_delta = (0.0 * delta, 1.0 * delta)
    x05 = scale(scale(divide(i_delta, 3.0), p2.one_third), lalpi.one_third)
    minus_i_half = (-0.0 * 0.5, -1.0 * 0.5)
    x1 = scale(scale(scale(minus_i_half, fma(chi_s, delta, chi_a)), p2.two_thirds), lalpi.two_thirds)
    x15 = scale(scale(divide(scale(i_delta, -17.0 / 28.0 + 5.0 * eta / 7.0), 3.0), p2.itself), lalpi.itself)
    a = fma(-43.0 / 21.0 * delta, sc, fma(139.0, eta, -79.0) / 42.0 * sigmac)
    i_a = (0.0 * a, 1.0 * a)
    w = fma(-2.0, logof2, -0.5)
    pi_plus_iw = (lalpi.itself + 0.0 * w, 1.0 * w)
    i_third = float(np.float32(1.0) / np.float32(3.0))
    i_third_delta = (0.0 * delta, i_third * delta)
    product = mul(i_third_delta, pi_plus_iw)
    x2 = scale(scale((i_a[0] + product[0], i_a[1] + product[1]), p2.four_thirds), lalpi.four_thirds)
    t25 = fma(-509.0, eta, -43.0) / 126.0 + 79.0 * eta * eta / 168.0
    x25 = scale(scale(divide(scale(i_delta, t25), 3.0), p2.five_thirds), lalpi.five_thirds)
    u = fma(6.0, eta, -17.0) / 28.0 * lalpi.itself
    v = fma(eta, -353.0 / 28.0 - 3.0 * logof2 / 7.0, 17.0 / 56.0) + 17.0 * logof2 / 14.0
    u_plus_iv = (u + 0.0 * v, 1.0 * v)
    x3 = scale(scale(divide(mul(i_delta, u_plus_iv), 3.0), p2.two), lalpi.two)
    xs = (x05, x1, x15, x2, x25, x3)
    return Struct(
        pntd_factor=8.0 * eta * lalpi.two_thirds * p2.two_thirds * lalpi.sqrt / math.sqrt(5.0),
        x_re=[x[0] for x in xs],
        x_im=[x[1] for x in xs],
        xdot5=gen.xdot5(wf, p2, lalpi),
        xdot6=gen.xdot6(wf, p2, lalpi),
        xdot65=gen.xdot65(wf, p2, lalpi),
        xdot7=gen.xdot7(wf, p2, lalpi),
        xdot75=gen.xdot75(wf, p2, lalpi),
        xdot8=gen.xdot8(wf, p2, lalpi),
        xdot8_log=gen.xdot8_log(wf, p2, lalpi),
        xdot85=gen.xdot85(wf, p2, lalpi),
        log2pi_two_thirds=math.log(lalpi.two_thirds * p2.two_thirds),
    )


def pn21_ansatz(p, c):
    """``IMRPhenomXHM_Inspiral_PNAmp_21Ansatz``: the time-domain PN
    amplitude of the (2,1) times the stationary-phase factor of TaylorT4
    (strain without ``amp0``)."""
    lalpi = POWERS_OF_LALPI
    two_to_m_one_sixths = 0.8908987181403393
    three_to_m_one_second = 0.5773502691896257
    x_to_m_one_four = two_to_m_one_sixths * lalpi.m_one_sixth * p.m_one_sixth
    terms = [p.one_third, p.two_thirds, p.itself, p.four_thirds, p.five_thirds, p.two]
    re = terms[0] * c.x_re[0]
    im = terms[0] * c.x_im[0]
    for k in range(1, 6):
        re = re + terms[k] * c.x_re[k]
        im = im + terms[k] * c.x_im[k]
    re = re * p.two_thirds * c.pntd_factor
    im = im * p.two_thirds * c.pntd_factor
    xdot = gen.xdot_t4(p, c)
    # xdot > 0 in the inspiral; elsewhere (where the ansatz is not used) a
    # finite placeholder keeps NaNs out of the gradients
    xdot = jnp.where(xdot > 0.0, xdot, 1.0)
    return 2.0 * lalpi.sqrt * three_to_m_one_second * jnp.hypot(re, im) * x_to_m_one_four / jnp.sqrt(xdot)


def pn_amp_rescaled(p, pn, amp_norm, pn21=None, use_famp_pn=False):
    """``IMRPhenomXHM_Inspiral_PNAmp_Ansatz`` with ``InspRescaleFactor = 1``:
    the PN amplitude divided by the leading order of the 22, ``ampNorm
    f^(-7/6)`` (the TaylorT4 amplitude of the (2,1), ``pn21``, where
    ``use_famp_pn``)."""
    terms = [p.one_third, p.two_thirds, p.itself, p.four_thirds, p.five_thirds, p.two]
    re = pn.pn_re[0]
    im = pn.pn_im[0]
    for k in range(6):
        re = re + terms[k] * pn.pn_re[k + 1]
        im = im + terms[k] * pn.pn_im[k + 1]
    series = jnp.hypot(re, im) * pn.pn_global_factor
    if pn21 is None:
        return series
    taylor = pn21_ansatz(p, pn21) / (amp_norm * p.m_seven_sixths)
    return jnp.where(use_famp_pn, taylor, series)


def phase_ansatz_int(mf, p, phase):
    """``IMRPhenomXHM_Inspiral_Phase_AnsatzInt``: the rescaled 22 inspiral
    phase, ``sum_i (phi_i + phiL_i log f) f^((i-5)/3)``."""
    freqs = [
        p.m_five_thirds, p.m_four_thirds, p.m_one, p.m_two_thirds, p.m_one_third, 1.0,
        p.one_third, p.two_thirds, mf, p.four_thirds, p.five_thirds, p.two, p.seven_thirds,
    ]
    log_mf = p.log
    philm = 0.0
    for i in range(13):
        philm = fma(fma(phase.phi_l[i], log_mf, phase.phi[i]), freqs[i], philm)
    return philm


_COEFFS = (-5.0 / 3.0, -4.0 / 3.0, -1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0,
           4.0 / 3.0, 5.0 / 3.0, 2.0, 7.0 / 3.0)


def phase_ansatz(mf, p, phase):
    """``IMRPhenomXHM_Inspiral_Phase_Ansatz``: the derivative of
    :func:`phase_ansatz_int`."""
    freqs = [
        p.m_eight_thirds, p.m_seven_thirds, p.m_two, p.m_five_thirds, p.m_four_thirds, p.m_one,
        p.m_two_thirds, p.m_one_third, 1.0, p.one_third, p.two_thirds, mf, p.four_thirds,
    ]
    log_mf = p.log
    dphilm = 0.0
    for i in range(13):
        coefficient = fma(phase.phi_l[i], log_mf, phase.phi[i])
        dphilm = fma(fma(coefficient, _COEFFS[i], phase.phi_l[i]), freqs[i], dphilm)
    return dphilm
