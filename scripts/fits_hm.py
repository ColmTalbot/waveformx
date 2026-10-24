# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Generated modules of the IMRPhenomXHM higher modes of release 122019
(see generate_fits.py): the quasi-normal-mode fits, the parameter-space fits
of the 122019 amplitude (fits version 122018) and of the phase, the
coefficients of the intermediate amplitude polynomial and the straight-line
pieces of the 122019 amplitude. The control flow lives in
``src/waveformjax/models/xpnr/hm/``."""

_PHASE_FITS = [f"phase_{mode}_p{i}" for mode in (21, 33, 32, 44) for i in range(1, 7)]

MODULES = {
    "hm_qnm": (
        "imrphenomx/hm_qnm.rs",
        "Quasi-normal-mode fits of the higher modes (``LALSimIMRPhenomXHM_qnm.c``): the ringdown\n"
        "and damping frequencies of the (2,1), (3,3), (3,2) and (4,4) modes in units of the remnant\n"
        "mass, and the spheroidal-spherical mixing coefficients of the (3,2) mode.",
        None, [], None, "",
    ),
    "hm_fits_122018": (
        "imrphenomx/hm_fits_122018.rs",
        "Parameter-space fits of the IMRPhenomXHM amplitude of the 122019 release (fits version\n"
        "122018): inspiral and intermediate collocation points and ringdown coefficients.",
        None, ["inspiral_fits", "intermediate_fits", "ringdown_fits"], None,
        '''

_MODES = ("21", "33", "32", "44")


def inspiral_fits(mode_int):
    """The inspiral collocation fits ``iv1``, ``iv2``, ``iv3`` of mode
    ``mode_int`` (index of (2,1), (3,3), (3,2), (4,4); ``pAmp->InspiralAmpFits``)."""
    mode = _MODES[mode_int]
    return tuple(globals()[f"insp_amp_{mode}_iv{i}"] for i in (1, 2, 3))


def intermediate_fits(mode_int):
    """The intermediate fits ``int1``, ``int2``, ``int0``, ``dint0`` of mode
    ``mode_int`` (``pAmp->IntermediateAmpFits``)."""
    mode = _MODES[mode_int]
    return tuple(globals()[f"inter_amp_{mode}_{name}"] for name in ("int1", "int2", "int0", "dint0"))


def ringdown_fits(mode_int):
    """The ringdown fits ``alambda``, ``lambda``, ``sigma`` of mode
    ``mode_int`` (``pAmp->RingdownAmpFits``)."""
    mode = _MODES[mode_int]
    return tuple(globals()[f"rd_amp_{mode}_{name}"] for name in ("alambda", "lambda", "sigma"))
''',
    ),
    "hm_amp_122019": (
        "imrphenomx/hm_amp_122019.rs",
        "Straight-line pieces of the IMRPhenomXHM amplitude of the 122019 release: the TaylorT4\n"
        "``xdot`` coefficients of the (2,1) inspiral, the pseudo-PN coefficients ``rho``, the\n"
        "intermediate root tests and the original ringdown ansatz.",
        [
            "xdot_t4", "p2_discriminant", "p3_q2", "p3_q3", "p3_discri", "p4_q1", "p4_q2", "p4_q3",
            "p4_sqrt_arg", "p4_bignum", "rho1_3", "rho2_3", "rho3_3", "rho1_2", "rho2_2", "rho1_1",
            "xdot5", "xdot6", "xdot65", "xdot7", "xdot75", "xdot8", "xdot8_log", "xdot85",
        ],
        [], [],
        '''

def rd_amp_ansatz0(ff, frd, fda, a0, lambda_, sigma, lc):
    """``IMRPhenomXHM_RD_Amp_Ansatz`` (version 0): the ringdown amplitude
    rescaled by the leading order of the 22, ``fda |a0| sigma exp(-dfr
    lambda / dfd) / (dfr^2 + dfd^2) f^(-lc)``."""
    dfr = ff - frd
    dfd = fda * sigma
    return fda * jnp.abs(a0) * sigma * jnp.exp(-dfr * lambda_ / dfd) / fma(dfr, dfr, dfd * dfd) * jnp.power(ff, -lc)


def rd_amp_dansatz0(ff, frd, fda, a0, lambda_, sigma, lc):
    """``IMRPhenomXHM_RD_Amp_DAnsatz`` (version 0): the derivative of
    :func:`rd_amp_ansatz0`."""
    dfd = fda * sigma
    numerator = jnp.abs(a0) * jnp.power(ff, -1.0 - lc) * fma(
        ff * ff * ff, lambda_,
        fma(ff * ff, fma(-2.0 * lambda_, frd, dfd * (2.0 + lc)),
            fma(dfd * lc, fma(frd, frd, dfd * dfd),
                ff * fma(dfd * dfd, lambda_, fma(frd * frd, lambda_, -(2.0 * dfd * frd * (1.0 + lc)))))))
    denom = fma(ff, ff, fma(-(ff * 2.0), frd, fma(frd, frd, dfd * dfd)))
    expon = jnp.exp(-(ff - frd) * lambda_ / dfd)
    return -numerator * expon / (denom * denom)
''',
    ),
    "hm_inspiral": (
        "imrphenomx/hm_inspiral.rs",
        "Inspiral phase of the IMRPhenomXHM higher modes (``LALSimIMRPhenomXHM_inspiral.c``): the\n"
        "linear-in-frequency correction for extreme mass ratios (version 122019) and ``LambdaPN``.",
        ["phase_21_lambda", "phase_33_lambda", "phase_32_lambda", "phase_44_lambda"],
        [], [],
        '''
import math


def phase_lambda_fit(mode_int, wf):
    """``pPhase->InspiralPhaseFits[modeInt]``: the linear-in-f phase term
    for extreme mass ratios of mode ``mode_int``."""
    mode = ("21", "33", "32", "44")[mode_int]
    return globals()[f"phase_{mode}_lambda"](wf)


def lambda_pn(eta, mode_tag):
    """``IMRPhenomXHM_Insp_Phase_LambdaPN``: the linear-in-f term from the
    complex PN amplitude, Eq. 4.12 of arXiv:2001.10914 (``mode_tag`` is
    ``10 l + m``, a Python int)."""
    pi = POWERS_OF_LALPI.itself
    if mode_tag == 21:
        output = 2.0 * pi * fma(-2.0, LN_2, -0.5)
    elif mode_tag == 33:
        output = 2.0 / 3.0 * pi * fma(6.0, math.log(1.5), -21.0 / 5.0)
    elif mode_tag == 32:
        output = -((2376.0 * pi * fma(22.0, eta, -5.0)) / fma(11880.0, eta, -3960.0))
    elif mode_tag == 44:
        output = 45045.0 * pi * fma(320.0 * fma(3.0, eta, -1.0), LN_2, fma(-1193.0, eta, 336.0)) / (
            2.0 * fma(5405400.0, eta, -1801800.0))
    else:
        output = 0.0
    return -output
''',
    ),
    "hm_intermediate": (
        "imrphenomx/hm_intermediate.rs",
        "Intermediate phase of the IMRPhenomXHM higher modes (``LALSimIMRPhenomXHM_intermediate.c``):\n"
        "the phase-derivative collocation fits ``p1`` ... ``p6`` (version 122019).",
        _PHASE_FITS, [], [],
        '''

def phase_fit(mode_int, i, wf):
    """``pPhase->IntermediatePhaseFits``: the phase-derivative collocation
    fit ``p(i+1)`` of mode ``mode_int``."""
    mode = ("21", "33", "32", "44")[mode_int]
    return globals()[f"phase_{mode}_p{i + 1}"](wf)
''',
    ),
    "hm_ringdown": (
        "imrphenomx/hm_ringdown.rs",
        "Ringdown phase of the IMRPhenomXHM higher modes (``LALSimIMRPhenomXHM_ringdown.c``,\n"
        "version 122019): the rescaled 22 ringdown and the spheroidal (3,2) ringdown.",
        [
            "phase_22_alpha2", "phase_22_alpha_l", "phase_32_spheroidal_time_shift",
            "phase_32_spheroidal_phase_shift", "phase_32_p1", "phase_32_p2", "phase_32_p3", "phase_32_p4",
        ],
        [], [],
        '''

def phase_32_fit(i, wf):
    """``pPhase->RingdownPhaseFits``: the spheroidal phase-derivative
    collocation fit ``p(i+1)`` of the (3,2) mode."""
    return globals()[f"phase_32_p{i + 1}"](wf)
''',
    ),
}
