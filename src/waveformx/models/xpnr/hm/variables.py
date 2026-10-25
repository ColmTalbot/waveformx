# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The variables of one higher mode, ``IMRPhenomXHM_SetHMWaveformVariables``
of ``LALSimIMRPhenomXHM_internals.c`` for ``PhenomXHMReleaseVersion =
122019``, after waveformrs's ``HMWaveformStruct::new`` and
``hm_amp_122019::set_waveform_variables``.

:func:`hm_waveform_struct` returns a :class:`~.._math.Struct` with the
static description of the mode (``ell``, ``emm``, ``mode_tag``,
``mode_int``, ``mixing_on``: Python values) and the traced variables
(frequencies in units of the total mass). The flags of the release that
depend on the binary (``amp_emr``, ``use_famp_pn``, the vetoes) are traced
booleans, or Python ``False`` for the modes that never use them.
"""

from __future__ import annotations

import jax.numpy as jnp

from .._generated import hm_qnm, utilities
from .._math import PI, Struct, fnms, real

__all__ = ["HIGHER_MODES", "hm_waveform_struct", "qnm_fits", "mixing_coefficients", "is_mode", "both", "where_mode"]

# the higher modes of IMRPhenomXHM in the order of LAL's modeInt
HIGHER_MODES = ((2, 1), (3, 3), (3, 2), (4, 4))

# eta of q = 70 (AmpEMR), and the limit between comparable and extreme mass
# ratios of the phase
ETA_AMP_EMR = 0.013886133703630232
ETA_EMR = 0.05


def is_mode(hm, *tags):
    """Whether the mode of ``hm`` (a :func:`hm_waveform_struct`, or a view
    of one) is one of ``tags`` (``10 l + m``): a Python bool for a mode
    known when tracing, a traced bool for a mode that is one of several
    (``hm.tags``, the modes computed together under ``jax.vmap``, see
    :func:`.mode.modes_coefficients`)."""
    possible = hm.tags if "tags" in hm else (hm.mode_tag,)
    inside = [tag for tag in possible if tag in tags]
    if len(inside) == len(possible):
        return True
    if not inside:
        return False
    hit = hm.mode_tag == inside[0]
    for tag in inside[1:]:
        hit = hit | (hm.mode_tag == tag)
    return hit


def both(condition, value):
    """``condition & value`` for a condition of :func:`is_mode` (``value``
    or ``False`` where it is a Python bool)."""
    if condition is True:
        return value
    if condition is False:
        return False
    return condition & value


def where_mode(condition, a, b):
    """``a`` where the condition of :func:`is_mode` holds, else ``b`` (a
    choice while tracing where it is a Python bool)."""
    if condition is True:
        return a
    if condition is False:
        return b
    return jnp.where(condition, a, b)


def qnm_fits(ell, emm):
    """The ringdown and damping frequency fits ``(fring, fdamp)`` of mode
    ``(ell, emm)`` as functions of the final spin, in units of the remnant
    mass (``evaluate_QNMfit_fring21`` ...; :mod:`.._generated.hm_qnm`)."""
    tag = 10 * ell + emm
    return getattr(hm_qnm, f"fring{tag}"), getattr(hm_qnm, f"fdamp{tag}")


def mixing_coefficients(af):
    """``IMRPhenomXHM_Initialize_MixingCoeffs``: the spheroidal-spherical
    mixing coefficients ``a222``, ``a223``, ``-a322``, ``-a323`` of the (3,2)
    mode for the final spin ``af`` (Berti & Klein 2014), each as ``(re,
    im)``, with the signs of the (3,2) ones flipped to the conventions of
    the hybrids (in double precision, where LAL stores them as
    ``COMPLEX8``)."""
    def value(x):
        return jnp.asarray(x, real())

    return (
        (value(hm_qnm.re_l2m2lp2(af)), value(hm_qnm.im_l2m2lp2(af))),
        (value(hm_qnm.re_l2m2lp3(af)), value(hm_qnm.im_l2m2lp3(af))),
        (-value(hm_qnm.re_l3m2lp2(af)), -value(hm_qnm.im_l3m2lp2(af))),
        (-value(hm_qnm.re_l3m2lp3(af)), -value(hm_qnm.im_l3m2lp3(af))),
    )


def hm_waveform_struct(wf, ell, emm):
    """``IMRPhenomXHM_SetHMWaveformVariables`` for mode ``(ell, emm)``,
    ``emm > 0`` (Python ints), from the waveform variables ``wf``, with the
    flags of release 122019.

    With the tuned co-precessing model of IMRPhenomXPNR (``wf.coprec.tuned``,
    possibly traced) the ringdown and damping frequencies use the precessing
    final spin ``wf.a_final_prec`` and the ringdown frequency is shifted by
    ``-emm wf.coprec.f_ring_eff_shift_divided_by_emm``; the mixing
    coefficients always use ``wf.a_final``."""
    if (ell, emm) not in HIGHER_MODES:
        raise ValueError(f"IMRPhenomXHM: mode ({ell},{emm}) is not a higher mode of the model")
    mode_tag = 10 * ell + emm
    mode_int = HIGHER_MODES.index((ell, emm))
    mixing_on = mode_tag == 32
    odd = mode_tag in (21, 33)
    if odd:
        amp_zero = (wf.q == 1.0) & (wf.chi1l == wf.chi2l)
    else:
        amp_zero = jnp.asarray(False)

    fring, fdamp = qnm_fits(ell, emm)
    f_ring_prec = fnms(float(emm), wf.coprec.f_ring_eff_shift_divided_by_emm, fring(wf.a_final_prec) / wf.m_final)
    f_damp_prec = fdamp(wf.a_final_prec) / wf.m_final
    f_ring = jnp.where(wf.coprec.tuned, f_ring_prec, fring(wf.a_final) / wf.m_final)
    f_damp = jnp.where(wf.coprec.tuned, f_damp_prec, fdamp(wf.a_final) / wf.m_final)

    psi4tostrain = utilities.psi4_to_strain(wf.eta, wf.s_tot_r, wf.dchi)

    # flags of release 122019 (IMRPhenomXHM_SetHMWaveformVariables)
    amp_emr = (wf.eta < ETA_AMP_EMR) & (wf.chi1l <= 0.9)
    inspiral_amp_veto = False
    intermediate_amp_veto = False
    ringdown_amp_veto = False
    use_famp_pn = False
    if mode_tag == 21:
        inspiral_amp_veto = wf.q < 8.0
        intermediate_amp_veto = inspiral_amp_veto
        ringdown_amp_veto = True
        # the TaylorT4 inspiral for q <~ 40 (LAL's comment says the opposite)
        use_famp_pn = wf.eta >= 0.0237954
    elif mode_tag == 32:
        ringdown_amp_veto = True

    return Struct(
        ell=ell,
        emm=emm,
        mode_tag=mode_tag,
        mode_int=mode_int,
        mixing_on=mixing_on,
        amp_zero=amp_zero,
        n_colloc_pts_inter_phase=6 if mixing_on else 5,
        eta_emr=ETA_EMR,
        chi_s=(wf.chi1l + wf.chi2l) * 0.5,
        chi_a=(wf.chi1l - wf.chi2l) * 0.5,
        f_meco_lm=wf.f_meco * float(emm) * 0.5,
        f_ring=f_ring,
        f_damp=f_damp,
        timeshift=0.0,
        phaseshift=0.0,
        phiref22=0.0,
        mixing_coeffs=mixing_coefficients(wf.a_final) if mixing_on else None,
        amp_norm=wf.amp_norm,
        amp0=wf.amp0,
        delta_t=-2.0 * PI * (500.0 + psi4tostrain),
        amp_emr=amp_emr,
        use_famp_pn=use_famp_pn,
        inspiral_amp_veto=inspiral_amp_veto,
        intermediate_amp_veto=intermediate_amp_veto,
        ringdown_amp_veto=ringdown_amp_veto,
    )
