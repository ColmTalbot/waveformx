# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The PNR precession of IMRPhenomXPNR (LALSimulation 6.2.0, optionally with
the fix of lalsuite !2593), after waveformrs's ``imrphenomx/pnr.rs``:
``IMRPhenomXGetAndSetPrecessionVariables`` with ``PhenomXPrecVersion`` 330
and ``PhenomXPFinalSpinMod`` 7, ``LALSimIMRPhenomX_PNR.c``,
``_PNR_internals.c``, ``_PNR_alpha.c``, ``_PNR_beta.c`` and the SpinTaylor
angles of version 330.

LAL changes the prescription of the Euler angles with the binary: version
330 (SpinTaylor angles, the default), 223 (MSA, when the SpinTaylor
integration fails) or 102 (NNLO, for in-plane spins below ``1e-6`` or when
the MSA system fails), and below ``1e-7`` it switches the PNR machinery off.
Here each prescription is a function of a *static* ``version`` (the
arguments called ``version`` below are Python ints), or of a
:class:`Fallback` (versions 102 and 223 together, chosen by a traced flag),
and the model (:mod:`.model`) chooses among them with
:func:`._transforms.switch`.

The set-up is split into :func:`common_setup` (before the SpinTaylor
integration, which decides whether version 330 can be used) and
:func:`finish_setup` (for a given version).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as _jax_core

from . import msa as _msa
from .msa import acos_safe
from . import nnlo as _nnlo
from . import pnr_angles as pa
from . import precession as _prec
from ._generated import pnr_fits as _pnr_fits
from ._generated import qnm as _qnm
from ._generated import utilities as _utilities
from ._math import (
    C_SI, FRAC_PI_2, FRAC_PI_4, G_SI, MTSUN_SI, PI, FencedModule, Struct, barrier, blocked_map, blocked_map_extra, blocked_scan, chunked_map, fenced, fma, fnms, hz_to_mf, mf_to_hz, powers,
    rotate_y, rotate_z, safe_div, safe_sqrt,
)
from .hm import qnm as _hm_qnm
from ._transforms import cond
from .spintaylor import angles as st_angles

__all__ = ["common_setup", "finish_setup"]

pnr_fits = FencedModule(_pnr_fits)
qnm = FencedModule(_qnm)
utilities = FencedModule(_utilities)
hm_qnm = FencedModule(_hm_qnm)

MAX_TOL_ATAN = 1.0e-15


def _copy(struct):
    return jax.tree_util.tree_map(lambda x: x, struct)


def hypot2(x, y):
    """``sqrt(x^2 + y^2)`` (with a finite derivative at zero)."""
    return safe_sqrt(fma(x, x, y * y), 1e-300)


def chi_in_plane(chi1, chi2):
    return safe_sqrt(fma(chi2[1], chi2[1], fma(chi2[0], chi2[0], fma(chi1[0], chi1[0], chi1[1] * chi1[1]))), 1e-300)


def atan2_tol(a, b, tol=MAX_TOL_ATAN):
    """``XLALSimIMRPhenomXatan2tol``: ``atan2(a, b)``, zero when both are
    below ``tol``."""
    small = (jnp.abs(a) < tol) & (jnp.abs(b) < tol)
    return jnp.where(small, 0.0, jnp.arctan2(jnp.where(small, 0.0, a), jnp.where(small, 1.0, b)))


# ---------------------------------------------------------------------------
# remnant and ringdown frequencies
# ---------------------------------------------------------------------------


def ringdown_frequency(ell, emm, wf):
    """``IMRPhenomXHM_GenerateRingdownFrequency``: the ringdown frequency of
    mode ``(ell, emm)`` with the effective shift of the tuned co-precessing
    model (``ell``, ``emm`` static, or traced with ``ell == emm``)."""
    if not (isinstance(ell, int) and isinstance(emm, int)):
        return jnp.select(
            [(ell == 2) & (emm == 2), (ell == 3) & (emm == 3)],
            [ringdown_frequency(2, 2, wf), ringdown_frequency(3, 3, wf)],
            ringdown_frequency(4, 4, wf),
        )
    tuned = wf.coprec.tuned
    afinal = jnp.where(tuned, wf.a_final_prec, wf.a_final)
    if (ell, emm) == (2, 2):
        return wf.f_ring
    fring = {
        (2, 1): hm_qnm.fring21, (3, 3): hm_qnm.fring33, (3, 2): hm_qnm.fring32, (4, 4): hm_qnm.fring44,
    }[(ell, emm)]
    f_ring = fring(afinal) / wf.m_final
    return jnp.where(tuned, fnms(float(emm), wf.coprec.f_ring_eff_shift_divided_by_emm, f_ring), f_ring)


# ---------------------------------------------------------------------------
# the common set-up
# ---------------------------------------------------------------------------


def mode_range(modes):
    """``(M_MIN, M_MAX, L_MAX_PNR)`` of a co-precessing mode array (static)."""
    if (4, 4) in modes:
        l_max = 4
    elif (3, 3) in modes or (3, 2) in modes:
        l_max = 3
    else:
        l_max = 2
    m_min = min([4] + [m for _, m in modes])
    m_max = max([1] + [m for _, m in modes])
    return m_min, m_max, l_max


def common_setup(wf, m1_si, m2_si, chi1, chi2, mode_range_, parity_fix, extended=False, smooth=False, direct=True,
                 phase_grid=False):
    """The part of ``IMRPhenomXGetAndSetPrecessionVariables`` before the
    SpinTaylor integration: the spin quantities, the decisions on the
    prescription that do not need the integration, and the start of the
    integration that the PNR angle interpolants need.

    ``m1_si``, ``m2_si`` (kg) and the spins (3-vectors) are ordered as
    ``XLALIMRPhenomXPCheckMassesAndSpins`` orders them; ``mode_range_`` is
    :func:`mode_range` of the co-precessing modes (static, or traced with
    ``M_MAX = L_MAX_PNR``, as for the J-frame modes). ``extended`` and
    ``smooth`` (static) select the extended SpinTaylor angles and their
    continuous variant, and for it ``direct`` and ``phase_grid`` (static)
    the direct SpinTaylor angles and the phase grid (see
    :func:`.model._decide`).
    Returns ``(prec, wf)``; ``wf`` gets the tuned co-precessing flag as LAL
    sets it.
    """
    wf = _copy(wf)
    chi1x, chi1y, chi1z = chi1
    chi2x, chi2y, chi2z = chi2
    chip_ = chi_in_plane(chi1, chi2)
    # IMRPhenomXSetWaveformVariables reads PNRUseTunedCoprec
    wf.coprec = wf.coprec.replace(tuned=jnp.asarray(True))

    m1 = m1_si / wf.m_tot_si
    m2 = m2_si / wf.m_tot_si
    big_m = m1 + m2
    m1_2 = m1 * m1
    m2_2 = m2 * m2
    q = m1 / m2
    eta = wf.eta
    pi_gm = PI * (m1_si + m2_si) * (G_SI / C_SI) / (C_SI * C_SI)
    chi1_norm = safe_sqrt(fma(chi1z, chi1z, fma(chi1x, chi1x, chi1y * chi1y)), 1e-300)
    chi2_norm = safe_sqrt(fma(chi2z, chi2z, fma(chi2x, chi2x, chi2y * chi2y)), 1e-300)
    s1x = chi1x * m1_2
    s1y = chi1y * m1_2
    s2x = chi2x * m2_2
    s2y = chi2y * m2_2
    s_tot_perp = hypot2(s1x + s2x, s1y + s2y)
    chi_tot_perp = s_tot_perp * (big_m * big_m) / m1_2
    s1_perp = m1_2 * hypot2(chi1x, chi1y)
    s2_perp = m2_2 * hypot2(chi2x, chi2y)
    a1 = 2.0 + (3.0 * m2) / (2.0 * m1)
    a2 = 2.0 + (3.0 * m1) / (2.0 * m2)
    asp1 = a1 * s1_perp
    asp2 = a2 * s2_perp
    num = jnp.where(asp2 > asp1, asp2, asp1)
    den = jnp.where(m2 > m1, a2 * m2_2, a1 * m1_2)
    chip = num / den
    m_min, m_max, l_max = mode_range_

    prec = Struct(
        chi_in_plane=chip_,
        # (the extended angles keep the PNR machinery and the SpinTaylor
        # angles for all in-plane spins)
        pnr_on=jnp.asarray(True) if extended else chip_ >= 1e-7,
        nnlo_from_start=jnp.asarray(False) if extended else chip_ < 1e-6,
        m1=m1, m2=m2, m1_2=m1_2, m2_2=m2_2, q=q,
        pi_gm=pi_gm,
        eta=eta,
        chi1=list(chi1),
        chi2=list(chi2),
        chi1_norm=chi1_norm,
        chi2_norm=chi2_norm,
        chi1_perp=hypot2(chi1x, chi1y),
        chi2_perp=hypot2(chi2x, chi2y),
        s1_perp=s1_perp,
        s2_perp=s2_perp,
        chi_tot_perp=chi_tot_perp,
        chi_p=chip,
        sl=fma(chi1z, m1_2, chi2z * m2_2),
        sperp=chip * m1_2,
        chi1_evolved=list(chi1),
        chi2_evolved=list(chi2),
        pnr_inspiral_scaling=jnp.asarray(False),
        pnr_hm_mflow=jnp.zeros(()),
        pnr_hm_mfhigh=jnp.zeros(()),
        parity_fix=parity_fix,
        extended=extended,
        smooth=smooth,
        direct_angles=bool(smooth and direct),
        phase_grid=bool(smooth and phase_grid),
        pole=jnp.asarray(1.0 - 1e-15 if smooth else 1.0),
        all_two_spin=jnp.asarray(smooth),
        transition_q=jnp.asarray(pa.PNR_Q_WINDOW_UPPER + (Q_SMOOTH_WIDTH if smooth else 0.0)),
        q12_weight=jnp.zeros(()),
        scaling_weight=jnp.zeros(()),
        error_spin=(chi1_norm > 1.0) | (chi2_norm > 1.0),
    )
    prec.l = _prec.three_pn(eta, wf.delta, chi1z, chi2z)
    # ---- the start of the SpinTaylor integration (version 330, with the
    # tuned angles) ----
    wf.delta_mf = jnp.where(wf.delta_f == 0.0, _prec.delta_mf_from_wfstruct(wf), wf.delta_mf)
    flow = wf.f_min
    if True:
        flow = jnp.where(wf.delta_f != 0.0, jnp.floor(wf.f_min / jnp.where(wf.delta_f != 0.0, wf.delta_f, 1.0)) * wf.delta_f, flow)
        fmin_hm_inspiral = flow * 2.0 / m_max
        # the PNR variables and the alpha and beta coefficients with a fake
        # version 223 to catch edge cases
        tmp = _copy(prec)
        set_pnr_variables(tmp, wf, 223)
        alpha_params = precompute_alpha_coefficients(tmp, wf, 223)
        beta_params = pa.beta_connection_frequencies(precompute_beta_coefficients(tmp, wf, 223), _soft(prec))
        mf_alpha_upper = alpha_params.a4 / 3.0
        mf_low_cut = (3.0 / 3.5) * mf_alpha_upper
        mf_high_cut = beta_params.mf_beta_lower
        mf_high_cut = jnp.where((mf_high_cut > wf.f_cut_def) | (mf_high_cut < 0.1 * wf.f_ring), wf.f_ring, mf_high_cut)
        mf_low_cut = jnp.where((mf_low_cut > wf.f_cut_def) | (mf_high_cut < mf_low_cut), mf_high_cut / 2.0, mf_low_cut)
        flow_alpha = mf_to_hz(mf_low_cut * 0.65 * m_max / 2.0, wf.m_tot)
        mf_rd_22 = wf.f_ring
        mf_rd_lm = ringdown_frequency(l_max, m_max, wf)
        fmin_hm_ringdown = mf_to_hz(hz_to_mf(flow, wf.m_tot) - (mf_rd_lm - mf_rd_22), wf.m_tot)
        flow_mapped = jnp.where((fmin_hm_ringdown < fmin_hm_inspiral) & (fmin_hm_ringdown > 0.0), fmin_hm_ringdown,
                                fmin_hm_inspiral)
        flow = jnp.where(flow_alpha < flow, fmin_hm_inspiral / 1.5, flow_mapped)
        delta_f_int = interpolation_delta_f(tmp, wf, flow, 330)
        integration_buffer = 1.4 * delta_f_int
        flow = jnp.where(fnms(2.0, delta_f_int, flow) < 0.0, flow / 2.0, fnms(2.0, delta_f_int, flow))
        flow = jnp.floor(flow / delta_f_int) * delta_f_int
    prec.integration_buffer = integration_buffer
    prec.st_flow = flow
    prec.st_flow_ok = ~jnp.isnan(flow) & (flow > 0.0)
    # the end of the integration: fRING + 8 fDAMP of the aligned-spin struct
    prec.st_fcut = mf_to_hz(wf.f_ring + 8.0 * wf.f_damp, wf.m_tot)
    prec.m_min, prec.m_max, prec.l_max_pnr = m_min, m_max, l_max
    return prec, wf


def spintaylor_solution(prec, wf, m1_si, m2_si, flow=None, fcut=None, active=True):
    """``IMRPhenomX_InspiralAngles_SpinTaylor`` from ``flow`` (default the
    start computed by :func:`common_setup`) to ``fcut`` (default ``(fRING +
    8 fDAMP) / M`` of the aligned-spin struct: LAL reads the struct as it is
    at the call, which the PNR set-up has changed when it integrates
    again)."""
    if flow is None:
        flow = prec.st_flow
    if fcut is None:
        fcut = prec.st_fcut
    flow_safe = jnp.where(prec.st_flow_ok, flow, wf.f_min)
    sol = st_angles.pn_solution(m1_si, m2_si, prec.chi1, prec.chi2, wf.f_ref, flow_safe, fcut, active,
                                extended=prec.extended, smooth=prec.smooth)
    sol.failed = sol.failed | ~prec.st_flow_ok
    return sol


# ---------------------------------------------------------------------------
# the set-up for a given version
# ---------------------------------------------------------------------------


class Fallback:
    """Versions 102 and 223 of the PNR machinery as one: 223 (MSA angles)
    where the traced ``msa`` is true, 102 (NNLO angles) elsewhere. The
    functions below take it in place of a static ``version``: they compute
    what the two versions compute differently both ways and select (the
    versions LAL falls back to share their code, which halves what is
    compiled for them). It compares unequal to 330 and refuses the other
    comparisons, which must go through :func:`_by_version`."""

    def __init__(self, msa):
        self.msa = msa

    def __eq__(self, other):
        if other == 330:
            return False
        raise TypeError("a Fallback version is compared through _by_version")

    def __ne__(self, other):
        return not self == other

    __hash__ = object.__hash__

    def __ge__(self, other):
        raise TypeError("a Fallback version is compared through _by_version")

    __gt__ = __le__ = __lt__ = __floordiv__ = __ge__


def _by_version(version, nnlo, msa, other=None):
    """``nnlo()`` for version 102, ``msa()`` for 223, ``other()`` for 330;
    for a :class:`Fallback`, both of the first two, selected leaf by leaf."""
    if isinstance(version, Fallback):
        a, b = msa(), nnlo()
        return jax.tree_util.tree_map(lambda x, y: jnp.where(version.msa, x, y), a, b)
    if version == 102:
        return nnlo()
    if version == 223:
        return msa()
    return other()


def _at_least_200(version):
    """``version >= 200`` (traced for a :class:`Fallback`)."""
    return version.msa if isinstance(version, Fallback) else version >= 200


def _soft(prec):
    """The half-width of the smooth connection of the PNR ``beta``
    (:func:`.pnr_angles.beta_connection_frequencies`), ``None`` for LAL's."""
    return pa.BETA_SOFT_WIDTH if prec.smooth else None


def two_spin(prec, version):
    """``IMRPhenomX_PNR_CheckTwoSpin``; for the smooth angles
    (``prec.all_two_spin``), the two-spin construction for every binary:
    LAL's changes the PNR ``beta`` by up to a mismatch of ~1e-1 at ``|chi2| =
    1e-3``, a step that does not vanish with the second spin."""
    lal = (prec.chi1_norm != 0.0) & (prec.chi2_norm >= 1.0e-3)
    return (prec.all_two_spin | lal) & _at_least_200(version)


def _off_poles(prec, costheta):
    """``costheta`` kept off ``+-1`` for the smooth angles: the fits of the
    single-spin system and the PN waveform ``beta`` take ``sqrt(1 -
    costheta^2)`` or ``acos(costheta)``, whose derivatives are infinite for
    aligned spins (a change of their sine below ``5e-8``). ``prec.pole``
    (traced) is 1, or ``1 - 1e-15`` for the smooth angles."""
    return jnp.where(prec.pole < 1.0, jnp.clip(costheta, -prec.pole, prec.pole), costheta)


def set_pnr_variables(prec, wf, version):
    """``IMRPhenomX_PNR_GetAndSetPNRVariables`` (in place)."""
    m1 = wf.m1 * wf.m_tot
    m2 = wf.m2 * wf.m_tot
    q = wf.q
    chi1 = prec.chi1_evolved if version == 330 else prec.chi1
    chi2 = prec.chi2_evolved if version == 330 else prec.chi2
    chi1x, chi1y, chi1z = chi1
    chi2x, chi2y, chi2z = chi2
    chieff = utilities.chi_eff(wf.eta, chi1z, chi2z)
    chipar = wf.m_tot * chieff / m1
    m1sq = m1 * m1
    m2sq = m2 * m2
    chis = hypot2(fma(m1sq, chi1x, m2sq * chi2x), fma(m1sq, chi1y, m2sq * chi2y)) / (m1 * m1)
    s = jnp.sin((q - 1.0) * PI)
    c = jnp.cos((q - 1.0) * PI)
    if version == 330:
        chiperp = chis
    else:
        chiperp = jnp.where(q <= 1.5, fma(s * s, prec.chi_p, c * c * chis), prec.chi_p)
    antisymmetric_chis = hypot2(fma(m1sq, chi1x, -(m2sq * chi2x)), fma(m1sq, chi1y, -(m2sq * chi2y))) / (m1 * m1)
    chiperp_antisymmetric = jnp.where(q <= 1.5, fma(s * s, prec.chi_p, c * c * antisymmetric_chis), prec.chi_p)
    chi_mag = hypot2(chipar, chiperp)
    prec.chi_single_spin = chi_mag
    chi_mag_antisymmetric = hypot2(chipar, chiperp_antisymmetric)
    prec.chi_single_spin_antisymmetric = chi_mag_antisymmetric
    costheta = jnp.where(chi_mag >= 1.0e-6, safe_div(chipar, chi_mag), 0.0)
    prec.theta_antisymmetric = jnp.where(
        chi_mag_antisymmetric >= 1.0e-6, acos_safe(safe_div(chipar, chi_mag_antisymmetric)), 0.0
    )
    prec.costheta_single_spin = _off_poles(prec, costheta)
    xfparr = utilities.final_spin_2017(wf.eta, chi_mag * costheta, 0.0)
    qfactor = q / (1.0 + q)
    xfperp = qfactor * qfactor * chi_mag * pa.sqrt_one_minus_square(costheta)
    xf = hypot2(xfparr, xfperp)
    prec.costheta_final_single_spin = jnp.where(xf > 1.0e-6, safe_div(xfparr, xf), 0.0)
    prec.pnr_hm_mflow = jnp.zeros(())
    prec.pnr_hm_mfhigh = jnp.zeros(())
    scaling = (q > pa.PNR_Q_WINDOW_UPPER) | (prec.chi_single_spin > pa.PNR_CHI_WINDOW_UPPER)
    if prec.smooth:
        # beyond q = 12, from LAL's angles at the edge of the transition
        # region to LAL's beyond it (see merged_beta_at, frequency_map)
        x = jnp.clip((q - pa.PNR_Q_WINDOW_UPPER) / Q_SMOOTH_WIDTH, 0.0, 1.0)
        prec.q12_weight = x * x * x * fma(x, fma(6.0, x, -15.0), 10.0)
        weight = jnp.where(prec.chi_single_spin > pa.PNR_CHI_WINDOW_UPPER, 1.0, prec.q12_weight)
    else:
        weight = jnp.where(scaling, 1.0, 0.0)
    prec.scaling_weight = weight
    prec.pnr_inspiral_scaling = weight >= 1.0
    return prec


@fenced
def set_coprec_params(prec, wf, tuned, version):
    """``IMRPhenomX_PNR_GetAndSetCoPrecParams`` (``wf.coprec`` in place);
    ``tuned`` is traced (the PNR machinery is on)."""
    a1 = prec.chi_single_spin
    theta_ls = acos_safe(prec.costheta_single_spin)
    pnr_window = jnp.where(tuned, pa.coprec_window(wf.q), 0.0)
    apply = jnp.where(tuned, 1.0, 0.0)
    dev = a1 * jnp.sin(theta_ls) * apply
    dev = jnp.where(tuned, dev * pnr_window, dev)
    if version == 330:
        coprec_eta = jnp.where(wf.eta >= 0.09876, wf.eta, fma(-(0.09876 - wf.eta), 0.1641, 0.09876))
        coprec_a1 = jnp.where(a1 <= 0.8, a1, 0.8 + (a1 - 0.8) / 12.0)
    else:
        coprec_eta = jnp.where(wf.eta >= 0.09876, wf.eta, 0.09876)
        coprec_a1 = jnp.where(a1 <= 0.8, a1, 0.8)
    coprec_a1 = jnp.where(coprec_a1 >= 0.2, coprec_a1, 0.2)
    e, a = coprec_eta, coprec_a1
    zero = jnp.zeros(())
    fit = lambda f: jnp.where(tuned, f(theta_ls, e, a), zero)  # noqa: E731
    wf.coprec = Struct(
        dev=dev,
        mu1=fit(pnr_fits.mu1_l2m2),
        mu2=fit(pnr_fits.mu2_l2m2),
        mu3=fit(pnr_fits.mu3_l2m2),
        nu4=fit(pnr_fits.nu4_l2m2),
        nu5=fit(pnr_fits.nu5_l2m2),
        nu6=fit(pnr_fits.nu6_l2m2),
        zeta1=fit(pnr_fits.zeta1_l2m2),
        zeta2=fit(pnr_fits.zeta2_l2m2),
        tuned=jnp.asarray(tuned),
        pnr_window=pnr_window,
        f_ring_eff_shift_divided_by_emm=zero,
    )
    return wf


def ringdown_pnr_beta(prec, wf):
    """``IMRPhenomX_PNR_GenerateRingdownPNRBeta``."""
    return pa.arctan_window(
        acos_safe(prec.costheta_final_single_spin)
        - pnr_fits.beta_bf(wf.eta, prec.chi_single_spin, prec.costheta_single_spin)
    )


@fenced
def set_remnant(prec, wf):
    """``IMRPhenomX_SetPrecessingRemnantParams`` with
    ``PhenomXPFinalSpinMod`` 7 (``wf`` in place)."""
    tuned = wf.coprec.tuned
    prec.beta_rd = jnp.where(tuned, ringdown_pnr_beta(prec, wf), 0.0)
    sign = jnp.where(jnp.cos(prec.beta_rd) < 0, -1.0, 1.0)
    afinal_prec = _prec.precessing_final_spin_2017(wf.eta, prec.chi1[2], prec.chi2[2], prec.chi_tot_perp)
    wf.a_final_prec = sign * jnp.abs(afinal_prec)
    window = wf.coprec.pnr_window
    wf.a_final = jnp.where(tuned, fma(window, wf.a_final_nonprec, (1.0 - window) * wf.a_final_prec), wf.a_final_prec)
    wf.a_final = jnp.where(jnp.abs(wf.a_final) > 1.0, jnp.sign(wf.a_final), wf.a_final)
    wf.a_final_prec = jnp.where(jnp.abs(wf.a_final_prec) > 1.0, jnp.sign(wf.a_final_prec), wf.a_final_prec)
    f_ring = qnm.fring22(wf.a_final) / wf.m_final
    f_damp = qnm.fdamp22(wf.a_final) / wf.m_final
    c = wf.coprec
    # APPLY_PNR_DEVIATIONS and the effective ringdown frequency
    f_ring_tuned = fnms(c.dev, c.nu5, f_ring)
    f_damp_tuned = fma(c.dev, c.nu6, f_damp)
    f_ring22_prec = qnm.fring22(wf.a_final_prec) / wf.m_final
    f_ring21_prec = hm_qnm.fring21(wf.a_final_prec) / wf.m_final
    shift = (1.0 - jnp.abs(jnp.cos(prec.beta_rd))) * (f_ring22_prec - f_ring21_prec)
    wf.coprec = c.replace(f_ring_eff_shift_divided_by_emm=jnp.where(tuned, shift, 0.0))
    wf.f_ring = jnp.where(tuned, fnms((1.0 - window) * 2.0, shift, f_ring_tuned), f_ring)
    wf.f_damp = jnp.where(tuned, f_damp_tuned, f_damp)
    return prec, wf


#: Half-widths of the bands in ``beta_min`` and ``beta_min / beta_max``
#: across which the smooth angles refine the grid of the PNR interpolants
#: (:func:`interpolation_delta_f`).
GRID_SOFT_BETA = 0.002
GRID_SOFT_RATIO = 0.05


@fenced
def interpolation_delta_f(prec, wf, f_min, version):
    """``IMRPhenomX_PNR_HMInterpolationDeltaF``: the spacing (Hz) of the
    uniform grid of the PNR angle interpolants."""
    error_tolerance = 0.01
    mf = hz_to_mf(f_min, wf.m_tot)
    eta_term = jnp.sqrt(fnms(4.0, wf.eta, 1.0))
    numerator = 3.0 * PI * mf * mf * mf * mf * mf * error_tolerance * (1.0 + eta_term)
    denominator = fma(13.0, eta_term, 7.0)
    constant = 4.0 * jnp.sqrt(2.0 / 5.0)
    delta_f_alpha = mf_to_hz(constant * jnp.sqrt(jnp.sqrt(numerator / denominator)), wf.m_tot)
    plain = jnp.maximum(delta_f_alpha, 1e-2)

    # two-spin oscillations: sample the period of the spin magnitude
    def two_spin_period(system, l_coeffs):
        g0, deltam, psi1, psi2 = _msa.psi_terms(system)
        v0 = jnp.cbrt(PI * mf)
        v02 = v0 * v0
        v06 = v02 * v02 * v02
        dpsi = g0 * deltam * PI / (4.0 * v06) * fma(psi2, v02, fma(2.0 * psi1, v0, 3.0))
        dpsi_inv = jnp.abs(safe_div(1.0, dpsi))
        l_fmin = wf.m_tot * wf.m_tot * _prec.l_ansatz(l_coeffs, v0, wf.eta / v0)
        beta_min = jnp.arctan2(jnp.abs(prec.s1_perp - prec.s2_perp), l_fmin + prec.sl)
        beta_max = jnp.arctan2(prec.s1_perp + prec.s2_perp, l_fmin + prec.sl)
        ratio = safe_div(beta_min, beta_max)
        if prec.smooth:
            # LAL samples four times more finely where L comes close to J
            # (the grid, and the waveform, jump there): the smooth angles
            # refine by a factor that goes from 1 to 4 across bands around
            # LAL's thresholds. (For vanishing in-plane spins, the ratio is
            # its limit, whose derivative does not divide by beta_max^2.)
            tiny = beta_max < 1e-100
            limit = safe_div(jnp.abs(prec.s1_perp - prec.s2_perp), prec.s1_perp + prec.s2_perp)
            ratio = jnp.where(tiny, limit, safe_div(beta_min, jnp.where(tiny, 1.0, beta_max)))
            w = pa._ramp(0.01 - beta_min, GRID_SOFT_BETA) * pa._ramp(0.55 - ratio, GRID_SOFT_RATIO)
            dpsi_inv = dpsi_inv / fma(3.0, w, 1.0)
        else:
            dpsi_inv = jnp.where((beta_min < 0.01) & (ratio < 0.55), dpsi_inv / 4.0, dpsi_inv)
        delta_f_twospin = mf_to_hz(dpsi_inv / 4.0, wf.m_tot)
        use_twospin = two_spin(prec, version) & (delta_f_twospin < delta_f_alpha) & ~jnp.isnan(dpsi)
        return jnp.where(use_twospin, jnp.maximum(delta_f_twospin, 1e-2), plain)

    if not isinstance(version, Fallback) and version // 100 == 3:
        result = two_spin_period(_msa.system(msa_inputs(prec, wf, prec.chi1, prec.chi2)),
                                 _prec.three_pn(prec.eta, wf.delta, prec.chi1[2], prec.chi2[2]))
    else:
        result = _by_version(version, lambda: plain, lambda: two_spin_period(prec.msa, prec.l))
    aligned = (prec.chi1_perp == 0.0) & (prec.chi2_perp == 0.0)
    return jnp.where(aligned, jnp.where(wf.delta_f != 0.0, wf.delta_f, 0.1), result)


def msa_inputs(prec, wf, chi1, chi2):
    return Struct(eta=wf.eta, m1=wf.m1, m2=wf.m2, m1_2=prec.m1_2, m2_2=prec.m2_2, chi1=list(chi1),
                  chi2=list(chi2), pi_gm=prec.pi_gm, f_ref=wf.f_ref)


def fit_parameters(prec, wf, version):
    """The single-spin ``eta`` and ``chi`` of the MR fits."""
    if version == 330:
        eta = jnp.where(wf.eta >= 0.09, prec.eta, 0.09)
        chiboundary = fma(-0.20, jnp.exp(-jnp.power((wf.q - 6.0) / 1.5, 8.0)), 0.80)
        chi = jnp.where(prec.chi_single_spin <= chiboundary, prec.chi_single_spin, chiboundary)
        return eta, chi
    return wf.eta, prec.chi_single_spin


@fenced
def precompute_alpha_coefficients(prec, wf, version):
    """``IMRPhenomX_PNR_precompute_alpha_coefficients``."""
    eta, chi = fit_parameters(prec, wf, version)
    costheta = prec.costheta_single_spin
    a2 = jnp.minimum(pnr_fits.alpha_a2(eta, chi, costheta), 0.0)
    a3 = jnp.maximum(jnp.abs(pnr_fits.alpha_a3(eta, chi, costheta)), 0.00001)
    max_a2 = PI * PI * jnp.sqrt(a3)
    a2 = jnp.where(a2 < -max_a2, -max_a2, a2)
    return pa.alpha_params(
        a1=jnp.abs(pnr_fits.alpha_a1(eta, chi, costheta)), a2=a2, a3=a3, a4=pnr_fits.alpha_a4(eta, chi, costheta)
    )


@fenced
def precompute_beta_coefficients(prec, wf, version):
    """``IMRPhenomX_PNR_precompute_beta_coefficients``."""
    eta, chi = fit_parameters(prec, wf, version)
    costheta = prec.costheta_single_spin
    costhetaf = prec.costheta_final_single_spin
    b4 = jnp.maximum(pnr_fits.beta_b4(eta, chi, costheta), 175.0)
    b2 = pnr_fits.beta_b2(eta, chi, costheta)
    return pa.beta_params(
        b0=acos_safe(costhetaf) - pnr_fits.beta_b0(eta, chi, costheta),
        b1=pnr_fits.beta_b1(eta, chi, costheta),
        b2=b2,
        b3=b2 * pnr_fits.beta_b3(eta, chi, costheta),
        b4=b4,
        b5=pnr_fits.beta_b5(eta, chi, costheta),
    )


def line_of_sight(wf):
    inc = wf.inclination
    phi_ref = wf.phi_ref_in
    return [jnp.sin(inc) * jnp.cos(FRAC_PI_2 - phi_ref), jnp.sin(inc) * jnp.sin(FRAC_PI_2 - phi_ref), jnp.cos(inc)]


def polarization_angle(prec, wf, nx_jf, nz_jf):
    """The polarization angle ``zeta`` (convention 1)."""
    inc = wf.inclination
    phi_ref = wf.phi_ref_in
    x_sf = [-jnp.cos(inc) * jnp.sin(phi_ref), -jnp.cos(inc) * jnp.cos(phi_ref), jnp.sin(inc)]
    tmp = rotate_z(-prec.kappa, rotate_y(-prec.theta_j_sf, rotate_z(-prec.phi_j_sf, x_sf)))
    x_dot_p = fma(tmp[2], -nx_jf, fma(tmp[0], nz_jf, tmp[1] * 0.0))
    x_dot_q = fma(tmp[2], 0.0, fma(tmp[0], 0.0, tmp[1] * 1.0))
    return jnp.arctan2(x_dot_q, x_dot_p)


def rotate_to_l_frame(lx, ly, lz, s1, s2):
    """The spins in the frame of ``L`` (lalsuite !2593)."""
    lperp_sq = fma(lx, lx, ly * ly)
    lperp = safe_sqrt(lperp_sq, 1e-300)
    lnorm = jnp.sqrt(fma(lz, lz, lperp_sq))
    cos_theta, sin_theta = lz / lnorm, lperp / lnorm
    has_perp = lperp > 0.0
    cos_phi = jnp.where(has_perp, safe_div(lx, lperp), 1.0)
    sin_phi = jnp.where(has_perp, safe_div(ly, lperp), 0.0)

    def rotate(s):
        xp = fma(s[0], cos_phi, s[1] * sin_phi)
        yp = fnms(s[0], sin_phi, s[1] * cos_phi)
        zp = s[2]
        return [fma(xp, cos_theta, -(zp * sin_theta)), yp, fma(xp, sin_theta, cos_theta * zp)]

    return rotate(s1), rotate(s2)


@fenced
def finish_setup(prec, wf, version, st=None):
    """The rest of ``IMRPhenomXGetAndSetPrecessionVariables`` for a static
    ``version`` (330 with the SpinTaylor solution ``st``, 223 or 102),
    returning new ``(prec, wf)``."""
    prec = _copy(prec)
    wf = _copy(wf)
    prec.st = st
    chi1, chi2 = prec.chi1, prec.chi2
    pnr_on = prec.pnr_on
    if version == 330:
        prec.fmin_integration = st.f_min_pn
        prec.mfmin_integration = hz_to_mf(st.f_min_pn, wf.m_tot)
        lx, ly, lz = st.lnhat_end
        if prec.parity_fix:
            prec.chi1_evolved, prec.chi2_evolved = rotate_to_l_frame(lx, ly, lz, st.s1_end, st.s2_end)
        else:
            # (L along z, aligned spins: phi = atan2(0, 0) with a zero
            # derivative rather than 0/0)
            perp = (lx != 0.0) | (ly != 0.0)
            phi = jnp.where(perp, jnp.arctan2(jnp.where(perp, ly, 0.0), jnp.where(perp, lx, 1.0)),
                            jax.lax.stop_gradient(jnp.arctan2(ly, lx)))
            theta = acos_safe(lz / jnp.sqrt(fma(lz, lz, fma(lx, lx, ly * ly))))
            prec.chi1_evolved = rotate_y(-theta, rotate_z(-phi, st.s1_end))
            prec.chi2_evolved = rotate_y(-theta, rotate_z(-phi, st.s2_end))
    else:
        prec.fmin_integration = jnp.zeros(())
        prec.mfmin_integration = jnp.zeros(())
    set_pnr_variables(prec, wf, 330 if version == 330 else 223)
    wf = set_coprec_params(prec, wf, pnr_on, version)
    if isinstance(version, Fallback) or version == 223:
        prec.msa = _msa.system(msa_inputs(prec, wf, chi1, chi2))
    else:
        prec.msa = None
    prec, wf = set_remnant(prec, wf)
    eta, delta = wf.eta, wf.delta
    big_m = prec.m1 + prec.m2
    if version == 330:
        prec.l = _prec.three_pn(eta, delta, chi1[2], chi2[2])
    else:
        prec.l = _by_version(version, lambda: _prec.three_pn(eta, delta, chi1[2], chi2[2]),
                             lambda: _prec.three_pn_msa(eta, delta, chi1[2], chi2[2]))
    prec.l_ref = big_m * big_m * _prec.l_ansatz(prec.l, wf.v_ref, wf.eta / wf.v_ref)
    m1_2, m2_2 = prec.m1_2, prec.m2_2
    j0 = [fma(m1_2, chi1[0], m2_2 * chi2[0]), fma(m1_2, chi1[1], m2_2 * chi2[1]), fma(m1_2, chi1[2], m2_2 * chi2[2]) + prec.l_ref]
    prec.j0 = j0
    prec.j0_norm = jnp.sqrt(fma(j0[2], j0[2], fma(j0[0], j0[0], j0[1] * j0[1])))
    prec.theta_j_sf = jnp.where(prec.j0_norm < 1e-10, 0.0, acos_safe(j0[2] / prec.j0_norm))
    prec.phi_j_sf = atan2_tol(j0[1], j0[0], MAX_TOL_ATAN)
    wf.phi0 = jnp.zeros_like(wf.phi0)
    n_sf = line_of_sight(wf)
    tmp = rotate_y(-prec.theta_j_sf, rotate_z(-prec.phi_j_sf, n_sf))
    prec.kappa = atan2_tol(tmp[1], tmp[0], MAX_TOL_ATAN)
    prec.alpha0 = PI - prec.kappa
    j0_dot_n = fma(j0[2], n_sf[2], fma(j0[0], n_sf[0], j0[1] * n_sf[1]))
    prec.theta_jn = acos_safe(j0_dot_n / prec.j0_norm)
    prec.zeta_polarization = polarization_angle(prec, wf, jnp.sin(prec.theta_jn), jnp.cos(prec.theta_jn))
    if isinstance(version, Fallback) or version // 100 == 1:
        m1 = prec.m1
        prec.nnlo = _nnlo.coefficients(
            eta, delta, prec.q, wf.chi_eff, prec.chi_p, m1, m1 * m1, m1 ** 3, m1 ** 4, m1 ** 5, m1 ** 6, m1 ** 8
        )
    else:
        prec.nnlo = None
    prec.epsilon0 = prec.phi_j_sf - PI
    omega_ref = wf.pi_m * wf.f_ref * 2.0 / 2.0

    def msa_offsets():
        angles = _msa.phi_zeta_costhetal(prec.msa, jnp.cbrt(omega_ref))
        return angles.phiz - prec.alpha0, angles.zeta - prec.epsilon0

    def nnlo_offsets():
        logomega_ref = jnp.log(omega_ref)
        omega_ref_cbrt = jnp.cbrt(omega_ref)
        alpha, epsilon = _nnlo.evaluate(prec.nnlo, omega_ref, logomega_ref, omega_ref_cbrt, omega_ref_cbrt * omega_ref_cbrt)
        return alpha - prec.alpha0, epsilon - prec.epsilon0

    def st_offsets():
        # version 330 without the NNLO coefficients: alpha = epsilon = 0
        return -prec.alpha0, -prec.epsilon0

    prec.alpha_offset, prec.epsilon_offset = _by_version(version, nnlo_offsets, msa_offsets, st_offsets)
    prec.use_mr_beta = jnp.asarray(False)
    return prec, wf


# ---------------------------------------------------------------------------
# SpinTaylor angles of version 330
# ---------------------------------------------------------------------------


def _rotation(prec):
    return (prec.phi_j_sf, prec.theta_j_sf, prec.kappa)


# Evaluations of the SpinTaylor splines and of the MSA angles, shared by the
# many functions that need them at the same frequencies within one trace
# (keyed by the identity of the traced inputs; cleared by clear_cache).
_EVAL_CACHE = {}


def clear_cache():
    """Forget the shared evaluations (call before tracing a new waveform)."""
    _EVAL_CACHE.clear()


def _trace():
    """The trace being built: a value computed inside a loop body (another
    trace) must not be reused outside it, nor the other way round."""
    return _jax_core.trace_ctx.trace


def _cached(kind, anchors, mf, compute):
    trace = _trace()
    key = (kind, id(trace), mf.__class__, id(mf)) + tuple(id(a) for a in anchors)
    entry = _EVAL_CACHE.get(key)
    if (entry is not None and entry[0] is mf and entry[3] is trace
            and all(x is y for x, y in zip(entry[1], anchors))):
        return entry[2]
    value = compute()
    _EVAL_CACHE[key] = (mf, tuple(anchors), value, trace)
    return value


def _cache_put(kind, anchors, mf, value):
    trace = _trace()
    _EVAL_CACHE[(kind, id(trace), mf.__class__, id(mf)) + tuple(id(a) for a in anchors)] = (
        mf, tuple(anchors), value, trace)


def _st_context(prec, wf):
    """The SpinTaylor solution, the J frame in which LAL built its splines
    (``IMRPhenomX_InterpolateAlphaBeta_SpinTaylor``, before
    ``IMRPhenomX_PNR_RemapThetaJSF`` changes the J frame), the scale from
    ``M omega`` to ``Mf``, the cache anchors and the spline context
    (:func:`.spintaylor.angles.spline_context`)."""
    st = prec.st
    rotation = prec.st_rotation if "st_rotation" in prec else _rotation(prec)
    # the solution's arrays determine the J frame of its splines (set once,
    # by initialize_euler_angles) and the scale
    anchors = (st.back.y, st.fwd.rows, st.fine.rows)
    mf_scale = MTSUN_SI * wf.m_tot / prec.pi_gm
    if "st_context" in prec:
        # built once by initialize_euler_angles
        return st, rotation, mf_scale, anchors, prec.st_context
    # (the smooth angles take alpha from the splines of LNhat's components)
    context = _cached("st_context", anchors[1:], anchors[0],
                      lambda: barrier(st_angles.spline_context(st, rotation, mf_scale, prec.smooth,
                                                               prec.direct_angles)))
    return st, rotation, mf_scale, anchors, context


#: Above this many points, the SpinTaylor splines are evaluated in chunks.
ST_MANY = 1024


def _st_anchors(prec):
    st = prec.st
    return (st.back.y, st.fwd.rows, st.fine.rows)


def _st_spline(prec, wf, mf):
    """The splines of the SpinTaylor ``alpha`` and ``cos(beta)`` (shared):
    ``(alpha, dalpha, cosbeta, dcosbeta, inside)`` at ``mf``."""
    if "st_bank" in prec:
        # the points evaluated together by st_bank
        for _, entry in prec.st_bank.items():
            if entry.mf is mf:
                return entry.values
    if jnp.size(mf) > ST_MANY:
        return _st_spline_many(prec, wf, mf, jnp.size(mf), PNR_GRID_CHUNK)

    def compute():
        st, rotation, mf_scale, _, context = _st_context(prec, wf)
        return st_angles.spline(st, rotation, mf, mf_scale, context)

    return barrier(_cached("st", _st_anchors(prec), mf, compute))


def _st_spline_many(prec, wf, mf, n, chunk):
    """:func:`_st_spline` at many points (the first ``n`` of ``mf``, a 1-d
    array)."""
    if mf.ndim != 1:
        flat = _st_spline_many(prec, wf, mf.reshape(-1), mf.size, chunk)
        return tuple(x.reshape(mf.shape) for x in flat)

    def compute():
        st, rotation, mf_scale, _, context = _st_context(prec, wf)
        return st_angles.spline_many(st, rotation, mf, n, chunk, mf_scale, context)

    return _cached("st", _st_anchors(prec), mf, compute)


def _st_spline_put(prec, wf, mf, value):
    """Record ``value`` as :func:`_st_spline` at ``mf``."""
    _cache_put("st", _st_anchors(prec), mf, value)


def _msa_angles(system, mf):
    """``IMRPhenomX_Return_phi_zeta_costhetaL_MSA`` at ``v = (pi Mf)^(1/3)``
    (shared)."""
    anchor = jax.tree_util.tree_leaves(system)
    return barrier(_cached("msa", (anchor[0], anchor[-1]), mf,
                           lambda: _msa.phi_zeta_costhetal(system, jnp.cbrt(PI * mf))))


def _euler_points(prec, wf, context):
    """The end of the inspiral angles (``fmax_inspiral``) and the points
    of the splines that :func:`initialize_euler_angles` reads."""
    mfmax_pn = context.last_mf
    fmax_inspiral = jnp.where(prec.pnr_on, mfmax_pn, mfmax_pn - wf.delta_mf)
    fmax_inspiral = jnp.where(fmax_inspiral > wf.f_ring - wf.f_damp, 1.020 * wf.f_meco, fmax_inspiral)
    return fmax_inspiral, jnp.stack([0.97 * fmax_inspiral, 0.99 * fmax_inspiral, 0.98 * fmax_inspiral, fmax_inspiral])


def _alpha_points(prec, wf, ap, version):
    """The bounds of the connection of the PNR ``alpha``
    (:func:`alpha_connection`) and the points of ``alpha`` it reads."""
    dmf = pa.PNR_DMF
    mf_alpha_upper = ap.a4 / 3.0
    mf_alpha_lower = (3.0 / 3.5) * mf_alpha_upper
    below = mf_alpha_upper < hz_to_mf(2.0, wf.m_tot)
    if version == 330:
        below = below | (mf_alpha_lower - 2.0 * dmf < prec.mfmin_integration)
    mf_alpha_lower = jnp.where(below, 100.0, mf_alpha_lower)
    mf_alpha_upper = jnp.where(below, 100.0, mf_alpha_upper)
    points = jnp.stack([mf_alpha_lower - dmf, mf_alpha_lower, mf_alpha_lower + dmf, mf_alpha_upper])
    return mf_alpha_lower, mf_alpha_upper, points


def _st_beta_points(prec, bp, ftrans_mrd):
    """The points of the SpinTaylor ``beta`` that
    :func:`st_beta_connection` reads, for the parameters ``bp`` with the
    connection frequencies."""
    dmf = pa.PNR_DMF
    ftrans = jnp.where(ftrans_mrd <= 0.9 * bp.mf_beta_lower, ftrans_mrd, 0.9 * bp.mf_beta_lower)
    mfa = jnp.where(ftrans - dmf < prec.mfmin_integration + 2.0 * dmf, prec.mfmin_integration + 2.0 * dmf, ftrans - dmf)
    return mfa, jnp.stack([mfa - dmf, mfa, mfa + dmf])


#: The step of the derivative of ``alpha`` at the transition of the
#: antisymmetric mode (:func:`.antisymmetric.antisymmetric_coefficients`).
ANTISYMMETRIC_DMF = 0.0005


def antisymmetric_points(wf):
    """The transition of the antisymmetric (2,2) mode and the points of
    ``alpha`` it reads."""
    mf_t = 0.85 * wf.f_ring
    return mf_t, jnp.stack([mf_t + ANTISYMMETRIC_DMF, mf_t - ANTISYMMETRIC_DMF, mf_t, wf.mf_ref])


@fenced
def st_bank(prec, wf):
    """The SpinTaylor splines at the points that the set-up of version 330
    and the antisymmetric mode read (:func:`initialize_euler_angles`,
    :func:`alpha_connection`, :func:`st_beta_connection`,
    :func:`.antisymmetric.antisymmetric_coefficients`), evaluated together:
    one evaluation of the splines to compile rather than one for each.
    Kept in ``prec.st_bank`` (in place, with the J frame of the splines and
    the alpha and beta parameters of :func:`populate_structs`), where
    :func:`_st_spline` finds the values of these very points."""
    prec.st_rotation = _rotation(prec)
    prec.st_context = _st_context(prec, wf)[4]
    fmax_inspiral, euler = _euler_points(prec, wf, prec.st_context)
    ap = precompute_alpha_coefficients(prec, wf, 330)
    bp = pa.beta_connection_frequencies(precompute_beta_coefficients(prec, wf, 330), _soft(prec))
    _, _, alpha = _alpha_points(prec, wf, ap, 330)
    _, beta = _st_beta_points(prec, bp, 0.98 * fmax_inspiral)
    _, antisymmetric = antisymmetric_points(wf)
    groups = dict(euler=euler, alpha=alpha, beta=beta, antisymmetric=antisymmetric)
    values = _st_spline(prec, wf, jnp.concatenate(list(groups.values())))
    bank, start = Struct(), 0
    for name, points in groups.items():
        n = points.shape[0]
        setattr(bank, name, Struct(mf=points, values=tuple(v[start:start + n] for v in values)))
        start += n
    prec.st_bank = bank
    prec.st_bank_params = Struct(alpha=ap, beta=bp)
    return prec


@fenced
def initialize_euler_angles(prec, wf):
    """``IMRPhenomX_Initialize_Euler_Angles`` and
    ``IMRPhenomX_InterpolateAlphaBeta_SpinTaylor`` for version 330: the
    end of the inspiral angles, the merger-ringdown continuations and the
    failure conditions of the interpolation of ``gamma`` (in place)."""
    st = prec.st
    # the J frame of the splines
    prec.st_rotation = _rotation(prec)
    buffer = prec.integration_buffer
    fmin_angles = (wf.f_min - buffer) * 2.0 / prec.m_max
    mf_max = jnp.where(wf.mf_max > fma(4.0, wf.f_damp, wf.f_ring), wf.mf_max, fma(4.0, wf.f_damp, wf.f_ring))
    mf_max_angles = (mf_max + hz_to_mf(buffer, wf.m_tot)) * 2.0 / prec.m_min
    fmax_angles = mf_to_hz(mf_max_angles, wf.m_tot)
    error = jnp.isnan(fmin_angles) | (fmin_angles <= 0.0)
    error = error | (fmax_angles <= 0.0) | (fmax_angles <= fmin_angles) | (wf.f_ref < fmin_angles)
    # the knots: Mf of the last one
    mf_scale = MTSUN_SI * wf.m_tot / prec.pi_gm
    context = _st_context(prec, wf)[4]
    # (kept: the splines are built once)
    prec.st_context = context
    mfmax_pn = context.last_mf
    mf_first = context.first_mf
    fmax_inspiral, q = _euler_points(prec, wf, context)
    if "st_bank" in prec:
        q = prec.st_bank.euler.mf
    ftrans_mrd = 0.98 * fmax_inspiral
    # alphaMRD_coeff from the spline of alpha (at 0.97, 0.99, 0.98 and 1
    # times fmax_inspiral)
    f1 = q[0]
    f2 = q[1]
    a, da, cb, dcb, inside = _st_spline(prec, wf, q)
    error = error | ~(inside[0] & inside[1])
    alpha1, dalpha1, alpha2 = -a[0], -da[0], -a[1]
    f1sq, f2sq = f1 * f1, f2 * f2
    f1cube = f1sq * f1
    f1_4, f2_4 = f1 ** 4, f2 ** 4
    den = 2.0 * (f1sq - f2sq) ** 2
    alpha_mrd = Struct(
        a=fma(2.0 * f2_4, alpha2, fma(f1cube * (f1 - f2) * (f1 + f2), dalpha1, 2.0 * fnms(2.0 * f1sq, f2sq, f1_4) * alpha1)) / den,
        b=(f1_4 * f2sq * fma(f1 * (f1 - f2) * (f1 + f2), dalpha1, 2.0 * f2sq * (-alpha1 + alpha2))) / den,
        c=(f1sq * fma(f1 * (-f1_4 + f2_4), dalpha1, 4.0 * f2_4 * (alpha1 - alpha2))) / den,
    )
    # betaMRD_coeff of version 330 (offset: the PNR ringdown beta)
    off = ringdown_pnr_beta(prec, wf)
    fdamp = wf.f_damp
    dfdamp = hm_qnm.fdamp21(wf.a_final) / wf.m_final - fdamp
    kappa = 2.0 * PI * dfdamp
    g1 = 0.97 * fmax_inspiral
    g2 = 0.98 * fmax_inspiral
    ef1 = jnp.exp(kappa * g1)
    ef2 = jnp.exp(kappa * g2)
    cosbeta1 = cb[0]
    cosbeta2, dcosbeta2 = cb[2], dcb[2]
    cosbetamax = cb[3]
    success = inside[3]
    flat = (jnp.abs(cosbeta1) > 1.0) | (jnp.abs(cosbeta2) > 1.0) | (jnp.abs(cosbetamax) > 1.0) | ~success
    beta1 = acos_safe(cosbeta1)
    beta2 = acos_safe(cosbeta2)
    sqrtarg = fnms(cosbeta2, cosbeta2, 1.0)
    dbeta2 = -dcosbeta2 / jnp.sqrt(jnp.where(sqrtarg <= 0.0, 1.0, sqrtarg))
    gden = (g1 - g2) ** 2
    g1_4 = g1 ** 4
    g2_3 = g2 ** 3
    g2_4 = g2 ** 4
    g1sq, g2sq = g1 * g1, g2 * g2
    k2 = fma(g2, kappa, 2.0)
    k3 = fma(g2, kappa, 3.0)
    k4 = fma(g2, kappa, 4.0)
    w_a = fma(-g2, k3, g1 * k4)
    z_a = fma(g2 * (-g1 + g2), dbeta2, w_a * (off - beta2))
    w_b = fma(-g2sq, k2, g1sq * k4)
    z_b = (g1 - g2) * g2 * (g1 + g2) * dbeta2 - w_b * (off - beta2)
    w_c = fma(-g2, k2, g1 * k3)
    z_c = fma(g2 * (-g1 + g2), dbeta2, w_c * (off - beta2))
    beta_mrd = Struct(
        a=fma(-(ef1 * g1_4), off - beta1, ef2 * g2_3 * z_a) / gden,
        b=fma(2.0 * ef1 * g1_4 * g2, off - beta1, ef2 * g2_3 * z_b) / gden,
        c=fma(-(ef1 * g1_4 * g2sq), off - beta1, ef2 * g1 * g2_4 * z_c) / gden,
        d=off,
        cosbeta_sign=jnp.where(cosbetamax < 0, -1.0, 1.0),
        flat=flat,
        dfdamp=dfdamp,
    )
    # IMRPhenomX_InterpolateGamma_SpinTaylor: the failure conditions (the
    # PNR angles do not use its gamma: the extended angles leave them out)
    mf_min = hz_to_mf(fmin_angles, wf.m_tot)
    delta_mf = _prec.delta_mf_from_wfstruct(wf)
    delta_f = mf_to_hz(delta_mf, wf.m_tot)
    i_start = jnp.floor(fmin_angles / delta_f)
    i_stop = jnp.floor(fmax_angles / delta_f) + 1.0
    output_length = i_stop - i_start
    gamma_error = output_length < 4
    # gamma_from_alpha_cosbeta evaluates the splines from mf_min up to
    # ftrans_MRD (in steps of delta_mf, four sub-steps each)
    last_inspiral = jnp.minimum(
        jnp.ceil((ftrans_mrd - mf_min) / delta_mf) - 1.0, output_length - 1.0
    )
    mf_last_eval = fma(jnp.maximum(last_inspiral, 1.0), delta_mf, mf_min) + delta_mf
    gamma_error = gamma_error | ((last_inspiral >= 1.0) & ((mf_min + delta_mf < mf_first) | (mf_last_eval > mfmax_pn)))
    mf_last = fma(output_length - 1.0, delta_mf, mf_min)
    gamma_contains = lambda mf: (mf >= mf_min) & (mf <= mf_last)  # noqa: E731
    gamma_error = gamma_error | ~gamma_contains(wf.mf_ref) | ~gamma_contains(ftrans_mrd)
    if not prec.extended:
        error = error | gamma_error
    mf_alpha_ref = jnp.where(wf.mf_ref <= ftrans_mrd, wf.mf_ref, ftrans_mrd)
    error = error | (mf_alpha_ref < mf_first) | (mf_alpha_ref > mfmax_pn)
    error = error | (ftrans_mrd < mf_first) | (ftrans_mrd > mfmax_pn)
    # LAL leaves the spline of alpha for alphaMRD at ftrans_MRD, where they
    # differ (alphaMRD matches the spline's value and slope at f1 and its
    # value at f2): alpha steps there in frequency, and the PNR grid samples
    # the step wherever a node crosses it. The smooth angles switch at f1,
    # where alpha is continuous with its derivative.
    ftrans_alpha = f1 if prec.smooth else ftrans_mrd
    prec.st_angles = Struct(ftrans_mrd=ftrans_mrd, ftrans_alpha=ftrans_alpha, alpha_mrd=alpha_mrd, beta_mrd=beta_mrd,
                            mf_first=mf_first, mfmax_pn=mfmax_pn)
    prec.error_euler = error
    return prec


def alpha_mrd(p, mf):
    """``alphaMRD``."""
    mf2 = mf * mf
    mf4 = mf2 * mf2
    return -(p.a + p.b / mf4 + p.c / mf2)


def beta_mrd(p, mf):
    """``betaMRD``."""
    kappa = 2.0 * PI * p.dfdamp
    mf2 = mf * mf
    mf3 = mf2 * mf
    value = fma(jnp.exp(-mf * kappa) / mf, p.a / mf + p.b / mf2 + p.c / mf3, p.d)
    return jnp.where(p.flat, jnp.arccos(p.cosbeta_sign), value)


def st_alpha(prec, wf, mf):
    """``alpha_SpinTaylor_IMR`` (version 330), with the error flag."""
    sa = prec.st_angles
    a, _, _, _, inside = _st_spline(prec, wf, mf)
    inspiral = mf < sa.ftrans_alpha
    value = jnp.where(inspiral, a, alpha_mrd(sa.alpha_mrd, mf)) + prec.alpha_offset
    error = inspiral & ~inside
    return value, error


def st_beta(prec, wf, mf, bp, use_mr_beta):
    """``beta_SpinTaylor_IMR`` (version 330), with the error flag."""
    sa = prec.st_angles
    ftrans = jnp.where(
        use_mr_beta,
        jnp.where(sa.ftrans_mrd <= 0.9 * bp.mf_beta_lower, sa.ftrans_mrd - 0.0005, fma(0.9, bp.mf_beta_lower, -0.0005)),
        sa.ftrans_mrd,
    )
    _, _, cb, _, inside = _st_spline(prec, wf, mf)
    inspiral = mf < ftrans
    cosbeta = jnp.clip(cb, -1.0, 1.0)
    value = jnp.where(
        inspiral, acos_safe(cb),
        jnp.where(use_mr_beta, pa.beta_connection(mf, bp), beta_mrd(sa.beta_mrd, mf)),
    )
    return value, inspiral & ~inside


# ---------------------------------------------------------------------------
# PN angles
# ---------------------------------------------------------------------------


def nnlo_beta(prec, v):
    """The NNLO opening angle at ``v`` (sign of ``L + SL`` ignored in ``s``)."""
    l = _prec.l_ansatz(prec.l, v, prec.eta / v)
    return _nnlo.nnlo_beta(l, prec.sperp, prec.sl)


def pn_alpha(prec, wf, mf, version):
    """``IMRPhenomX_PNR_GetPNAlphaAtFreq``, with the error flag."""
    no_error = jnp.zeros(jnp.shape(mf), bool)

    def nnlo():
        omega = PI * mf
        omega_cbrt = jnp.cbrt(omega)
        alpha, _ = _nnlo.evaluate(prec.nnlo, omega, jnp.log(omega), omega_cbrt, omega_cbrt * omega_cbrt)
        return alpha - prec.alpha_offset, no_error

    def st():
        alpha, error = st_alpha(prec, wf, mf)
        return alpha, error | ~jnp.isfinite(alpha)

    return _by_version(version, nnlo, lambda: (_msa_angles(prec.msa, mf).phiz, no_error), st)


def _single_beta(prec, mf):
    """``acos(cos(theta_L))`` of the single-spin system (``SingleSpin``) at
    ``v = (pi Mf)^(1/3)``."""
    single = prec.single
    cos_l_msa = _msa_angles(single.msa, mf).cos_theta_l
    anchor = jax.tree_util.tree_leaves(single.msa)
    cos_l_102 = _cached("msa102", (anchor[0], anchor[-1]), mf,
                        lambda: _msa.cos_theta_l_with(single.msa, jnp.cbrt(PI * mf), single.l_102))
    cos_l = jnp.where(single.error, cos_l_102, cos_l_msa)
    return acos_safe(cos_l)


def pn_beta(prec, wf, mf, bp, version, use_mr_beta):
    """``IMRPhenomX_PNR_GetPNBetaAtFreq``: the PN ``beta`` with the two-spin
    oscillations tapered below the lower connection frequency."""
    no_error = jnp.zeros(jnp.shape(mf), bool)

    def tapered(beta_full, version):
        if prec.single is None:
            return beta_full
        beta_single = _single_beta(prec, mf)
        oscillations = beta_full - beta_single
        e = jnp.cos(2.0 * PI * mf / (4.0 * bp.mf_beta_lower))
        tapered = jnp.where(mf <= bp.mf_beta_lower, fma(oscillations, e * e, beta_single), beta_single)
        return jnp.where(two_spin(prec, version), tapered, beta_full)

    def st():
        beta_full, error = st_beta(prec, wf, mf, bp, use_mr_beta)
        return tapered(beta_full, 330), error | ~jnp.isfinite(beta_full)

    return _by_version(
        version, lambda: (nnlo_beta(prec, jnp.cbrt(PI * mf)), no_error),
        lambda: (tapered(acos_safe(_msa_angles(prec.msa, mf).cos_theta_l), 223), no_error), st)


def pn_beta_fulltwospin(prec, wf, mf, bp, version, use_mr_beta):
    """``IMRPhenomX_PNR_GetPNBetaAtFreq_fulltwospin``."""
    no_error = jnp.zeros(jnp.shape(mf), bool)

    def st():
        beta, error = st_beta(prec, wf, mf, bp, use_mr_beta)
        return beta, error | ~jnp.isfinite(beta)

    return _by_version(version, lambda: (nnlo_beta(prec, jnp.cbrt(PI * mf)), no_error),
                       lambda: (acos_safe(_msa_angles(prec.msa, mf).cos_theta_l), no_error), st)


def waveform_beta(prec, wf, mf, pn_beta_value):
    """``IMRPhenomX_PNR_PNWaveformBetaWrapper``."""
    big_m = wf.m_tot
    m1 = wf.m1 * big_m
    m2 = wf.m2 * big_m
    chi_parr = (m1 + m2) * wf.chi_eff / m1
    v = jnp.cbrt(PI * mf)
    l_norm = wf.eta / v
    l_3pn = big_m * big_m * _prec.l_ansatz(prec.l, v, l_norm)
    chi_temp = pa.chi_calc(m1, l_3pn, prec.j0_norm, prec.l_ref, chi_parr, pn_beta_value)
    costheta_temp = _off_poles(prec, safe_div(chi_parr, chi_temp))
    return pa.pn_waveform_beta(mf, pn_beta_value, m1, m2, chi_temp, costheta_temp)


# ---------------------------------------------------------------------------
# connections of the PNR angles
# ---------------------------------------------------------------------------


@fenced
def alpha_connection(prec, wf, ap, version):
    """``IMRPhenomX_PNR_alpha_connection_parameters``; returns the new
    parameters and the error flag."""
    dmf = pa.PNR_DMF
    mf_alpha_lower, mf_alpha_upper, points = _alpha_points(prec, wf, ap, version)
    if version == 330 and "st_bank" in prec:
        points = prec.st_bank.alpha.mf
    ap = ap.replace(mf_alpha_lower=mf_alpha_lower, mf_alpha_upper=mf_alpha_upper)
    values, errors = pn_alpha(prec, wf, points, version)
    a1, alpha_lower, a3, alpha_upper = values[0], values[1], values[2], values[3]
    a4 = pa.mr_alpha(mf_alpha_upper - dmf, ap)
    a6 = pa.mr_alpha(mf_alpha_upper + dmf, ap)
    derivative_alpha_lower = (a3 - a1) / (2.0 * dmf)
    derivative_alpha_upper = (a6 - a4) / (2.0 * dmf)
    args = (mf_alpha_lower, mf_alpha_upper, alpha_lower, alpha_upper, derivative_alpha_lower, derivative_alpha_upper)
    interp = [pa.alpha_interpolation_0(*args), pa.alpha_interpolation_1(*args), pa.alpha_interpolation_2(*args),
              pa.alpha_interpolation_3(*args)]
    interp = [jnp.where(jnp.isnan(x), 0.0, x) for x in interp]
    ap = ap.replace(alpha_interp=interp)
    mr_alpha_at_upper = pa.mr_alpha(mf_alpha_upper, ap)
    ap = ap.replace(
        alpha_mr_offset=alpha_upper - mr_alpha_at_upper,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        derivative_alpha_lower=derivative_alpha_lower,
        derivative_alpha_upper=derivative_alpha_upper,
    )
    return ap, jnp.any(errors)


@fenced
def beta_connection(prec, wf, bp, version):
    """``IMRPhenomX_PNR_beta_connection_parameters`` (versions 102, 223)."""
    dmf = pa.PNR_DMF
    bp = pa.beta_connection_frequencies(bp, _soft(prec))
    mf_beta_lower = bp.mf_beta_lower
    points = jnp.stack([mf_beta_lower - dmf, mf_beta_lower, mf_beta_lower + dmf])
    beta, errors = pn_beta(prec, wf, points, bp, version, jnp.asarray(False))
    wb = waveform_beta(prec, wf, points, beta)
    b1, beta_lower, b3 = wb[0], wb[1], wb[2]
    b4 = pa.mr_beta(mf_beta_lower - dmf, bp)
    beta_upper = pa.mr_beta(mf_beta_lower, bp)
    b6 = pa.mr_beta(mf_beta_lower + dmf, bp)
    derivative_beta_lower = (b3 - b1) / (2.0 * dmf)
    derivative_beta_upper = (b6 - b4) / (2.0 * dmf)
    r1 = pa.beta_rescaling_1(mf_beta_lower, beta_lower, beta_upper, derivative_beta_lower, derivative_beta_upper)
    r2 = pa.beta_rescaling_2(mf_beta_lower, beta_lower, beta_upper, derivative_beta_lower, derivative_beta_upper)
    bp = bp.replace(
        beta_lower=beta_lower, beta_upper=beta_upper, derivative_beta_lower=derivative_beta_lower,
        derivative_beta_upper=derivative_beta_upper, beta_rescale_1=jnp.where(jnp.isnan(r1), 0.0, r1),
        beta_rescale_2=jnp.where(jnp.isnan(r2), 0.0, r2),
    )
    return bp.replace(attach_weight=pa.attach_weight(bp)), jnp.any(errors)


@fenced
def st_beta_connection(prec, wf, bp):
    """``IMRPhenomX_ST_PNR_beta_connection_parameters`` (version 330)."""
    dmf = pa.PNR_DMF
    if "st_bank" in prec:
        # (with the connection frequencies)
        bp = prec.st_bank_params.beta
    else:
        bp = pa.beta_connection_frequencies(bp, _soft(prec))
    mfa, pts = _st_beta_points(prec, bp, prec.st_angles.ftrans_mrd)
    if "st_bank" in prec:
        pts = prec.st_bank.beta.mf
        mfa = pts[1]
    mfb = bp.mf_beta_lower
    _, _, cb, _, inside = _st_spline(prec, wf, pts)
    st_b = acos_safe(cb)
    error = ~jnp.all(inside)
    ba1, ba, ba2 = st_b[0], st_b[1], st_b[2]
    omega = PI * mfb
    domega = PI * dmf
    xs = jnp.stack([omega - domega, omega, omega + domega])
    if prec.single is None:
        main_msa = _msa.system(msa_inputs(prec, wf, prec.chi1, prec.chi2))
        msa_b = acos_safe(_msa.cos_theta_l_with(main_msa, jnp.cbrt(xs), prec.l))
    else:
        main_msa = _msa.system(msa_inputs(prec, wf, prec.chi1, prec.chi2))
        msa_main = acos_safe(_msa.cos_theta_l_with(main_msa, jnp.cbrt(xs), prec.l))
        msa_single = _single_beta(prec, xs / PI)
        msa_b = jnp.where(two_spin(prec, 330), msa_single, msa_main)
    bb1, bb, bb2 = msa_b[0], msa_b[1], msa_b[2]
    dba = (ba2 - ba1) / (2.0 * dmf)
    dbb = (bb2 - bb1) / (2.0 * dmf)
    # (sic: the SpinTaylor beta is passed for both ends)
    interp = [pa.st_beta_coeffs_0(mfa, mfb, ba, ba, dba, dbb), pa.st_beta_coeffs_1(mfa, mfb, ba, ba, dba, dbb),
              pa.st_beta_coeffs_2(mfa, mfb, ba, ba, dba, dbb), pa.st_beta_coeffs_3(mfa, mfb, ba, ba, dba, dbb)]
    bp = bp.replace(beta_interp=interp)
    bbnew = jnp.where(two_spin(prec, 330), bb, ba)
    points = jnp.stack([mfb - dmf, mfb, mfb + dmf])
    wb = waveform_beta(prec, wf, points, jnp.stack([bb1, bbnew, bb2]))
    bc1, bc, bc2 = wb[0], wb[1], wb[2]
    bd1 = pa.mr_beta(mfb - dmf, bp)
    bd = pa.mr_beta(mfb, bp)
    bd2 = pa.mr_beta(mfb + dmf, bp)
    dbc = (bc2 - bc1) / (2.0 * dmf)
    dbd = (bd2 - bd1) / (2.0 * dmf)
    r1 = pa.beta_rescaling_1(mfb, bc, bd, dbc, dbd)
    r2 = pa.beta_rescaling_2(mfb, bc, bd, dbc, dbd)
    bp = bp.replace(
        beta_lower=bc, beta_upper=bd, derivative_beta_lower=dbc, derivative_beta_upper=dbd,
        beta_rescale_1=jnp.where(jnp.isnan(r1), 0.0, r1), beta_rescale_2=jnp.where(jnp.isnan(r2), 0.0, r2),
    )
    return bp.replace(attach_weight=pa.attach_weight(bp, _soft(prec))), error


@fenced
def populate_structs(prec, wf, version):
    """``IMRPhenomX_PNR_PopulateStructs``: the single-spin system of the
    two-spin map and the alpha and beta parameters (in place on ``prec``:
    ``single``, ``alpha_params``, ``beta_params``, ``error_structs``)."""
    if isinstance(version, Fallback) or version >= 200:
        chi = prec.chi_single_spin
        costheta = prec.costheta_single_spin
        chi1 = [chi * safe_sqrt(1.0 - costheta * costheta, 1e-300), jnp.zeros_like(chi), chi * costheta]
        system = _msa.system(msa_inputs(prec, wf, chi1, [jnp.zeros_like(chi)] * 3))
        prec.single = Struct(
            msa=system, error=_msa.msa_error(system), l_102=_prec.three_pn(wf.eta, wf.delta, chi1[2], 0.0)
        )
    else:
        prec.single = None
    banked = version == 330 and "st_bank" in prec
    ap = prec.st_bank_params.alpha if banked else precompute_alpha_coefficients(prec, wf, version)
    ap, error_a = alpha_connection(prec, wf, ap, version)
    bp = None if banked else precompute_beta_coefficients(prec, wf, version)
    if version == 330:
        bp, error_b = st_beta_connection(prec, wf, bp)
    else:
        bp, error_b = beta_connection(prec, wf, bp, version)
    prec.alpha_params = ap
    prec.beta_params = bp
    prec.error_structs = error_a | error_b
    return prec


# ---------------------------------------------------------------------------
# the PNR angles
# ---------------------------------------------------------------------------


def _in_calibration(prec, wf):
    return (wf.q <= pa.PNR_Q_WINDOW_LOWER) & (prec.chi_single_spin <= pa.PNR_CHI_WINDOW_LOWER)


def _in_transition(prec, wf):
    # (for the smooth angles, up to q = 12 + Q_SMOOTH_WIDTH)
    return (wf.q <= prec.transition_q) & (prec.chi_single_spin <= pa.PNR_CHI_WINDOW_UPPER)


#: The smooth angles go from LAL's at the edge of the transition region (q =
#: 12) to LAL's beyond it over this range of the mass ratio.
Q_SMOOTH_WIDTH = 2.0


def frequency_map(prec, mf, ell, emm, mf_low, mf_high, mf_rd_22, mf_rd_lm):
    """:func:`.pnr_angles.linear_frequency_map` with the binary's inspiral
    scaling; for the smooth angles between q = 12 and 12 +
    :data:`Q_SMOOTH_WIDTH`, the linear map blended into the scaling with
    ``prec.scaling_weight``."""
    value, error = pa.linear_frequency_map(mf, ell, emm, mf_low, mf_high, mf_rd_22, mf_rd_lm, prec.pnr_inspiral_scaling)
    w = prec.scaling_weight
    partial = (w > 0.0) & (w < 1.0)
    linear, linear_error = pa.linear_frequency_map(mf, ell, emm, mf_low, mf_high, mf_rd_22, mf_rd_lm, jnp.asarray(False))
    blended = fma(w, 2.0 * mf / emm - linear, linear)
    return jnp.where(partial, blended, value), jnp.where(partial, linear_error, error)


def pnr_alpha_at(prec, wf, mf, ap, version):
    """``IMRPhenomX_PNR_GeneratePNRAlphaAtMf``."""
    pn, error = pn_alpha(prec, wf, mf, version)
    mr = pa.mr_alpha(mf, ap) + ap.alpha_mr_offset
    inter = pa.intermediate_alpha(mf, ap)
    below = mf <= ap.mf_alpha_lower
    value = jnp.where(below, pn, jnp.where(mf >= ap.mf_alpha_upper, mr, inter))
    return value, error & below


def merged_alpha_at(prec, wf, mf, ap, version):
    """``IMRPhenomX_PNR_GenerateMergedPNRAlphaAtMf``."""
    pnr_window = pa.angles_window(wf.q, prec.chi_single_spin)
    msa_window = 1.0 - pnr_window
    pn, error = pn_alpha(prec, wf, mf, version)
    pnr = jnp.where(mf >= ap.mf_alpha_upper, pa.mr_alpha(mf, ap) + ap.alpha_mr_offset, pa.intermediate_alpha(mf, ap))
    value = jnp.where(mf <= ap.mf_alpha_lower, pn, fma(pnr_window, pnr, msa_window * pn))
    return value, error


def region_alpha_at(prec, wf, mf, ap, version):
    """The ``alpha`` of the region of parameter space: PNR, merged or PN."""
    pnr, e1 = pnr_alpha_at(prec, wf, mf, ap, version)
    merged, e2 = merged_alpha_at(prec, wf, mf, ap, version)
    pn, e3 = pn_alpha(prec, wf, mf, version)
    cal = _in_calibration(prec, wf)
    tra = _in_transition(prec, wf)
    value = jnp.where(cal, pnr, jnp.where(tra, merged, pn))
    error = jnp.where(cal, e1, jnp.where(tra, e2, e3))
    return value, error


def region_alpha_many(prec, wf, mf, ap, version, n, st_alpha=None):
    """:func:`region_alpha_at` at the first ``n`` (traced) of many points
    ``mf`` (a 1-d array), by chunks (:func:`~._math.chunked_map`; zeros
    beyond the chunks needed). ``st_alpha`` is the SpinTaylor ``alpha`` of
    the splines at ``mf`` and whether it lies within the knots, if computed
    already (``st_extra`` of :func:`angle_interpolants`)."""

    def put(mf, st_alpha):
        if st_alpha is not None:
            # (only alpha is read: the other values are NaN)
            a, inside = st_alpha
            nan = jnp.full_like(a, jnp.nan)
            _st_spline_put(prec, wf, mf, (a, nan, nan, nan, inside))

    if mf.shape[0] <= PNR_GRID_CHUNK:
        put(mf, st_alpha)
        return region_alpha_at(prec, wf, mf, ap, version)
    fallback = isinstance(version, Fallback)

    def evaluate(x, args):
        prec, wf, ap, msa = args
        x, st_alpha = x
        if st_alpha is not None:
            a, inside = st_alpha
            nan = jnp.full_like(a, jnp.nan)
            _st_spline_put(prec, wf, x, (a, nan, nan, nan, inside))
        return region_alpha_at(prec, wf, x, ap, Fallback(msa) if fallback else version)

    return chunked_map(evaluate, (mf, st_alpha), n, PNR_GRID_CHUNK, (prec, wf, ap, version.msa if fallback else None))


def _held(bp, held, free):
    """``beta`` above ``mf_beta_upper``: held at its value there (LAL), or
    with the smooth connection, held with the weight ``bp.beta_freeze``."""
    return jnp.where(bp.beta_freeze >= 1.0, held, fma(bp.beta_freeze, held - free, free))


def pnr_beta_at(prec, wf, mf, bp, version, use_mr_beta):
    """``IMRPhenomX_PNR_GeneratePNRBetaAtMf``."""
    beta_pn, error = pn_beta(prec, wf, mf, bp, version, use_mr_beta)
    beta_waveform = waveform_beta(prec, wf, mf, beta_pn)
    low = pa.arctan_window(beta_waveform * pa.rescale_beta(mf, bp))
    if isinstance(version, Fallback) or version == 223:
        msa = version.msa if isinstance(version, Fallback) else True
        low = jnp.where(msa & (bp.beta_lower < 0.01 * bp.beta_upper),
                        pa.arctan_window(pa.mr_beta(bp.mf_beta_lower, bp)), low)
    upper = pa.arctan_window(pa.mr_beta(bp.mf_beta_upper, bp))
    mid = pa.arctan_window(pa.mr_beta(mf, bp))
    below = mf <= bp.mf_beta_lower
    value = jnp.where(below, low, jnp.where(mf >= bp.mf_beta_upper, _held(bp, upper, mid), mid))
    return value, error & below


def merged_beta_at(prec, wf, mf, bp, version, smooth=False, beyond=None):
    """``IMRPhenomX_PNR_GenerateMergedPNRBetaAtMf``; with ``smooth``
    (static), blended into LAL's ``beta`` beyond the transition region
    (:func:`beta_no_mr`, or ``beyond``, its value and error flag if
    computed already) with ``prec.q12_weight`` (between q = 12 and 12 +
    :data:`Q_SMOOTH_WIDTH`)."""
    pnr_window = pa.angles_window(wf.q, prec.chi_single_spin)
    msa_window = 1.0 - pnr_window
    below = mf <= bp.mf_beta_lower
    if version != 330:
        beta_pn, error = pn_beta(prec, wf, mf, bp, version, jnp.asarray(False))
        beta_waveform = waveform_beta(prec, wf, mf, beta_pn)
        pnr_beta = beta_waveform * pa.rescale_beta(mf, bp)
        pn_b = beta_waveform
        beta_waveform_high = beta_waveform
        error_high = error
    else:
        beta_pn1, e1 = pn_beta(prec, wf, mf, bp, version, jnp.asarray(True))
        beta_waveform1 = waveform_beta(prec, wf, mf, beta_pn1)
        rescaled_beta1 = beta_waveform1 * pa.rescale_beta(mf, bp)
        beta_pn2, e2 = pn_beta(prec, wf, mf, bp, version, jnp.asarray(False))
        beta_waveform2 = waveform_beta(prec, wf, mf, beta_pn2)
        pnr_beta = rescaled_beta1
        pn_b = beta_waveform2
        error = e1 | e2
        beta_waveform_high = beta_waveform2
        error_high = e2
    low = pa.arctan_window(fma(pnr_window, pnr_beta, msa_window * pn_b))
    mr_mf = pa.mr_beta(mf, bp)
    mr = jnp.where(mf >= bp.mf_beta_upper, _held(bp, pa.mr_beta(bp.mf_beta_upper, bp), mr_mf), mr_mf)
    high = pa.arctan_window(fma(pnr_window, mr, msa_window * beta_waveform_high))
    value, error = jnp.where(below, low, high), jnp.where(below, error, error_high)
    if smooth:
        if beyond is None:
            beyond = beta_no_mr(prec, wf, mf, bp, version, jnp.asarray(False))
        beyond, error_beyond = beyond
        u = prec.q12_weight
        value = jnp.where(u > 0.0, fma(u, beyond - value, value), value)
        error = error | ((u > 0.0) & error_beyond)
    return value, error


def beta_no_mr(prec, wf, mf, bp, version, use_mr_beta):
    """``IMRPhenomX_PNR_GeneratePNRBetaNoMR``."""
    beta, error = pn_beta_fulltwospin(prec, wf, mf, bp, version, use_mr_beta)
    return pa.arctan_window(waveform_beta(prec, wf, mf, beta)), error


def uniform_angles(prec, wf, mf, version, st_values=None, smooth=False):
    """``IMRPhenomX_PNR_GeneratePNRAngles_UniformFrequencies`` without
    ``gamma``: ``alpha`` and ``beta`` at ``mf`` (an array) and the error
    flag of each point. ``st_values`` are the SpinTaylor splines at ``mf``
    (version 330), if computed already.

    LAL's ``beta`` depends on the region of the binary (calibration,
    transition, outside) and on whether the MR ``beta`` is attached: only
    the variant of the binary is computed (:func:`._transforms.switch`;
    under ``jax.vmap``, those of the batch). With ``smooth`` (static), the
    ``beta`` with and without the MR ``beta`` are blended with the weight
    ``bp.attach_weight`` where it lies between 0 and 1."""
    prec, wf = barrier((prec, wf))
    ap, bp = prec.alpha_params, prec.beta_params
    weight = bp.attach_weight
    attach = weight > 0.0
    cal = _in_calibration(prec, wf)
    tra = _in_transition(prec, wf)
    if st_values is not None:
        _st_spline_put(prec, wf, mf, st_values)
    # alpha: the regions share the PN alpha
    a_cal, ea_cal = pnr_alpha_at(prec, wf, mf, ap, version)
    a_tra, ea_tra = merged_alpha_at(prec, wf, mf, ap, version)
    a_pn, ea_pn = pn_alpha(prec, wf, mf, version)
    alpha = jnp.where(cal, a_cal, jnp.where(tra, a_tra, a_pn))
    error_alpha = jnp.where(cal, ea_cal, jnp.where(tra, ea_tra, ea_pn))

    # beta: in the calibration region with the MR beta attached, in the
    # transition region with it attached, else without (for the smooth
    # angles, also blends of the first two with the last). Each of the
    # three is computed only where the binary needs it
    # (:func:`._transforms.cond`; under jax.vmap, where some binary of the
    # batch does): one copy of each to compile.
    fallback = isinstance(version, Fallback)
    case = jnp.where(attach & cal, 0, jnp.where(attach & tra, 1, 2))
    if smooth:
        partial = (weight > 0.0) & (weight < 1.0)
        case = jnp.where(partial & cal, 3, jnp.where(partial & tra, 4, case))
    in_tra = (case == 1) | (case == 4)
    need_detached = (case >= 2) | (in_tra & (prec.q12_weight > 0.0)) if smooth else case == 2

    def guarded(beta_of, needed, extra=None):
        def run(operands):
            prec, wf, mf, bp, msa, st_values, extra = operands
            v = Fallback(msa) if fallback else version
            if st_values is not None:
                _st_spline_put(prec, wf, mf, st_values)
            return beta_of(prec, wf, mf, bp, v, extra)

        def skip(operands):
            mf = operands[2]
            return jnp.zeros(mf.shape), jnp.zeros(mf.shape, bool)

        return cond(needed, run, skip, (prec, wf, mf, bp, version.msa if fallback else None, st_values, extra))

    detached = guarded(lambda prec, wf, mf, bp, v, _: beta_no_mr(prec, wf, mf, bp, v, jnp.asarray(False)),
                       need_detached)
    calibration = guarded(lambda prec, wf, mf, bp, v, _: pnr_beta_at(prec, wf, mf, bp, v, jnp.asarray(True)),
                          (case == 0) | (case == 3))
    transition = guarded(lambda prec, wf, mf, bp, v, beyond: merged_beta_at(prec, wf, mf, bp, v, smooth, beyond),
                         in_tra, detached if smooth else None)

    def blend(attached):
        return fma(bp.attach_weight, attached[0] - detached[0], detached[0]), attached[1] | detached[1]

    choices = [calibration, transition, detached]
    if smooth:
        choices += [blend(calibration), blend(transition)]
    beta = jnp.select([case == i for i in range(len(choices))], [c[0] for c in choices])
    error_beta = jnp.select([case == i for i in range(len(choices))], [c[1] for c in choices])
    return alpha, beta, error_alpha | error_beta


# ---------------------------------------------------------------------------
# the interpolants of the PNR angles
# ---------------------------------------------------------------------------

#: The capacity of the uniform grid of the PNR angle interpolants (LAL's
#: grid has ``(f_max - f_min) / deltaF`` points with ``deltaF >= 0.01 Hz``;
#: the cost is that of the points used).
PNR_GRID_CAPACITY = 524288
#: The points of the grid evaluated together (chunks beyond LAL's grid are
#: skipped under ``jax.jit``).
PNR_GRID_CHUNK = 1024


@fenced
def interpolation_grid(prec, wf, modes, version):
    """The uniform frequency grid of ``IMRPhenomX_PNR_GeneratePNRAngleInterpolants``
    (Hz): ``(i_start, n, delta_f)`` with the grid ``(i_start + k) delta_f``,
    ``k < n``, and the error flag. Also sets ``pnr_hm_mflow`` and
    ``pnr_hm_mfhigh`` (in place). ``modes`` is the static list of
    co-precessing modes ``(l, m' > 0)``."""
    ap, bp = prec.alpha_params, prec.beta_params
    mfhigh = bp.mf_beta_lower
    mflow = ap.mf_alpha_lower
    mfhigh = jnp.where((mfhigh > wf.f_cut_def) | (mfhigh < 0.1 * wf.f_ring), wf.f_ring, mfhigh)
    mflow = jnp.where((mflow > wf.f_cut_def) | (mfhigh < mflow), mfhigh / 2.0, mflow)
    prec.pnr_hm_mfhigh = mfhigh
    prec.pnr_hm_mflow = mflow

    uniform = wf.delta_f != 0.0
    f_min_22 = wf.f_min
    f_min_from_delta_f = jnp.floor(f_min_22 / jnp.where(uniform, wf.delta_f, 1.0)) * wf.delta_f
    f_min_22 = jnp.where(uniform & (f_min_from_delta_f < f_min_22), f_min_from_delta_f, f_min_22)
    f_max_22 = jnp.where(uniform, wf.f_max_prime, wf.f_max)
    f_min, f_max = f_min_22, f_max_22
    error = jnp.asarray(False)
    vanishing_odd = (wf.q == 1.0) & (wf.chi1l == wf.chi2l)
    for ell, emm in modes:
        if (ell, emm) == (2, 2):
            continue
        mf_rd_22 = wf.f_ring
        mf_rd_lm = ringdown_frequency(ell, emm, wf)
        mf_low, mf_high = pa.linear_frequency_map_transition_frequencies(emm, mf_rd_22, mf_rd_lm, mflow, mfhigh)
        lo, e1 = frequency_map(prec, hz_to_mf(f_min_22, wf.m_tot), ell, emm, mf_low, mf_high, mf_rd_22, mf_rd_lm)
        hi, e2 = frequency_map(prec, hz_to_mf(f_max_22, wf.m_tot), ell, emm, mf_low, mf_high, mf_rd_22, mf_rd_lm)
        skip = vanishing_odd if emm % 2 else jnp.asarray(False)
        lo = mf_to_hz(lo, wf.m_tot)
        hi = mf_to_hz(hi, wf.m_tot)
        f_min = jnp.where(~skip & (lo < f_min), lo, f_min)
        f_max = jnp.where(~skip & (hi > f_max), hi, f_max)
        error = error | (~skip & (e1 | e2))
    delta_f_int = interpolation_delta_f(prec, wf, f_min, version)
    f_min = jnp.where(fnms(2.0, delta_f_int, f_min) < 0.0, f_min / 2.0, fnms(2.0, delta_f_int, f_min))
    f_max = fma(2.0, delta_f_int, f_max)
    i_start = jnp.floor(f_min / delta_f_int)
    i_stop = jnp.floor(f_max / delta_f_int) + 1.0
    n = i_stop - i_start
    error = error | (n < 3) | (n > PNR_GRID_CAPACITY)
    return i_start, n, delta_f_int, error


#: Points on each side of a chunk in the local solves of the spline
#: equations (the influence of a point decays as ``(2 - sqrt(3))^k``:
#: ``~1e-18`` here).
SPLINE_OVERLAP = 32


def _natural_spline_chunk(y, start, n, h):
    """The second derivatives of GSL's natural cubic spline through ``y[k]``
    at ``k h``, ``k < n``, for the chunk of :data:`PNR_GRID_CHUNK` points
    from ``start``: the equations of the chunk and of
    :data:`SPLINE_OVERLAP` points on either side (with LAL's natural end
    conditions where they reach the ends of the grid) are solved, zero
    beyond ``n``."""
    cap = y.shape[0]
    chunk = PNR_GRID_CHUNK
    width = chunk + 2 * SPLINE_OVERLAP
    offsets = np.arange(width) - SPLINE_OVERLAP
    p = start + offsets
    window = y[jnp.clip(p, 0, cap - 1)]
    inside = (p >= 0) & (p < n)
    # the window's own ends (inside the grid): natural (the error decays
    # before the chunk)
    interior = (p >= 1) & (p < n - 1) & (offsets > -SPLINE_OVERLAP) & (offsets < width - 1 - SPLINE_OVERLAP)
    y_prev = jnp.concatenate([window[:1], window[:-1]])
    y_next = jnp.concatenate([window[1:], window[-1:]])
    lower = jnp.where(interior, h, 0.0).at[0].set(0.0)
    upper = jnp.where(interior, h, 0.0).at[-1].set(0.0)
    diag = jnp.where(interior, 4.0 * h, 1.0)
    rhs = jnp.where(interior, 6.0 * ((y_next - window) / h - (window - y_prev) / h), 0.0)
    m = jax.lax.linalg.tridiagonal_solve(lower, diag, upper, rhs[:, None])[:, 0]
    m = jnp.where(inside, m, 0.0)
    return m[SPLINE_OVERLAP:SPLINE_OVERLAP + chunk]


def _uniform_natural_splines(ys, n, h, n_eval):
    """:func:`_natural_spline_chunk` of each of the arrays ``ys`` for the
    chunks needed (:func:`~._math.blocked_map`, the cost that of the ``n``
    points): the second derivatives, zero beyond ``n``."""
    cap = ys[0].shape[0]
    chunk = PNR_GRID_CHUNK

    def local(index, args):
        ys, n, h = args
        return [_natural_spline_chunk(y, index * chunk, n, h) for y in ys]

    out = blocked_map(local, jnp.arange(cap // chunk), jnp.ceil(n_eval / chunk), (list(ys), n, h))
    return [m.reshape(cap) for m in out]


def _spline_at(y, m, n, x0, h, x):
    """Value and derivative of the uniform spline at ``x``."""
    k = jnp.clip(jnp.floor((x - x0) / h), 0, n - 2).astype(jnp.int64)
    xk = x0 + k * h
    y0, y1 = y[k], y[k + 1]
    m0, m1 = m[k], m[k + 1]
    dx = x - xk
    b = (y1 - y0) / h - h * (m1 + 2.0 * m0) / 6.0
    c = m0 / 2.0
    d = (m1 - m0) / (6.0 * h)
    return y0 + dx * (b + dx * (c + dx * d)), b + dx * (2.0 * c + 3.0 * dx * d)


#: The largest spacing (in ``Mf``) of the phase grid of the PNR
#: interpolants (``WaveformOptions.extended_phase_grid``): its spacing
#: grows like ``f^2`` (constant steps of the precession and nutation phases
#: at leading order) from LAL's spacing at its start up to this.
PHASE_GRID_MF = 1e-4


def _phase_grid(f0, f_end, h, m_tot):
    """The nodes ``f_k = F(s0 + k ds)`` of the phase grid from ``f0`` to
    (at least) ``f_end``, with ``s(f) = -1/f + f/fc^2``: spacing ``h`` at
    ``f0``, ``~h (f / f0)^2`` above it and ``mf_to_hz(PHASE_GRID_MF)`` (or
    ``h``, if larger) at high frequencies."""
    df_max = jnp.maximum(mf_to_hz(PHASE_GRID_MF, m_tot), h * (1.0 + 1e-6))
    fc2 = f0 * f0 * (df_max / h - 1.0)
    ds = h * (1.0 / (f0 * f0) + 1.0 / fc2)
    grid = Struct(fc2=fc2, ds=ds)
    grid.s0 = _phase_s(grid, f0)
    n = jnp.ceil((_phase_s(grid, f_end) - grid.s0) / ds) + 1.0
    grid.n = n
    grid.f_first = f0
    grid.f_last = _phase_f(grid, grid.s0 + (n - 1.0) * ds)
    return grid


def _phase_s(grid, f):
    return -1.0 / f + f / grid.fc2


def _phase_f(grid, s):
    """The inverse of :func:`_phase_s` (the stable root for either sign)."""
    root = jnp.sqrt(s * s + 4.0 / grid.fc2)
    low = 2.0 / jnp.where(s <= 0.0, root - s, 1.0)
    high = 0.5 * grid.fc2 * (s + root)
    return jnp.where(s <= 0.0, low, high)


def angle_interpolants(prec, wf, grid, version, active=True, extra_mf=None, n_extra=None):
    """``IMRPhenomX_PNR_GeneratePNRAngleInterpolants``: the PNR ``alpha`` and
    ``beta`` on the uniform grid, their splines and ``gamma`` integrated
    from them with Boole's rule. Returns a :class:`~._math.Struct` of the
    three splines and the error flag. Where ``active`` (traced) is false,
    nothing is evaluated (the prescription is not LAL's: under ``jax.vmap``
    every prescription is computed).

    The grid has room for :data:`PNR_GRID_CAPACITY` points; everything is
    computed in chunks of :data:`PNR_GRID_CHUNK` points, for the chunks
    holding LAL's ``n`` points only.

    With ``extra_mf`` (a 1-d array of ``Mf``, version 330), the SpinTaylor
    ``alpha`` of the splines at the first ``n_extra`` (traced, default all)
    of those points too, in the same loop (one evaluation of the splines to
    compile): ``st_extra``, the values and whether they lie within the
    knots, as :func:`_st_spline` gives them (the antisymmetric mode's
    points, see :func:`region_alpha_many`; zeros beyond the chunks
    needed)."""
    i_start, n, delta_f_int, error = grid
    prec, wf = barrier((prec, wf))
    n_eval = jnp.where(active, n, 0.0)
    chunk = PNR_GRID_CHUNK
    n_chunks = PNR_GRID_CAPACITY // chunk
    x0 = i_start * delta_f_int
    h = delta_f_int
    smooth = prec.smooth
    node_map = None
    if smooth and prec.phase_grid:
        node_map = _phase_grid(x0, (i_start + n - 1.0) * h, h, wf.m_tot)
        n = jnp.minimum(node_map.n, n)
        n_eval = jnp.where(active, n, 0.0)

    def frequencies(k, i_start, n, h):
        return (i_start + jnp.minimum(k, n - 1.0)) * h

    # the angles on the grid (the flag of a Fallback version passed in: the
    # loop's body must not close over batched values)
    fallback = isinstance(version, Fallback)

    def angles(index, args):
        prec, wf, i_start, n, h, msa, node_map = args
        version_ = Fallback(msa) if fallback else version
        k = index * chunk + jnp.arange(chunk, dtype=jnp.float64)
        valid = k < n
        if node_map is None:
            mf = hz_to_mf(frequencies(k, i_start, n, h), wf.m_tot)
        else:
            mf = hz_to_mf(_phase_f(node_map, node_map.s0 + jnp.minimum(k, n - 1.0) * node_map.ds), wf.m_tot)
        st_values = None
        if version == 330:
            # the SpinTaylor splines at the chunk's points, shared by the
            # angles below
            st, rotation, mf_scale, _, context = _st_context(prec, wf)
            st_values = st_angles.spline(st, rotation, mf, mf_scale, context)
        alpha, beta, err = uniform_angles(prec, wf, mf, version_, st_values, smooth)
        return jnp.where(valid, alpha, 0.0), jnp.where(valid, beta, 0.0), jnp.any(err & valid)

    args = (prec, wf, i_start, n, h, version.msa if fallback else None, node_map)
    st_extra = None
    if extra_mf is None or version != 330:
        alpha, beta, errors = blocked_map(angles, jnp.arange(n_chunks), jnp.ceil(n_eval / chunk), args)
    else:
        n_extra_chunks = -(-extra_mf.shape[0] // chunk)
        padded = jnp.concatenate([extra_mf, jnp.broadcast_to(extra_mf[-1:], (n_extra_chunks * chunk - extra_mf.shape[0],))])

        def angles_or_extra(index, mf_extra, is_extra, args):
            prec, wf, i_start, n, h, _, node_map = args
            k = index * chunk + jnp.arange(chunk, dtype=jnp.float64)
            valid = k < n
            if node_map is None:
                mf = hz_to_mf(frequencies(k, i_start, n, h), wf.m_tot)
            else:
                mf = hz_to_mf(_phase_f(node_map, node_map.s0 + jnp.minimum(k, n - 1.0) * node_map.ds), wf.m_tot)
            mf = jnp.where(is_extra, mf_extra, mf)
            st, rotation, mf_scale, _, context = _st_context(prec, wf)
            st_values = st_angles.spline(st, rotation, mf, mf_scale, context)

            def on_grid(operands):
                prec, wf, mf, st_values, valid = operands
                alpha, beta, err = uniform_angles(prec, wf, mf, version, st_values, smooth)
                return jnp.where(valid, alpha, 0.0), jnp.where(valid, beta, 0.0), jnp.any(err & valid)

            def skip(operands):
                mf = operands[2]
                return jnp.zeros(mf.shape), jnp.zeros(mf.shape), jnp.asarray(False)

            values = cond(is_extra, skip, on_grid, (prec, wf, mf, st_values, valid))
            return values, (st_values[0], st_values[4])

        needed = n_extra_chunks if n_extra is None else jnp.ceil(jnp.minimum(n_extra, extra_mf.shape[0]) / chunk)
        (alpha, beta, errors), st_extra = blocked_map_extra(
            angles_or_extra, jnp.arange(n_chunks), jnp.ceil(n_eval / chunk), padded.reshape(n_extra_chunks, chunk),
            args, jnp.where(active, needed, 0))
        st_extra = tuple(x.reshape(-1)[:extra_mf.shape[0]] for x in st_extra)
    error = error | jnp.any(errors)
    alpha, beta = alpha.reshape(-1), beta.reshape(-1)
    if node_map is not None:
        # splines in the index of the nodes
        splines = _geometric_interpolants(alpha, beta, jnp.zeros(()), n, jnp.ones(()), n_eval, error)
        splines.node_map = node_map
        splines.st_extra = st_extra
        return splines
    if smooth:
        splines = _geometric_interpolants(alpha, beta, i_start, n, h, n_eval, error)
        splines.st_extra = st_extra
        return splines
    m_alpha, m_beta = _uniform_natural_splines([alpha, beta], n, h, n_eval)

    # gamma from the minimal rotation condition with Boole's rule, interval
    # by interval, summed chunk by chunk
    def boole(index, total, args):
        alpha, m_alpha, beta, m_beta, i_start, n, h, x0 = args
        cap = alpha.shape[0]
        # the intervals [k, k + 1] of the chunk: values at k and k + 1
        i = index * chunk + jnp.arange(chunk)
        here, after = jnp.clip(i, 0, cap - 1), jnp.clip(i + 1, 0, cap - 1)
        t1 = frequencies(here.astype(jnp.float64), i_start, n, h)
        t2 = frequencies(after.astype(jnp.float64), i_start, n, h)
        y0, y1, m0, m1 = alpha[here], alpha[after], m_alpha[here], m_alpha[after]
        yb0, yb1, mb0, mb1 = beta[here], beta[after], m_beta[here], m_beta[after]
        points = jnp.stack([t2, fma(3.0, t1, t2) / 4.0, (t1 + t2) / 2.0, t1, fma(3.0, t2, t1) / 4.0])
        dx = points - (x0 + i * h)
        b = (y1 - y0) / h - h * (m1 + 2.0 * m0) / 6.0
        dalpha = b + dx * (m0 + (m1 - m0) / (2.0 * h) * dx)
        bb = (yb1 - yb0) / h - h * (mb1 + 2.0 * mb0) / 6.0
        beta_x = yb0 + dx * (bb + dx * (mb0 / 2.0 + dx * (mb1 - mb0) / (6.0 * h)))
        f = pa.alphadot_cosbeta(dalpha, beta_x)
        total_f = fma(7.0, f[0], fma(32.0, f[1], fma(12.0, f[2], fma(7.0, f[3], 32.0 * f[4]))))
        increments = jnp.where(i + 1 < n, 1.0 / 90.0 * (t2 - t1) * total_f, 0.0)
        # gamma at the chunk's points: the integral up to each
        sums = jnp.cumsum(increments)
        gamma = total + jnp.concatenate([jnp.zeros(1), sums[:-1]])
        return gamma, total + sums[-1]

    gamma, _ = blocked_scan(boole, jnp.arange(n_chunks), jnp.ceil(n_eval / chunk), jnp.zeros(()),
                            (alpha, m_alpha, beta, m_beta, i_start, n, h, x0))
    gamma = gamma.reshape(-1)
    (m_gamma,) = _uniform_natural_splines([gamma], n, h, n_eval)
    return Struct(alpha=alpha, beta=beta, gamma=gamma, m_alpha=m_alpha, m_beta=m_beta, m_gamma=m_gamma, n=n, x0=x0,
                  h=h, error=error, st_extra=st_extra)


#: Half-width in ``cos(beta)`` of the band around ``beta = pi / 2`` across
#: which the interpolants of the smooth angles go from the form regular at
#: ``beta = 0`` to the one regular at ``beta = pi``.
HEMISPHERE_SOFT = 0.3


def _wrap(x):
    """``x`` modulo ``2 pi``, in ``[-pi, pi]``."""
    return x - 2 * PI * jnp.round(x / (2 * PI))


def _geometric_interpolants(alpha, beta, i_start, n, h, n_eval, error):
    """The interpolants of the smooth angles: instead of LAL's splines of
    ``alpha``, ``beta`` and ``gamma``, splines of the unit vector ``(u, v,
    w) = (sin(beta) cos(alpha), sin(beta) sin(alpha), cos(beta))`` and of
    ``gamma + alpha`` and ``gamma - alpha``, from which
    :func:`spline_angles` takes ``alpha = atan2(v, u)`` and ``gamma``.

    Where L passes close to J (or to -J), ``alpha`` swings by nearly ``pi``
    between the points of the grid, and the integrand of LAL's ``gamma``
    peaks there: LAL's splines cannot follow either, and their errors jump
    when the branch of the swing flips with the parameters. ``u``, ``v``,
    ``w`` are smooth there, and so is ``gamma + alpha`` near ``beta = 0``
    and ``gamma - alpha`` near ``beta = pi``: across each interval of the
    grid, ``gamma`` (LAL's ``-int alpha' cos(beta) df``) changes by

    * ``-d alpha + int alpha' (1 - cos(beta))`` for ``beta < pi / 2``, and
    * ``d alpha - int alpha' (1 + cos(beta))`` for ``beta > pi / 2``,

    where ``d alpha`` is the change of ``alpha`` modulo ``2 pi`` and
    ``alpha' (1 -+ cos(beta)) = (u v' - v u') / (r (r +- w))`` (``r =
    |(u, v, w)|``) is regular in its hemisphere (Boole's rule, as LAL's
    ``gamma``). The two are blended across a band of ``cos(beta)``
    (:data:`HEMISPHERE_SOFT`), where both are regular and agree. The
    angles enter the waveform through integer multiples of ``alpha`` and
    ``gamma`` only, so that their branches do not matter; ``gamma`` is
    LAL's up to a constant, which the offset at the reference frequency
    removes."""
    chunk = PNR_GRID_CHUNK
    n_chunks = PNR_GRID_CAPACITY // chunk
    x0 = i_start * h
    sin_beta = jnp.sin(beta)
    u, v, w = sin_beta * jnp.cos(alpha), sin_beta * jnp.sin(alpha), jnp.cos(beta)
    m_u, m_v, m_w = _uniform_natural_splines([u, v, w], n, h, n_eval)
    raw = st_angles.small_atan2(v, u)

    def frequencies(k, i_start, n, h):
        return (i_start + jnp.minimum(k, n - 1.0)) * h

    def boole(index, totals, args):
        u, m_u, v, m_v, w, m_w, raw, i_start, n, h, x0 = args
        cap = u.shape[0]
        i = index * chunk + jnp.arange(chunk)
        here, after = jnp.clip(i, 0, cap - 1), jnp.clip(i + 1, 0, cap - 1)
        t1 = frequencies(here.astype(jnp.float64), i_start, n, h)
        t2 = frequencies(after.astype(jnp.float64), i_start, n, h)
        points = jnp.stack([t2, fma(3.0, t1, t2) / 4.0, (t1 + t2) / 2.0, t1, fma(3.0, t2, t1) / 4.0])
        dx = points - (x0 + i * h)

        def piece(y, m):
            y0, y1, m0, m1 = y[here], y[after], m[here], m[after]
            b = (y1 - y0) / h - h * (m1 + 2.0 * m0) / 6.0
            value = y0 + dx * (b + dx * (m0 / 2.0 + dx * (m1 - m0) / (6.0 * h)))
            return value, b + dx * (m0 + (m1 - m0) / (2.0 * h) * dx)

        def boole_sum(f):
            return 1.0 / 90.0 * (t2 - t1) * fma(7.0, f[0], fma(32.0, f[1], fma(12.0, f[2], fma(7.0, f[3], 32.0 * f[4]))))

        (ux, du), (vx, dv), (wx, _) = piece(u, m_u), piece(v, m_v), piece(w, m_w)
        r = safe_sqrt(ux * ux + vx * vx + wx * wx, 1e-300)
        twist = ux * dv - vx * du
        north = boole_sum(safe_div(twist, r * (r + wx)))
        south = boole_sum(safe_div(twist, r * (r - wx)))
        d_alpha = _wrap(raw[after] - raw[here])
        weight = pa._ramp(wx[2] / r[2], HEMISPHERE_SOFT)
        d_gamma = fma(weight, north - d_alpha, (1.0 - weight) * (d_alpha - south))
        used = i + 1 < n
        increments = jnp.stack([jnp.where(used, d_gamma, 0.0), jnp.where(used, d_alpha, 0.0)])
        sums = jnp.cumsum(increments, axis=1)
        values = totals[:, None] + jnp.concatenate([jnp.zeros((2, 1)), sums[:, :-1]], axis=1)
        return values, totals + sums[:, -1]

    first = jnp.stack([jnp.zeros(()), raw[0]])
    values, _ = blocked_scan(boole, jnp.arange(n_chunks), jnp.ceil(n_eval / chunk), first,
                             (u, m_u, v, m_v, w, m_w, raw, i_start, n, h, x0))
    # (chunks, 2, chunk) -> (2, capacity)
    values = jnp.moveaxis(values, 1, 0).reshape(2, -1)
    gamma, alpha_unwrapped = values[0], values[1]
    plus, minus = gamma + alpha_unwrapped, gamma - alpha_unwrapped
    m_plus, m_minus, m_unwrapped = _uniform_natural_splines([plus, minus, alpha_unwrapped], n, h, n_eval)
    return Struct(u=u, v=v, w=w, m_u=m_u, m_v=m_v, m_w=m_w, plus=plus, m_plus=m_plus, minus=minus, m_minus=m_minus,
                  unwrapped=alpha_unwrapped, m_unwrapped=m_unwrapped, n=n, x0=x0, h=h, error=error)


def spline_angles(splines, f):
    """The PNR ``alpha``, ``beta`` and ``gamma`` at ``f`` (Hz) and whether
    ``f`` lies on the grid (GSL fails outside). For the interpolants of
    the smooth angles (:func:`_geometric_interpolants`), ``alpha`` and
    ``gamma`` are known modulo ``2 pi``."""
    inside = (f >= splines.x0) & (f <= splines.x0 + (splines.n - 1.0) * splines.h)
    if "u" in splines:
        x = f
        if "node_map" in splines:
            m = splines.node_map
            inside = (f >= m.f_first) & (f <= m.f_last)
            x = (_phase_s(m, f) - m.s0) / m.ds

        def at(name):
            return _spline_at(getattr(splines, name), getattr(splines, "m_" + name), splines.n, splines.x0, splines.h,
                              x)[0]

        u, v, w = at("u"), at("v"), at("w")
        a = st_angles.small_atan2(v, u)
        b = st_angles._atan2(safe_sqrt(u * u + v * v, 1e-300), w)
        # alpha on the branch of the unwrapped spline (which matters only
        # in the band, where both forms of gamma are used and alpha is
        # smooth)
        unwrapped = at("unwrapped")
        a_u = unwrapped + _wrap(a - unwrapped)
        weight = pa._ramp(w / safe_sqrt(u * u + v * v + w * w, 1e-300), HEMISPHERE_SOFT)
        gamma = fma(weight, at("plus") - a_u, (1.0 - weight) * (at("minus") + a_u))
        return a, b, gamma, inside
    a, _ = _spline_at(splines.alpha, splines.m_alpha, splines.n, splines.x0, splines.h, f)
    b, _ = _spline_at(splines.beta, splines.m_beta, splines.n, splines.x0, splines.h, f)
    g, _ = _spline_at(splines.gamma, splines.m_gamma, splines.n, splines.x0, splines.h, f)
    return a, b, g, inside


@fenced
def remap_theta_jsf(prec, wf, beta_ref):
    """``IMRPhenomX_PNR_RemapThetaJSF`` (in place)."""
    skip = prec.j0_norm < 1e-10
    # (the smooth angles remap for all in-plane spins: the remapped frame
    # tends to LAL's as they vanish, without LAL's step at 1e-3)
    small = jnp.asarray(False) if prec.smooth else chi_in_plane(prec.chi1, prec.chi2) < 1e-3
    theta_j_sf = jnp.where(skip | small, prec.theta_j_sf, beta_ref)
    n_sf = line_of_sight(wf)
    tmp = rotate_y(-theta_j_sf, rotate_z(-prec.phi_j_sf, n_sf))
    kappa = atan2_tol(tmp[1], tmp[0], MAX_TOL_ATAN)
    alpha0 = PI - kappa
    n_jf = rotate_z(-kappa, tmp)
    remapped = _copy(prec)
    remapped.theta_j_sf = theta_j_sf
    remapped.kappa = kappa
    remapped.alpha0 = alpha0
    theta_jn = acos_safe(n_jf[2])
    zeta = polarization_angle(remapped, wf, n_jf[0], n_jf[2])
    prec.alpha_offset = jnp.where(skip, prec.alpha_offset, prec.alpha_offset - jnp.where(small, prec.alpha0, alpha0))
    keep = skip | small
    prec.theta_j_sf = jnp.where(keep, prec.theta_j_sf, theta_j_sf)
    prec.kappa = jnp.where(keep, prec.kappa, kappa)
    prec.alpha0 = jnp.where(keep, prec.alpha0, alpha0)
    prec.theta_jn = jnp.where(keep, prec.theta_jn, theta_jn)
    prec.zeta_polarization = jnp.where(keep, prec.zeta_polarization, zeta)
    return prec


def nnlo_angles(prec, omega):
    """The NNLO angles ``(alpha, cos(beta), epsilon)`` at ``omega = 2 pi Mf /
    m'`` without the PNR angles (``Get_alpha_beta_epsilon``)."""
    logomega = jnp.log(omega)
    omega_cbrt = jnp.cbrt(omega)
    alpha, epsilon = _nnlo.evaluate(prec.nnlo, omega, logomega, omega_cbrt, omega_cbrt * omega_cbrt)
    v = omega_cbrt
    l = _prec.l_ansatz(prec.l, v, prec.eta / v)
    s = prec.sperp / (l + prec.sl)
    cos_beta = jnp.where(l + prec.sl < 0, -1.0, 1.0) / jnp.sqrt(1.0 + s * s)
    return alpha - prec.alpha_offset, cos_beta, epsilon - prec.epsilon_offset
