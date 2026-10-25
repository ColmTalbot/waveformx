# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The PNR precession of IMRPhenomXPNR (LALSimulation 6.2.0 with the fix of
lalsuite !2593), after waveformrs's ``imrphenomx/pnr.rs``:
``IMRPhenomXGetAndSetPrecessionVariables`` with ``PhenomXPrecVersion`` 330
and ``PhenomXPFinalSpinMod`` 7, ``LALSimIMRPhenomX_PNR.c``,
``_PNR_internals.c``, ``_PNR_alpha.c``, ``_PNR_beta.c`` and the SpinTaylor
angles of version 330.

The model uses the SpinTaylor angles (version 330) for every binary, with
LAL's discrete switches made continuous (the "smooth" angles below: the
two-spin construction for every binary, the connection of the PNR
``beta``, the edge of the transition region at ``q = 12``, ...). LAL's own
prescriptions for small in-plane spins and failed integrations (versions
102, NNLO, and 223, MSA) are not used; the functions keep a *static*
``version`` (a Python int) where LAL's code depends on it, as the set-up
of version 330 itself evaluates parts of version 223.

The set-up is split into :func:`common_setup` (before the SpinTaylor
integration) and :func:`finish_setup`.
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
    FRAC_PI_2, MTSUN_SI, PI, FencedModule, Struct, barrier, blocked_map, blocked_map_extra, blocked_scan,
    chunked_map, dbl_epsilon, fenced, fma, fnms, hz_to_mf, integer, mf_to_hz, real, rotate_y, rotate_z, safe_div,
    safe_sqrt, tiny,
)
from .hm import qnm as _hm_qnm
from ._transforms import cond
from .spintaylor import angles as st_angles
from .spintaylor import lean as st_lean

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
    return safe_sqrt(fma(x, x, y * y), tiny())


def chi_in_plane(chi1, chi2):
    return safe_sqrt(fma(chi2[1], chi2[1], fma(chi2[0], chi2[0], fma(chi1[0], chi1[0], chi1[1] * chi1[1]))), tiny())


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


def common_setup(wf, m1_si, m2_si, chi1, chi2, mode_range_, phase_grid=False, grid_range=None, reference=False):
    """The part of ``IMRPhenomXGetAndSetPrecessionVariables`` before the
    SpinTaylor integration: the spin quantities and the start of the
    integration that the PNR angle interpolants need.

    ``m1_si``, ``m2_si`` (solar masses) and the spins
    (3-vectors) are ordered as ``XLALIMRPhenomXPCheckMassesAndSpins`` orders
    them; ``mode_range_`` is :func:`mode_range` of the co-precessing modes
    (static, or traced with ``M_MAX = L_MAX_PNR``, as for the J-frame
    modes). ``phase_grid`` (static) selects the phase grid of the PNR
    interpolants (:func:`angle_interpolants`); ``grid_range`` (static,
    :func:`mode_range` of the modes of the grid of the PNR interpolants)
    lowers the start of the one integration by the ratio of the largest
    ``m``, where the grid reaches lower than ``mode_range_`` (the J-frame
    modes; LAL integrates again there), and ``reference`` (static) marks
    the mass-scaled reference frequency (``WaveformOptions.reference_mf``),
    which can lie below the band: the integration then starts below it too
    (:data:`REFERENCE_FLOW`). Returns ``(prec, wf)``; ``wf`` gets the tuned
    co-precessing flag as LAL sets it.
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
    pi_gm = PI * (m1_si + m2_si) * MTSUN_SI
    chi1_norm = safe_sqrt(fma(chi1z, chi1z, fma(chi1x, chi1x, chi1y * chi1y)), tiny())
    chi2_norm = safe_sqrt(fma(chi2z, chi2z, fma(chi2x, chi2x, chi2y * chi2y)), tiny())
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
        pnr_hm_mflow=jnp.zeros((), real()),
        pnr_hm_mfhigh=jnp.zeros((), real()),
        reference=reference,
        phase_grid=phase_grid,
        pole=jnp.asarray(1.0 - (1e-15 if real() == np.float64 else dbl_epsilon())),
        transition_q=jnp.asarray(pa.PNR_Q_WINDOW_UPPER + Q_SMOOTH_WIDTH),
        q12_weight=jnp.zeros((), real()),
        scaling_weight=jnp.zeros((), real()),
        error_spin=(chi1_norm > 1.0) | (chi2_norm > 1.0),
    )
    prec.l = _prec.three_pn(eta, wf.delta, chi1z, chi2z)
    # ---- the start of the SpinTaylor integration (version 330, with the
    # tuned angles) ----
    wf.delta_mf = jnp.where(wf.delta_f == 0.0, _prec.delta_mf_from_wfstruct(wf), wf.delta_mf)
    flow = wf.f_min
    flow = jnp.where(wf.delta_f != 0.0, jnp.floor(wf.f_min / jnp.where(wf.delta_f != 0.0, wf.delta_f, 1.0)) * wf.delta_f, flow)
    fmin_hm_inspiral = flow * 2.0 / m_max
    # the PNR variables and the alpha and beta coefficients with a fake
    # version 223 to catch edge cases
    tmp = _copy(prec)
    set_pnr_variables(tmp, wf, 223)
    alpha_params = precompute_alpha_coefficients(tmp, wf, 223)
    beta_params = pa.beta_connection_frequencies(precompute_beta_coefficients(tmp, wf, 223))
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
    # one integration (see model._spintaylor_setup): lower by the ratio of the
    # largest m of the grid's modes (the frequency maps of the modes scale
    # as 2 / m: the J-frame modes), and by half for heavy binaries, whose
    # grid can start lower still where the frequency maps of the higher
    # modes approach their ringdown maps (LAL integrates again there; of
    # 1400 random binaries, one, of 450 solar masses at Mf = 0.006, needed
    # a factor 0.66). Heavy binaries take few steps.
    m_grid = m_max if grid_range is None else grid_range[1]
    heavy = hz_to_mf(flow, wf.m_tot) > LEAN_HEAVY_MF
    prec.st_flow_lean = flow * m_max / m_grid * jnp.where(heavy, 0.5, 1.0)
    if reference:
        # the grid of the PNR interpolants reaches down to f_ref less
        # REFERENCE_NODES nodes, no lower than ~0.31 f_ref
        # (interpolation_grid)
        # (where LAL's start fails, as for heavy binaries with m' = 1 or
        # 3 only, the start below f_ref, which the grid then needs)
        low = REFERENCE_FLOW * wf.f_ref
        lal_ok = ~jnp.isnan(flow) & (flow > 0.0)
        prec.st_flow_lean = soft_min(jnp.where(lal_ok, prec.st_flow_lean, low), low, 0.1 * low)
    prec.st_flow_ok = ~jnp.isnan(flow) & (flow > 0.0)
    if reference:
        prec.st_flow_ok = ~jnp.isnan(prec.st_flow_lean) & (prec.st_flow_lean > 0.0)
    # the end of the integration: fRING + 8 fDAMP of the aligned-spin struct
    prec.st_fcut = mf_to_hz(wf.f_ring + 8.0 * wf.f_damp, wf.m_tot)
    prec.m_min, prec.m_max, prec.l_max_pnr = m_min, m_max, l_max
    return prec, wf


#: The model with the mass-scaled reference integrates from at most
#: this fraction of ``f_ref`` (:func:`common_setup`).
REFERENCE_FLOW = 0.25
#: The nodes of the grid of the PNR interpolants at least on each side of
#: the mass-scaled reference frequency (:func:`interpolation_grid`).
REFERENCE_NODES = 4


def soft_min(a, b, soft):
    """``min(a, b)``, blended (C2) across ``|a - b| < soft``: between the
    two, at most ``min(a, b) + soft``."""
    w = pa._ramp(b - a, soft)
    return fma(w, a - b, b)


def soft_max(a, b, soft):
    """``max(a, b)``, blended as :func:`soft_min`."""
    return -soft_min(-a, -b, soft)


#: Above this ``Mf`` of the start of the SpinTaylor integration, the
#: integration starts from half of it (:func:`common_setup`).
LEAN_HEAVY_MF = 0.004


def spintaylor_solution(prec, wf, m1, m2, active=True):
    """The SpinTaylor solution (:func:`.spintaylor.lean.solution`, in
    place of ``IMRPhenomX_InspiralAngles_SpinTaylor``) from the start
    computed by :func:`common_setup` to ``(fRING + 8 fDAMP) / M`` of the
    aligned-spin struct."""
    flow_safe = jnp.where(prec.st_flow_ok, prec.st_flow_lean, wf.f_min)
    sol = st_lean.solution(m1, m2, prec.chi1, prec.chi2, wf.f_ref, flow_safe, prec.st_fcut, active)
    sol.failed = sol.failed | ~prec.st_flow_ok
    return sol


# ---------------------------------------------------------------------------
# the set-up for a given version
# ---------------------------------------------------------------------------


def _by_version(version, nnlo, msa, other=None):
    """``nnlo()`` for version 102, ``msa()`` for 223, ``other()`` for 330."""
    if version == 102:
        return nnlo()
    if version == 223:
        return msa()
    return other()


def two_spin(prec, version):
    """``IMRPhenomX_PNR_CheckTwoSpin`` with the two-spin construction for
    every binary of versions 223 and 330: LAL's (for ``|chi2| >= 1e-3``)
    changes the PNR ``beta`` by up to a mismatch of ~1e-1 at ``|chi2| =
    1e-3``, a step that does not vanish with the second spin."""
    return jnp.asarray(version >= 200)


def _off_poles(prec, costheta):
    """``costheta`` kept off ``+-1`` by ``prec.pole`` (``1 - 1e-15``, ``1 -
    FLT_EPSILON`` in float32): the fits of the single-spin system and the
    PN waveform ``beta`` take ``sqrt(1 - costheta^2)`` or
    ``acos(costheta)``, whose derivatives are infinite for aligned spins (a
    change of their sine below ``5e-8``)."""
    return jnp.clip(costheta, -prec.pole, prec.pole)


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
    # (zero for aligned spins, not hypot2's floor: the ratios of the PNR beta
    # divide by its square)
    chis = _prec.hypot_safe(fma(m1sq, chi1x, m2sq * chi2x), fma(m1sq, chi1y, m2sq * chi2y)) / (m1 * m1)
    s = jnp.sin((q - 1.0) * PI)
    c = jnp.cos((q - 1.0) * PI)
    if version == 330:
        chiperp = chis
    else:
        chiperp = jnp.where(q <= 1.5, fma(s * s, prec.chi_p, c * c * chis), prec.chi_p)
    antisymmetric_chis = _prec.hypot_safe(fma(m1sq, chi1x, -(m2sq * chi2x)), fma(m1sq, chi1y, -(m2sq * chi2y))) / (m1 * m1)
    chiperp_antisymmetric = jnp.where(q <= 1.5, fma(s * s, prec.chi_p, c * c * antisymmetric_chis), prec.chi_p)
    chi_mag = hypot2(chipar, chiperp)
    prec.chi_single_spin = chi_mag
    prec.chi_parallel_single_spin = chipar
    prec.chi_perp_single_spin = chiperp
    chi_mag_antisymmetric = hypot2(chipar, chiperp_antisymmetric)
    prec.chi_single_spin_antisymmetric = chi_mag_antisymmetric
    costheta = jnp.where(chi_mag >= 1.0e-6, safe_div(chipar, chi_mag), 0.0)
    prec.costheta_single_spin = _off_poles(prec, costheta)
    # (the polar angles and sines from the components of the spins: LAL's
    # acos and sqrt(1 - cos^2) of their cosines lose half the digits of
    # small in-plane spins, and those kept off the poles leave a residual
    # in-plane spin, and an opening angle, for aligned spins; LAL's
    # choices below |chi| = 1e-6, which tend to these, are not kept)
    prec.sintheta_single_spin = safe_div(chiperp, chi_mag)
    prec.theta_antisymmetric = _prec._atan2_safe(chiperp_antisymmetric, chipar)
    xfparr = utilities.final_spin_2017(wf.eta, chipar, 0.0)
    qfactor = q / (1.0 + q)
    xfperp = qfactor * qfactor * chiperp
    prec.theta_final_single_spin = _prec._atan2_safe(xfperp, xfparr)
    prec.pnr_hm_mflow = jnp.zeros((), real())
    prec.pnr_hm_mfhigh = jnp.zeros((), real())
    # beyond q = 12, from LAL's angles at the edge of the transition region
    # to LAL's beyond it (see merged_beta_at, frequency_map)
    x = jnp.clip((q - pa.PNR_Q_WINDOW_UPPER) / Q_SMOOTH_WIDTH, 0.0, 1.0)
    prec.q12_weight = x * x * x * fma(x, fma(6.0, x, -15.0), 10.0)
    weight = jnp.where(prec.chi_single_spin > pa.PNR_CHI_WINDOW_UPPER, 1.0, prec.q12_weight)
    prec.scaling_weight = weight
    prec.pnr_inspiral_scaling = weight >= 1.0
    return prec


@fenced
def set_coprec_params(prec, wf, tuned, version):
    """``IMRPhenomX_PNR_GetAndSetCoPrecParams`` (``wf.coprec`` in place);
    ``tuned`` is traced (the PNR machinery is on)."""
    a1 = prec.chi_single_spin
    # (the polar angle of the single spin from its components: LAL's acos
    # of its cosine, kept off the poles, loses half the digits of small
    # in-plane spins, and is pi / 2 below |chi| = 1e-6, where dev vanishes)
    theta_ls = _prec._atan2_safe(prec.chi_perp_single_spin, prec.chi_parallel_single_spin)
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
    zero = jnp.zeros((), real())
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


#: The fits of the PNR coefficients of the form ``chi sin(theta) g(eta, chi,
#: cos(theta))``: the table of the polynomial and ``g`` of it.
_IN_PLANE_FITS = {
    "alpha_a1": (_pnr_fits.COEFF_A1, lambda p: p * p),
    "alpha_a2": (_pnr_fits.COEFF_A2, lambda p: -(p * p)),
    "beta_b0": (_pnr_fits.COEFF_B0, lambda p: p),
    "beta_b1": (_pnr_fits.COEFF_B1, jnp.exp),
    "beta_b2": (_pnr_fits.COEFF_B2, lambda p: -jnp.exp(p)),
    "beta_bf": (_pnr_fits.COEFF_BF, lambda p: p),
}


def in_plane_fit(name, eta, chi, costheta, sintheta):
    """The fit ``pnr_fits.<name>`` (one of :data:`_IN_PLANE_FITS`) with
    ``sin(theta)`` given: LAL's takes it from ``costheta``, which for
    aligned spins is kept off the poles, so that the fit, which vanishes
    with the in-plane spin, does not."""
    coeff, g = _IN_PLANE_FITS[name]
    return chi * sintheta * g(pnr_fits.evaluate(coeff, eta, chi, costheta))


def ringdown_pnr_beta(prec, wf):
    """``IMRPhenomX_PNR_GenerateRingdownPNRBeta``."""
    return pa.beta_window(
        prec.theta_final_single_spin
        - in_plane_fit("beta_bf", wf.eta, prec.chi_single_spin, prec.costheta_single_spin, prec.sintheta_single_spin)
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
#: across which the grid of the PNR interpolants is refined
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
        # (at equal masses, deltam == 0, psi1 and psi2 are infinite: LAL's
        # dpsi is NaN and its grid the coarse one without the two-spin
        # oscillations, a step in the waveform of ~1e-3 at q = 1; dpsi
        # diverges as q -> 1, so that the limit is the finest grid)
        dpsi_inv = jnp.where(deltam == 0.0, 0.0, jnp.abs(safe_div(1.0, dpsi)))
        l_fmin = wf.m_tot * wf.m_tot * _prec.l_ansatz(l_coeffs, v0, wf.eta / v0)
        beta_min = jnp.arctan2(jnp.abs(prec.s1_perp - prec.s2_perp), l_fmin + prec.sl)
        beta_max = jnp.arctan2(prec.s1_perp + prec.s2_perp, l_fmin + prec.sl)
        # LAL samples four times more finely where L comes close to J (the
        # grid, and the waveform, jump there): here by a factor that goes
        # from 1 to 4 across bands around LAL's thresholds. (For vanishing
        # in-plane spins, the ratio is its limit, whose derivative does not
        # divide by beta_max^2.)
        tiny = beta_max < (1e-100 if real() == np.float64 else 1e-30)
        limit = safe_div(jnp.abs(prec.s1_perp - prec.s2_perp), prec.s1_perp + prec.s2_perp)
        ratio = jnp.where(tiny, limit, safe_div(beta_min, jnp.where(tiny, 1.0, beta_max)))
        w = pa._ramp(0.01 - beta_min, GRID_SOFT_BETA) * pa._ramp(0.55 - ratio, GRID_SOFT_RATIO)
        dpsi_inv = dpsi_inv / fma(3.0, w, 1.0)
        delta_f_twospin = mf_to_hz(dpsi_inv / 4.0, wf.m_tot)
        use_twospin = two_spin(prec, version) & (delta_f_twospin < delta_f_alpha) & ~jnp.isnan(dpsi)
        if real().itemsize == 4:
            # (in float32 the MSA system of in-plane spins below 1e-6
            # underflows, and its derivatives are not finite: the
            # oscillations it estimates are negligible there)
            use_twospin = use_twospin & (prec.chi_in_plane >= 1e-6)
        return jnp.where(use_twospin, jnp.maximum(delta_f_twospin, 1e-2), plain)

    if version // 100 == 3:
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
    costheta, sintheta = prec.costheta_single_spin, prec.sintheta_single_spin
    a2 = jnp.minimum(in_plane_fit("alpha_a2", eta, chi, costheta, sintheta), 0.0)
    a3 = jnp.maximum(jnp.abs(pnr_fits.alpha_a3(eta, chi, costheta)), 0.00001)
    max_a2 = PI * PI * jnp.sqrt(a3)
    a2 = jnp.where(a2 < -max_a2, -max_a2, a2)
    return pa.alpha_params(
        a1=jnp.abs(in_plane_fit("alpha_a1", eta, chi, costheta, sintheta)), a2=a2, a3=a3, a4=pnr_fits.alpha_a4(eta, chi, costheta)
    )


@fenced
def precompute_beta_coefficients(prec, wf, version):
    """``IMRPhenomX_PNR_precompute_beta_coefficients``."""
    eta, chi = fit_parameters(prec, wf, version)
    costheta, sintheta = prec.costheta_single_spin, prec.sintheta_single_spin
    b4 = jnp.maximum(pnr_fits.beta_b4(eta, chi, costheta), 175.0)
    b2 = in_plane_fit("beta_b2", eta, chi, costheta, sintheta)
    return pa.beta_params(
        b0=prec.theta_final_single_spin - in_plane_fit("beta_b0", eta, chi, costheta, sintheta),
        b1=in_plane_fit("beta_b1", eta, chi, costheta, sintheta),
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
    # (zero for aligned spins: a floor would tilt the spins by it)
    lperp = _prec.hypot_safe(lx, ly)
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
    if version == 330:
        prec.fmin_integration = st.f_min_pn
        prec.mfmin_integration = hz_to_mf(st.f_min_pn, wf.m_tot)
        lx, ly, lz = st.lnhat_end
        prec.chi1_evolved, prec.chi2_evolved = rotate_to_l_frame(lx, ly, lz, st.s1_end, st.s2_end)
    else:
        prec.fmin_integration = jnp.zeros((), real())
        prec.mfmin_integration = jnp.zeros((), real())
    set_pnr_variables(prec, wf, 330 if version == 330 else 223)
    wf = set_coprec_params(prec, wf, jnp.asarray(True), version)
    if version == 223:
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
    # (LAL's acos(J0z / |J0|) and acos(J0.N / |J0|) lose half the digits of
    # small angles, those of small in-plane spins: their atan2 forms)
    theta_j_sf = _prec._atan2_safe(_prec.hypot_safe(j0[0], j0[1]), j0[2])
    prec.theta_j_sf = jnp.where(prec.j0_norm < 1e-10, 0.0, theta_j_sf)
    prec.phi_j_sf = atan2_tol(j0[1], j0[0], MAX_TOL_ATAN)
    wf.phi0 = jnp.zeros_like(wf.phi0)
    n_sf = line_of_sight(wf)
    tmp = rotate_y(-prec.theta_j_sf, rotate_z(-prec.phi_j_sf, n_sf))
    prec.kappa = atan2_tol(tmp[1], tmp[0], MAX_TOL_ATAN)
    prec.alpha0 = PI - prec.kappa
    j0_dot_n = fma(j0[2], n_sf[2], fma(j0[0], n_sf[0], j0[1] * n_sf[1]))
    j0_cross_n = [fnms(j0[2], n_sf[1], j0[1] * n_sf[2]), fnms(j0[0], n_sf[2], j0[2] * n_sf[0]),
                  fnms(j0[1], n_sf[0], j0[0] * n_sf[1])]
    prec.theta_jn = _prec._atan2_safe(_prec.hypot_safe(*j0_cross_n), j0_dot_n)
    prec.zeta_polarization = polarization_angle(prec, wf, jnp.sin(prec.theta_jn), jnp.cos(prec.theta_jn))
    if version // 100 == 1:
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
    anchors = _st_anchors(prec)
    mf_scale = MTSUN_SI * wf.m_tot / prec.pi_gm
    if "st_context" in prec:
        # built once by initialize_euler_angles
        return st, rotation, mf_scale, anchors, prec.st_context
    context = _cached("st_context", anchors[1:], anchors[0],
                      lambda: barrier(st_lean.context(st, rotation, mf_scale)))
    return st, rotation, mf_scale, anchors, context


#: Above this many points, the SpinTaylor splines are evaluated in chunks.
ST_MANY = 1024


def _st_anchors(prec):
    st = prec.st
    return (st.rows, st.h, st.t0)


def _st_spline(prec, wf, mf):
    """The splines of the SpinTaylor ``alpha`` and ``beta`` (shared):
    ``(alpha, dalpha, beta, dbeta, inside)`` at ``mf``."""
    if "st_bank" in prec:
        # the points evaluated together by st_bank
        for _, entry in prec.st_bank.items():
            if entry.mf is mf:
                return entry.values
    if jnp.size(mf) > ST_MANY:
        return _st_spline_many(prec, wf, mf, jnp.size(mf), PNR_GRID_CHUNK)

    def compute():
        st, rotation, mf_scale, _, context = _st_context(prec, wf)
        return st_lean.angles(st, rotation, mf, mf_scale, context)

    return barrier(_cached("st", _st_anchors(prec), mf, compute))


def _st_spline_many(prec, wf, mf, n, chunk):
    """:func:`_st_spline` at many points (the first ``n`` of ``mf``, a 1-d
    array)."""
    if mf.ndim != 1:
        flat = _st_spline_many(prec, wf, mf.reshape(-1), mf.size, chunk)
        return tuple(x.reshape(mf.shape) for x in flat)

    def compute():
        st, rotation, mf_scale, _, context = _st_context(prec, wf)
        return chunked_map(lambda mf, args: st_lean.angles(args[0], args[1], mf, args[2], args[3]), mf, n, chunk,
                           (st, rotation, mf_scale, context))

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
    fmax_inspiral = mfmax_pn
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
    bp = pa.beta_connection_frequencies(precompute_beta_coefficients(prec, wf, 330))
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
    if getattr(prec, "reference", False):
        # (the mass-scaled reference: the angles are needed down to f_ref)
        fmin_angles = jnp.minimum(fmin_angles, wf.f_ref)
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
    # (the angles and their derivatives here are the SpinTaylor solution's,
    # exact: LAL reads them from its splines near their last knots, whose
    # natural end condition puts the derivatives 0.6-5% off; the merger-
    # ringdown angles, and so heavy binaries, differ from LAL's by mismatches
    # up to ~2e-4, an accepted difference)
    # alphaMRD_coeff from the spline of alpha (at 0.97, 0.99, 0.98 and 1
    # times fmax_inspiral)
    f1 = q[0]
    f2 = q[1]
    a, da, b, db, inside = _st_spline(prec, wf, q)
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
    # (beta from the SpinTaylor solution directly, LAL's acos of its
    # cos(beta), whose |cos(beta)| > 1 from the rounding of its spline LAL
    # also tests for)
    beta1 = b[0]
    beta2, dbeta2 = b[2], db[2]
    flat = ~inside[3]
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
        cosbeta_sign=jnp.where(b[3] > 0.5 * PI, -1.0, 1.0),
        flat=flat,
        dfdamp=dfdamp,
    )
    mf_alpha_ref = jnp.where(wf.mf_ref <= ftrans_mrd, wf.mf_ref, ftrans_mrd)
    error = error | (mf_alpha_ref < mf_first) | (mf_alpha_ref > mfmax_pn)
    error = error | (ftrans_mrd < mf_first) | (ftrans_mrd > mfmax_pn)
    # LAL leaves the spline of alpha for alphaMRD at ftrans_MRD, where they
    # differ (alphaMRD matches the spline's value and slope at f1 and its
    # value at f2): alpha steps there in frequency, and the PNR grid samples
    # the step wherever a node crosses it. Here the switch is at f1, where
    # alpha is continuous with its derivative.
    ftrans_alpha = f1
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
    _, _, beta, _, inside = _st_spline(prec, wf, mf)
    inspiral = mf < ftrans
    value = jnp.where(
        inspiral, beta,
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
    ``v = (pi Mf)^(1/3)``: ``atan2(S_perp, L + S_par)``.

    That is LAL's MSA ``cos(theta_L) = (J^2 + L^2 - S^2) / (2 J L)`` with
    ``J^2 = L^2 + 2 L S_par + S^2``, for a single spin, whose magnitude
    ``S`` is constant (the two roots ``S_-^2 = S_+^2``), without its
    cancellations: LAL takes ``S_+^2 = S_-^2 + 1e-9`` for the degenerate
    roots, which leaves ``beta ~ 3e-5`` for aligned spins. ``S_perp`` and
    ``S_par`` are the components of the effective spin, not those of its
    polar angle kept off the poles."""
    single = prec.single
    system = single.msa
    v = jnp.cbrt(PI * mf)
    l_norm = system.eta / v
    l_msa = _msa.l_norm_3pn(system, v, v * v, l_norm)
    l_102 = _prec.l_ansatz(single.l_102, v, l_norm)
    l = jnp.where(single.error, l_102, l_msa)
    # (m1 / M)^2
    m1_2 = system.eta / system.qq
    return jnp.arctan2(m1_2 * prec.chi_perp_single_spin, fma(m1_2, prec.chi_parallel_single_spin, l))


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
    # (the in-plane part of the effective spin of pa.chi_calc, which with
    # chi_parr is all that pa.pn_waveform_beta reads: LAL's polar angle,
    # kept off the poles, left a residual beta for aligned spins)
    chi_perp = jnp.abs((prec.j0_norm - (prec.l_ref - l_3pn)) * jnp.sin(pn_beta_value)) / (m1 * m1)
    return pa.pn_waveform_beta_components(mf, pn_beta_value, m1, m2, chi_parr, chi_perp)


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
    bp = pa.beta_connection_frequencies(bp)
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
        bp = pa.beta_connection_frequencies(bp)
    mfa, pts = _st_beta_points(prec, bp, prec.st_angles.ftrans_mrd)
    if "st_bank" in prec:
        pts = prec.st_bank.beta.mf
        mfa = pts[1]
    mfb = bp.mf_beta_lower
    _, _, st_b, _, inside = _st_spline(prec, wf, pts)
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
    return bp.replace(attach_weight=pa.attach_weight(bp)), error


@fenced
def populate_structs(prec, wf, version):
    """``IMRPhenomX_PNR_PopulateStructs``: the single-spin system of the
    two-spin map and the alpha and beta parameters (in place on ``prec``:
    ``single``, ``alpha_params``, ``beta_params``, ``error_structs``)."""
    if version >= 200:
        chi = prec.chi_single_spin
        costheta = prec.costheta_single_spin
        chi1 = [chi * safe_sqrt(1.0 - costheta * costheta, tiny()), jnp.zeros_like(chi), chi * costheta]
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
    # (up to q = 12 + Q_SMOOTH_WIDTH)
    return (wf.q <= prec.transition_q) & (prec.chi_single_spin <= pa.PNR_CHI_WINDOW_UPPER)


#: The angles go from LAL's at the edge of the transition region (q = 12)
#: to LAL's beyond it over this range of the mass ratio.
Q_SMOOTH_WIDTH = 2.0


def frequency_map(prec, mf, ell, emm, mf_low, mf_high, mf_rd_22, mf_rd_lm):
    """:func:`.pnr_angles.linear_frequency_map` with the binary's inspiral
    scaling; between q = 12 and 12 + :data:`Q_SMOOTH_WIDTH`, the linear map
    blended into the scaling with ``prec.scaling_weight``."""
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

    def evaluate(x, args):
        prec, wf, ap = args
        x, st_alpha = x
        put(x, st_alpha)
        return region_alpha_at(prec, wf, x, ap, version)

    return chunked_map(evaluate, (mf, st_alpha), n, PNR_GRID_CHUNK, (prec, wf, ap))


def _held(bp, held, free):
    """``beta`` above ``mf_beta_upper``: held at its value there with the
    weight ``bp.beta_freeze`` (LAL's: held or not)."""
    return jnp.where(bp.beta_freeze >= 1.0, held, fma(bp.beta_freeze, held - free, free))


def pnr_beta_at(prec, wf, mf, bp, version, use_mr_beta):
    """``IMRPhenomX_PNR_GeneratePNRBetaAtMf``."""
    beta_pn, error = pn_beta(prec, wf, mf, bp, version, use_mr_beta)
    beta_waveform = waveform_beta(prec, wf, mf, beta_pn)
    low = pa.beta_window(beta_waveform * pa.rescale_beta(mf, bp))
    if version == 223:
        low = jnp.where(bp.beta_lower < 0.01 * bp.beta_upper, pa.beta_window(pa.mr_beta(bp.mf_beta_lower, bp)), low)
    upper = pa.beta_window(pa.mr_beta(bp.mf_beta_upper, bp))
    mid = pa.beta_window(pa.mr_beta(mf, bp))
    below = mf <= bp.mf_beta_lower
    value = jnp.where(below, low, jnp.where(mf >= bp.mf_beta_upper, _held(bp, upper, mid), mid))
    return value, error & below


def merged_beta_at(prec, wf, mf, bp, version, beyond=None):
    """``IMRPhenomX_PNR_GenerateMergedPNRBetaAtMf``, blended into LAL's
    ``beta`` beyond the transition region (:func:`beta_no_mr`, or
    ``beyond``, its value and error flag if computed already) with
    ``prec.q12_weight`` (between q = 12 and 12 + :data:`Q_SMOOTH_WIDTH`)."""
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
    low = pa.beta_window(fma(pnr_window, pnr_beta, msa_window * pn_b))
    mr_mf = pa.mr_beta(mf, bp)
    mr = jnp.where(mf >= bp.mf_beta_upper, _held(bp, pa.mr_beta(bp.mf_beta_upper, bp), mr_mf), mr_mf)
    high = pa.beta_window(fma(pnr_window, mr, msa_window * beta_waveform_high))
    value, error = jnp.where(below, low, high), jnp.where(below, error, error_high)
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
    return pa.beta_window(waveform_beta(prec, wf, mf, beta)), error


def uniform_angles(prec, wf, mf, version, st_values=None):
    """``IMRPhenomX_PNR_GeneratePNRAngles_UniformFrequencies`` without
    ``gamma``: ``alpha`` and ``beta`` at ``mf`` (an array) and the error
    flag of each point. ``st_values`` are the SpinTaylor splines at ``mf``
    (version 330), if computed already.

    LAL's ``beta`` depends on the region of the binary (calibration,
    transition, outside) and on whether the MR ``beta`` is attached: only
    the variant of the binary is computed (:func:`._transforms.switch`;
    under ``jax.vmap``, those of the batch); the ``beta`` with and without
    the MR ``beta`` are blended with the weight ``bp.attach_weight`` where
    it lies between 0 and 1."""
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
    # transition region with it attached, else without, and blends of the
    # first two with the last. Each of the three is computed only where the
    # binary needs it (:func:`._transforms.cond`; under jax.vmap, where some
    # binary of the batch does): one copy of each to compile.
    case = jnp.where(attach & cal, 0, jnp.where(attach & tra, 1, 2))
    partial = (weight > 0.0) & (weight < 1.0)
    case = jnp.where(partial & cal, 3, jnp.where(partial & tra, 4, case))
    in_tra = (case == 1) | (case == 4)
    need_detached = (case >= 2) | (in_tra & (prec.q12_weight > 0.0))

    def guarded(beta_of, needed, extra=None):
        def run(operands):
            prec, wf, mf, bp, st_values, extra = operands
            if st_values is not None:
                _st_spline_put(prec, wf, mf, st_values)
            return beta_of(prec, wf, mf, bp, version, extra)

        def skip(operands):
            mf = operands[2]
            return jnp.zeros(mf.shape, real()), jnp.zeros(mf.shape, bool)

        return cond(needed, run, skip, (prec, wf, mf, bp, st_values, extra))

    detached = guarded(lambda prec, wf, mf, bp, v, _: beta_no_mr(prec, wf, mf, bp, v, jnp.asarray(False)),
                       need_detached)
    calibration = guarded(lambda prec, wf, mf, bp, v, _: pnr_beta_at(prec, wf, mf, bp, v, jnp.asarray(True)),
                          (case == 0) | (case == 3))
    transition = guarded(lambda prec, wf, mf, bp, v, beyond: merged_beta_at(prec, wf, mf, bp, v, beyond),
                         in_tra, detached)

    def blend(attached):
        return fma(bp.attach_weight, attached[0] - detached[0], detached[0]), attached[1] | detached[1]

    choices = [calibration, transition, detached, blend(calibration), blend(transition)]
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
    reference = getattr(prec, "reference", False)
    if reference and (2, 2) not in modes:
        # (the mass-scaled reference: the band of the modes only; LAL's
        # grid holds the band of the (2,2) mode whichever modes are asked
        # for, below the start of the integration where m' = 1 only)
        f_min, f_max = jnp.full_like(f_min_22, jnp.inf), jnp.full_like(f_max_22, -jnp.inf)
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
    if reference:
        # (the odd modes all vanish: the band of the (2,2) mode)
        f_min = jnp.where(jnp.isfinite(f_min), f_min, f_min_22)
        f_max = jnp.where(jnp.isfinite(f_max), f_max, f_max_22)
        f_min, f_max, delta_f_int = _reference_grid(prec, wf, f_min, f_max, version)
    else:
        delta_f_int = interpolation_delta_f(prec, wf, f_min, version)
        f_min = jnp.where(fnms(2.0, delta_f_int, f_min) < 0.0, f_min / 2.0, fnms(2.0, delta_f_int, f_min))
        f_max = fma(2.0, delta_f_int, f_max)
    i_start = jnp.floor(f_min / delta_f_int)
    i_stop = jnp.floor(f_max / delta_f_int) + 1.0
    n = i_stop - i_start
    error = error | (n < 3) | (n > PNR_GRID_CAPACITY)
    return i_start, n, delta_f_int, error


def _reference_grid(prec, wf, f_min, f_max, version):
    """The ends and the spacing of the grid of the PNR interpolants with the
    mass-scaled reference frequency, from the ends ``f_min``, ``f_max`` that
    the modes need: the grid holds ``f_ref`` with :data:`REFERENCE_NODES`
    nodes on each side. As LAL's, the spacing is that at the lower end, and
    the grid reaches two nodes beyond the band; the ends follow the soft
    minimum and maximum (:func:`soft_min`), so that they are continuous
    with their derivatives where the reference crosses the band. For heavy
    binaries, whose spacing is a sizeable fraction of the frequency, the
    spacing is at most ``f_ref / (2 REFERENCE_NODES)`` (the grid then
    starts above ~0.31 ``f_ref``)."""
    f_ref = wf.f_ref
    low = soft_min(f_min, f_ref, 0.05 * f_ref)
    delta_f_int = interpolation_delta_f(prec, wf, low, version)
    cap = f_ref / (2.0 * REFERENCE_NODES)
    delta_f_int = soft_min(delta_f_int, cap, 0.1 * cap)
    band_low = jnp.where(fnms(2.0, delta_f_int, f_min) < 0.0, f_min / 2.0, fnms(2.0, delta_f_int, f_min))
    f_min = soft_min(band_low, fnms(float(REFERENCE_NODES), delta_f_int, f_ref), delta_f_int)
    f_max = soft_max(fma(2.0, delta_f_int, f_max), fma(float(REFERENCE_NODES), delta_f_int, f_ref), delta_f_int)
    return f_min, f_max, delta_f_int


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
    k = jnp.clip(jnp.floor((x - x0) / h), 0, n - 2).astype(integer())
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
    from them with Boole's rule (on the phase grid, with
    ``prec.phase_grid``, the interpolants of :func:`_geometric_interpolants`).
    Returns a :class:`~._math.Struct` of the interpolants and the error
    flag. Where ``active`` (traced) is false, nothing is evaluated.

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
    node_map = None
    if prec.phase_grid:
        node_map = _phase_grid(x0, (i_start + n - 1.0) * h, h, wf.m_tot)
        n = jnp.minimum(node_map.n, n)
        n_eval = jnp.where(active, n, 0.0)

    def frequencies(k, i_start, n, h):
        return (i_start + jnp.minimum(k, n - 1.0)) * h

    # the angles on the grid
    def angles(index, args):
        prec, wf, i_start, n, h, node_map = args
        k = index * chunk + jnp.arange(chunk, dtype=real())
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
            st_values = st_lean.angles(st, rotation, mf, mf_scale, context)
        alpha, beta, err = uniform_angles(prec, wf, mf, version, st_values)
        return jnp.where(valid, alpha, 0.0), jnp.where(valid, beta, 0.0), jnp.any(err & valid)

    args = (prec, wf, i_start, n, h, node_map)
    st_extra = None
    if extra_mf is None or version != 330:
        alpha, beta, errors = blocked_map(angles, jnp.arange(n_chunks), jnp.ceil(n_eval / chunk), args)
    else:
        n_extra_chunks = -(-extra_mf.shape[0] // chunk)
        padded = jnp.concatenate([extra_mf, jnp.broadcast_to(extra_mf[-1:], (n_extra_chunks * chunk - extra_mf.shape[0],))])

        def angles_or_extra(index, mf_extra, is_extra, args):
            prec, wf, i_start, n, h, node_map = args
            k = index * chunk + jnp.arange(chunk, dtype=real())
            valid = k < n
            if node_map is None:
                mf = hz_to_mf(frequencies(k, i_start, n, h), wf.m_tot)
            else:
                mf = hz_to_mf(_phase_f(node_map, node_map.s0 + jnp.minimum(k, n - 1.0) * node_map.ds), wf.m_tot)
            mf = jnp.where(is_extra, mf_extra, mf)
            st, rotation, mf_scale, _, context = _st_context(prec, wf)
            st_values = st_lean.angles(st, rotation, mf, mf_scale, context)

            def on_grid(operands):
                prec, wf, mf, st_values, valid = operands
                alpha, beta, err = uniform_angles(prec, wf, mf, version, st_values)
                return jnp.where(valid, alpha, 0.0), jnp.where(valid, beta, 0.0), jnp.any(err & valid)

            def skip(operands):
                mf = operands[2]
                return jnp.zeros(mf.shape, real()), jnp.zeros(mf.shape, real()), jnp.asarray(False)

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
        splines = _geometric_interpolants(alpha, beta, jnp.zeros((), real()), n, jnp.ones((), real()), n_eval, error)
        splines.node_map = node_map
        splines.st_extra = st_extra
        return splines
    # LAL's natural splines of alpha and beta, and gamma from them (on the
    # phase grid the interpolants above, in the index of the nodes)
    splines = _lal_interpolants(alpha, beta, i_start, n, h, n_eval, error)
    splines.st_extra = st_extra
    return splines


def _lal_interpolants(alpha, beta, i_start, n, h, n_eval, error):
    """LAL's interpolants of the PNR angles: natural cubic splines of
    ``alpha`` and ``beta`` on the uniform grid ``(i_start + k) h``, ``k <
    n``, and of ``gamma`` integrated from them with Boole's rule
    (``IMRPhenomX_PNR_GeneratePNRGamma``, zero at the first node).

    The interpolants of the unit vector of L (:func:`_geometric_interpolants`)
    would follow L passing close to J, but where LAL's grid lets ``alpha``
    turn by up to ~3 rad between nodes (at its lower end) they cut across
    its circle, by up to a mismatch of 5e-4 with LAL, and refining the grid
    until they do not costs more than LAL's splines."""
    chunk = PNR_GRID_CHUNK
    n_chunks = PNR_GRID_CAPACITY // chunk
    x0 = i_start * h

    def frequencies(k, i_start, n, h):
        return (i_start + jnp.minimum(k, n - 1.0)) * h

    m_alpha, m_beta = _uniform_natural_splines([alpha, beta], n, h, n_eval)

    # gamma from the minimal rotation condition with Boole's rule, interval
    # by interval, summed chunk by chunk
    def boole(index, total, args):
        alpha, m_alpha, beta, m_beta, i_start, n, h, x0 = args
        cap = alpha.shape[0]
        # the intervals [k, k + 1] of the chunk: values at k and k + 1
        i = index * chunk + jnp.arange(chunk)
        here, after = jnp.clip(i, 0, cap - 1), jnp.clip(i + 1, 0, cap - 1)
        t1 = frequencies(here.astype(real()), i_start, n, h)
        t2 = frequencies(after.astype(real()), i_start, n, h)
        y0, y1, m0, m1 = alpha[here], alpha[after], m_alpha[here], m_alpha[after]
        yb0, yb1, mb0, mb1 = beta[here], beta[after], m_beta[here], m_beta[after]
        points = jnp.stack([t2, fma(3.0, t1, t2) / 4.0, (t1 + t2) / 2.0, t1, fma(3.0, t2, t1) / 4.0])
        dx = points - (x0 + i * h)
        b = (y1 - y0) / h - h * (m1 + 2.0 * m0) / 6.0
        dalpha = b + dx * (m0 + (m1 - m0) / (2.0 * h) * dx)
        bb = (yb1 - yb0) / h - h * (mb1 + 2.0 * mb0) / 6.0
        beta_x = yb0 + dx * (bb + dx * (mb0 / 2.0 + dx * (mb1 - mb0) / (6.0 * h)))
        f = -dalpha * jnp.cos(beta_x)
        total_f = fma(7.0, f[0], fma(32.0, f[1], fma(12.0, f[2], fma(7.0, f[3], 32.0 * f[4]))))
        increments = jnp.where(i + 1 < n, 1.0 / 90.0 * (t2 - t1) * total_f, 0.0)
        # gamma at the chunk's points: the integral up to each
        sums = jnp.cumsum(increments)
        gamma = total + jnp.concatenate([jnp.zeros(1, real()), sums[:-1]])
        return gamma, total + sums[-1]

    gamma, _ = blocked_scan(boole, jnp.arange(n_chunks), jnp.ceil(n_eval / chunk), jnp.zeros((), real()),
                            (alpha, m_alpha, beta, m_beta, i_start, n, h, x0))
    gamma = gamma.reshape(-1)
    (m_gamma,) = _uniform_natural_splines([gamma], n, h, n_eval)
    return Struct(alpha=alpha, beta=beta, gamma=gamma, m_alpha=m_alpha, m_beta=m_beta, m_gamma=m_gamma, n=n, x0=x0,
                  h=h, error=error)


#: Half-width in ``cos(beta)`` of the band around ``beta = pi / 2`` across
#: which the interpolants of the PNR angles go from the form regular at
#: ``beta = 0`` to the one regular at ``beta = pi``.
HEMISPHERE_SOFT = 0.3


def _wrap(x):
    """``x`` modulo ``2 pi``, in ``[-pi, pi]``."""
    return x - 2 * PI * jnp.round(x / (2 * PI))


def _geometric_interpolants(alpha, beta, i_start, n, h, n_eval, error):
    """The interpolants of the PNR angles: instead of LAL's splines of
    ``alpha``, ``beta`` and ``gamma``, splines of the unit vector ``(u, v,
    w) = (sin(beta) cos(alpha), sin(beta) sin(alpha), cos(beta))`` and of
    ``gamma + alpha`` and ``gamma - alpha`` (as the sums of the splines of
    ``gamma`` and of the unwrapped ``alpha``, splines being linear in their
    values: two arrays of the grid fewer), from which
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
        t1 = frequencies(here.astype(real()), i_start, n, h)
        t2 = frequencies(after.astype(real()), i_start, n, h)
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
        r = safe_sqrt(ux * ux + vx * vx + wx * wx, tiny())
        twist = ux * dv - vx * du
        north = boole_sum(safe_div(twist, r * (r + wx)))
        south = boole_sum(safe_div(twist, r * (r - wx)))
        d_alpha = _wrap(raw[after] - raw[here])
        weight = pa._ramp(wx[2] / r[2], HEMISPHERE_SOFT)
        d_gamma = fma(weight, north - d_alpha, (1.0 - weight) * (d_alpha - south))
        used = i + 1 < n
        increments = jnp.stack([jnp.where(used, d_gamma, 0.0), jnp.where(used, d_alpha, 0.0)])
        sums = jnp.cumsum(increments, axis=1)
        values = totals[:, None] + jnp.concatenate([jnp.zeros((2, 1), real()), sums[:, :-1]], axis=1)
        # (two arrays: the chunks of each make the grid without a transpose)
        return (values[0], values[1]), totals + sums[:, -1]

    first = jnp.stack([jnp.zeros((), real()), raw[0]])
    (gamma, alpha_unwrapped), _ = blocked_scan(boole, jnp.arange(n_chunks), jnp.ceil(n_eval / chunk), first,
                                               (u, m_u, v, m_v, w, m_w, raw, i_start, n, h, x0))
    gamma, alpha_unwrapped = gamma.reshape(-1), alpha_unwrapped.reshape(-1)
    m_gamma, m_unwrapped = _uniform_natural_splines([gamma, alpha_unwrapped], n, h, n_eval)
    return Struct(u=u, v=v, w=w, m_u=m_u, m_v=m_v, m_w=m_w, gamma=gamma, m_gamma=m_gamma,
                  unwrapped=alpha_unwrapped, m_unwrapped=m_unwrapped, n=n, x0=x0, h=h, error=error)


def spline_angles(splines, f):
    """The PNR ``alpha``, ``beta`` and ``gamma`` at ``f`` (Hz) and whether
    ``f`` lies on the grid (GSL fails outside): LAL's splines, or the
    interpolants of :func:`_geometric_interpolants` (the phase grid), for
    which ``alpha`` and ``gamma`` are known modulo ``2 pi``."""
    inside = (f >= splines.x0) & (f <= splines.x0 + (splines.n - 1.0) * splines.h)
    if "u" not in splines:
        a, _ = _spline_at(splines.alpha, splines.m_alpha, splines.n, splines.x0, splines.h, f)
        b, _ = _spline_at(splines.beta, splines.m_beta, splines.n, splines.x0, splines.h, f)
        g, _ = _spline_at(splines.gamma, splines.m_gamma, splines.n, splines.x0, splines.h, f)
        return a, b, g, inside
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
    b = st_angles._atan2(safe_sqrt(u * u + v * v, tiny()), w)
    # alpha on the branch of the unwrapped spline (which matters only
    # in the band, where both forms of gamma are used and alpha is
    # smooth)
    # (with d = a_u - unwrapped: gamma + alpha - a_u = gamma - d and
    # gamma - alpha + a_u = gamma + d)
    d = _wrap(a - at("unwrapped"))
    weight = pa._ramp(w / safe_sqrt(u * u + v * v + w * w, tiny()), HEMISPHERE_SOFT)
    g = at("gamma")
    gamma = fma(weight, g - d, (1.0 - weight) * (g + d))
    return a, b, gamma, inside


@fenced
def remap_theta_jsf(prec, wf, beta_ref):
    """``IMRPhenomX_PNR_RemapThetaJSF`` (in place)."""
    skip = prec.j0_norm < 1e-10
    # (remapped for all in-plane spins: the remapped frame tends to LAL's as
    # they vanish, without LAL's step at 1e-3)
    small = jnp.asarray(False)
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
    theta_jn = _prec._atan2_safe(_prec.hypot_safe(n_jf[0], n_jf[1]), n_jf[2])
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
