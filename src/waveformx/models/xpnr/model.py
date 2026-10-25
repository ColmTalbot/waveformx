# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR: the polarizations (``SimInspiralChooseFDWaveformSequence``),
after waveformrs's ``imrphenomx/xpnr.rs``.

IMRPhenomXPNR is IMRPhenomXPHM with the PNR angles, the tuned co-precessing
(2,2) mode, the antisymmetric (2,|2|) contribution, the 122019 release of
the higher modes, ``PhenomXPrecVersion`` 330 and ``PhenomXPFinalSpinMod`` 7,
here with the fix of lalsuite !2593 (the mirror symmetry of the waveform).

LAL's physics is kept (the fits, the ansatzes, the PNR construction of the
angles, the calibrated choices of the higher-mode amplitudes), but not its
numerical procedure:

* the SpinTaylor angles for every binary: the PNR machinery is kept for all
  in-plane spins, where LAL switches to the NNLO or MSA angles below
  ``1e-6`` or when its integration fails, and the SpinTaylor solution goes
  on where LAL's stops (a reference frequency beyond the PN energy test);
* the precession equations integrated once with a high-order method
  (:mod:`.spintaylor.lean`), from low enough a frequency that LAL's second
  integration from the start of the grid of the PNR interpolants is not
  needed, and the angles evaluated from that solution rather than from
  LAL's splines through its samples;
* LAL's discrete switches of the angles made continuous (the end of the
  inspiral, the two-spin construction for every binary, the connection of
  the PNR ``beta``, the edge of the PNR transition region at ``q = 12``);
* no opening angle where the in-plane spins vanish (no arctan window of
  the PNR ``beta`` within ``[0, pi]``, the single-spin ``beta`` without
  the perturbed MSA roots, the in-plane spin of the PNR fits not kept off
  zero), the small angles taken as ``atan2`` of components rather than
  ``acos`` of cosines, and the limit of the grid of the PNR interpolants
  at equal masses, so that the waveform is continuous there;
* exact derivatives in the higher modes where LAL takes finite
  differences, and no emulation of LAL's arithmetic (masses in kg,
  single-precision storage, the roots of GSL's companion matrix).

The version that reproduces LAL's numerics step for step is the tag
``lal-faithful``. The J-frame modes (:func:`jframe_modes`) are LAL's
single-mode functions: a set-up for every range of modes, mapped with
``jax.lax.map``.

The derivatives are forward-mode Jacobians (:mod:`._transforms`):
``jax.grad``, ``jax.jvp`` and ``jax.jacfwd`` work with respect to the
binary. Where the model fails the polarizations are NaN.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from ..._options import REFERENCE_MF, strain_scale
from . import hm, pnr
from . import pnr_angles as pa
from ._math import (
    MTSUN_SI, Struct, barrier, complex_, hz_to_mf, mass_unit, materialized, mf_to_hz, precision, precision_of, real,
)
from ._transforms import differentiable
from .antisymmetric import antisymmetric_waveform
from .internals import errors as wf_errors
from .internals import waveform_struct
from .xas import h22, phif_ref, xas_setup

__all__ = ["DEFAULT_MODES", "lal_order", "ordered_binary", "polarizations", "setup_waveform"]

#: The co-precessing modes ``(l, m' > 0)`` in the order of LAL's mode loop.
DEFAULT_MODES = ((2, 1), (2, 2), (3, 2), (3, 3), (4, 4))


def lal_order(modes):
    """The co-precessing modes among ``modes`` in LAL's order."""
    return tuple(mode for mode in DEFAULT_MODES if mode in modes)


def coprecessing_modes(mode_array):
    """The co-precessing modes ``(l, |m'|)`` of a LAL mode array (LAL twists
    up both signs of ``m'`` whichever sign is listed); all modes for
    ``None``."""
    if mode_array is None:
        return DEFAULT_MODES
    return lal_order({(ell, abs(emm)) for ell, emm in mode_array})


def ordered_binary(binary):
    """``XLALIMRPhenomXPCheckMassesAndSpins``: the bodies ordered with the
    first strictly heavier (equal masses are swapped), as traced values."""
    m1 = jnp.asarray(binary.mass_1, dtype=real())
    m2 = jnp.asarray(binary.mass_2, dtype=real())
    s1 = jnp.asarray(binary.spin_1, dtype=real())
    s2 = jnp.asarray(binary.spin_2, dtype=real())
    keep = m1 > m2
    return (
        jnp.where(keep, m1, m2),
        jnp.where(keep, m2, m1),
        jnp.where(keep, s1, s2),
        jnp.where(keep, s2, s1),
    )


def reference_frequency(binary, reference_mf=None):
    """The reference frequency (Hz) of ``binary``: ``binary.f_ref`` (LAL's
    convention, zero for the first frequency), or ``reference_mf / M``
    with the mass-scaled reference (``WaveformOptions.reference_mf``)."""
    if reference_mf is None:
        return jnp.asarray(binary.f_ref, dtype=real())
    m_tot = jnp.asarray(binary.mass_1, dtype=real()) + jnp.asarray(binary.mass_2, dtype=real())
    return reference_mf / (m_tot * MTSUN_SI)


def setup_waveform(binary, delta_f, f_min, f_max, amp_scale=1.0, reference_mf=None):
    """``IMRPhenomXSetWaveformVariables`` as the precessing drivers call it
    (ordered bodies, the 122019 release of the higher modes), with the
    reference frequency of :func:`reference_frequency`. With the
    mass-scaled reference, the binary is first carried to the canonical
    reference ``REFERENCE_MF`` (:func:`.reference.to_canonical`), where
    the model takes everything that depends on the reference."""
    if reference_mf is not None and reference_mf != REFERENCE_MF:
        from .reference import to_canonical

        binary, reference_mf = to_canonical(binary, reference_mf), REFERENCE_MF
    m1, m2, chi1, chi2 = ordered_binary(binary)
    wf = waveform_struct(
        m1 * mass_unit(), m2 * mass_unit(), chi1[2], chi2[2], delta_f, reference_frequency(binary, reference_mf),
        jnp.asarray(binary.phi_ref, dtype=real()), f_min, f_max,
        jnp.asarray(binary.luminosity_distance, dtype=real()),
        jnp.asarray(binary.inclination, dtype=real()), amp_scale,
    )
    return wf, m1 * mass_unit(), m2 * mass_unit(), [chi1[0], chi1[1], chi1[2]], [chi2[0], chi2[1], chi2[2]]


# ---------------------------------------------------------------------------
# twisting up
# ---------------------------------------------------------------------------

SQRT2 = 2.0 ** 0.5
SQRT5 = 5.0 ** 0.5
SQRT6 = 2.44948974278317788
SQRT7 = 7.0 ** 0.5
SQRT10 = 10.0 ** 0.5
SQRT14 = 14.0 ** 0.5
SQRT15 = 15.0 ** 0.5
SQRT30 = 30.0 ** 0.5
SQRT70 = 70.0 ** 0.5


def half_angles(beta):
    """``cos(beta / 2)`` and ``sin(beta / 2)``: from ``beta`` itself, not
    LAL's ``sqrt((1 -+ cos(beta)) / 2)``, which loses ``beta`` below ~1e-8
    (``cos(beta)`` rounds to 1) and with it the derivatives for small
    in-plane spins."""
    return jnp.cos(0.5 * beta), jnp.sin(0.5 * beta)


def wigner_d(ell, mprime, c, s):
    """``d^l_{k, m'}(beta)`` for ``k = -l .. l`` from the powers ``c[k] =
    cos(beta/2)^k``, ``s[k] = sin(beta/2)^k`` (``IMRPhenomXPHMTwistUp``)."""
    if (ell, mprime) == (2, 2):
        return [s[4], 2.0 * c[1] * s[3], SQRT6 * s[2] * c[2], 2.0 * c[3] * s[1], c[4]]
    if (ell, mprime) == (2, 1):
        return [2.0 * c[1] * s[3], 3.0 * c[2] * s[2] - s[4], SQRT6 * (c[3] * s[1] - c[1] * s[3]),
                c[2] * (c[2] - 3.0 * s[2]), -2.0 * c[3] * s[1]]
    if (ell, mprime) == (3, 3):
        return [s[6], SQRT6 * c[1] * s[5], SQRT15 * c[2] * s[4], 2.0 * SQRT5 * c[3] * s[3], SQRT15 * c[4] * s[2],
                SQRT6 * c[5] * s[1], c[6]]
    if (ell, mprime) == (3, 2):
        return [SQRT6 * c[1] * s[5], s[4] * (5.0 * c[2] - s[2]), SQRT10 * s[3] * (2.0 * c[3] - c[1] * s[2]),
                SQRT30 * c[2] * (c[2] - s[2]) * s[2], SQRT10 * c[3] * (c[2] * s[1] - 2.0 * s[3]),
                c[4] * (c[2] - 5.0 * s[2]), -1.0 * SQRT6 * c[5] * s[1]]
    if (ell, mprime) == (4, 4):
        return [s[8], 2.0 * SQRT2 * c[1] * s[7], 2.0 * SQRT7 * c[2] * s[6], 2.0 * SQRT14 * c[3] * s[5],
                SQRT70 * c[4] * s[4], 2.0 * SQRT14 * c[5] * s[3], 2.0 * SQRT7 * c[6] * s[2], 2.0 * SQRT2 * c[7] * s[1],
                c[8]]
    raise ValueError(f"mode ({ell},{mprime}) is not twisted up")


def spherical_harmonics(theta, ell):
    """``XLALSpinWeightedSphericalHarmonic(theta, 0, -2, l, m)`` for ``m = -l
    .. l`` (real), as ``SphericalHarmonics::new`` writes them."""
    from ..._harmonics import spin_weighted_spherical_harmonic

    return [jnp.real(spin_weighted_spherical_harmonic(theta, 0.0, ell, m)) for m in range(-ell, ell + 1)]


def twist_factors(alpha, beta, epsilon, mprime):
    """The factors shared by the modes with the same ``m'``: the powers of
    ``exp(i alpha)``, of the half angles and ``exp(-i m' epsilon) / 2``."""
    e1 = jnp.exp(1j * alpha)
    e2 = e1 * e1
    e3 = e1 * e2
    e4 = e1 * e3
    powers = [jnp.ones_like(e1), e1, e2, e3, e4]
    cb, sb = half_angles(beta)
    c = [jnp.ones_like(cb)]
    s = [jnp.ones_like(sb)]
    for _ in range(1, 9):
        c.append(cb * c[-1])
        s.append(sb * s[-1])
    eps = jnp.exp(-1j * mprime * epsilon) * 0.5
    return Struct(cos_k=[jnp.real(p) for p in powers], sin_k=[jnp.imag(p) for p in powers], c=c, s=s, eps=eps)


def twist_up(ell, mprime, twist, ylm, h, symmetry):
    """``IMRPhenomXPHMTwistUp`` for ``h = h_{l,-m'}`` with
    ``PolarizationSymmetry`` ``symmetry`` (+1, or -1 for the antisymmetric
    (2,2) mode): the contributions ``(h_+, h_x)`` in the J frame."""
    d = wigner_d(ell, mprime, twist.c, twist.s)
    sigma = symmetry if ell % 2 == 0 else -symmetry
    # (h_+ and h_x stacked, behind a barrier: one kernel computes the terms
    # they share, rather than XLA copying them into a kernel for each)
    re = im = 0.0
    for index in range(2 * ell + 1):
        y = ylm[index]
        k = index - ell
        dm = d[2 * ell - index] if (k + mprime) % 2 == 0 else -d[2 * ell - index]
        a = dm * y
        b = sigma * d[index] * y
        cos = twist.cos_k[abs(k)]
        sin = -twist.sin_k[abs(k)] if k < 0 else twist.sin_k[abs(k)]
        total = a + b
        re = re + total * jnp.stack([cos, sin])
        im = im + jnp.stack([(b - a) * sin, (a - b) * cos])
    hp, hc = materialized(twist.eps * h * (re + 1j * im))
    return hp, hc


def rotate_polarizations(zeta, hp, hc):
    cos = jnp.cos(2.0 * zeta)
    sin = jnp.sin(2.0 * zeta)
    return hp * cos + hc * sin, hc * cos - hp * sin


# ---------------------------------------------------------------------------
# the prescriptions
# ---------------------------------------------------------------------------


def _mode_angles_pnr(prec, wf, splines, ell, emm, mf):
    """The PNR angles of co-precessing mode ``(ell, emm)`` at ``mf``: at the
    frequency of the (2,2) mode given by the frequency map."""
    if (ell, emm) == (2, 2):
        mf_mapped = mf
        map_error = jnp.zeros(jnp.shape(mf), bool)
    else:
        mf_rd_lm = pnr.ringdown_frequency(ell, emm, wf)
        low, high = pa.linear_frequency_map_transition_frequencies(emm, wf.f_ring, mf_rd_lm, prec.pnr_hm_mflow,
                                                                   prec.pnr_hm_mfhigh)
        mf_mapped, map_error = pnr.frequency_map(prec, mf, ell, emm, low, high, wf.f_ring, mf_rd_lm)
    f_mapped = mf_to_hz(mf_mapped, wf.m_tot)
    alpha, beta, gamma, inside = pnr.spline_angles(splines, f_mapped)
    return alpha - prec.alpha_offset, beta, -gamma - prec.epsilon_offset, map_error | ~inside


def _prepare_pnr(prec, wf, grid, version, error, active=True, extra_mf=None, n_extra=None):
    """The PNR angle interpolants (``IMRPhenomX_PNR_GeneratePNRAngleInterpolants``)
    on ``grid`` (:func:`.pnr.interpolation_grid`), with the offsets at the
    reference frequency and the remapped J frame (and the SpinTaylor
    ``alpha`` at the first ``n_extra`` of ``extra_mf``, see
    :func:`.pnr.angle_interpolants`)."""
    splines = barrier(pnr.angle_interpolants(prec, wf, grid, version, active, extra_mf, n_extra))
    error = error | splines.error
    alpha_ref, beta_ref, gamma_ref, inside = pnr.spline_angles(splines, wf.f_ref)
    error = error | ~inside | ~jnp.isfinite(beta_ref)
    prec.alpha_offset = alpha_ref
    prec.epsilon_offset = -gamma_ref - prec.epsilon0
    prec = pnr.remap_theta_jsf(prec, wf, beta_ref)
    return prec, splines, error


def _spintaylor_setup(base, wf0, m1, m2, modes):
    """The set-up of version 330 up to the grid of the PNR interpolants:
    the SpinTaylor solution, ``IMRPhenomX_Initialize_Euler_Angles``,
    ``IMRPhenomX_PNR_PopulateStructs`` and the grid. One integration, from
    low enough a frequency (:func:`.pnr.common_setup`) that LAL's second
    integration from the start of the grid is not needed (an error where it
    would be). Returns ``(prec, wf, grid, flags)`` with ``flags`` the
    failure of the integration (``failed``), the capacities exceeded
    (``incomplete``) and the errors of the set-up (``error``)."""
    st = pnr.spintaylor_solution(base, wf0, m1, m2)
    prec, wf = pnr.finish_setup(base, wf0, 330, st)
    prec = pnr.st_bank(prec, wf)
    prec = pnr.initialize_euler_angles(prec, wf)
    prec = pnr.populate_structs(prec, wf, 330)
    grid = pnr.interpolation_grid(prec, wf, modes, 330)
    f0 = grid[0] * grid[2]
    error = prec.error_structs | grid[3] | prec.error_euler | (prec.fmin_integration > f0)
    flags = Struct(failed=st.failed, incomplete=st.incomplete, error=error)
    return prec, wf, grid, flags


#: The fields of the waveform struct that the co-precessing modes read
#: after the precession set-up has modified it.
WF_FIELDS = (
    "m1_si", "m2_si", "m_tot_si", "m1", "m2", "m_tot", "q", "eta", "delta", "eta2", "eta3", "chi1l", "chi2l",
    "chi1l2l", "chi1l2", "chi1l3", "chi2l2", "chi2l3", "chi_eff", "chi_pn_hat", "s_tot_r", "dchi", "dchi_half",
    "sigma_l", "sl", "f_ref", "phi_ref_in", "phi0", "beta", "mf_ref", "pi_m", "v_ref", "m_sec", "delta_f",
    "delta_mf", "f_cut_def", "f_min", "f_max", "mf_max", "f_cut", "f_max_prime", "distance", "inclination", "amp0",
    "amp_norm", "dphase0", "coprec", "hm_release", "m_final", "a_final", "a_final_nonprec", "a_final_prec",
    "f_ring", "f_damp", "f_meco", "f_isco",
)


def _angle_data(prec, wf, splines, modes, frequencies):
    """What the precession contributes to the waveform: for each
    co-precessing mode the Euler angles ``(alpha, beta, epsilon)`` at
    ``frequencies``, the antisymmetric part of the (2,2) mode, the line of
    sight in the J frame, the polarization angle, the waveform struct as the
    set-up left it, and the error flag."""
    mf = wf.m_sec * frequencies
    mf_cut = wf.f_max_prime * wf.m_sec
    keep = mf <= mf_cut
    # LAL skips the odd modes where they vanish
    vanishing_odd = (wf.q == 1.0) & (wf.chi1l == wf.chi2l)
    error = jnp.asarray(False)
    alphas, betas, epsilons = [], [], []
    for ell, emm in modes:
        alpha, beta, epsilon, err = _mode_angles_pnr(prec, wf, splines, ell, emm, mf)
        if emm % 2:
            err = err & ~vanishing_odd
        error = error | jnp.any(err & keep)
        alphas.append(alpha)
        betas.append(beta)
        epsilons.append(epsilon)
    if (2, 2) in modes:
        amp, phase, err, xas22 = antisymmetric_waveform(prec, wf, frequencies, 330, splines.st_extra)
        error = error | err
        h_as = amp * jnp.exp(1j * phase)
    else:
        h_as = jnp.zeros(frequencies.shape, complex_())
        xas = xas_setup(wf)
        xas22 = Struct(amp=xas.amp, phase=xas.phase, linb=xas.linb)
    wf_out = Struct(**{name: getattr(wf, name) for name in WF_FIELDS})
    wf_out.hm_release = jnp.asarray(122019)
    return Struct(
        wf=wf_out, alpha=jnp.stack(alphas), beta=jnp.stack(betas), epsilon=jnp.stack(epsilons), h_as=h_as,
        theta_jn=prec.theta_jn, zeta=prec.zeta_polarization, error=error, xas22=xas22,
    )


def _coprecessing_modes(wf, modes, frequencies, xas22=None):
    """The co-precessing modes ``h_{l,-m'}`` at ``frequencies`` (no cutoff).
    ``xas22`` are the amplitude, phase and time-shift coefficients of the
    (2,2) mode if computed already (``data.xas22`` of :func:`_angle_data`)."""
    wf = barrier(wf)
    mf = wf.m_sec * frequencies
    if xas22 is None:
        xas = xas_setup(wf)
    else:
        xas = Struct(amp=xas22.amp, phase=xas22.phase, linb=xas22.linb, lina=0.0, inveta=1.0 / wf.eta,
                     amp_prefactor=wf.amp0 * wf.amp_norm)
    phase22 = xas.phase
    higher = [mode for mode in modes if mode != (2, 2)]
    coefficients = dict(zip(higher, hm.modes_coefficients(wf, phase22, higher)))
    out = []
    for ell, emm in modes:
        if (ell, emm) == (2, 2):
            out.append(h22(mf, wf, xas, phif_ref(wf, xas), mf_cut=jnp.inf))
        else:
            out.append(hm.mode_h(coefficients[(ell, emm)], mf))
    return materialized(out)


def _twist_polarizations(data, modes, frequencies):
    """``IMRPhenomXPHM_hplushcross``: the co-precessing modes (and the
    antisymmetric part of the (2,2) mode) twisted up into ``(h_+, h_x)``."""
    data = materialized(data)
    wf = data.wf
    mf = wf.m_sec * frequencies
    keep = mf <= wf.f_max_prime * wf.m_sec
    vanishing_odd = (wf.q == 1.0) & (wf.chi1l == wf.chi2l)
    ylms = {ell: spherical_harmonics(data.theta_jn, ell) for ell in (2, 3, 4)}
    hp = jnp.zeros(frequencies.shape, complex_())
    hc = jnp.zeros(frequencies.shape, complex_())
    for index, ((ell, emm), h) in enumerate(zip(modes, _coprecessing_modes(wf, modes, frequencies, data.xas22))):
        twist = materialized(twist_factors(data.alpha[index], data.beta[index], data.epsilon[index], emm))
        p, c = twist_up(ell, emm, twist, ylms[ell], h, 1.0)
        if (ell, emm) == (2, 2):
            pa_, ca_ = twist_up(ell, emm, twist, ylms[ell], data.h_as, -1.0)
            p = p + pa_
            c = c + ca_
        if emm % 2:
            p = jnp.where(vanishing_odd, 0.0, p)
            c = jnp.where(vanishing_odd, 0.0, c)
        hp = hp + jnp.where(keep, p, 0.0)
        hc = hc + jnp.where(keep, c, 0.0)
    return rotate_polarizations(data.zeta, hp, hc)


def _twist_jframe(data, modes, jmodes, frequencies, hs):
    """``IMRPhenomXPHMTwistUpOneMode``: the J-frame modes ``jmodes`` as
    arrays ``(positive, negative)`` of shape ``(len(jmodes), n)`` holding
    ``h_lm(f)`` and ``h_lm(-f)``. Each co-precessing mode ``h = h_{l,-m'}``
    with antisymmetric part ``h_as`` contributes ``e^{-i m' epsilon} (h +
    h_as) e^{-i m alpha} d^l_{m,-m'}`` and ``e^{i m' epsilon} (-1)^l conj(h
    - h_as) e^{-i m alpha} d^l_{m,m'}``. ``hs`` holds the co-precessing
    modes ``{(l, m'): h}`` of the ``l`` of ``jmodes``."""
    data = barrier(data)
    wf = data.wf
    mf = wf.m_sec * frequencies
    keep = mf <= wf.f_max_prime * wf.m_sec
    vanishing_odd = (wf.q == 1.0) & (wf.chi1l == wf.chi2l)
    ells = {ell for ell, _ in jmodes}
    selected = [(i, mode) for i, mode in enumerate(modes) if mode[0] in ells]
    hs = [hs[mode] for _, mode in selected]
    zero = jnp.zeros(frequencies.shape, complex_())
    positive = [zero for _ in jmodes]
    negative = [zero for _ in jmodes]
    for (index, (ell, emm)), h in zip(selected, hs):
        h_as = data.h_as if (ell, emm) == (2, 2) else jnp.zeros_like(h)
        cb, sb = half_angles(data.beta[index])
        c = [jnp.ones_like(cb)]
        s = [jnp.ones_like(sb)]
        for _ in range(1, 9):
            c.append(cb * c[-1])
            s.append(sb * s[-1])
        d = wigner_d(ell, emm, c, s)
        e = jnp.exp(1j * emm * data.epsilon[index])
        sigma = 1.0 if ell % 2 == 0 else -1.0
        factor_pos = (h + h_as) / e
        factor_neg = e * sigma * jnp.conj(h - h_as)
        skip = vanishing_odd if emm % 2 else jnp.asarray(False)
        for j, (l, m) in enumerate(jmodes):
            if l != ell:
                continue
            rotation = jnp.exp(-1j * m * data.alpha[index])
            k = m + l
            d_minus = d[2 * l - k] if (m + emm) % 2 == 0 else -d[2 * l - k]
            positive[j] = positive[j] + jnp.where(keep & ~skip, factor_pos * (rotation * d_minus), 0.0)
            negative[j] = negative[j] + jnp.where(keep & ~skip, factor_neg * (rotation * d[k]), 0.0)
    return jnp.stack(positive), jnp.stack(negative)


def _pipeline_330(setup, modes, frequencies, active=True):
    """The angle data (:func:`_angle_data`) from :func:`_spintaylor_setup`."""
    prec, wf, grid, flags = setup
    prec, wf = barrier((pnr._copy(prec), wf))
    # (the antisymmetric mode reads the SpinTaylor alpha at the frequencies
    # below its transition at 0.85 fRING: evaluated with the grid)
    extra_mf, n_extra = None, None
    if (2, 2) in modes:
        extra_mf = hz_to_mf(frequencies, wf.m_tot)
        n_extra = jnp.sum(extra_mf < pnr.antisymmetric_points(wf)[0])
    prec, splines, error = barrier(_prepare_pnr(prec, wf, grid, 330, flags.error, active, extra_mf, n_extra))
    data = _angle_data(prec, wf, splines, modes, frequencies)
    data.error = data.error | error | flags.incomplete
    return data


def _decide(binary, frequencies, delta_f, f_min, f_max, modes, mode_range_=None, check_reference=True,
            phase_grid=False, amp_scale=1.0, reference_mf=None):
    """Set up the binary and compute the angle data. ``phase_grid``
    (static) selects the phase grid of the PNR interpolants
    (:func:`.pnr.angle_interpolants`), ``reference_mf`` (static) the
    mass-scaled reference frequency (``WaveformOptions.reference_mf``):
    the reference may then lie outside the frequencies, and a non-zero
    ``binary.f_ref`` is an error."""
    reference = reference_mf is not None
    wf, m1, m2, chi1, chi2 = setup_waveform(binary, delta_f, f_min, f_max, amp_scale, reference_mf)
    wf.hm_release = 122019
    error = wf_errors(wf)
    if reference:
        error = error | (jnp.asarray(binary.f_ref, dtype=real()) != 0.0)
    elif check_reference:
        # tuned angles: f_min <= f_ref <= f_max
        upper = jnp.where(delta_f > 0, wf.f_max_prime, f_max)
        error = error | ~((wf.f_ref >= f_min) & (wf.f_ref <= upper))
    if mode_range_ is None:
        mode_range_ = pnr.mode_range(modes)
    base, wf0 = barrier(pnr.common_setup(wf, m1, m2, chi1, chi2, mode_range_, phase_grid, pnr.mode_range(modes),
                                         reference))
    error = error | base.error_spin | (wf0.f_max_prime <= wf0.f_min)
    setup = _spintaylor_setup(base, wf0, m1, m2, modes)
    st_failed = setup[3].failed
    data = _pipeline_330(setup, modes, frequencies, ~st_failed)
    data.error = data.error | error | st_failed
    return data


def _polarizations(binary, frequencies, options):
    with precision(precision_of(binary)):
        return _polarizations_in(binary, frequencies, options)


def _polarizations_in(binary, frequencies, options):
    pnr.clear_cache()
    frequencies = jnp.asarray(frequencies, dtype=real())
    modes = coprecessing_modes(options.mode_array)
    data = _decide(binary, frequencies, 0.0, frequencies[0], frequencies[-1], modes,
                   phase_grid=options.extended_phase_grid, amp_scale=strain_scale(options),
                   reference_mf=options.reference_mf)
    hp, hc = _twist_polarizations(data, modes, frequencies)
    nan = jnp.asarray(jnp.nan + 1j * jnp.nan, complex_())
    return jnp.where(data.error, nan, hp), jnp.where(data.error, nan, hc)


_POLARIZATIONS = {}


def polarizations(binary, frequencies, options):
    """``(h_+, h_x)`` of IMRPhenomXPNR at ``frequencies`` (Hz, positive and
    increasing), as ``SimInspiralChooseFDWaveformSequence``: zero above
    ``min(f_max, 0.3 / M)`` (``f_max`` the last frequency); NaN where the
    model fails. Float32 for a binary of float32 parameters. With
    ``options.reference_mf``, the binary is given at the mass-scaled
    reference frequency ``reference_mf / M`` (and ``binary.f_ref`` must be
    zero).

    Under ``jax.vmap`` the binaries are computed one after the other. The
    derivatives with respect to the binary come from forward-mode Jacobians
    (:func:`._transforms.differentiable`): ``jax.grad``, ``jax.jvp`` and
    ``jax.jacfwd`` all work."""
    if options not in _POLARIZATIONS:
        _POLARIZATIONS[options] = differentiable(functools.partial(_polarizations, options=options))
    return _POLARIZATIONS[options](binary, frequencies)


def jframe_groups(modes, custom_mode_array):
    """The J-frame modes ``(l, m)`` (``l`` among the co-precessing modes,
    ``|m| <= l``) grouped by the mode range of their set-up: as in LAL,
    without a mode array each J-frame mode has its own range ``M_MIN =
    max(1, |m|)``, ``M_MAX = L_MAX_PNR = l``."""
    ells = sorted({ell for ell, _ in modes})
    groups = {}
    for ell in ells:
        for m in range(-ell, ell + 1):
            key = pnr.mode_range(modes) if custom_mode_array else (max(1, abs(m)), ell, ell)
            groups.setdefault(key, []).append((ell, m))
    return groups


_JFRAME_MODES = {}


def jframe_modes(binary, frequencies, options, delta_f=0.0, f_min=None, f_max=None):
    """The J-frame modes of ``XLALSimIMRPhenomXPHMFrequencySequenceOneMode``
    (``delta_f = 0``) or ``XLALSimIMRPhenomXPHMOneMode`` with the flags of
    IMRPhenomXPNR at ``frequencies``: a dict ``{(l, m): (h(f), h(-f))}``,
    ``thetaJN`` and ``zeta`` (of the first set-up), NaN where the model fails.

    Without a mode array LAL sets the binary up again for every J-frame
    mode, with a mode range depending on the mode (:func:`jframe_groups`);
    here the set-ups of the groups run in a ``jax.lax.map`` over the mode
    ranges, so that they are compiled once. The derivatives come from
    forward-mode Jacobians, as for :func:`polarizations`."""
    key = (options, delta_f, f_min, f_max)
    if key not in _JFRAME_MODES:
        _JFRAME_MODES[key] = differentiable(functools.partial(
            _jframe_modes, options=options, delta_f=delta_f, f_min=f_min, f_max=f_max))
    return _JFRAME_MODES[key](binary, frequencies)


def _jframe_modes(binary, frequencies, options, delta_f, f_min, f_max):
    with precision(precision_of(binary)):
        return _jframe_modes_in(binary, frequencies, options, delta_f, f_min, f_max)


def _jframe_modes_in(binary, frequencies, options, delta_f, f_min, f_max):
    pnr.clear_cache()
    frequencies = jnp.asarray(frequencies, dtype=real())
    modes = coprecessing_modes(options.mode_array)
    custom = options.mode_array is not None
    if f_min is None:
        f_min = frequencies[0]
    if f_max is None:
        f_max = frequencies[-1]
    groups = jframe_groups(modes, custom)
    results = []
    if custom:
        ((key, jmodes),) = groups.items()
        data = _decide(binary, frequencies, delta_f, f_min, f_max, modes, mode_range_=key, check_reference=False,
                       phase_grid=options.extended_phase_grid, amp_scale=strain_scale(options),
                       reference_mf=options.reference_mf)
        hs = dict(zip(modes, _coprecessing_modes(data.wf, modes, frequencies, data.xas22)))
        results.append((jmodes, data, hs))
    else:
        keys = list(groups)
        ells = sorted({ell for ell, _ in modes})
        by_ell = {ell: [mode for mode in modes if mode[0] == ell] for ell in ells}
        width = max(len(v) for v in by_ell.values())
        zero = jnp.zeros(frequencies.shape, complex_())

        def modes_of(ell):
            def evaluate(wf):
                hs = _coprecessing_modes(wf, by_ell[ell], frequencies)
                return jnp.stack(hs + [zero] * (width - len(hs)))
            return evaluate

        def one(args):
            key, ell_index = args
            pnr.clear_cache()
            data = _decide(binary, frequencies, delta_f, f_min, f_max, modes, mode_range_=(key[0], key[1], key[2]),
                           check_reference=False, phase_grid=options.extended_phase_grid,
                           amp_scale=strain_scale(options), reference_mf=options.reference_mf)
            hs = jax.lax.switch(ell_index, [modes_of(ell) for ell in ells], data.wf)
            return data, hs

        params = jnp.asarray(keys, dtype=real())
        ell_index = jnp.asarray([ells.index(key[2]) for key in keys])
        datas, hs_all = jax.lax.map(one, (params, ell_index))
        for g, key in enumerate(keys):
            data = jax.tree_util.tree_map(lambda x: x[g], datas)
            hs = {mode: hs_all[g, i] for i, mode in enumerate(by_ell[key[2]])}
            results.append((groups[key], data, hs))
    out = {}
    nan = jnp.asarray(jnp.nan + 1j * jnp.nan, complex_())
    for jmodes, data, hs in results:
        pos, neg = _twist_jframe(data, modes, tuple(jmodes), frequencies, hs)
        for index, mode in enumerate(jmodes):
            out[mode] = (jnp.where(data.error, nan, pos[index]), jnp.where(data.error, nan, neg[index]))
    return out, results[0][1].theta_jn, results[0][1].zeta
