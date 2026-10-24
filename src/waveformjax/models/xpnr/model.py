# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR: the polarizations (``SimInspiralChooseFDWaveformSequence``),
after waveformrs's ``imrphenomx/xpnr.rs``.

IMRPhenomXPNR is IMRPhenomXPHM with the PNR angles, the tuned co-precessing
(2,2) mode, the antisymmetric (2,|2|) contribution, the 122019 release of
the higher modes, ``PhenomXPrecVersion`` 330 and ``PhenomXPFinalSpinMod`` 7.
The precession prescription LAL ends up with depends on the binary:

* in-plane spins below ``1e-7``: plain IMRPhenomXPHM with the NNLO angles
  (no PNR angles, deviations or antisymmetric modes);
* below ``1e-6``: the PNR machinery with the NNLO angles (102);
* otherwise the SpinTaylor angles (330), unless LAL's integration fails:
  then the MSA angles (223), or the NNLO angles if the MSA system cannot be
  initialised.

They are three branches of :func:`._transforms.switch` (the PNR machinery
of versions 102 and 223 is one branch, :class:`.pnr.Fallback`), which under
``jax.vmap`` computes a branch for the whole batch only if some binary
takes it. The SpinTaylor integration (:mod:`.spintaylor`) happens once, or
twice where LAL integrates again from the start of the grid of the PNR
interpolants (:func:`_spintaylor_setup`). The J-frame modes
(:func:`jframe_modes`) are LAL's single-mode functions: a set-up for every
range of modes, mapped with ``jax.lax.map``.

The derivatives are forward-mode Jacobians (:mod:`._transforms`):
``jax.grad``, ``jax.jvp`` and ``jax.jacfwd`` work with respect to the
binary. Failures that LAL reports as errors give NaN polarizations.

Without version fallbacks (``WaveformOptions.version_fallback``) only the
SpinTaylor branch is computed, and the polarizations are NaN where LAL
would switch. The extended angles (``WaveformOptions.extended_spintaylor``)
are that branch with the static flags ``extended`` and ``smooth`` of the
set-up (:func:`.pnr.common_setup`): the SpinTaylor solution goes on where
LAL's stops (:func:`.spintaylor.angles.pn_solution`), every binary keeps the
PNR machinery, and LAL's discrete switches are made continuous (see
``report/extended`` for what changes and by how much). The lean model
(``WaveformOptions.lean``, :mod:`.lean`) is the extended one with the
static flag ``lean``: the SpinTaylor solution of :mod:`.spintaylor.lean`
in one integration (:func:`_lean_setup`) and none of LAL's arithmetic
emulations.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from ..._options import REFERENCE_MF, strain_scale
from . import hm, pnr
from . import pnr_angles as pa
from ._math import (
    MTSUN_SI, PI, Struct, barrier, complex_, hz_to_mf, mass_unit, materialized, mf_to_hz, narrowed, precision, precision_of, real,
    safe_sqrt, tiny,
)
from ._math import (
    lean as lean_mode,
)
from ._transforms import cond, differentiable, switch
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


def half_angles(cos_beta):
    """``cos(beta / 2)`` and ``sin(beta / 2)`` from ``cos(beta)``."""
    return safe_sqrt(jnp.abs(1.0 + cos_beta) / 2.0, tiny()), safe_sqrt(jnp.abs(1.0 - cos_beta) / 2.0, tiny())


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


def twist_factors(alpha, cos_beta, epsilon, mprime):
    """The factors shared by the modes with the same ``m'``: the powers of
    ``exp(i alpha)``, of the half angles and ``exp(-i m' epsilon) / 2``."""
    e1 = jnp.exp(1j * alpha)
    e2 = e1 * e1
    e3 = e1 * e2
    e4 = e1 * e3
    powers = [jnp.ones_like(e1), e1, e2, e3, e4]
    cb, sb = half_angles(cos_beta)
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
    return alpha - prec.alpha_offset, jnp.cos(beta), -gamma - prec.epsilon_offset, map_error | ~inside


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


#: The fields of the precession struct replaced when LAL integrates the
#: SpinTaylor equations again (``IMRPhenomX_PNR_GeneratePNRAngleInterpolants``).
REINTEGRATED = ("st", "fmin_integration", "mfmin_integration", "chi1_evolved", "chi2_evolved")


def _partition(tree):
    """The traced leaves of ``tree`` and what rebuilds it from them."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    dynamic = [isinstance(leaf, jax.Array) for leaf in leaves]
    values = [leaf for leaf, d in zip(leaves, dynamic) if d]
    return values, (treedef, [None if d else leaf for leaf, d in zip(leaves, dynamic)], dynamic)


def _combine(values, meta):
    treedef, static, dynamic = meta
    values = iter(values)
    return jax.tree_util.tree_unflatten(treedef, [next(values) if d else s for s, d in zip(static, dynamic)])


def _tree_where(condition, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(condition, x, y) if isinstance(x, jax.Array) else x, a, b)


def _like_input(zero, anchor):
    """``zero`` (an array of zeros) made to depend on the traced scalar
    ``anchor`` (non-zero tangents under ``jax.jvp``, batched under
    ``jax.vmap`` where ``anchor`` is)."""
    if jnp.issubdtype(zero.dtype, jnp.inexact):
        return zero + 0.0 * anchor
    return jnp.where(anchor == anchor, zero, zero)


def _spintaylor_setup(base, wf0, m1_si, m2_si, modes, needs_st):
    """The set-up of version 330 up to the grid of the PNR interpolants:
    the SpinTaylor solution, ``IMRPhenomX_Initialize_Euler_Angles``,
    ``IMRPhenomX_PNR_PopulateStructs`` and the grid. Where the grid starts
    below the solution, LAL integrates again from the start of the grid and
    repeats the first two (keeping the rest of the set-up and the grid).
    Both integrations are one compiled loop body: a ``jax.lax.scan`` of two
    iterations, each skipped by :func:`._transforms.cond` when not needed.

    Returns ``(prec, wf, grid, flags)`` with ``flags`` the failure of the
    first integration (``failed``, where LAL falls back to the MSA angles),
    the capacities exceeded (``incomplete``) and the errors of the set-up
    and of a second integration (``error``)."""

    def body(first, previous, run, base, wf0, m1_si, m2_si):
        prev_prec, prev_wf, prev_grid, prev_flags = previous
        flow = jnp.where(first, base.st_flow, prev_flags.f0)
        if prev_wf is None:
            fcut = base.st_fcut
        else:
            fcut = jnp.where(first, base.st_fcut, mf_to_hz(prev_wf.f_ring + 8.0 * prev_wf.f_damp, prev_wf.m_tot))
        st = pnr.spintaylor_solution(base, wf0, m1_si, m2_si, flow=flow, fcut=fcut, active=run)
        fresh, wf = pnr.finish_setup(base, wf0, 330, st)
        prec = pnr._copy(fresh)
        if prev_prec is not None:
            for name in fresh.keys():
                if name not in REINTEGRATED:
                    setattr(prec, name, _tree_where(first, getattr(fresh, name), getattr(prev_prec, name)))
            wf = _tree_where(first, wf, prev_wf)
        prec = pnr.st_bank(prec, wf)
        prec = pnr.initialize_euler_angles(prec, wf)
        prec = pnr.populate_structs(prec, wf, 330)
        grid = pnr.interpolation_grid(prec, wf, modes, 330)
        if prev_prec is not None:
            grid = _tree_where(first, grid, prev_grid)
            prec.pnr_hm_mflow = jnp.where(first, prec.pnr_hm_mflow, prev_prec.pnr_hm_mflow)
            prec.pnr_hm_mfhigh = jnp.where(first, prec.pnr_hm_mfhigh, prev_prec.pnr_hm_mfhigh)
        error = prec.error_structs | grid[3] | prec.error_euler
        f0 = grid[0] * grid[2]
        flags = Struct(
            f0=f0,
            again=prec.fmin_integration > f0,
            failed=jnp.where(first, st.failed, prev_flags.failed),
            incomplete=jnp.where(first, st.incomplete, prev_flags.incomplete),
            error=jnp.where(first, error, prev_flags.error | error | st.failed | st.incomplete),
        )
        return prec, wf, grid, flags

    false = jnp.asarray(False)
    start = Struct(f0=jnp.zeros((), real()), again=false, failed=false, incomplete=false, error=false)
    inputs = (base, wf0, m1_si, m2_si)
    holder = {}

    def structure(inputs):
        values, holder["meta"] = _partition(body(jnp.asarray(True), (None, None, None, start), True, *inputs))
        return values

    shapes = jax.eval_shape(lambda: structure(inputs))
    meta = holder["meta"]

    def compute(operands):
        first, carry, run, inputs = operands
        values, _ = _partition(body(first, _combine(carry, meta), run, *inputs))
        return [jnp.asarray(v).astype(narrowed(s.dtype)) for v, s in zip(values, shapes)]

    def step(carry, index):
        first = index == 0
        flags = _combine(carry, meta)[3]
        run = jnp.where(first, needs_st, needs_st & ~flags.failed & flags.again)
        # (batched: the integration runs for the whole batch if any binary
        # needs it, with no steps for the others)
        return cond(run, compute, lambda operands: operands[1], (first, carry, run, inputs)), None

    # (the initial carry depends on the inputs, so that under jax.jvp and
    # jax.vmap its tangents and batching are those of the iterations: JAX
    # then traces the body once rather than until they agree)
    anchor = jnp.asarray(m1_si, real())
    carry = [_like_input(jnp.zeros(s.shape, narrowed(s.dtype)), anchor) for s in shapes]
    carry, _ = jax.lax.scan(step, carry, jnp.arange(2))
    return _combine(carry, meta)


def _lean_setup(base, wf0, m1, m2, modes, needs_st):
    """:func:`_spintaylor_setup` of the lean model: one integration, from
    low enough a frequency (:func:`.pnr.common_setup`) that LAL's second
    integration from the start of the grid of the PNR interpolants is not
    needed (an error where it would be). The two passes of
    :func:`_spintaylor_setup` compile to ~8% more instructions."""
    st = pnr.spintaylor_solution(base, wf0, m1, m2, active=needs_st)
    prec, wf = pnr.finish_setup(base, wf0, 330, st)
    prec = pnr.st_bank(prec, wf)
    prec = pnr.initialize_euler_angles(prec, wf)
    prec = pnr.populate_structs(prec, wf, 330)
    grid = pnr.interpolation_grid(prec, wf, modes, 330)
    f0 = grid[0] * grid[2]
    error = prec.error_structs | grid[3] | prec.error_euler | (prec.fmin_integration > f0)
    flags = Struct(f0=f0, again=jnp.asarray(False), failed=st.failed, incomplete=st.incomplete, error=error)
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


def _angle_data(prec, wf, splines, modes, frequencies, version, tuned):
    """What a prescription contributes to the waveform: for each
    co-precessing mode the Euler angles ``(alpha, cos(beta), epsilon)`` at
    ``frequencies``, the antisymmetric part of the (2,2) mode, the line of
    sight in the J frame, the polarization angle, the waveform struct as the
    set-up left it, and the error flag. The same structure for every
    prescription, so that :func:`._transforms.switch` can choose among them and the
    modes are built once."""
    mf = wf.m_sec * frequencies
    mf_cut = wf.f_max_prime * wf.m_sec
    keep = mf <= mf_cut
    # LAL skips the odd modes where they vanish
    vanishing_odd = (wf.q == 1.0) & (wf.chi1l == wf.chi2l)
    error = jnp.asarray(False)
    alphas, cos_betas, epsilons = [], [], []
    for ell, emm in modes:
        if tuned:
            alpha, cos_beta, epsilon, err = _mode_angles_pnr(prec, wf, splines, ell, emm, mf)
            if emm % 2:
                err = err & ~vanishing_odd
            error = error | jnp.any(err & keep)
        else:
            alpha, cos_beta, epsilon = pnr.nnlo_angles(prec, PI * mf * 2.0 / emm)
        alphas.append(alpha)
        cos_betas.append(cos_beta)
        epsilons.append(epsilon)
    if (2, 2) in modes and tuned:
        amp, phase, err, xas22 = antisymmetric_waveform(prec, wf, frequencies, version,
                                                        splines.st_extra if version == 330 else None)
        error = error | err
        h_as = amp * jnp.exp(1j * phase)
    else:
        h_as = jnp.zeros(frequencies.shape, complex_())
        xas = xas_setup(wf)
        xas22 = Struct(amp=xas.amp, phase=xas.phase, linb=xas.linb)
    wf_out = Struct(**{name: getattr(wf, name) for name in WF_FIELDS})
    wf_out.hm_release = jnp.asarray(122019)
    if prec.smooth:
        # (a marker: the higher modes take exact derivatives where LAL takes
        # finite differences, see hm.finite_differences)
        wf_out.exact_derivatives = jnp.zeros((), real())
    return Struct(
        wf=wf_out, alpha=jnp.stack(alphas), cos_beta=jnp.stack(cos_betas), epsilon=jnp.stack(epsilons), h_as=h_as,
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
        twist = materialized(twist_factors(data.alpha[index], data.cos_beta[index], data.epsilon[index], emm))
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
        cb, sb = half_angles(data.cos_beta[index])
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


def _pipeline(base, wf_base, modes, frequencies, version, pnr_on, active=True):
    """The angle data of one prescription (:func:`_angle_data`) of version
    102 or 223."""
    prec, wf = barrier(pnr.finish_setup(base, wf_base, version))
    if pnr_on:
        prec = barrier(pnr.populate_structs(prec, wf, version))
        grid = pnr.interpolation_grid(prec, wf, modes, version)
        error = prec.error_structs | grid[3]
        prec, splines, error = barrier(_prepare_pnr(prec, wf, grid, version, error, active))
    else:
        splines, error = None, jnp.asarray(False)
    data = _angle_data(prec, wf, splines, modes, frequencies, version, pnr_on)
    data.error = data.error | error
    return data


def _pipeline_330(setup, modes, frequencies, active=True):
    """The angle data of version 330 from :func:`_spintaylor_setup`."""
    prec, wf, grid, flags = setup
    prec, wf = barrier((pnr._copy(prec), wf))
    # (the antisymmetric mode reads the SpinTaylor alpha at the frequencies
    # below its transition at 0.85 fRING: evaluated with the grid)
    extra_mf, n_extra = None, None
    if (2, 2) in modes:
        extra_mf = hz_to_mf(frequencies, wf.m_tot)
        n_extra = jnp.sum(extra_mf < pnr.antisymmetric_points(wf)[0])
    prec, splines, error = barrier(_prepare_pnr(prec, wf, grid, 330, flags.error, active, extra_mf, n_extra))
    data = _angle_data(prec, wf, splines, modes, frequencies, 330, True)
    data.error = data.error | error | flags.incomplete
    return data


def _decide(binary, frequencies, delta_f, f_min, f_max, modes, parity_fix, mode_range_=None, check_reference=True,
            fallback=True, extended=False, smooth=None, direct=True, phase_grid=False, amp_scale=1.0, lean=False,
            reference_mf=None):
    """Set up the binary and compute the angle data of the prescription LAL
    would use. Without ``fallback`` (static), only the SpinTaylor angles
    (version 330) are computed, with the error flag set where LAL would use
    another prescription. With ``extended`` (static, implies no
    ``fallback``), the SpinTaylor angles are extended beyond LAL's and,
    with ``smooth`` (static, default ``extended``), made continuous in the
    parameters (:func:`.spintaylor.angles.pn_solution`); for the smooth
    angles, ``direct`` and ``phase_grid`` (static) select the direct
    SpinTaylor angles (:func:`.spintaylor.angles.direct_spline`) and the
    phase grid of the PNR interpolants (:func:`.pnr.angle_interpolants`).
    ``lean`` (static) is the lean model (:mod:`.lean`): the smooth, direct
    angles with the parity fix and the lean SpinTaylor solution; for it,
    ``reference_mf`` (static) the mass-scaled reference frequency
    (``WaveformOptions.reference_mf``): the reference may then lie outside
    the frequencies, and a non-zero ``binary.f_ref`` is an error."""
    if lean:
        extended, smooth, direct, parity_fix = True, True, True, True
    fallback = fallback and not extended
    smooth = extended if smooth is None else smooth
    reference = reference_mf is not None
    wf, m1_si, m2_si, chi1, chi2 = setup_waveform(binary, delta_f, f_min, f_max, amp_scale, reference_mf)
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
    base, wf0 = barrier(pnr.common_setup(wf, m1_si, m2_si, chi1, chi2, mode_range_, parity_fix, extended, smooth,
                                         direct, phase_grid, lean, pnr.mode_range(modes), reference))
    error = error | base.error_spin | (wf0.f_max_prime <= wf0.f_min)

    # the SpinTaylor integration, where LAL attempts it
    needs_st = base.pnr_on & ~base.nnlo_from_start
    setup = (_lean_setup if lean else _spintaylor_setup)(base, wf0, m1_si, m2_si, modes, needs_st)
    st_failed = needs_st & setup[3].failed
    if not fallback:
        data = _pipeline_330(setup, modes, frequencies, needs_st & ~st_failed)
        data.error = data.error | error | ~needs_st | st_failed
        return data
    system = pnr._msa.system(pnr.msa_inputs(base, wf0, base.chi1, base.chi2))
    msa_error = pnr._msa.msa_error(system)
    # 0: PNR off (NNLO angles); 1: NNLO angles in PNR; 2: MSA; 3: SpinTaylor
    index = jnp.where(
        ~base.pnr_on, 0,
        jnp.where(base.nnlo_from_start | (st_failed & msa_error), 1, jnp.where(st_failed | ~needs_st, 2, 3)),
    )

    # (the flags are true in the branch taken; under jax.vmap a branch is
    # computed for the whole batch if any binary takes it, with no grid for
    # the others). Versions 102 and 223 of the PNR machinery are one branch
    # (pnr.Fallback).
    def no_pnr(operands):
        base, wf0, _, index, frequencies = operands
        return _pipeline(base, wf0, modes, frequencies, 102, False, index == 0)

    def fallback(operands):
        base, wf0, _, index, frequencies = operands
        return _pipeline(base, wf0, modes, frequencies, pnr.Fallback(index == 2), True, (index == 1) | (index == 2))

    def st_ok(operands):
        _, _, setup, index, frequencies = operands
        return _pipeline_330(setup, modes, frequencies, index == 3)

    data = switch(jnp.where(index == 0, 0, jnp.where(index == 3, 2, 1)), [no_pnr, fallback, st_ok],
                  (base, wf0, setup, index, frequencies))
    data.error = data.error | error
    return data


def _polarizations(binary, frequencies, options):
    with precision(precision_of(binary)), lean_mode(options.lean):
        return _polarizations_in(binary, frequencies, options)


def _polarizations_in(binary, frequencies, options):
    pnr.clear_cache()
    frequencies = jnp.asarray(frequencies, dtype=real())
    modes = coprecessing_modes(options.mode_array)
    data = _decide(binary, frequencies, 0.0, frequencies[0], frequencies[-1], modes, options.pnr_parity_fix,
                   fallback=options.version_fallback, extended=options.extended_spintaylor,
                   direct=options.extended_direct_angles, phase_grid=options.extended_phase_grid,
                   amp_scale=strain_scale(options), lean=options.lean, reference_mf=options.reference_mf)
    hp, hc = _twist_polarizations(data, modes, frequencies)
    nan = jnp.asarray(jnp.nan + 1j * jnp.nan, complex_())
    return jnp.where(data.error, nan, hp), jnp.where(data.error, nan, hc)


_POLARIZATIONS = {}


def polarizations(binary, frequencies, options):
    """``(h_+, h_x)`` of IMRPhenomXPNR at ``frequencies`` (Hz, positive and
    increasing), as ``SimInspiralChooseFDWaveformSequence``: zero above
    ``min(f_max, 0.3 / M)`` (``f_max`` the last frequency); NaN where LAL
    fails.

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
    ``thetaJN`` and ``zeta`` (of the first set-up), NaN where LAL fails.

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
    with precision(precision_of(binary)), lean_mode(options.lean):
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
    parity_fix = options.pnr_parity_fix
    results = []
    if custom:
        ((key, jmodes),) = groups.items()
        data = _decide(binary, frequencies, delta_f, f_min, f_max, modes, parity_fix, mode_range_=key,
                       check_reference=False, fallback=options.version_fallback,
                       extended=options.extended_spintaylor, direct=options.extended_direct_angles,
                       phase_grid=options.extended_phase_grid, amp_scale=strain_scale(options), lean=options.lean,
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
            data = _decide(binary, frequencies, delta_f, f_min, f_max, modes, parity_fix,
                           mode_range_=(key[0], key[1], key[2]), check_reference=False,
                           fallback=options.version_fallback, extended=options.extended_spintaylor,
                           direct=options.extended_direct_angles, phase_grid=options.extended_phase_grid,
                           amp_scale=strain_scale(options), lean=options.lean, reference_mf=options.reference_mf)
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
