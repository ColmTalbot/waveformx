# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR (:mod:`..model`) cut into stages, in JAX.

The MLX port runs the model's own JAX code for everything that is a
function of one binary's scalars, or of one binary's scalars and one point:
each such stage is traced to a jaxpr once (:mod:`._translate`) and evaluated
with MLX operations, for a batch of binaries and many points at once by
broadcasting. What does not fit that form is written for MLX by hand: the
integration of the precession equations (a loop), the angles read from its
dense output (searches in tables), the splines of the PNR interpolants and
the moving average of the antisymmetric mode. This module holds the stages
as plain JAX functions of pytrees, and :func:`reference` composes them with
the JAX versions of the hand-written parts: it reproduces
:func:`waveformx.models.xpnr.model.polarizations` (the tests check it to
rounding), and its intermediate values are the reference of the MLX port.

The stages, in order (``prec`` and ``wf`` are the model's precession and
waveform structs):

=====  ====================================================================
``A``  the binary to the common set-up and the inputs of the integration
``B``  (by hand) the integration: :func:`..spintaylor.lean.solution`
``C1`` ``finish_setup`` and the 15 points of the set-up's SpinTaylor angles
``B2`` (by hand) the SpinTaylor angles at points (:func:`st_angles`)
``C2`` the rest of the set-up of version 330, up to the grid of the PNR
       interpolants
``D``  per grid point: the PNR ``alpha`` and ``beta`` (``uniform_angles``)
``S``  (by hand) the interpolants: :func:`..pnr._lal_interpolants`
``E``  the offsets at the reference frequency, the remapped J frame and the
       coefficients of the modes and of the antisymmetric mode
``F1`` per frequency: the frequencies at which each mode reads the angles,
       and the amplitude ratio of the antisymmetric mode
``F2`` per frequency: the angles, the modes, the twist and the
       polarizations
=====  ====================================================================
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .. import hm, model, pnr
from .. import pnr_angles as pa
from .._math import (
    MTSUN_SI, PI, Struct, fma, hz_to_mf, mf_to_hz, powers, precision, real, safe_div,
)
from ..amp22 import amp_ansatz
from ..antisymmetric import _kappa_function, antisymmetric_coefficients, smooth_amp_ratio
from ..internals import errors as wf_errors
from ..phase22 import phase_xas
from ..spintaylor import angles as st_angles
from ..spintaylor import lean as st_lean
from ..xas import h22, phif_ref

__all__ = ["BANK_GROUPS", "stage_a", "stage_c1", "stage_c2", "grid_point", "stage_e", "stage_f1", "stage_f2",
           "spline_tail", "grid_frequencies", "reference", "binary_leaves", "BINARY_FIELDS"]

#: The fields of a :class:`~waveformx.Binary`, in order.
BINARY_FIELDS = ("mass_1", "mass_2", "spin_1", "spin_2", "luminosity_distance", "inclination", "phi_ref", "f_ref")
#: The groups of points at which the set-up reads the SpinTaylor angles
#: (:func:`..pnr.st_bank`), and their sizes.
BANK_GROUPS = (("euler", 4), ("alpha", 4), ("beta", 3), ("antisymmetric", 4))
#: The interpolants of the PNR angles (:func:`..pnr._lal_interpolants`).
SPLINES = ("alpha", "beta", "gamma")


def binary_leaves(binary):
    """The fields of a binary as a dict."""
    return {name: getattr(binary, name) for name in BINARY_FIELDS}


def _binary(leaves):
    from ...._binary import Binary

    return Binary(**{name: leaves[name] for name in BINARY_FIELDS})


# ---------------------------------------------------------------------------
# A: the common set-up
# ---------------------------------------------------------------------------


def stage_a(leaves, f_min, f_max, modes, amp_scale, reference_mf=None):
    """``model._decide`` of the model up to the integration: returns
    ``(base, wf0, ode, error)`` with ``ode`` the inputs of
    :func:`..spintaylor.lean.solution` (``m1``, ``m2`` in solar masses, the
    spins ``chi1``, ``chi2``, ``f_ref``, ``f_low``, ``f_cut``, whether the
    integration is needed and whether its start is valid);
    ``reference_mf`` the mass-scaled reference frequency, if any."""
    pnr.clear_cache()
    binary = _binary(leaves)
    wf, m1, m2, chi1, chi2 = model.setup_waveform(binary, 0.0, f_min, f_max, amp_scale, reference_mf)
    wf.hm_release = 122019
    error = wf_errors(wf)
    if reference_mf is None:
        error = error | ~((wf.f_ref >= f_min) & (wf.f_ref <= f_max))
    else:
        error = error | (binary.f_ref != 0.0)
    mode_range = pnr.mode_range(modes)
    base, wf0 = pnr.common_setup(wf, m1, m2, chi1, chi2, mode_range, False, mode_range, reference_mf is not None)
    error = error | base.error_spin | (wf0.f_max_prime <= wf0.f_min)
    needs_st = jnp.asarray(True)
    flow_safe = jnp.where(base.st_flow_ok, base.st_flow_lean, wf0.f_min)
    ode = Struct(m1=m1, m2=m2, chi1=jnp.stack(base.chi1), chi2=jnp.stack(base.chi2), f_ref=wf0.f_ref, f_low=flow_safe,
                 f_cut=base.st_fcut, active=needs_st, flow_ok=base.st_flow_ok)
    return base, wf0, ode, error


def solution_summary(sol, flow_ok):
    """What the scalar set-up reads of the SpinTaylor solution: the spins
    and LNhat at the end of the inspiral, the start of the solution and its
    range in ``M omega``, and its failure flags."""
    return Struct(
        s1_end=jnp.stack(sol.s1_end), s2_end=jnp.stack(sol.s2_end), lnhat_end=jnp.stack(sol.lnhat_end),
        f_min_pn=sol.f_min_pn, omega_first=sol.omega_first, omega_last=sol.omega_last,
        failed=sol.failed | ~flow_ok, incomplete=sol.incomplete,
    )


def _st_stub(summary):
    """A stand-in for the SpinTaylor solution in ``prec.st``: its scalars,
    and markers for the functions that look for the arrays (which the
    stages never read)."""
    dummy = jnp.zeros((1,), real())
    return Struct(lean=jnp.zeros((), real()), y=dummy, dense=dummy, h=dummy, f_min_pn=summary.f_min_pn,
                  lnhat_end=list(summary.lnhat_end), s1_end=list(summary.s1_end), s2_end=list(summary.s2_end))


def _st_range(prec, wf, summary):
    """The scale from ``M omega`` to ``Mf`` and the range of the solution
    in ``Mf`` (:func:`..spintaylor.lean.context`)."""
    mf_scale = MTSUN_SI * wf.m_tot / prec.pi_gm
    first_mf = jnp.cbrt(summary.omega_first) ** 3 * mf_scale
    last_mf = jnp.cbrt(summary.omega_last) ** 3 * mf_scale
    return mf_scale, first_mf, last_mf


# ---------------------------------------------------------------------------
# C1, C2: the set-up of version 330
# ---------------------------------------------------------------------------


def stage_c1(base, wf0, summary):
    """``finish_setup`` and :func:`..pnr.st_bank` up to the evaluation of
    the SpinTaylor angles: returns ``(prec, wf, points, rotation,
    st_range)`` with the 15 points (``Mf``) of :data:`BANK_GROUPS`, the J
    frame of the angles and ``(mf_scale, first_mf, last_mf)``."""
    pnr.clear_cache()
    prec, wf = pnr.finish_setup(base, wf0, 330, _st_stub(summary))
    prec.st_rotation = pnr._rotation(prec)
    mf_scale, first_mf, last_mf = _st_range(prec, wf, summary)
    context = Struct(lean=jnp.zeros((), real()), first_mf=first_mf, last_mf=last_mf)
    fmax_inspiral, euler = pnr._euler_points(prec, wf, context)
    ap = pnr.precompute_alpha_coefficients(prec, wf, 330)
    bp = pa.beta_connection_frequencies(pnr.precompute_beta_coefficients(prec, wf, 330))
    _, _, alpha = pnr._alpha_points(prec, wf, ap, 330)
    _, beta = pnr._st_beta_points(prec, bp, 0.98 * fmax_inspiral)
    _, antisymmetric = pnr.antisymmetric_points(wf)
    prec.st_bank_params = Struct(alpha=ap, beta=bp)
    points = jnp.concatenate([euler, alpha, beta, antisymmetric])
    return prec, wf, points, jnp.stack(prec.st_rotation), jnp.stack([mf_scale, first_mf, last_mf])


def _with_bank(prec, points, values, st_range):
    """``prec`` with the SpinTaylor angles at the bank's points
    (:func:`..pnr.st_bank`) and the range of the solution."""
    _, first_mf, last_mf = st_range[0], st_range[1], st_range[2]
    prec.st_context = Struct(lean=jnp.zeros((), real()), first_mf=first_mf, last_mf=last_mf)
    bank, start = Struct(), 0
    for name, n in BANK_GROUPS:
        setattr(bank, name, Struct(mf=points[start:start + n], values=tuple(v[start:start + n] for v in values)))
        start += n
    prec.st_bank = bank
    return prec


def stage_c2(prec, wf, points, values, st_range, modes):
    """The rest of ``model._spintaylor_setup`` with the SpinTaylor angles at the
    bank's points ``values`` (``(alpha, dalpha, cosbeta, dcosbeta,
    inside)``, each of shape ``(15,)``): returns ``(prec, wf, grid,
    error)`` with ``grid = (i_start, n, delta_f)`` of the PNR interpolants."""
    pnr.clear_cache()
    prec = _with_bank(prec, points, values, st_range)
    prec = pnr.initialize_euler_angles(prec, wf)
    prec = pnr.populate_structs(prec, wf, 330)
    grid = pnr.interpolation_grid(prec, wf, modes, 330)
    f0 = grid[0] * grid[2]
    error = prec.error_structs | grid[3] | prec.error_euler | (prec.fmin_integration > f0)
    return prec, wf, jnp.stack([grid[0], grid[1], grid[2]]), error


# ---------------------------------------------------------------------------
# D: the PNR angles on the grid, per point
# ---------------------------------------------------------------------------


def grid_frequencies(k, grid, m_tot):
    """``Mf`` of the points ``k`` (real) of the grid of the PNR
    interpolants (the last point repeated beyond ``n``)."""
    i_start, n, h = grid[0], grid[1], grid[2]
    return hz_to_mf((i_start + jnp.minimum(k, n - 1.0)) * h, m_tot)


def grid_point(prec, wf, mf, st_values):
    """``uniform_angles`` at one point ``mf`` with
    the SpinTaylor angles ``st_values`` there: ``(alpha, beta, error)``."""
    pnr.clear_cache()
    return pnr.uniform_angles(prec, wf, mf, 330, tuple(st_values))


# ---------------------------------------------------------------------------
# E: offsets, remapped J frame and the coefficients of the modes
# ---------------------------------------------------------------------------


def spline_tail(values):
    """``(alpha, beta, gamma)`` of :func:`..pnr.spline_angles` from the
    values of the interpolants (:data:`SPLINES`) at a point."""
    alpha, beta, gamma = values
    return alpha, beta, gamma


def stage_e(prec, wf, at_ref, inside_ref, modes):
    """``_prepare_pnr`` after the interpolants (``at_ref``: the values of
    :data:`SPLINES` at ``f_ref``; ``inside_ref``: whether ``f_ref`` lies on
    the grid) and the frequency-independent parts of ``_angle_data`` and
    ``_twist_polarizations``. Returns ``(prec, wf, extra, error)``."""
    pnr.clear_cache()
    alpha_ref, beta_ref, gamma_ref = spline_tail(list(at_ref))
    error = ~inside_ref | ~jnp.isfinite(beta_ref)
    prec.alpha_offset = alpha_ref
    prec.epsilon_offset = -gamma_ref - prec.epsilon0
    prec = pnr.remap_theta_jsf(prec, wf, beta_ref)
    extra = Struct()
    if (2, 2) in modes:
        co = antisymmetric_coefficients(prec, wf, 330)
        error = error | co.error
        extra.co = co
        amp22, phase22, linb = co.amp22, co.phase22, co.linb
    else:
        from ..xas import xas_setup

        xas = xas_setup(wf)
        amp22, phase22, linb = xas.amp, xas.phase, xas.linb
    wf_out = Struct(**{name: getattr(wf, name) for name in model.WF_FIELDS})
    wf_out.hm_release = jnp.asarray(122019)
    xas = Struct(amp=amp22, phase=phase22, linb=linb, lina=0.0, inveta=1.0 / wf_out.eta,
                 amp_prefactor=wf_out.amp0 * wf_out.amp_norm)
    higher = [mode for mode in modes if mode != (2, 2)]
    extra.coefficients = list(hm.modes_coefficients(wf_out, phase22, higher))
    extra.xas = xas
    extra.phif = phif_ref(wf_out, xas)
    extra.ylms = {ell: model.spherical_harmonics(prec.theta_jn, ell) for ell in sorted({e for e, _ in modes})}
    extra.wf_out = wf_out
    return prec, wf, extra, error


# ---------------------------------------------------------------------------
# F1, F2: per frequency
# ---------------------------------------------------------------------------


def stage_f1(prec, wf, f, modes):
    """At one frequency ``f`` (Hz): the frequencies (Hz) at which each mode
    reads the PNR angles and whether the map failed, ``Mf`` of the
    antisymmetric mode and its amplitude ratio before the moving average."""
    pnr.clear_cache()
    mf = wf.m_sec * f
    mapped, errors = [], []
    for ell, emm in modes:
        if (ell, emm) == (2, 2):
            mf_mapped, map_error = mf, jnp.zeros((), bool)
        else:
            mf_rd_lm = pnr.ringdown_frequency(ell, emm, wf)
            low, high = pa.linear_frequency_map_transition_frequencies(emm, wf.f_ring, mf_rd_lm, prec.pnr_hm_mflow,
                                                                       prec.pnr_hm_mfhigh)
            mf_mapped, map_error = pnr.frequency_map(prec, mf, ell, emm, low, high, wf.f_ring, mf_rd_lm)
        mapped.append(mf_to_hz(mf_mapped, wf.m_tot))
        errors.append(map_error)
    mf_as = hz_to_mf(f, wf.m_tot)
    if (2, 2) in modes:
        kappa_at, kappa_rd = _kappa_function(prec, wf)
        below = mf_as < wf.f_ring
        kappa = jnp.where(below, kappa_at(jnp.cbrt(PI * jnp.where(below, mf_as, wf.f_ring))), kappa_rd)
    else:
        kappa = jnp.zeros((), real())
    return jnp.stack(mapped), jnp.stack(errors), mf_as, kappa


def stage_f2(prec, wf, extra, f, spline_values, inside, map_errors, kappa, st_alpha, st_inside, modes):
    """At one frequency ``f`` (Hz): ``(h_+, h_x)`` as ``(re, im)`` pairs
    and whether the point makes the waveform fail. ``spline_values``
    (``(n_modes, 6)``) and ``inside`` are the interpolants at the mapped
    frequencies of the modes, ``kappa`` the smoothed amplitude ratio and
    ``st_alpha``, ``st_inside`` the SpinTaylor ``alpha`` at ``Mf`` of the
    antisymmetric mode."""
    pnr.clear_cache()
    wf_out = extra.wf_out
    mf = wf.m_sec * f
    keep = mf <= wf.f_max_prime * wf.m_sec
    vanishing_odd = (wf.q == 1.0) & (wf.chi1l == wf.chi2l)
    error = jnp.zeros((), bool)
    angles = []
    for index, (ell, emm) in enumerate(modes):
        a, b, g = spline_tail([spline_values[index, j] for j in range(len(SPLINES))])
        err = map_errors[index] | ~inside[index]
        if emm % 2:
            err = err & ~vanishing_odd
        error = error | (err & keep)
        angles.append((a - prec.alpha_offset, jnp.cos(b), -g - prec.epsilon_offset))
    h_as = None
    if (2, 2) in modes:
        co = extra.co
        mf_as = hz_to_mf(f, wf.m_tot)
        nan = jnp.full_like(st_alpha, jnp.nan)
        pnr._st_spline_put(prec, wf, mf_as, (st_alpha, nan, nan, nan, st_inside))
        alpha_pnr, err_alpha = pnr.region_alpha_at(prec, wf, mf_as, co.alpha_params, 330)
        used = mf_as < co.mf_t
        error = error | (err_alpha & used & (mf_as <= wf.f_max_prime * wf.m_sec))
        p = powers(mf)
        phi = phase_xas(mf, p, co.phase22) * co.inveta
        phi = phi + (fma(co.linb, mf, co.lina) + co.phiref22)
        a = amp_ansatz(mf, p, co.amp22)
        phi_as = jnp.where(mf < co.mf_t, fma(co.a0, mf, phi / 2.0 + alpha_pnr) + co.phi_a0, phi + co.phi_b0)
        amp = co.amp_prefactor * p.m_seven_sixths * (kappa * a)
        h_as = amp * jnp.exp(1j * (co.phase_offset + phi_as))
    coefficients = dict(zip([m for m in modes if m != (2, 2)], extra.coefficients))
    hp = jnp.zeros((), complex)
    hc = jnp.zeros((), complex)
    mf_modes = wf_out.m_sec * f
    for index, (ell, emm) in enumerate(modes):
        if (ell, emm) == (2, 2):
            h = h22(mf_modes, wf_out, extra.xas, extra.phif, mf_cut=jnp.inf)
        else:
            h = hm.mode_h(coefficients[(ell, emm)], mf_modes)
        twist = model.twist_factors(*angles[index], emm)
        p_, c_ = model.twist_up(ell, emm, twist, extra.ylms[ell], h, 1.0)
        if (ell, emm) == (2, 2):
            pa_, ca_ = model.twist_up(ell, emm, twist, extra.ylms[ell], h_as, -1.0)
            p_ = p_ + pa_
            c_ = c_ + ca_
        if emm % 2:
            p_ = jnp.where(vanishing_odd, 0.0, p_)
            c_ = jnp.where(vanishing_odd, 0.0, c_)
        hp = hp + jnp.where(keep, p_, 0.0)
        hc = hc + jnp.where(keep, c_, 0.0)
    hp, hc = model.rotate_polarizations(prec.zeta_polarization, hp, hc)
    return jnp.stack([jnp.real(hp), jnp.imag(hp), jnp.real(hc), jnp.imag(hc)]), error


# ---------------------------------------------------------------------------
# the parts by hand: their arithmetic at one point (or step, or interval),
# after the searches and gathers that the MLX port does itself
# ---------------------------------------------------------------------------


def st_newton(omega, y0, dense, om1):
    """``_time_at_omega`` of :mod:`..spintaylor.lean` in the step holding
    ``M omega = omega`` (``y0``: ``M omega`` and LNhat at its start;
    ``dense``: their dense output; ``om1``: ``M omega`` at its end): the
    position ``x`` in the step (not differentiated)."""
    from ..spintaylor import trajectory

    omega, y0, dense, om1 = jax.lax.stop_gradient((omega, y0, dense, om1))
    om0 = y0[0]
    x = jnp.clip(safe_div(omega - om0, om1 - om0), 0.0, 1.0)

    def omega_at(x):
        return trajectory._dense_eval(dense, y0, x)[..., 0]

    def newton(_, x):
        value, slope = jax.jvp(omega_at, (x,), (jnp.ones_like(x),))
        return jnp.clip(x - safe_div(value - omega, slope), 0.0, 1.0)

    return jax.lax.fori_loop(0, st_lean.NEWTON, newton, x)


def st_point_angles(st_range, rotation, mf, y0, dense, t0, h, x, node):
    """:func:`..spintaylor.lean.angles` at one point ``mf`` in its step
    (``y0``, ``dense``, ``t0``, ``h``) at ``x`` (:func:`st_newton`), with
    ``node`` the continuous ``alpha`` of the node before it: ``(alpha,
    dalpha, cosbeta, dcosbeta, inside)``."""
    from ..spintaylor import trajectory
    from ..spintaylor.angles import _rotate, small_atan2, small_square

    mf_scale, first_mf, last_mf = st_range[0], st_range[1], st_range[2]
    omega = mf / mf_scale
    t = jax.lax.stop_gradient(t0 + x * h)
    h_safe = jnp.where(h != 0.0, h, 1.0)

    def state(t):
        return trajectory._dense_eval(dense, y0, (t - t0) / h_safe)

    y, dydt, t = st_angles._refined(state, t, omega)
    rot = (rotation[0], rotation[1], rotation[2])
    lx, ly, lz = _rotate(y[..., 1], y[..., 2], y[..., 3], *rot)
    dlx, dly, dlz = _rotate(dydt[..., 1], dydt[..., 2], dydt[..., 3], *rot)
    raw = small_atan2(ly, lx)
    d = raw - node
    alpha = node + d - 2 * PI * jnp.round(jax.lax.stop_gradient(d) / (2 * PI))
    dmf_dt = mf_scale * dydt[..., 0]
    r2 = lx * lx + ly * ly
    big = r2 > small_square()
    dalpha_dt = jnp.where(big, lx * dly - ly * dlx, 0.0) / jnp.where(big, r2, 1.0)
    inside = (mf >= first_mf) & (mf <= last_mf)
    return alpha, safe_div(dalpha_dt, dmf_dt), lz, safe_div(dlz, dmf_dt), inside


def st_node_raw(rotation, lnhat):
    """``atan2`` of LNhat in the J frame at a node (``lean.context``)."""
    from ..spintaylor.angles import _rotate, small_atan2

    x, yy, _ = _rotate(lnhat[0], lnhat[1], lnhat[2], rotation[0], rotation[1], rotation[2])
    return small_atan2(yy, x)


def grid_point_mf(grid, wf, k):
    """``Mf`` of point ``k`` (real) of the grid."""
    return grid_frequencies(k, grid, wf.m_tot)


def boole_interval(grid, i, here, after):
    """The increment of ``gamma`` over the interval ``[i, i + 1]`` (``i``
    real) of the grid (``boole`` of :func:`..pnr._lal_interpolants`):
    ``here`` and ``after`` are ``alpha``, ``beta`` and their second
    derivatives at both ends (4 each). Zero beyond the grid."""
    i_start, n, h = grid[0], grid[1], grid[2]
    x0 = i_start * h

    def frequencies(k):
        return (i_start + jnp.minimum(k, n - 1.0)) * h

    t1 = frequencies(i)
    t2 = frequencies(i + 1.0)
    points = jnp.stack([t2, fma(3.0, t1, t2) / 4.0, (t1 + t2) / 2.0, t1, fma(3.0, t2, t1) / 4.0])
    dx = points - (x0 + i * h)
    y0, y1, m0, m1 = here[0], after[0], here[2], after[2]
    yb0, yb1, mb0, mb1 = here[1], after[1], here[3], after[3]
    b = (y1 - y0) / h - h * (m1 + 2.0 * m0) / 6.0
    dalpha = b + dx * (m0 + (m1 - m0) / (2.0 * h) * dx)
    bb = (yb1 - yb0) / h - h * (mb1 + 2.0 * mb0) / 6.0
    beta_x = yb0 + dx * (bb + dx * (mb0 / 2.0 + dx * (mb1 - mb0) / (6.0 * h)))
    f = -dalpha * jnp.cos(beta_x)
    total_f = fma(7.0, f[0], fma(32.0, f[1], fma(12.0, f[2], fma(7.0, f[3], 32.0 * f[4]))))
    used = i + 1.0 < n
    return (jnp.where(used, 1.0 / 90.0 * (t2 - t1) * total_f, 0.0),)


def spline_point(grid, x, k, ends, curvatures):
    """A uniform spline (:func:`..pnr._spline_at`) at ``x`` in the interval
    ``k`` (real, already clipped) with the values ``ends`` and second
    derivatives ``curvatures`` at its ends."""
    i_start, h = grid[0], grid[2]
    x0 = i_start * h
    xk = x0 + k * h
    y0, y1 = ends[0], ends[1]
    m0, m1 = curvatures[0], curvatures[1]
    dx = x - xk
    b = (y1 - y0) / h - h * (m1 + 2.0 * m0) / 6.0
    c = m0 / 2.0
    d = (m1 - m0) / (6.0 * h)
    return y0 + dx * (b + dx * (c + dx * d))


def spline_point6(grid, x, k, y0, y1, m0, m1):
    """The interpolants (:data:`SPLINES`) at ``x`` in the interval
    ``k``, from their values and second derivatives at its ends, and
    whether ``x`` lies on the grid (:func:`..pnr.spline_angles`)."""
    i_start, n, h = grid[0], grid[1], grid[2]
    x0 = i_start * h
    values = jnp.stack([spline_point(grid, x, k, jnp.stack([y0[j], y1[j]]), jnp.stack([m0[j], m1[j]]))
                        for j in range(len(SPLINES))])
    inside = (jax.lax.stop_gradient(x) >= x0) & (jax.lax.stop_gradient(x) <= x0 + (n - 1.0) * h)
    return values, inside


def stage_f2_pieces(prec, wf, extra, grid, f, x, k, map_errors, kappa, st_alpha, st_inside, *pieces, modes):
    """:func:`stage_f2` with the interpolants evaluated at the frequencies
    ``x`` of the modes (Hz, in the intervals ``k``) from ``pieces``: for
    each mode, the values and second derivatives of the interpolants
    at both ends of its interval (``y0``, ``y1``, ``m0``, ``m1``)."""
    values, inside = [], []
    for m in range(len(modes)):
        y0, y1, m0, m1 = pieces[4 * m:4 * m + 4]
        v, i = spline_point6(grid, x[m], k[m], y0, y1, m0, m1)
        values.append(v)
        inside.append(i)
    return stage_f2(prec, wf, extra, f, jnp.stack(values), jnp.stack(inside), map_errors, kappa, st_alpha, st_inside,
                    modes)


# ---------------------------------------------------------------------------
# the stages composed in JAX
# ---------------------------------------------------------------------------


def _is_dynamic(leaf):
    """Whether a leaf of a pytree is a value (an array of any library, or
    a shape) rather than a Python constant of the model's structs."""
    return not isinstance(leaf, (bool, int, float, complex, str, type(None)))


def partition(tree):
    """The array leaves of a pytree and a hashable description of the rest
    (its structure and its Python leaves)."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    dynamic = tuple(_is_dynamic(leaf) for leaf in leaves)
    values = [leaf for leaf, d in zip(leaves, dynamic) if d]
    static = tuple(None if d else leaf for leaf, d in zip(leaves, dynamic))
    return values, (treedef, static, dynamic)


def named_leaves(tree, prefix=""):
    """``(path, leaf)`` of the array leaves of a pytree of structs, lists,
    tuples and dicts, in the order of :func:`partition` (paths like
    ``.alpha_params.a1`` or ``[2]``)."""
    out = []
    if isinstance(tree, Struct):
        for name in sorted(tree.keys()):
            out += named_leaves(getattr(tree, name), f"{prefix}.{name}")
    elif isinstance(tree, dict):
        for name in sorted(tree):
            out += named_leaves(tree[name], f"{prefix}[{name!r}]")
    elif isinstance(tree, (list, tuple)):
        for i, item in enumerate(tree):
            out += named_leaves(item, f"{prefix}[{i}]")
    elif tree is not None and _is_dynamic(tree):
        out.append((prefix, tree))
    return out


def combine(values, meta):
    """The pytree of :func:`partition` from its array leaves."""
    treedef, static, dynamic = meta
    values = iter(values)
    return jax.tree_util.tree_unflatten(treedef, [next(values) if d else s for s, d in zip(static, dynamic)])


class Staged:
    """``function(*pytrees)`` under ``jax.jit`` in a given precision, with
    the Python leaves of its arguments and results kept static (the
    structs of the model carry Python flags). :meth:`jvp` gives its
    forward-mode derivatives along several directions."""

    def __init__(self, function, dtype):
        self.function = function
        self.dtype = np.dtype(dtype)
        self.cache = {}

    def _flat(self, args):
        parts = [partition(a) for a in args]
        metas = tuple(meta for _, meta in parts)
        if metas not in self.cache:
            holder = {}

            def flat(*values):
                it = iter(values)
                trees = [combine([next(it) for _ in vals], meta) for (vals, _), meta in zip(parts, metas)]
                with precision(self.dtype):
                    out, holder["meta"] = partition(self.function(*trees))
                return out

            self.cache[metas] = (jax.jit(flat), holder)
        jitted, holder = self.cache[metas]
        return jitted, holder, [v for vals, _ in parts for v in vals]

    def __call__(self, *args):
        jitted, holder, flat = self._flat(args)
        return combine(jitted(*flat), holder["meta"])

    def jvp(self, args, tangents):
        """``(outputs, output_tangents)`` for the tangents of the array
        leaves of ``args`` (one flat list per argument, ``None`` for zero,
        each with a leading axis of directions); the output tangents are a
        flat list (``None`` for integers and booleans)."""
        jitted, holder, flat = self._flat(args)
        flat_t = [t for ts in tangents for t in ts]
        floating = [jnp.issubdtype(jnp.result_type(v), jnp.inexact) for v in flat]
        n_dir = next(t.shape[0] for t in flat_t if t is not None)
        given = [jnp.zeros((n_dir,) + jnp.shape(v), jnp.result_type(v)) if t is None
                 else jnp.asarray(t, jnp.result_type(v)) for v, t, f in zip(flat, flat_t, floating) if f]

        def one(*dir_tangents):
            it = iter(dir_tangents)
            ts = [next(it) if f else np.zeros(jnp.shape(v), jax.dtypes.float0) for v, f in zip(flat, floating)]
            out, t_out = jax.jvp(jitted, flat, ts)
            return out, [t if jnp.issubdtype(t.dtype, jnp.inexact) else None for t in t_out]

        out, t_out = jax.vmap(one, out_axes=(None, 0))(*given)
        return combine(out, holder["meta"]), t_out


class Dual:
    """A pytree and the flat list of the tangents of its array leaves
    (``None``, or with a leading axis of directions)."""

    def __init__(self, tree, tangents):
        self.tree = tree
        self.tangents = tangents

    def split(self):
        """The elements of a tuple-valued pytree as :class:`Dual`."""
        out, start = [], 0
        for item in self.tree:
            n = len(partition(item)[0])
            out.append(Dual(item, self.tangents[start:start + n]))
            start += n
        return out


def _call(staged, *duals):
    """``staged`` on :class:`Dual` arguments (tangents if any are given)."""
    if all(all(t is None for t in d.tangents) for d in duals):
        out = staged(*[d.tree for d in duals])
        return Dual(out, [None] * len(partition(out)[0]))
    out, t_out = staged.jvp([d.tree for d in duals], [d.tangents for d in duals])
    return Dual(out, t_out)


def _const(tree):
    return Dual(tree, [None] * len(partition(tree)[0]))


@functools.lru_cache(maxsize=None)
def _compiled(dtype_name, modes, amp_scale, reference_mf=None):
    """The stages and the parts by hand as jitted JAX functions (in the
    precision ``dtype_name``)."""
    dtype = np.dtype(dtype_name)

    def under(function):
        return Staged(function, dtype)

    def a(leaves, f):
        return stage_a(leaves, f[0], f[-1], modes, amp_scale, reference_mf)

    def b(ode):
        sol = st_lean.solution(ode.m1, ode.m2, list(ode.chi1), list(ode.chi2), ode.f_ref, ode.f_low, ode.f_cut,
                               ode.active)
        return sol, solution_summary(sol, ode.flow_ok)

    def b2(sol, rotation, st_range, mf):
        mf_scale = st_range[0]
        ctx = st_lean.context(sol, tuple(rotation), mf_scale)
        return st_lean.angles(sol, tuple(rotation), mf, mf_scale, ctx)

    def c2(prec, wf, points, values, st_range):
        return stage_c2(prec, wf, points, values, st_range, modes)

    def grid_mf(k, grid, wf):
        return grid_frequencies(k, grid, wf.m_tot)

    def d(prec, wf, mf, st_values):
        return jax.vmap(grid_point, in_axes=(None, None, 0, 0))(prec, wf, mf, st_values)

    def s(alpha, beta, grid):
        i_start, n, h = grid[0], grid[1], grid[2]
        pad = pnr.PNR_GRID_CAPACITY - alpha.shape[0]
        return pnr._lal_interpolants(jnp.pad(alpha, (0, pad)), jnp.pad(beta, (0, pad)), i_start, n, h, n,
                                     jnp.asarray(False))

    def spline_values(splines, f):
        out = [pnr._spline_at(getattr(splines, name), getattr(splines, "m_" + name), splines.n, splines.x0,
                              splines.h, f)[0] for name in SPLINES]
        inside = (f >= splines.x0) & (f <= splines.x0 + (splines.n - 1.0) * splines.h)
        return jnp.stack(out, axis=-1), inside

    def at_ref(splines, wf):
        return spline_values(splines, wf.f_ref)

    def e(prec, wf, values):
        return stage_e(prec, wf, values[0], values[1], modes)

    def f1(prec, wf, f):
        return jax.vmap(stage_f1, in_axes=(None, None, 0, None))(prec, wf, f, modes)

    def f2(prec, wf, extra, f, values, mapped, kappa_s, st_f):
        values, inside = values
        _, map_errors, _, _ = mapped
        return jax.vmap(stage_f2, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, None))(
            prec, wf, extra, f, values, inside, map_errors, kappa_s, st_f[0], st_f[4], modes)

    def smooth(mapped, f):
        return smooth_amp_ratio(mapped[3], f) if (2, 2) in modes else mapped[3]

    def assemble(h, flags):
        error = functools.reduce(jnp.logical_or, [jnp.any(x) for x in flags])
        hp = jnp.where(error, jnp.nan, h[0][:, 0] + 1j * h[0][:, 1])
        hc = jnp.where(error, jnp.nan, h[0][:, 2] + 1j * h[0][:, 3])
        return hp, hc, error

    return Struct(a=under(a), b=under(b), c1=under(stage_c1), b2=under(b2), c2=under(c2), grid_mf=under(grid_mf),
                  d=under(d), s=under(s), spline_values=under(spline_values), at_ref=under(at_ref), e=under(e),
                  f1=under(f1), f2=under(f2), smooth=under(smooth), assemble=under(assemble))


#: The parameters of a binary along which :func:`reference` differentiates
#: (the order of the flat leaves of :data:`BINARY_FIELDS`, spins expanded).
N_PARAMETERS = 12


def reference(binary, frequencies, mode_array=None, scaled_strain=True, keep=False, tangents=False, reference_mf=None):
    """The polarizations from the stages composed in JAX (eagerly,
    stage by stage, with the sizes of the binary), in the precision of the
    binary: ``(h_+, h_x)``, and with ``keep`` also a dict of the
    intermediate values of every stage (:class:`Dual`, with their
    derivatives along the twelve parameters of the binary if
    ``tangents``; ``reference_mf`` the mass-scaled reference frequency)."""
    from ...._options import STRAIN_SCALE
    from .._math import precision_of

    dtype = precision_of(binary)
    modes = model.coprecessing_modes(mode_array)
    st = _compiled(dtype.name, modes, STRAIN_SCALE if scaled_strain else 1.0, reference_mf)
    leaves = {name: jnp.asarray(value, dtype) for name, value in binary_leaves(binary).items()}
    if tangents:
        eye = np.eye(N_PARAMETERS)
        # (the flat leaves of the dict are in sorted order of the names)
        columns = {"mass_1": eye[0], "mass_2": eye[1], "spin_1": eye[2:5].T, "spin_2": eye[5:8].T,
                   "luminosity_distance": eye[8], "inclination": eye[9], "phi_ref": eye[10], "f_ref": eye[11]}
        binary_t = [jnp.asarray(columns[name], dtype) for name in sorted(leaves)]
        x = Dual(leaves, binary_t)
    else:
        x = _const(leaves)
    f = _const(jnp.asarray(frequencies, dtype))
    out = {}
    base, wf0, ode, error_a = _call(st.a, x, f).split()
    sol, summary = _call(st.b, ode).split()
    prec1, wf1, points, rotation, st_range = _call(st.c1, base, wf0, summary).split()
    bank = _call(st.b2, sol, rotation, st_range, points)
    prec, wf, grid, error_c = _call(st.c2, prec1, wf1, points, bank, st_range).split()
    n = int(grid.tree[1])
    k = _const(jnp.arange(n, dtype=dtype))
    mf_grid = _call(st.grid_mf, k, grid, wf)
    st_grid = _call(st.b2, sol, rotation, st_range, mf_grid)
    alpha, beta, err_grid = _call(st.d, prec, wf, mf_grid, st_grid).split()
    splines = _call(st.s, alpha, beta, grid)
    at_ref = _call(st.at_ref, splines, wf)
    prec_e, wf_e, extra, error_e = _call(st.e, prec, wf, at_ref).split()
    mapped = _call(st.f1, prec_e, wf_e, f)
    values = _call(st.spline_values, splines, Dual(mapped.tree[0], mapped.tangents[:1]))
    kappa_s = _call(st.smooth, mapped, f)
    st_f = _call(st.b2, sol, rotation, st_range, Dual(mapped.tree[2], mapped.tangents[2:3]))
    h = _call(st.f2, prec_e, wf_e, extra, f, values, mapped, kappa_s, st_f)
    flags = _const((error_a.tree, ~ode.tree.active, summary.tree.failed, error_c.tree, summary.tree.incomplete,
                    err_grid.tree, error_e.tree, h.tree[1]))
    hp, hc, error = _call(st.assemble, h, flags).split()
    if keep:
        out.update(base=base, wf0=wf0, ode=ode, error_a=error_a, sol=sol, summary=summary, prec_c1=prec1, wf_c1=wf1,
                   points=points, rotation=rotation, st_range=st_range, bank=bank, prec_c2=prec, wf_c2=wf, grid=grid,
                   error_c=error_c, mf_grid=mf_grid, st_grid=st_grid, alpha=alpha, beta=beta, err_grid=err_grid,
                   splines=splines, at_ref=at_ref, prec_e=prec_e, wf_e=wf_e, extra=extra, error_e=error_e,
                   mapped=mapped, values=values, kappa_s=kappa_s, st_f=st_f, h=h, error=error, hp=hp, hc=hc)
        return (hp.tree, hc.tree), out
    return hp.tree, hc.tree
