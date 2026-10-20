# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The SpinTaylor solution behind the Euler angles of IMRPhenomXPHM/XPNR
(``IMRPhenomX_InspiralAngles_SpinTaylor`` and the splines of
``IMRPhenomX_InterpolateAlphaBeta_SpinTaylor`` in
``LALSimIMRPhenomX_precession.c``; waveformrs's
``imrphenomx/spintaylor_angles.rs``).

LAL evolves the orbit-averaged SpinTaylorT4 equations from ``f_ref``
backwards to ``f_min - 0.5 Hz`` and forwards to the end of the inspiral,
sampled every ``delta_t = 5 / f_cut`` (``f_cut`` the ringdown frequency
plus eight damping widths); it then repeats the end of the inspiral from
ten samples before its end with ten times finer sampling, and keeps the
samples of both runs as the knots of natural cubic splines of ``alpha`` and
``cos(beta)`` against ``Mf`` in the J frame.

Here (see :mod:`.trajectory` and :mod:`.replica`):

* the solution comes from a dense, differentiable DOP853 integration
  (:func:`.trajectory.integrate`), evaluated at LAL's sample times, which
  are a lattice: ``t_k = k delta_t`` from ``f_ref`` (without LAL's sample
  at ``-delta_t``, which ``appendTS`` drops) and ``t_c + j delta_t / 10``
  for the fine run;
* where each run of LAL ends depends on its exact sequence of adaptive
  steps, so the number of samples of the forward run and of the fine run
  come from :func:`.replica.evolve`, which repeats LAL's steps; the number
  of samples of the backward run, where LAL stops at a frequency, follows
  from the solution;
* the splines have as many knots as LAL has samples (millions for light
  binaries), but a natural cubic spline at ``x`` depends on knots further
  than ``k`` intervals away only through factors ``~(2 - sqrt(3))^k``. So
  :func:`spline` evaluates the spline at ``x`` from the
  ``2 WINDOW + 2`` knots around ``x`` (with LAL's natural end conditions
  where the window reaches the ends): the same values to ``~1e-15``, with a
  cost independent of the number of samples.

The knot values differ from LAL's by LAL's own integration error (``~1e-9``,
the tolerances are ``1e-12`` per step); everything else is LAL's
construction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._math import C_SI, G_SI, MSUN_SI, MTSUN_SI, PI, REAL4_EPS, Struct, fma, safe_div
from . import dynamics, replica, trajectory

__all__ = ["pn_solution", "knot_times", "knot_states", "spline", "WINDOW"]

#: Knots on each side of an interval used to evaluate the splines.
WINDOW = 16
#: Precession angle (rad) per step of the DOP853 integration.
STEP_ANGLE = 0.1
#: Capacities (steps) of the backward and forward DOP853 integrations and the
#: limit on LAL's adaptive steps in the replica of a run.
BACKWARD_CAPACITY = 4096
FORWARD_CAPACITY = 16384
FINE_CAPACITY = 4096
MAX_REPLICA_ITERATIONS = 1000000
#: Iterations of LAL's backward run tried to detect that it stops at once.
EARLY_STOP_ITERATIONS = 8


def _initial_state(m1_si, m2_si, chi1, chi2, f_start, msec):
    m_tot = m1_si + m2_si
    norm1 = (m1_si / m_tot) ** 2
    norm2 = (m2_si / m_tot) ** 2
    return jnp.stack([
        jnp.zeros_like(f_start), PI * msec * f_start, jnp.zeros_like(f_start), jnp.zeros_like(f_start),
        jnp.ones_like(f_start), norm1 * chi1[0], norm1 * chi1[1], norm1 * chi1[2], norm2 * chi2[0],
        norm2 * chi2[1], norm2 * chi2[2], jnp.ones_like(f_start), jnp.zeros_like(f_start), jnp.zeros_like(f_start),
    ])


def _du_per_step(m1_si, m2_si):
    """The increment of ``u = 1 / (M omega)`` that covers ``STEP_ANGLE`` of
    the leading-order precession of L, ``dalpha/du = 5 (2 + 3 m2 / 2 m1) /
    96`` (smaller near merger and with spin-spin terms; the margin of
    :data:`STEP_ANGLE` covers both)."""
    rate = 5.0 * (2.0 + 1.5 * m2_si / m1_si) / 96.0
    return STEP_ANGLE / rate


def _time_at_omega(traj, omega, backwards):
    """The time at which ``M omega`` of the solution equals ``omega``
    (arrays), by locating the step and inverting the dense output with
    Newton's method (a fixed number of iterations)."""
    n = jnp.maximum(traj.n, 1)
    cap = traj.t.shape[0]
    node_omega = traj.y[:, 1]
    # node omega decreases backwards in time and increases forwards
    key = -node_omega if backwards else node_omega
    key = jnp.where(jnp.arange(cap) < n, key, jnp.inf)
    query = -omega if backwards else omega
    i = jnp.clip(jnp.searchsorted(key, query, side="right") - 1, 0, n - 1)
    t0 = traj.t[i]
    h = traj.h[i]
    dense = traj.dense[i]
    y0 = traj.y[i][..., 1:5]
    omega0 = traj.y[i][..., 1]
    omega1 = jnp.where(i + 1 < n, traj.y[jnp.minimum(i + 1, cap - 1), 1], traj.y_last[1])
    x = jnp.clip(safe_div(omega - omega0, omega1 - omega0), 0.0, 1.0)
    x = jax.lax.stop_gradient(x)

    def omega_at(x):
        return trajectory._dense_eval(dense, y0, x)[..., 0]

    for _ in range(8):
        value, slope = jax.jvp(omega_at, (x,), (jnp.ones_like(x),))
        x = x - safe_div(value - omega, slope)
    return t0 + x * h


def pn_solution(m1_si, m2_si, chi1, chi2, f_ref, f_min, f_cut):
    """``IMRPhenomX_InspiralAngles_SpinTaylor`` (version 330, black holes):
    the SpinTaylor solution from ``f_min`` (Hz, including LAL's buffer) through
    ``f_ref`` to the end of the inspiral, for masses in kg (heavier first)
    and the spins ``chi1``, ``chi2`` (3-vectors in the L frame at ``f_ref``).
    ``f_cut`` (Hz) is ``(fRING + 8 fDAMP) / M`` of the waveform struct.

    The forward and fine runs are LAL's steps (:func:`.replica.evolve_recorded`),
    so their samples are LAL's to rounding; the backward run, which is long
    and ends at a frequency, is the DOP853 solution.

    Returns a :class:`~.._math.Struct` describing LAL's knots (the lattice,
    see :func:`knot_times`), the solutions, the spins and LNhat at the last
    knot, the frequency of the first knot ``f_min_pn``, ``failed`` (LAL's
    integration fails, where its waveform drivers fall back to the MSA
    angles) and ``incomplete`` (the capacities of this implementation were
    exceeded).
    """
    m_tot = m1_si + m2_si
    msec = m_tot / MSUN_SI * MTSUN_SI
    pi_gm = PI * m_tot * (G_SI / C_SI) / (C_SI * C_SI)
    # coarse and fine sampling (LAL reads PhenomXPSpinTaylorCoarseFactor = 10
    # as a float)
    dt_coarse = 0.5 * 10.0 / f_cut / msec
    dt_fine = 0.5 / f_cut / msec

    no_backward = (f_ref < REAL4_EPS) | (jnp.abs(f_ref - f_min) < REAL4_EPS)
    bad_order = ~no_backward & ~(f_ref - f_min > REAL4_EPS)
    f_start = jnp.where(no_backward, f_min, f_ref)
    y_ref = _initial_state(m1_si, m2_si, chi1, chi2, f_start, msec)
    omega_ref = y_ref[1]

    # LAL's forward coarse run, step for step
    c_fwd = dynamics.coefficients(m1_si, m2_si, f_start, f_cut)
    fwd = replica.evolve_recorded(c_fwd, y_ref, dt_coarse, MAX_REPLICA_ITERATIONS, FORWARD_CAPACITY)
    n_fwd = _trim_forward(fwd, c_fwd, dt_coarse)

    # the backward run: does it stop at once (f_ref beyond the validity of
    # the PN equations)?
    c_back = dynamics.coefficients(m1_si, m2_si, f_ref, f_min - 0.5)
    start = replica.evolve(c_back, y_ref, -dt_coarse, EARLY_STOP_ITERATIONS)
    stopped = ~start.exhausted & (start.code != dynamics.ST_TEST_FREQBOUND)

    # the backward solution down to the sample below f_min - 0.5 Hz
    du_step = _du_per_step(m1_si, m2_si)
    omega_low = PI * msec * (f_min - 0.5)
    omega_low_safe = jnp.where((omega_low > 0) & (omega_low < omega_ref), omega_low, 0.5 * omega_ref)
    k_dot_low = dynamics.derivatives(c_fwd, y_ref.at[1].set(omega_low_safe))[0][1]
    # one coarse interval below f_min - 0.5 Hz, and the shortfall of the step rule
    u_low = 1.0 / omega_low_safe + 1.5 * dt_coarse * k_dot_low / omega_low_safe ** 2
    range_back = jax.lax.stop_gradient(jnp.where(no_backward, 1.0, u_low - 1.0 / omega_ref) * 1.03)
    n_back_steps = jnp.clip(jnp.ceil(range_back / du_step), 1, BACKWARD_CAPACITY).astype(jnp.int64)
    du_back = range_back / n_back_steps
    back = trajectory.integrate(c_back, y_ref, du_back, n_back_steps, BACKWARD_CAPACITY)

    # the samples of the backward run: the first one below omega_low
    t_low = _time_at_omega(back, omega_low_safe, True)
    k_low = jnp.floor(-t_low / dt_coarse) + 1.0
    n_b = jnp.where(no_backward, 0.0, jnp.where(stopped, start.n_samples, k_low + 1.0))
    covered_back = no_backward | stopped | (-back.t_last >= k_low * dt_coarse)

    # the coarse sequence (appendTS drops the last two backward samples)
    split = jnp.where(no_backward, 0.0, n_b - 2.0)
    len_low = jnp.where(no_backward, n_fwd, split + n_fwd + 1.0)
    nbuffer = jnp.minimum(9.0, len_low - 1.0)
    copy_length = len_low - 1.0 - nbuffer
    in_forward = copy_length >= split
    t_c = jnp.where(in_forward, (copy_length - split) * dt_coarse, -(n_b - 1.0 - copy_length) * dt_coarse)
    zero_sample = ~no_backward & (copy_length >= split + n_fwd)

    # the fine run from the sample at t_c, step for step
    state_c = jnp.where(
        t_c >= 0, replica.sample(fwd, jnp.maximum(t_c, 0.0)[None])[0],
        trajectory.evaluate_full(back, jnp.minimum(t_c, 0.0)[None], True)[0],
    )
    omega_c = state_c[1]
    # v^3 / piGM with v = cbrt(M omega): the frequency of the sample
    f_trans = jnp.cbrt(omega_c) ** 3 / pi_gm
    bad_trans = zero_sample | ~(f_trans > 0)
    f_trans_safe = jnp.where(bad_trans, f_start, f_trans)
    y_fine = state_c.at[0].set(0.0).at[1].set(PI * msec * f_trans_safe)
    c_fine = dynamics.coefficients(m1_si, m2_si, f_trans_safe, f_cut)
    fine = replica.evolve_recorded(c_fine, y_fine, dt_fine, MAX_REPLICA_ITERATIONS, FINE_CAPACITY)
    n_fine = _trim_forward(fine, c_fine, dt_fine)

    len_pn = copy_length + n_fine
    last = replica.sample(fine, ((n_fine - 1.0) * dt_fine)[None])[0]
    norm1 = (m1_si / m_tot) ** 2
    norm2 = (m2_si / m_tot) ** 2
    sol = Struct(
        back=back,
        fwd=fwd,
        fine=fine,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
        no_backward=no_backward,
        n_b=n_b,
        split=split,
        n_fwd=n_fwd,
        copy_length=copy_length,
        t_c=t_c,
        n_fine=n_fine,
        len_pn=len_pn,
        omega_ref=omega_ref,
        msec=msec,
        pi_gm=pi_gm,
        s1_end=[last[5] / norm1, last[6] / norm1, last[7] / norm1],
        s2_end=[last[8] / norm2, last[9] / norm2, last[10] / norm2],
        lnhat_end=[last[2], last[3], last[4]],
    )
    # the first knot and the check that the solution starts below f_min
    omega_first = knot_states(sol, jnp.zeros(()))[0]
    f_min_pn = jnp.cbrt(omega_first) ** 3 / pi_gm
    # the failures of IMRPhenomX_InspiralAngles_SpinTaylor (LAL's drivers
    # then fall back to the MSA angles); the return codes of the
    # integrator are not failures
    sol.f_min_pn = f_min_pn
    sol.failed = (
        bad_order
        | bad_trans
        | (~no_backward & (n_b <= 1))
        | (len_pn < 4)
        | (f_min_pn < 0)
        | (f_min_pn > f_min)
        | ~(f_min > 0)
        | (~no_backward & (f_min - 0.5 < 0))
    )
    # limits of this implementation
    sol.incomplete = fwd.exhausted | fine.exhausted | ~covered_back
    return sol


def _trim_forward(rec, c, deltat):
    """The number of samples of a forward run after LAL's trimming at
    ``f_end`` (``XLALSimInspiralSpinTaylorPNEvolveOrbit``): samples beyond
    ``omega_end`` are dropped except the first. Only samples of the last
    step can be beyond (the stopping test fires at the end of the first step
    that crosses ``omega_end``)."""
    n = rec.n_samples
    w_end = c.omega_end
    last = jnp.maximum(rec.accepted - 1, 0)
    t0_last = rec.t0[last]

    def omega_at(index):
        return replica.sample(rec, (index * deltat)[None])[0, 1]

    # bisection for the first sample of the last step with omega > w_end
    lo = jnp.clip(jnp.floor(t0_last / deltat) + 1.0, 0.0, n)
    hi = n
    for _ in range(64):
        mid = jnp.floor((lo + hi) / 2.0)
        above = omega_at(jnp.minimum(mid, n - 1.0)) > w_end
        hi = jnp.where(above & (mid < hi), mid, hi)
        lo = jnp.where(above, lo, jnp.minimum(mid + 1.0, hi))
    first_above = hi
    trimmed = jnp.where(first_above < n, first_above + 1.0, n)
    applies = (c.f_end > c.f_start) & (c.f_end != 0.0)
    return jax.lax.stop_gradient(jnp.where(applies, trimmed, n))


def knot_times(sol, index):
    """The times (units of the total mass, ``t = 0`` at ``f_ref``) of LAL's
    knots ``index`` (float arrays of integers ``0 .. len_pn - 1``)."""
    coarse = index < sol.copy_length
    in_back = index < sol.split
    t_back = -(sol.n_b - 1.0 - index) * sol.dt_coarse
    t_fwd = (index - sol.split) * sol.dt_coarse
    t_fine = sol.t_c + (index - sol.copy_length) * sol.dt_fine
    return jnp.where(coarse, jnp.where(in_back, t_back, t_fwd), t_fine)


def knot_states(sol, index):
    """``M omega`` and LNhat at the knots ``index``: LAL's samples of the
    forward and fine runs, the DOP853 solution for the backward run."""
    t = knot_times(sol, index)
    coarse = index < sol.copy_length
    in_back = coarse & (index < sol.split)
    yb = trajectory.evaluate(sol.back, jnp.minimum(t, 0.0), True)
    yf = replica.sample(sol.fwd, jnp.maximum(t, 0.0))[..., 1:5]
    yfine = replica.sample(sol.fine, jnp.maximum(t - sol.t_c, 0.0))[..., 1:5]
    return jnp.where(in_back[..., None], yb, jnp.where(coarse[..., None], yf, yfine))


def index_at_time(sol, t):
    """The knot interval containing the time ``t``: ``i`` with ``t_i <= t <
    t_{i+1}`` (approximately: the windows of :func:`spline` absorb an
    off-by-one)."""
    split, copy_length = sol.split, sol.copy_length
    i_back = sol.n_b - 1.0 - jnp.ceil(-t / sol.dt_coarse)
    i_fwd = split + jnp.floor(t / sol.dt_coarse)
    i_coarse = jnp.where(t < -1.5 * sol.dt_coarse, i_back, jnp.where(t < 0.0, split - 1.0, i_fwd))
    i_fine = copy_length + jnp.floor((t - sol.t_c) / sol.dt_fine)
    i = jnp.where(t >= sol.t_c, i_fine, i_coarse)
    return jnp.clip(i, 0.0, sol.len_pn - 2.0)


def _thomas(lower, diag, upper, rhs):
    """Solve tridiagonal systems along the last axis (lists of arrays)."""
    n = len(diag)
    c = [None] * n
    d = [None] * n
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / m
    x = [None] * n
    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def spline(sol, rotation, mf, mf_scale):
    """The natural cubic splines of LAL through its knots of ``alpha``
    (unwrapped) and ``cos(beta)`` against ``Mf`` in the J frame, and their
    derivatives, at ``mf`` (an array).

    ``rotation`` is ``(phiJ_Sf, thetaJ_Sf, kappa)``: LNhat is rotated by
    ``-phiJ_Sf`` about ``z``, ``-thetaJ_Sf`` about ``y`` and ``-kappa``
    about ``z``. ``mf_scale`` converts ``M omega`` to the knots' ``Mf``
    (``Mf = v^3 / piGM * M``). Returns ``(alpha, dalpha, cosbeta,
    dcosbeta, inside)`` where ``inside`` is whether ``mf`` lies within the
    knots (GSL fails outside).
    """
    phi_j, theta_j, kappa = rotation
    omega = mf / mf_scale
    backwards = omega < sol.omega_ref
    omega = jax.lax.stop_gradient(omega)
    tb = _time_at_omega(sol.back, jnp.where(backwards, omega, sol.omega_ref * 0.999), True)
    tf = _forward_time_at_omega(sol.fwd, omega, 0.0)
    omega_c = sol.fine.y[0, 1]
    t_fine = _forward_time_at_omega(sol.fine, omega, sol.t_c)
    t = jnp.where(backwards, tb, jnp.where(omega >= omega_c, t_fine, tf))
    # the backward solution is only used before t_c
    t = jnp.where(backwards & (omega >= omega_c), t_fine, t)
    t = jax.lax.stop_gradient(t)
    i0 = index_at_time(sol, t)
    offsets = jnp.arange(-WINDOW, WINDOW + 2, dtype=jnp.float64)
    # the window, shifted to lie inside the knots
    start = jnp.clip(i0 - WINDOW, 0.0, jnp.maximum(sol.len_pn - (2 * WINDOW + 2), 0.0))
    idx = start[..., None] + offsets + WINDOW
    valid = idx <= sol.len_pn - 1.0
    idx = jnp.minimum(idx, sol.len_pn - 1.0)
    state = knot_states(sol, idx)
    om = state[..., 0]
    lx, ly, lz = _rotate(state[..., 1], state[..., 2], state[..., 3], phi_j, theta_j, kappa)
    knot_mf = jnp.cbrt(om) ** 3 * mf_scale
    alpha_raw = jnp.arctan2(ly, lx)
    cosb = lz
    # unwrap within the window, then the branch of the window's first knot
    dalpha = jnp.diff(alpha_raw, axis=-1)
    dalpha = jnp.where(dalpha > PI, dalpha - 2 * PI, jnp.where(dalpha < -PI, dalpha + 2 * PI, dalpha))
    alpha = jnp.concatenate([alpha_raw[..., :1], alpha_raw[..., :1] + jnp.cumsum(dalpha, axis=-1)], axis=-1)
    alpha = alpha + 2 * PI * _branch(sol, rotation, idx[..., 0], alpha_raw[..., 0])[..., None]
    # pad beyond the last knot: repeat it with increasing abscissae (decoupled)
    width = jnp.max(knot_mf, axis=-1, keepdims=True) - jnp.min(knot_mf, axis=-1, keepdims=True)
    pad = jnp.cumsum(jnp.where(valid, 0.0, 1.0), axis=-1)
    knot_mf = jnp.where(valid, knot_mf, jnp.max(jnp.where(valid, knot_mf, -jnp.inf), axis=-1, keepdims=True) + pad * width)
    # LAL's natural end conditions where the window reaches the ends of the
    # knots, else the second derivative estimated from the window's end
    true_start = start == 0.0
    true_end = (start + 2 * WINDOW + 1) >= sol.len_pn - 1.0
    a_val, a_der = _natural_spline_eval(knot_mf, alpha, valid, mf, true_start, true_end)
    b_val, b_der = _natural_spline_eval(knot_mf, cosb, valid, mf, true_start, true_end)
    inside = (mf >= _first_mf(sol, mf_scale)) & (mf <= _last_mf(sol, mf_scale))
    return a_val, a_der, b_val, b_der, inside


def _rotate(x, y, z, phi_j, theta_j, kappa):
    """LNhat in the J frame: ``R_z(-kappa) R_y(-thetaJ) R_z(-phiJ)``."""
    c1, s1 = jnp.cos(-phi_j), jnp.sin(-phi_j)
    x1 = x * c1 - y * s1
    y1 = x * s1 + y * c1
    c2, s2 = jnp.cos(-theta_j), jnp.sin(-theta_j)
    x2 = x1 * c2 + z * s2
    z2 = -x1 * s2 + z * c2
    c3, s3 = jnp.cos(-kappa), jnp.sin(-kappa)
    x3 = x2 * c3 - y1 * s3
    y3 = x2 * s3 + y1 * c3
    return x3, y3, z2


def _first_mf(sol, mf_scale):
    om = knot_states(sol, jnp.zeros(()))[..., 0]
    return jnp.cbrt(om) ** 3 * mf_scale


def _last_mf(sol, mf_scale):
    om = knot_states(sol, sol.len_pn - 1.0)[..., 0]
    return jnp.cbrt(om) ** 3 * mf_scale


def _branch(sol, rotation, index, alpha_raw):
    """The number of turns ``n`` such that ``alpha_raw + 2 pi n`` is LAL's
    unwrapped ``alpha`` at knot ``index``: the continuous ``alpha`` of the
    solution, followed along the steps of the integrations (each covers a
    small fraction of a precession cycle), relative to the first knot,
    where LAL starts unwrapping."""
    phi_j, theta_j, kappa = rotation
    tables = _alpha_tables(sol, rotation)
    t = knot_times(sol, index)
    alpha_cont = _continuous_alpha(sol, tables, t, alpha_raw)
    state0 = knot_states(sol, jnp.zeros(()))
    lx0, ly0, _ = _rotate(state0[1], state0[2], state0[3], phi_j, theta_j, kappa)
    raw0 = jnp.arctan2(ly0, lx0)
    alpha_first = _continuous_alpha(sol, tables, knot_times(sol, jnp.zeros(())), raw0)
    shift = jnp.round((raw0 - alpha_first) / (2 * PI)) * 2 * PI
    return jnp.round((alpha_cont + shift - alpha_raw) / (2 * PI))


def _unwrap_nodes(lx, ly, lz, rotation):
    x, y, _ = _rotate(lx, ly, lz, *rotation)
    raw = jnp.arctan2(y, x)
    d = jnp.diff(raw)
    d = jnp.where(d > PI, d - 2 * PI, jnp.where(d < -PI, d + 2 * PI, d))
    return jnp.concatenate([raw[:1], raw[:1] + jnp.cumsum(d)])


def _alpha_tables(sol, rotation):
    """The continuous ``alpha`` at the nodes of the backward solution and
    of the forward run (both unwrapped from ``alpha`` at ``t = 0``)."""
    back = sol.back
    fwd = sol.fwd
    return (
        _unwrap_nodes(back.y[:, 2], back.y[:, 3], back.y[:, 4], rotation),
        _unwrap_nodes(fwd.y[:, 2], fwd.y[:, 3], fwd.y[:, 4], rotation),
    )


def _continuous_alpha(sol, tables, t, alpha_raw):
    """The continuous ``alpha`` at time ``t`` where its value modulo ``2
    pi`` is ``alpha_raw``: the value at the last node before ``t`` plus the
    wrapped difference."""
    back, fwd = sol.back, sol.fwd
    # backward nodes: times decreasing from 0
    nb = jnp.maximum(back.n, 1)
    key_b = jnp.where(jnp.arange(back.t.shape[0]) < nb, -back.t, jnp.inf)
    ib = jnp.clip(jnp.searchsorted(key_b, -t, side="right") - 1, 0, nb - 1)
    nf = jnp.maximum(fwd.accepted, 1)
    key_f = jnp.where(jnp.arange(fwd.t0.shape[0]) < nf, fwd.t0, jnp.inf)
    i_f = jnp.clip(jnp.searchsorted(key_f, t, side="right") - 1, 0, nf - 1)
    node = jnp.where(t < 0, tables[0][ib], tables[1][i_f])
    d = alpha_raw - node
    return node + d - 2 * PI * jnp.round(d / (2 * PI))


def _forward_time_at_omega(rec, omega, offset):
    """An estimate (within a step) of the time at which ``M omega`` of a
    forward run reaches ``omega``."""
    n = jnp.maximum(rec.accepted, 1)
    cap = rec.t0.shape[0]
    key = jnp.where(jnp.arange(cap) < n, rec.y[:, 1], jnp.inf)
    i = jnp.clip(jnp.searchsorted(key, omega, side="right") - 1, 0, n - 1)
    w0 = rec.y[i, 1]
    w1 = jnp.where(i + 1 < n, rec.y[jnp.minimum(i + 1, cap - 1), 1], rec.y_end[1])
    x = jnp.clip(safe_div(omega - w0, w1 - w0), 0.0, 1.0)
    return offset + rec.t0[i] + x * rec.h[i]


def _second_difference(x0, x1, x2, y0, y1, y2):
    """Twice the second divided difference: the second derivative of the
    parabola through three points."""
    return 2.0 * ((y2 - y1) / (x2 - x1) - (y1 - y0) / (x1 - x0)) / (x2 - x0)


def _natural_spline_eval(x, y, valid, xq, natural_start=True, natural_end=True):
    """A cubic spline through ``(x, y)`` (the last axis; knots beyond the
    valid ones are decoupled and ignored) evaluated with its derivative at
    ``xq`` (the leading axes). Its second derivative vanishes at the ends
    (GSL's natural spline) where ``natural_start``/``natural_end``, and is
    estimated from the three end knots otherwise (the ends of a window
    inside LAL's knots)."""
    n = x.shape[-1]
    h = jnp.diff(x, axis=-1)
    hs = [h[..., i] for i in range(n - 1)]
    ys = [y[..., i] for i in range(n)]
    vs = [valid[..., i] for i in range(n)]
    lower = [jnp.zeros_like(ys[0])] * n
    diag = [jnp.ones_like(ys[0])] * n
    upper = [jnp.zeros_like(ys[0])] * n
    rhs = [jnp.zeros_like(ys[0])] * n
    for i in range(1, n - 1):
        interior = vs[i + 1]
        lower[i] = jnp.where(interior, hs[i - 1], 0.0)
        diag[i] = jnp.where(interior, 2.0 * (hs[i - 1] + hs[i]), 1.0)
        upper[i] = jnp.where(interior, hs[i], 0.0)
        rhs[i] = jnp.where(interior, 6.0 * ((ys[i + 1] - ys[i]) / hs[i] - (ys[i] - ys[i - 1]) / hs[i - 1]), 0.0)
    rhs[0] = jnp.where(natural_start, 0.0, _second_difference(x[..., 0], x[..., 1], x[..., 2], ys[0], ys[1], ys[2]))
    rhs[n - 1] = jnp.where(
        natural_end, 0.0, _second_difference(x[..., n - 3], x[..., n - 2], x[..., n - 1], ys[n - 3], ys[n - 2], ys[n - 1])
    )
    m = _thomas(lower, diag, upper, rhs)
    m = jnp.stack(m, axis=-1)
    # the interval of xq
    n_valid = jnp.sum(valid, axis=-1)
    k = jnp.clip(jnp.sum((x[..., 1:] <= xq[..., None]) & valid[..., 1:], axis=-1), 0, jnp.maximum(n_valid - 2, 0))
    take = lambda a, k: jnp.take_along_axis(a, k[..., None], axis=-1)[..., 0]  # noqa: E731
    x0, x1 = take(x, k), take(x, k + 1)
    y0, y1 = take(y, k), take(y, k + 1)
    m0, m1 = take(m, k), take(m, k + 1)
    hk = x1 - x0
    dx = xq - x0
    # GSL's cspline: y0 + dx (b + dx (c + dx d)) with c = m0 / 2
    b = (y1 - y0) / hk - hk * (m1 + 2.0 * m0) / 6.0
    c = m0 / 2.0
    d = (m1 - m0) / (6.0 * hk)
    value = y0 + dx * (b + dx * (c + dx * d))
    derivative = b + dx * (2.0 * c + 3.0 * dx * d)
    return value, derivative
