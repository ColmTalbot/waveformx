# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The SpinTaylor solution of IMRPhenomXPNR (:mod:`..model`).

The physics is that of LAL's ``IMRPhenomX_InspiralAngles_SpinTaylor``: the
orbit-averaged SpinTaylorT4 equations from the reference frequency
backwards to ``f_min - 0.5 Hz`` and forwards to the end of the inspiral. What LAL's numerics add to it is left
out:

* both directions are one DOP853 integration with dense output
  (:mod:`.trajectory`), in equal steps of ``u = 1 / (M omega)``: one loop,
  the backward steps first, then the forward steps from the reference
  state, until the stopping conditions of
  ``XLALSimInspiralSpinTaylorStoppingTest`` are met (rather than LAL's
  adaptive Runge-Kutta-Fehlberg steps, its sampling and its finer re-run
  of the end of the inspiral);
* the end of the inspiral is where the first stopping condition is met
  on the dense output, less half a step of LAL's fine sampling (where
  LAL's last sample lies on average), continuous in the parameters. When
  the conditions already fail at the reference frequency (beyond the
  validity of the PN equations, where LAL gives up the SpinTaylor angles),
  the end is where they are first met on the backward solution;
* the angles are evaluated from the dense output directly, rather than
  from LAL's splines through its samples, so there are no knots: the
  solution starts where the backward run ends.

The steps are rows of one record, the backward ones first; the functions
below read them in the order of time (``order``). A row holds what the
angles read (the state at the start of the step, the step in time, the
dense output and ``M omega`` at the end), not the derivative at the start
of the step, which :func:`_end` recomputes for the few steps it takes again:
the record, with a copy for every tangent of a forward-mode derivative, is
among the largest arrays of the model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._generated import dop853
from .._math import C_SI, MTSUN_SI, PI, Struct, g_si, integer, isolated_step, mass_unit, real, safe_div
from .._transforms import while_loop
from . import dynamics, trajectory
from .angles import _du_per_step, _refined, small_atan2, small_square, _rotate

__all__ = ["solution", "context", "angles", "CAPACITY"]

#: The steps of the backward and forward integrations together.
CAPACITY = 8192
#: Bisection iterations locating the end of the inspiral within a step.
BISECTIONS = 48
#: The largest step forwards, as a fraction of ``u = 1 / (M omega)``.
STEP_FRACTION = 0.1
#: The bisection iteration whose bracket gives the slope of the Newton step.
SECANT = 28
#: Newton iterations locating the time at which ``M omega`` is reached.
NEWTON = 6
#: The columns of a row of the record: the state (14), the step in time,
#: the dense output (``N_DENSE`` x 4) and ``M omega`` at the end.
N_STATE = 14
H_COLUMN = N_STATE
DENSE_COLUMNS = slice(N_STATE + 1, N_STATE + 1 + trajectory.N_DENSE * len(trajectory.DENSE_COMPONENTS))
OMEGA_COLUMN = DENSE_COLUMNS.stop
ROW = OMEGA_COLUMN + 1


def _rhs(c, y):
    return dynamics.derivatives(c, y)[0]


def margin(c, y):
    """Positive while the inspiral goes on by the stopping conditions of
    ``XLALSimInspiralSpinTaylorStoppingTest`` that need no derivative: the
    smallest of the margins of ``M omega`` below its bound, of the PN energy
    test and of ``v < 1``."""
    test, v = dynamics.energy_test(c, y)
    return jnp.minimum(jnp.minimum(c.omega_end - y[1], test), 1.0 - v)


def going(c, y):
    """:func:`margin` and ``domega/dt`` still increasing (the last
    condition): positive while the inspiral goes on, continuous in the
    state."""
    dydt = _rhs(c, y)
    _, ddomega = jax.jvp(lambda x: _rhs(c, x)[1], (y,), (dydt,))
    return jnp.minimum(margin(c, y), ddomega)


def _states(sol, r):
    """The states at the start of the steps (rows) ``r``."""
    return sol.rows[r, :N_STATE]


def _dense(sol, r):
    """The dense output of the steps (rows) ``r``."""
    return sol.rows[r, DENSE_COLUMNS].reshape(jnp.shape(r) + (trajectory.N_DENSE, len(trajectory.DENSE_COMPONENTS)))


def _integrate(c, y_ref, k_ref, du_back, n_back, du_fwd, beyond, active):
    """The record of ``n_back`` backward steps of ``du_back`` in ``u`` from
    the reference state, then of forward steps of ``du_fwd`` from it until
    the stopping conditions are met at the end of a step (none if
    ``beyond``). Rows: the state at the start of the step, the step in
    time, the dense output and ``M omega`` at the end."""
    rows = jnp.zeros((CAPACITY, ROW), real())
    limit = jnp.where(active, CAPACITY, 0)

    def test(consts, small):
        n_back, limit = consts[3], consts[6]
        i, _, _, stopped = small
        return (i < limit) & ((i < n_back) | ~stopped)

    def advance(state, consts):
        y, k0, i, _ = state
        c, y_ref, k_ref, n_back, du_back, du_fwd = consts
        restart = i == n_back
        y = jnp.where(restart, y_ref, y)
        k0 = jnp.where(restart, k_ref, k0)
        # (forwards, at most a fraction STEP_FRACTION of u: equal steps in
        # u are too long for the dense output where u becomes small, near
        # v = 1 for large aligned spins)
        du = jnp.where(i < n_back, du_back, jnp.maximum(du_fwd, -STEP_FRACTION / y[1]))
        h = trajectory._step_in_time(du, y[1], k0[1])
        k, y_new = trajectory._stages_looped(c, y, h, k0)
        dense = trajectory._dense_looped(y, h, k, y_new, trajectory.DENSE_COMPONENTS)
        k_new = k[dop853.N_STAGES]
        # (the stopping conditions at the end of the step; domega/dt
        # decreasing over the step, as LAL tests it, which happens at most a
        # step after its maximum: see _end)
        stop = (i >= n_back) & (~(margin(c, y_new) > 0.0) | ~(k_new[1] > k0[1]))
        row = jnp.concatenate([y, h[None], dense.reshape(-1), y_new[1:2]])
        return (y_new, k_new, i, stop), row.astype(real())

    def body(consts, small, rows, active):
        i = small[0]
        # (the step in a loop of its own: see _math.isolated_step)
        (y, k0, _, stop), row = isolated_step(advance, (small[1], small[2], i, small[3]), (consts[:6],), rows[i])
        # (inactive under jax.vmap: the row written back unchanged)
        row = jnp.where(active, row, rows[i])
        rows = jax.lax.dynamic_update_slice(rows, row[None], (i, jnp.zeros_like(i)))
        return (i + 1, y, k0, stop), rows

    consts = (c, y_ref, k_ref, n_back, du_back, du_fwd, limit)
    small = (jnp.zeros((), integer()), y_ref, k_ref, beyond)
    (n, y_last, _, stopped), rows = while_loop(test, body, consts, small, rows)
    return rows, n, stopped


def solution(m1, m2, chi1, chi2, f_ref, f_low, f_cut, active=True):
    """The SpinTaylor solution from ``f_low - 0.5 Hz`` through ``f_ref`` to
    the end of the inspiral, for masses in :func:`.._math.mass_unit`
    (heavier first) and the spins ``chi1``, ``chi2`` (3-vectors in the L
    frame at ``f_ref``); ``f_cut`` (Hz) bounds the frequency. Returns a
    :class:`~.._math.Struct` with what :func:`context` and :func:`angles`
    need, the spins and LNhat at the end of the inspiral, the frequency
    ``f_min_pn`` where the solution starts, ``failed`` and ``incomplete``
    (the capacity was exceeded). Where ``active`` (traced) is false, no
    steps are taken."""
    m_tot = m1 + m2
    msec = m_tot / mass_unit() * MTSUN_SI
    pi_gm = PI * m_tot * (g_si() / C_SI) / (C_SI * C_SI)
    # LAL's fine sampling of the end of the inspiral
    dt_fine = 0.5 / f_cut / msec
    y_ref = _initial_state(m1, m2, chi1, chi2, f_ref, msec)
    c = dynamics.coefficients(m1, m2, f_ref, f_cut)
    k_ref = _rhs(c, y_ref)
    beyond = ~(margin(c, y_ref) > 0.0)
    omega_ref = y_ref[1]
    omega_low = PI * msec * (f_low - 0.5)
    failed = ~(omega_low > 0.0) | ~(omega_low < omega_ref)
    omega_low = jnp.where(failed, 0.5 * omega_ref, omega_low)
    # equal steps in u covering a little more than the range (the steps
    # cover nearly du each; from beyond the energy test, less well)
    du_step = _du_per_step(m1, m2)
    cover = jnp.where(beyond, 1.25, 1.03)
    range_back = jax.lax.stop_gradient((1.0 / omega_low - 1.0 / omega_ref) * cover)
    n_back = jnp.clip(jnp.ceil(range_back / du_step), 1, CAPACITY // 2).astype(integer())
    du_back = range_back / n_back.astype(real())
    rows, n, stopped = _integrate(c, y_ref, k_ref, du_back, n_back, -du_step, beyond, active)
    h, omega_end, omega_start = rows[:, H_COLUMN], rows[:, OMEGA_COLUMN], rows[:, 1]

    # the steps in the order of time: row(p) = n_back - 1 - p for p < n_back
    p = jnp.arange(CAPACITY)
    valid = p < n
    order = jnp.where(p < n_back, n_back - 1 - p, p)
    order = jnp.clip(order, 0, CAPACITY - 1)
    is_back = jnp.arange(CAPACITY) < n_back
    taken = jnp.arange(CAPACITY) < n
    h = jnp.where(taken, h, 0.0)
    t_back = jnp.cumsum(jnp.where(is_back, h, 0.0))
    t_fwd = jnp.cumsum(jnp.where(~is_back & taken, h, 0.0))
    t0 = jnp.where(is_back, t_back, t_fwd) - h
    # time and M omega at the start (low end) of each interval, in order
    big = jnp.inf
    t_low = jnp.where(valid, jnp.where(p < n_back, (t0 + h)[order], t0[order]), big)
    omega_low_end = jnp.where(valid, jnp.where(p < n_back, omega_end[order], omega_start[order]), big)
    sol = Struct(
        lean=jnp.zeros((), real()), rows=rows, h=h, omega_end=omega_end, t0=t0, n=n, n_back=n_back,
        order=order, t_low=t_low, omega_low=omega_low_end, c=c, omega_ref=omega_ref, msec=msec, pi_gm=pi_gm,
    )
    t_end, last = _end(sol, beyond, 0.5 * dt_fine)
    norm1 = (m1 / m_tot) ** 2
    norm2 = (m2 / m_tot) ** 2
    sol.t_end = t_end
    sol.s1_end = [last[5] / norm1, last[6] / norm1, last[7] / norm1]
    sol.s2_end = [last[8] / norm2, last[9] / norm2, last[10] / norm2]
    sol.lnhat_end = [last[2], last[3], last[4]]
    sol.omega_first = omega_low_end[0]
    sol.omega_last = last[1]
    sol.f_min_pn = jnp.cbrt(sol.omega_first) ** 3 / pi_gm
    sol.failed = failed | ~(f_ref > 0.0)
    sol.incomplete = (n >= CAPACITY) & ~stopped | (sol.omega_first > PI * msec * (f_low - 0.5))
    return sol


def _initial_state(m1, m2, chi1, chi2, f_start, msec):
    m_tot = m1 + m2
    norm1 = (m1 / m_tot) ** 2
    norm2 = (m2 / m_tot) ** 2
    zero = jnp.zeros_like(f_start)
    one = jnp.ones_like(f_start)
    return jnp.stack([
        zero, PI * msec * f_start, zero, zero, one, norm1 * chi1[0], norm1 * chi1[1], norm1 * chi1[2],
        norm2 * chi2[0], norm2 * chi2[1], norm2 * chi2[2], one, zero, zero,
    ])


def _full_dense(sol, rows):
    """The states at the start of the steps ``rows``, the steps and the
    dense output of the whole state (the derivative at the start and the
    stages recomputed)."""
    y, h, t0 = _states(sol, rows), sol.h[rows], sol.t0[rows]

    def one(y, h):
        k, y_new = trajectory._stages_looped(sol.c, y, h, _rhs(sol.c, y))
        return trajectory._dense_looped(y, h, k, y_new)

    return y, jax.vmap(one)(y, h), t0, h


def _end(sol, beyond, back_off):
    """The end of the inspiral, ``back_off`` before the time at which the
    stopping conditions are first met (:func:`going`), and the whole state
    there. The conditions are met within the last forward step or the one
    before (where ``domega/dt`` peaks), or (``beyond``) within the
    backward step where the backward solution first passes them: the three
    steps up to it (in time) are searched by bisection on their dense
    output, and the root refined by a Newton step for the derivatives
    (those of the root), in one loop (one copy of the conditions)."""
    c = sol.c
    n_back, n = sol.n_back, sol.n

    # beyond: the first backward step (from the reference) whose end passes
    def search(_, bounds):
        lo, hi = bounds  # step lo's end fails, step hi's end passes
        mid = (lo + hi) // 2
        ok = margin(c, _states(sol, jnp.clip(mid + 1, 0, jnp.maximum(n_back - 1, 0)))) > 0.0
        return jnp.where(ok, lo, mid), jnp.where(ok, mid, hi)

    zero = jnp.zeros((), integer())
    _, j_back = jax.lax.fori_loop(0, 14, search, (zero - 1, jnp.maximum(n_back - 2, 0)))
    offsets = jnp.arange(2, -1, -1, dtype=integer())
    rows = jnp.where(beyond, jnp.clip(jnp.maximum(j_back, 0) + offsets, 0, jnp.maximum(n_back - 1, 0)),
                     jnp.clip(n - 1 - offsets, n_back, jnp.maximum(n - 1, n_back)))
    rows = jax.lax.stop_gradient(rows)
    y0, dense, t0, h = _full_dense(sol, rows)
    lo_t = jnp.minimum(t0, t0 + h)
    hi_t = jnp.maximum(t0, t0 + h)
    h_safe = jnp.where(h != 0.0, h, 1.0)

    def state(t):
        k = jnp.clip(jnp.sum(lo_t <= t) - 1, 0, 2)
        return trajectory._dense_eval(dense[k], y0[k], (t - t0[k]) / h_safe[k])

    t_a, t_b = jax.lax.stop_gradient((lo_t[0], hi_t[2]))
    last = BISECTIONS + 2

    def body(i, carry):
        # iterations 0 and 1: the ends; then bisection (positive before the
        # root); the secant slope of a narrow bracket is kept for the last
        # iteration, a Newton step with derivatives
        lo, hi, g_lo, g_hi, slope, root = carry
        x = jnp.where(i == 0, t_b, jnp.where(i == 1, t_a, jnp.where(i == last, root, 0.5 * (lo + hi))))
        x = jnp.where(i == last, x, jax.lax.stop_gradient(x))
        g = going(c, state(x))
        # (beyond the validity of the equations, the state can overflow)
        g = jnp.where(jnp.isnan(g), -1.0, g)
        bisecting = (i >= 2) & (i < last)
        ok = g > 0.0
        g_hi = jnp.where(i == 0, g, jnp.where(bisecting & ~ok, g, g_hi))
        g_lo = jnp.where(i == 1, g, jnp.where(bisecting & ok, g, g_lo))
        lo = jnp.where(bisecting & ok, x, lo)
        hi = jnp.where(bisecting & ~ok, x, hi)
        slope = jnp.where(i == SECANT, jax.lax.stop_gradient((g_hi - g_lo) / (hi - lo)), slope)
        newton = x - g / jnp.where(slope != 0.0, slope, 1.0)
        root = jnp.where(i == last, newton, jax.lax.stop_gradient(0.5 * (lo + hi)))
        return lo, hi, g_lo, g_hi, slope, root

    zero_r = jnp.zeros((), real())
    init = (t_a, t_b, zero_r, zero_r, zero_r, 0.5 * (t_a + t_b))
    lo, hi, g_lo, g_hi, slope, root = jax.lax.fori_loop(0, last + 1, body, init)
    # no root within the steps (the conditions fail throughout): their start
    root = jnp.where(g_lo > 0.0, root, t_a)
    # the end, in the step holding it (the steps near the end can be
    # shorter than the back-off)
    t_end = jnp.maximum(root - back_off, sol.t_low[0])
    p = jnp.clip(jnp.searchsorted(sol.t_low, jax.lax.stop_gradient(t_end), side="right") - 1, 0,
                 jnp.maximum(n - 1, 0))
    y0, dense, t0, h = _full_dense(sol, sol.order[p][None])
    h_safe = jnp.where(h[0] != 0.0, h[0], 1.0)
    return t_end, trajectory._dense_eval(dense[0], y0[0], (t_end - t0[0]) / h_safe)


def context(sol, rotation, mf_scale):
    """What :func:`angles` needs besides the points: the continuous
    ``alpha`` at the starts of the steps (in the order of time) and the
    range of the solution."""
    lnhat = sol.rows[sol.order, 2:5]
    x, yy, _ = _rotate(lnhat[:, 0], lnhat[:, 1], lnhat[:, 2], *rotation)
    valid = jnp.arange(CAPACITY) < sol.n
    raw = small_atan2(jnp.where(valid, yy, 0.0), jnp.where(valid, x, 1.0))
    d = jnp.diff(raw)
    d = jnp.where(d > PI, d - 2 * PI, jnp.where(d < -PI, d + 2 * PI, d))
    alpha = jnp.concatenate([raw[:1], raw[:1] + jnp.cumsum(d)])
    node_t = jnp.where(valid, sol.t0[sol.order], jnp.inf)
    first_mf = jnp.cbrt(sol.omega_first) ** 3 * mf_scale
    last_mf = jnp.cbrt(sol.omega_last) ** 3 * mf_scale
    return Struct(lean=jnp.zeros((), real()), alpha=alpha, node_t=node_t, first_mf=first_mf, last_mf=last_mf)


def _time_at_omega(sol, omega):
    """The step (in the order of time) holding ``M omega = omega`` and the
    time there: Newton's method on the dense output (not differentiated)."""
    sol, omega = jax.lax.stop_gradient((sol, omega))
    p = jnp.clip(jnp.searchsorted(sol.omega_low, omega, side="right") - 1, 0, jnp.maximum(sol.n - 1, 0))
    r = sol.order[p]
    y0 = sol.rows[r, 1:5]
    om0 = y0[..., 0]
    om1 = sol.omega_end[r]
    dense = _dense(sol, r)
    x = jnp.clip(safe_div(omega - om0, om1 - om0), 0.0, 1.0)

    def omega_at(x):
        return trajectory._dense_eval(dense, y0, x)[..., 0]

    def newton(_, x):
        value, slope = jax.jvp(omega_at, (x,), (jnp.ones_like(x),))
        return jnp.clip(x - safe_div(value - omega, slope), 0.0, 1.0)

    x = jax.lax.fori_loop(0, NEWTON, newton, x)
    return r, sol.t0[r] + x * sol.h[r]


def angles(sol, rotation, mf, mf_scale, ctx):
    """The SpinTaylor ``alpha`` (continuous, on the branch of the first
    step) and ``cos(beta)`` and their derivatives at ``mf``: the time at
    which ``M omega = mf / mf_scale`` on the dense output, one Newton step
    with derivatives (those of the root), and LNhat in the J frame there.
    Returns ``(alpha, dalpha, cosbeta, dcosbeta, inside)`` as
    :func:`.angles.spline`."""
    omega = mf / mf_scale
    r, t = _time_at_omega(sol, omega)
    y0 = sol.rows[r, 1:5]
    dense = _dense(sol, r)
    t0, h = sol.t0[r], sol.h[r]
    h_safe = jnp.where(h != 0.0, h, 1.0)

    def state(t):
        return trajectory._dense_eval(dense, y0, (t - t0) / h_safe)

    y, dydt, t = _refined(state, t, omega)
    lx, ly, lz = _rotate(y[..., 1], y[..., 2], y[..., 3], *rotation)
    dlx, dly, dlz = _rotate(dydt[..., 1], dydt[..., 2], dydt[..., 3], *rotation)
    raw = small_atan2(ly, lx)
    # the continuous alpha: the node before t plus the wrapped difference
    k = jnp.clip(jnp.searchsorted(ctx.node_t, jax.lax.stop_gradient(t), side="right") - 1, 0,
                 jnp.maximum(sol.n - 1, 0))
    node = ctx.alpha[k]
    d = raw - node
    alpha = node + d - 2 * PI * jnp.round(jax.lax.stop_gradient(d) / (2 * PI))
    dmf_dt = mf_scale * dydt[..., 0]
    r2 = lx * lx + ly * ly
    big = r2 > small_square()
    dalpha_dt = jnp.where(big, lx * dly - ly * dlx, 0.0) / jnp.where(big, r2, 1.0)
    inside = (mf >= ctx.first_mf) & (mf <= ctx.last_mf)
    return alpha, safe_div(dalpha_dt, dmf_dt), lz, safe_div(dlz, dmf_dt), inside
