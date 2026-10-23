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

* LAL's samples are a lattice in time: ``t_k = k delta_t`` from ``f_ref``
  (without LAL's sample at ``-delta_t``, which ``appendTS`` drops) and
  ``t_c + j delta_t / 10`` for the fine run;
* the forward and fine runs are LAL's adaptive steps, repeated by
  :func:`.replica.evolve_recorded` (where each run ends depends on the
  exact sequence of steps), with LAL's interpolant between them;
* the backward run, which is long and ends at a frequency, is a dense,
  differentiable DOP853 integration (:func:`.trajectory.integrate`)
  evaluated at the lattice;
* the splines have as many knots as LAL has samples (millions for light
  binaries). Up to :data:`GLOBAL_KNOTS` knots, :func:`spline_table` builds
  them as GSL does, from all knots at once. Beyond, a natural cubic spline
  at ``x`` depends on knots further than ``k`` intervals away only through
  factors ``~(2 - sqrt(3))^k``, so :func:`windowed_spline` evaluates it at
  ``x`` from the ``2 WINDOW + 2`` knots around ``x`` (with LAL's natural end
  conditions where the window reaches the ends): the same values to
  ``~1e-15``, with a cost per point independent of the number of samples.

The knots of the backward run differ from LAL's by LAL's own integration
error (``~1e-9``, the tolerances are ``1e-12`` per step); everything else is
LAL's construction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._math import C_SI, G_SI, MSUN_SI, MTSUN_SI, PI, REAL4_EPS, Struct, blocked_scan, chunked_map, fenced, fma, safe_div
from .._transforms import cond
from . import dynamics, replica, trajectory

__all__ = ["pn_solution", "knot_times", "knot_states", "spline", "spline_context", "spline_many", "spline_table",
           "windowed_spline", "WINDOW",
           "GLOBAL_KNOTS"]

#: Knots on each side of an interval used to evaluate the splines in windows.
WINDOW = 16
#: The number of knots up to which the splines are built from all knots.
GLOBAL_KNOTS = 8192
#: The knots computed together when building the splines from all knots.
KNOT_CHUNK = 2048
#: Precession angle (rad) per step of the DOP853 integration.
STEP_ANGLE = 0.1
#: Capacities (steps) of the backward and forward DOP853 integrations and the
#: limit on LAL's adaptive steps in the replica of a run.
BACKWARD_CAPACITY = 4096
FORWARD_CAPACITY = 65536
FINE_CAPACITY = 4096
#: The variables of the state kept for every step of the forward run (the
#: phase, M omega and LNhat: those of the knots), and the steps of its end
#: kept whole (from which the fine run starts: ~250 steps in practice).
KNOT_VARIABLES = 5
FORWARD_RING = 1024
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
    Newton's method (a fixed number of iterations). Not differentiated: it
    places LAL's samples."""
    traj, omega = jax.lax.stop_gradient((traj, omega))
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

    def newton(_, x):
        value, slope = jax.jvp(omega_at, (x,), (jnp.ones_like(x),))
        return x - safe_div(value - omega, slope)

    x = jax.lax.fori_loop(0, 8, newton, x)
    return t0 + x * h


@fenced
def pn_solution(m1_si, m2_si, chi1, chi2, f_ref, f_min, f_cut, active=True, extended=False, smooth=False):
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
    exceeded). Where ``active`` (traced) is false, the integrations take no
    steps (under ``jax.vmap``, a branch of :func:`.._transforms.cond` runs for
    the whole batch if any binary takes it).

    With ``extended`` (static), a reference frequency beyond the PN energy
    test (where LAL's backward run stops at once and its drivers leave the
    SpinTaylor angles) is not a failure: the backward run goes on from
    ``f_ref`` (its equations stay regular there), and the knots end at the
    last sample of the backward run that passes the energy test, from which
    the fine run goes on to the end of the inspiral, as LAL's does for a
    reference frequency just below the test (:func:`_last_valid_sample`).
    Elsewhere the solution is LAL's.

    With ``smooth`` (static), the inspiral ends half a step of the fine
    lattice before the condition that stops the fine run is met on its
    interpolant (:func:`_stop_time`; LAL's last sample lies up to a step
    before it, depending on where the lattice falls), rather than at its
    last sample, and the knots of the fine run are counted back from there
    (``t_stop - j delta_t``) rather than from its start: the spins and
    LNhat at the end, the last knot and the knots near it (where the angles
    of the merger and ringdown are fitted) are continuous in the
    parameters. A knot still appears or leaves where the
    fine run meets the coarse knots, which changes the splines by their
    interpolation error.
    """
    max_iterations = jnp.where(active, MAX_REPLICA_ITERATIONS, 0)
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
    fwd = replica.evolve_recorded(c_fwd, y_ref, dt_coarse, max_iterations, FORWARD_CAPACITY, components=KNOT_VARIABLES,
                                  ring=FORWARD_RING)
    n_fwd = _trim_forward(fwd, c_fwd, dt_coarse)

    # the backward run: does it stop at once (f_ref beyond the validity of
    # the PN equations)?
    c_back = dynamics.coefficients(m1_si, m2_si, f_ref, f_min - 0.5)
    start = replica.evolve(c_back, y_ref, -dt_coarse, jnp.where(active, EARLY_STOP_ITERATIONS, 0))
    stopped = ~start.exhausted & (start.code != dynamics.ST_TEST_FREQBOUND)
    # (extended: f_ref beyond the energy test only stops LAL's backward run)
    beyond = stopped & (start.code == dynamics.ST_TEST_ENERGY) if extended else jnp.asarray(False)
    stopped = stopped & ~beyond

    # the backward solution down to the sample below f_min - 0.5 Hz
    du_step = _du_per_step(m1_si, m2_si)
    omega_low = PI * msec * (f_min - 0.5)
    omega_low_safe = jnp.where((omega_low > 0) & (omega_low < omega_ref), omega_low, 0.5 * omega_ref)
    k_dot_low = dynamics.derivatives(c_fwd, y_ref.at[1].set(omega_low_safe))[0][1]
    # one coarse interval below f_min - 0.5 Hz, and the shortfall of the step rule
    u_low = 1.0 / omega_low_safe + 1.5 * dt_coarse * k_dot_low / omega_low_safe ** 2
    # (from beyond the energy test, the steps' leading-order estimate of the
    # time per step in u is poor in the first steps: a larger margin)
    margin = jnp.where(beyond, 1.25, 1.03)
    range_back = jax.lax.stop_gradient(jnp.where(no_backward, 1.0, u_low - 1.0 / omega_ref) * margin)
    n_back_steps = jnp.clip(jnp.ceil(range_back / du_step), 1, BACKWARD_CAPACITY).astype(jnp.int64)
    n_back_steps = jnp.where(active, n_back_steps, 0)
    du_back = range_back / jnp.maximum(n_back_steps, 1)
    back = trajectory.integrate(c_back, y_ref, du_back, n_back_steps, BACKWARD_CAPACITY)

    # the samples of the backward run: the first one below omega_low
    t_low = _time_at_omega(back, omega_low_safe, True)
    k_low = jnp.floor(-t_low / dt_coarse) + 1.0
    n_b = jnp.where(no_backward, 0.0, jnp.where(stopped, start.n_samples, k_low + 1.0))
    covered_back = no_backward | stopped | (-back.t_last >= k_low * dt_coarse)

    # the coarse sequence (appendTS drops the last two backward samples)
    split = jnp.where(no_backward, 0.0, n_b - 2.0)
    len_low = jnp.where(no_backward, n_fwd, split + n_fwd + 1.0)
    if extended:
        # beyond the energy test: the backward samples up to the last valid
        # one (t = -k_e delta_t), without the forward run
        k_e = _last_valid_sample(back, c_back, dt_coarse, n_b)
        split = jnp.where(beyond, n_b - k_e, split)
        len_low = jnp.where(beyond, split, len_low)
    nbuffer = jnp.minimum(9.0, len_low - 1.0)
    copy_length = len_low - 1.0 - nbuffer
    in_forward = copy_length >= split
    t_c = jnp.where(in_forward, (copy_length - split) * dt_coarse, -(n_b - 1.0 - copy_length) * dt_coarse)
    zero_sample = ~no_backward & (copy_length >= split + n_fwd) & ~beyond

    # the fine run from the sample at t_c, step for step
    forward_c, recorded_c = replica.sample_full(fwd, jnp.maximum(t_c, 0.0)[None])
    state_c = jnp.where(t_c >= 0, forward_c[0], trajectory.evaluate_full(back, jnp.minimum(t_c, 0.0)[None], True)[0])
    omega_c = state_c[1]
    # v^3 / piGM with v = cbrt(M omega): the frequency of the sample
    f_trans = jnp.cbrt(omega_c) ** 3 / pi_gm
    bad_trans = zero_sample | ~(f_trans > 0)
    f_trans_safe = jnp.where(bad_trans, f_start, f_trans)
    y_fine = state_c.at[0].set(0.0).at[1].set(PI * msec * f_trans_safe)
    c_fine = dynamics.coefficients(m1_si, m2_si, f_trans_safe, f_cut)
    fine = replica.evolve_recorded(c_fine, y_fine, dt_fine, max_iterations, FINE_CAPACITY)
    n_fine = _trim_forward(fine, c_fine, dt_fine)

    if smooth:
        # LAL's last sample lies up to a step of the lattice before the end:
        # half a step before it on average
        t_stop = jnp.maximum(_stop_time(fine, c_fine) - 0.5 * dt_fine, 0.0)
        n_fine = jax.lax.stop_gradient(jnp.floor(t_stop / dt_fine) + 1.0)
    else:
        t_stop = (n_fine - 1.0) * dt_fine
    len_pn = copy_length + n_fine
    last = replica.sample(fine, t_stop[None])[0]
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
        # the time of the last knot from the start of the fine run, from
        # which the fine knots are counted back
        t_stop=t_stop,
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
    sol.incomplete = fwd.exhausted | fine.exhausted | ~covered_back | ((t_c >= 0) & ~recorded_c[0])
    return sol


def _stop_time(rec, c):
    """The time (from the start of the forward run ``rec``) at which the
    condition of ``XLALSimInspiralSpinTaylorStoppingTest`` that ended the
    run is met, on LAL's interpolant of its last step (LAL's run ends at the
    end of that step, after which the test first fails): ``M omega`` reaches
    ``omega_end``, the PN energy test or ``v = 1`` crosses zero, or
    ``domega/dt`` peaks. A bisection, then a Newton step for the
    derivatives (those of the root). Continuous in the parameters, unlike
    the end of the run; the end of the run for other codes."""
    last = jnp.maximum(rec.accepted - 1, 0)
    t0, h, y0, k1, k6 = replica.steps(rec, last)
    y1 = rec.y_end
    code = rec.code

    def condition(t):
        # positive while the run goes on
        y = replica.hermite(t0, h, y0, y1, k1, k6, t)
        test, v = dynamics.energy_test(c, y)
        dydt, _ = dynamics.derivatives(c, y)
        _, ddomega = jax.jvp(lambda x: dynamics.derivatives(c, x)[0][1], (y,), (dydt,))
        return jnp.select(
            [code == dynamics.ST_TEST_FREQBOUND, code == dynamics.ST_TEST_ENERGY, code == dynamics.ST_TEST_LARGEV,
             code == dynamics.ST_TEST_OMEGADOUBLEDOT],
            [c.omega_end - y[1], test, 1.0 - v, ddomega],
            jnp.ones_like(t),
        )

    def bisect(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        going = condition(mid) > 0.0
        return jnp.where(going, mid, lo), jnp.where(going, hi, mid)

    end = t0 + h
    fires = (condition(jax.lax.stop_gradient(end)) <= 0.0) & (condition(jax.lax.stop_gradient(t0)) > 0.0)
    lo, hi = jax.lax.fori_loop(0, 60, bisect, jax.lax.stop_gradient((t0, end)))
    root = jax.lax.stop_gradient(0.5 * (lo + hi))
    value, slope = jax.jvp(condition, (root,), (jnp.ones_like(root),))
    root = root - value / jnp.where(slope != 0.0, slope, 1.0)
    return jnp.where(fires, root, end)


def _last_valid_sample(back, c, deltat, n_b):
    """The smallest ``k`` for which the sample ``t = -k deltat`` of the
    backward solution passes the PN energy test of
    ``XLALSimInspiralSpinTaylorStoppingTest`` (which only fails at high
    frequencies, so that it holds for every sample below): a bisection over
    the samples ``0 .. n_b - 1``. Not differentiated: it chooses the knots."""
    back = jax.lax.stop_gradient(back)

    def valid(k):
        y = trajectory.evaluate_full(back, -k * deltat, True)
        return dynamics.energy_test(c, y)[0] >= 0.0

    def bisect(_, bounds):
        lo, hi = bounds  # sample lo fails, sample hi passes
        mid = jnp.floor((lo + hi) / 2.0)
        ok = valid(mid)
        return jnp.where(ok, lo, mid), jnp.where(ok, mid, hi)

    _, k = jax.lax.fori_loop(0, 64, bisect, (jnp.zeros(()), jnp.maximum(n_b - 1.0, 1.0)))
    return k


def _trim_forward(rec, c, deltat):
    """The number of samples of a forward run after LAL's trimming at
    ``f_end`` (``XLALSimInspiralSpinTaylorPNEvolveOrbit``): samples beyond
    ``omega_end`` are dropped except the first. Only samples of the last
    step can be beyond (the stopping test fires at the end of the first step
    that crosses ``omega_end``)."""
    n = rec.n_samples
    w_end = c.omega_end
    last = jnp.maximum(rec.accepted - 1, 0)
    # the samples searched lie in the last step: its interpolant
    # (replica.sample) of M omega
    t0_last, h_last, y0, k1, k6 = replica.steps(rec, last)
    y0, k1, k6, y1 = y0[1], k1[1], k6[1], rec.y_end[1]

    def omega_at(index):
        return replica.hermite(t0_last, h_last, y0, y1, k1, k6, index * deltat)

    # bisection for the first sample of the last step with omega > w_end
    lo = jnp.clip(jnp.floor(t0_last / deltat) + 1.0, 0.0, n)
    hi = n

    def bisect(_, bounds):
        lo, hi = bounds
        mid = jnp.floor((lo + hi) / 2.0)
        above = omega_at(jnp.minimum(mid, n - 1.0)) > w_end
        return jnp.where(above, lo, jnp.minimum(mid + 1.0, hi)), jnp.where(above & (mid < hi), mid, hi)

    _, first_above = jax.lax.fori_loop(0, 64, bisect, (lo, hi))
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
    t_fine = sol.t_c + sol.t_stop - (sol.len_pn - 1.0 - index) * sol.dt_fine
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
    first_fine = sol.t_stop - (sol.len_pn - 1.0 - copy_length) * sol.dt_fine
    i_fine = copy_length + jnp.floor((t - sol.t_c - first_fine) / sol.dt_fine)
    i = jnp.where(t >= sol.t_c, i_fine, i_coarse)
    return jnp.clip(i, 0.0, sol.len_pn - 2.0)


def spline_context(sol, rotation, mf_scale, geometric=False):
    """What the splines of a solution need besides the evaluation points:
    the splines through all knots (:func:`spline_table`), the continuous
    ``alpha`` at the integration nodes and the branch of the first knot
    (for :func:`windowed_spline`) and the range of the knots.

    With ``geometric`` (static), ``alpha`` is taken from splines of the
    components of LNhat in the J frame (:func:`_geometric_alpha`) rather
    than from LAL's spline of ``alpha``."""
    phi_j, theta_j, kappa = rotation
    tables = _alpha_tables(sol, rotation)
    state0 = knot_states(sol, jnp.zeros(()))
    lx0, ly0, _ = _rotate(state0[1], state0[2], state0[3], phi_j, theta_j, kappa)
    raw0 = _atan2(ly0, lx0)
    alpha_first = _continuous_alpha(sol, tables, knot_times(sol, jnp.zeros(())), raw0)
    table = spline_table(sol, rotation, mf_scale, geometric)
    context = Struct(table=table, tables=tables, shift=jnp.round((raw0 - alpha_first) / (2 * PI)) * 2 * PI,
                     first_mf=table.first_mf, last_mf=table.last_mf, small=table.small)
    if geometric:
        # (a marker: the presence of the field is the static flag)
        context.geometric = jnp.zeros(())
    return context


def spline(sol, rotation, mf, mf_scale, context=None):
    """The natural cubic splines of LAL through its knots of ``alpha``
    (unwrapped) and ``cos(beta)`` against ``Mf`` in the J frame, and their
    derivatives, at ``mf`` (an array of a few points): from
    :func:`spline_table` up to :data:`GLOBAL_KNOTS` knots, else from
    :func:`windowed_spline` (under ``jax.vmap``, each where some binary of
    the batch needs it).
    Returns ``(alpha, dalpha, cosbeta, dcosbeta, inside)``."""
    if context is None:
        context = spline_context(sol, rotation, mf_scale)
    return _spline_either(mf, (sol, rotation, mf_scale, context))


def spline_many(sol, rotation, mf, n, chunk, mf_scale, context=None):
    """:func:`spline` at many points ``mf`` (an array of static length, the
    first ``n`` needed): the windows (in chunks of ``chunk``, see
    :func:`~.._math.chunked_map`) only where the knots are too many for
    :func:`spline_table`."""
    if context is None:
        context = spline_context(sol, rotation, mf_scale)

    return chunked_map(_spline_either, mf, n, chunk, (sol, rotation, mf_scale, context))


def _spline_either(mf, args):
    """:func:`table_eval` or :func:`windowed_spline` (under ``jax.vmap``,
    each only if some binary of the batch needs it)."""

    def from_table(operands):
        mf, (_, _, _, context) = operands
        return table_eval(context.table, mf)

    def from_windows(operands):
        mf, (sol, rotation, mf_scale, context) = operands
        return windowed_spline(sol, rotation, mf, mf_scale, context)

    return cond(args[3].small, from_table, from_windows, (mf, args))


def spline_table(sol, rotation, mf_scale, geometric=False):
    """GSL's natural cubic splines (``gsl_interp_cspline``) through all of
    LAL's knots of ``alpha`` (unwrapped by ``XLALSimIMRPhenomXUnwrapArray``)
    and ``cos(beta)`` against ``Mf``, when there are at most
    :data:`GLOBAL_KNOTS` knots (``small``; otherwise a placeholder). Returns a
    :class:`~.._math.Struct` of the knots ``x``, the values and the second
    derivatives; with ``geometric`` (static), also those of the J-frame
    components ``lx``, ``ly`` of LNhat."""
    small = sol.len_pn <= GLOBAL_KNOTS
    n = jnp.where(small, sol.len_pn, 0.0)
    k = jnp.arange(GLOBAL_KNOTS, dtype=jnp.float64)
    valid = k < n
    index = jnp.minimum(k, jnp.maximum(n - 1.0, 0.0))
    state = chunked_map(lambda i, sol: knot_states(sol, i), index, n, KNOT_CHUNK, sol)
    om = jnp.where(valid, state[:, 0], 1.0)
    lx, ly, lz = _rotate(state[:, 1], state[:, 2], state[:, 3], *rotation)
    lx = jnp.where(valid, lx, 1.0)
    ly = jnp.where(valid, ly, 0.0)
    cosb = jnp.where(valid, lz, 0.0)
    knot_mf = jnp.cbrt(om) ** 3 * mf_scale
    raw = _atan2(ly, lx)
    d = jnp.diff(raw)
    d = jnp.where(d > PI, d - 2 * PI, jnp.where(d < -PI, d + 2 * PI, d))
    d = jnp.where(valid[1:], d, 0.0)
    alpha = raw[0] + jnp.concatenate([jnp.zeros(1), jnp.cumsum(d)])
    # beyond the knots: increasing abscissae and decoupled rows
    last = jnp.maximum(n - 1.0, 0.0).astype(jnp.int64)
    x = jnp.where(valid, knot_mf, jnp.where(small, knot_mf[last], 0.0) + (k - n + 1.0))
    alpha = jnp.where(valid, alpha, 0.0)
    table = Struct(x=x, alpha=alpha, cosb=cosb, m_alpha=_natural_second_derivatives(x, alpha, n),
                   m_cosb=_natural_second_derivatives(x, cosb, n), n=jnp.where(small, n, float(GLOBAL_KNOTS)),
                   small=small, first_mf=_first_mf(sol, mf_scale), last_mf=_last_mf(sol, mf_scale))
    if geometric:
        table.lx, table.ly = lx, ly
        table.m_lx = _natural_second_derivatives(x, lx, n)
        table.m_ly = _natural_second_derivatives(x, ly, n)
    return table


def _natural_second_derivatives(x, y, n):
    """The second derivatives of the natural cubic spline through the first
    ``n`` points ``(x, y)`` (zero beyond)."""
    k = jnp.arange(x.shape[0])
    interior = (k >= 1) & (k < n - 1)
    h = jnp.diff(x)
    h_prev = jnp.concatenate([h[:1], h])
    h_next = jnp.concatenate([h, h[-1:]])
    dy = jnp.diff(y)
    s_prev = jnp.concatenate([dy[:1], dy]) / h_prev
    s_next = jnp.concatenate([dy, dy[-1:]]) / h_next
    lower = jnp.where(interior, h_prev, 0.0)
    upper = jnp.where(interior, h_next, 0.0)
    diag = jnp.where(interior, 2.0 * (h_prev + h_next), 1.0)
    rhs = jnp.where(interior, 6.0 * (s_next - s_prev), 0.0)
    return jax.lax.linalg.tridiagonal_solve(lower, diag, upper, rhs[:, None])[:, 0]


def table_eval(table, mf):
    """The splines of :func:`spline_table` and their derivatives at ``mf``
    (GSL's ``cspline`` evaluation), and whether ``mf`` lies within the
    knots."""
    i = jnp.clip(jnp.searchsorted(table.x, mf, side="right") - 1, 0, (table.n - 2.0).astype(jnp.int64))
    x0, x1 = table.x[i], table.x[i + 1]
    hk = x1 - x0
    dx = mf - x0
    out = []
    pairs = [(table.alpha, table.m_alpha), (table.cosb, table.m_cosb)]
    if "lx" in table:
        pairs += [(table.lx, table.m_lx), (table.ly, table.m_ly)]
    for y, m in pairs:
        y0, y1 = y[i], y[i + 1]
        m0, m1 = m[i], m[i + 1]
        b = (y1 - y0) / hk - hk * (m1 + 2.0 * m0) / 6.0
        c = m0 / 2.0
        d = (m1 - m0) / (6.0 * hk)
        out += [y0 + dx * (b + dx * (c + dx * d)), b + dx * (2.0 * c + 3.0 * dx * d)]
    inside = (mf >= table.first_mf) & (mf <= table.last_mf)
    if "lx" in table:
        out[0], out[1] = _geometric_alpha(out[0], *out[4:])
    return out[0], out[1], out[2], out[3], inside


def windowed_spline(sol, rotation, mf, mf_scale, context):
    """The natural cubic splines of LAL through its knots of ``alpha``
    (unwrapped) and ``cos(beta)`` against ``Mf`` in the J frame, and their
    derivatives, at ``mf`` (an array), each point from the ``2 WINDOW + 2``
    knots around it.

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
    omega_c = replica.omega(sol.fine, 0)
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
    alpha_raw = _atan2(ly, lx)
    cosb = lz
    # unwrap within the window, then the branch of the window's first knot
    dalpha = jnp.diff(alpha_raw, axis=-1)
    dalpha = jnp.where(dalpha > PI, dalpha - 2 * PI, jnp.where(dalpha < -PI, dalpha + 2 * PI, dalpha))
    alpha = jnp.concatenate([alpha_raw[..., :1], alpha_raw[..., :1] + jnp.cumsum(dalpha, axis=-1)], axis=-1)
    alpha = alpha + 2 * PI * _branch(sol, context, idx[..., 0], alpha_raw[..., 0])[..., None]
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
    if "geometric" in context:
        x_val, x_der = _natural_spline_eval(knot_mf, lx, valid, mf, true_start, true_end)
        y_val, y_der = _natural_spline_eval(knot_mf, ly, valid, mf, true_start, true_end)
        a_val, a_der = _geometric_alpha(a_val, x_val, x_der, y_val, y_der)
    inside = (mf >= context.first_mf) & (mf <= context.last_mf)
    return a_val, a_der, b_val, b_der, inside


def _geometric_alpha(alpha, lx, dlx, ly, dly):
    """``alpha`` and its derivative from the splines ``lx``, ``ly`` of the
    J-frame components of LNhat (and their derivatives), on the branch of
    LAL's spline ``alpha``. Where LNhat passes close to J, ``alpha`` swings
    by nearly ``pi`` within a few knots: LAL's spline of the unwrapped
    ``alpha`` cannot follow the swing, and the branch of the swing that the
    unwrapping picks flips with the parameters. The components are smooth
    there, and ``atan2`` of their splines follows the swing."""
    raw = small_atan2(ly, lx)
    d = raw - alpha
    value = alpha + d - 2 * PI * jnp.round(d / (2 * PI))
    r2 = lx * lx + ly * ly
    return value, jnp.where(r2 > SMALL_SQUARE, lx * dly - ly * dlx, 0.0) / jnp.where(r2 > SMALL_SQUARE, r2, 1.0)


def _atan2(y, x):
    """``atan2(y, x)``, zero with a zero derivative where both vanish (LNhat
    along J: aligned spins), where ``atan2`` has none."""
    both = (x == 0.0) & (y == 0.0)
    return jnp.arctan2(jnp.where(both, 0.0, y), jnp.where(both, 1.0, x))


#: Below this ``x^2 + y^2``, the derivative of ``atan2(y, x)`` would divide
#: by an underflowing square.
SMALL_SQUARE = 1e-280


def small_atan2(y, x):
    """:func:`_atan2`, also zero (with a zero derivative) where ``x^2 +
    y^2 < SMALL_SQUARE``."""
    small = x * x + y * y < SMALL_SQUARE
    return jnp.arctan2(jnp.where(small, 0.0, y), jnp.where(small, 1.0, x))


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


def _branch(sol, context, index, alpha_raw):
    """The number of turns ``n`` such that ``alpha_raw + 2 pi n`` is LAL's
    unwrapped ``alpha`` at knot ``index``: the continuous ``alpha`` of the
    solution, followed along the steps of the integrations (each covers a
    small fraction of a precession cycle), relative to the first knot,
    where LAL starts unwrapping (``context`` from :func:`spline_context`)."""
    alpha_cont = _continuous_alpha(sol, context.tables, knot_times(sol, index), alpha_raw)
    return jnp.round((alpha_cont + context.shift - alpha_raw) / (2 * PI))


def _unwrap_nodes(lx, ly, lz, rotation, n):
    x, y, _ = _rotate(lx, ly, lz, *rotation)
    # the nodes beyond the n integrated ones are zeros: keep atan2 (and its
    # derivative) finite there
    valid = jnp.arange(x.shape[0]) < n
    raw = _atan2(jnp.where(valid, y, 0.0), jnp.where(valid, x, 1.0))
    d = jnp.diff(raw)
    d = jnp.where(d > PI, d - 2 * PI, jnp.where(d < -PI, d + 2 * PI, d))
    return jnp.concatenate([raw[:1], raw[:1] + jnp.cumsum(d)])


def _alpha_tables(sol, rotation):
    """The continuous ``alpha`` at the nodes of the backward solution and
    of the forward run (both unwrapped from ``alpha`` at ``t = 0``)."""
    back = sol.back
    return (
        _unwrap_nodes(back.y[:, 2], back.y[:, 3], back.y[:, 4], rotation, jnp.maximum(back.n, 1)),
        _unwrap_steps(sol.fwd, rotation),
    )


#: The steps of a forward run unwrapped together.
UNWRAP_CHUNK = 1024


def _unwrap_steps(rec, rotation):
    """:func:`_unwrap_nodes` for the starts of the steps of a forward run
    (:func:`.replica.evolve_recorded`), chunk by chunk over the steps taken
    only (the record has room for many more)."""
    cap = rec.rows.shape[0]
    chunk = min(UNWRAP_CHUNK, cap)
    n_chunks = -(-cap // chunk)

    def unwrap(index, carry, args):
        rows, rotation = args
        last_raw, last_alpha, first = carry
        start = jnp.minimum(index * chunk, cap - chunk)
        lnh = jax.lax.dynamic_slice(rows, (start, 4), (chunk, 3))
        x, y, _ = _rotate(lnh[:, 0], lnh[:, 1], lnh[:, 2], *rotation)
        raw = _atan2(y, x)
        previous = jnp.concatenate([jnp.where(first, raw[:1], last_raw[None]), raw[:-1]])
        d = raw - previous
        d = jnp.where(d > PI, d - 2 * PI, jnp.where(d < -PI, d + 2 * PI, d))
        alpha = jnp.where(first, raw[0], last_alpha) + jnp.cumsum(d)
        return alpha, (raw[-1], alpha[-1], jnp.zeros((), bool))

    zero = jnp.zeros(())
    alpha, _ = blocked_scan(unwrap, jnp.arange(n_chunks), jnp.ceil(jnp.maximum(rec.accepted, 1) / chunk),
                            (zero, zero, jnp.ones((), bool)), (rec.rows, rotation))
    return alpha.reshape(-1)[:cap]


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
    i_f = jnp.clip(jnp.searchsorted(fwd.key_t, t, side="right") - 1, 0, nf - 1)
    node = jnp.where(t < 0, tables[0][ib], tables[1][i_f])
    d = alpha_raw - node
    return node + d - 2 * PI * jnp.round(d / (2 * PI))


def _forward_time_at_omega(rec, omega, offset):
    """An estimate (within a step) of the time at which ``M omega`` of a
    forward run reaches ``omega``."""
    n = jnp.maximum(rec.accepted, 1)
    cap = rec.rows.shape[0]
    i = jnp.clip(jnp.searchsorted(rec.key_omega, omega, side="right") - 1, 0, n - 1)
    t0, h = rec.rows[i, 0], rec.rows[i, 1]
    w0 = replica.omega(rec, i)
    w1 = jnp.where(i + 1 < n, replica.omega(rec, jnp.minimum(i + 1, cap - 1)), rec.y_end[1])
    x = jnp.clip(safe_div(omega - w0, w1 - w0), 0.0, 1.0)
    return offset + t0 + x * h


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
    slope = jnp.diff(y, axis=-1) / h
    interior = valid[..., 2:]
    zero = jnp.zeros_like(x[..., :1])
    one = jnp.ones_like(x[..., :1])
    lower = jnp.concatenate([zero, jnp.where(interior, h[..., :-1], 0.0), zero], axis=-1)
    diag = jnp.concatenate([one, jnp.where(interior, 2.0 * (h[..., :-1] + h[..., 1:]), 1.0), one], axis=-1)
    upper = jnp.concatenate([zero, jnp.where(interior, h[..., 1:], 0.0), zero], axis=-1)
    start = jnp.where(natural_start, 0.0, _second_difference(x[..., 0], x[..., 1], x[..., 2], y[..., 0], y[..., 1],
                                                              y[..., 2]))
    end = jnp.where(natural_end, 0.0, _second_difference(x[..., n - 3], x[..., n - 2], x[..., n - 1], y[..., n - 3],
                                                          y[..., n - 2], y[..., n - 1]))
    rhs = jnp.concatenate([
        jnp.broadcast_to(start, x.shape[:-1])[..., None],
        jnp.where(interior, 6.0 * (slope[..., 1:] - slope[..., :-1]), 0.0),
        jnp.broadcast_to(end, x.shape[:-1])[..., None],
    ], axis=-1)
    m = jax.lax.linalg.tridiagonal_solve(lower, diag, upper, rhs[..., None])[..., 0]
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
