# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""LAL's adaptive integration of the SpinTaylorT4 equations, step for step.

``XLALSimInspiralSpinTaylorPNEvolveOrbit`` integrates with
``XLALAdaptiveRungeKutta4Hermite`` (``LALAdaptiveRungeKuttaIntegrator.c``):
GSL's ``rkf45`` stepper with the standard step control (``y`` only,
tolerances ``1e-12``), ``gsl_odeiv_evolve_apply``, up to six retries with a
ten times smaller step when the derivative fails, samples every
``delta_t`` from the step's Hermite-like interpolant, and the SpinTaylor
stopping test after every step. The output ends with the samples of the step
at which the stopping test fires, so where it ends depends on the exact
sequence of steps. This module repeats that sequence (after waveformrs's
``spintaylor/integrator.rs``) in a ``jax.lax.while_loop``:

* :func:`evolve` returns the number of samples and where the integration
  stopped (not differentiated: its inputs pass ``jax.lax.stop_gradient``);
* :func:`evolve_recorded` also records every accepted step, from which
  :func:`sample` interpolates LAL's samples. It is differentiable (in
  forward mode) with respect to the coefficients and the initial state, the
  accepted steps being held fixed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .._math import DBL_MIN, Struct, isolated
from .._transforms import while_loop
from . import dynamics

__all__ = ["evolve", "evolve_recorded", "omega", "sample", "sample_full", "steps", "ABSOLUTE_TOLERANCE",
           "RELATIVE_TOLERANCE"]

ABSOLUTE_TOLERANCE = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-12
RETRIES = 6

# Runge-Kutta-Fehlberg coefficients of GSL's rkf45.c
AH = (1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0)
B3 = (3.0 / 32.0, 9.0 / 32.0)
B4 = (1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0)
B5 = (8341.0 / 4104.0, -32832.0 / 4104.0, 29440.0 / 4104.0, -845.0 / 4104.0)
B6 = (-6080.0 / 20520.0, 41040.0 / 20520.0, -28352.0 / 20520.0, 9295.0 / 20520.0, -5643.0 / 20520.0)
C1 = 902880.0 / 7618050.0
C3 = 3953664.0 / 7618050.0
C4 = 3855735.0 / 7618050.0
C5 = -1371249.0 / 7618050.0
C6 = 277020.0 / 7618050.0
EC = (0.0, 1.0 / 360.0, 0.0, -128.0 / 4275.0, -2197.0 / 75240.0, 1.0 / 50.0, 2.0 / 55.0)


#: The stages of ``rkf45_apply`` as rows of weights of ``(k1, .., k6)``: the
#: states at which ``k2 .. k6`` are evaluated, then the new state (at which
#: the derivative is evaluated last).
STAGES = np.array([
    [AH[0], 0.0, 0.0, 0.0, 0.0, 0.0],
    [B3[0], B3[1], 0.0, 0.0, 0.0, 0.0],
    [B4[0], B4[1], B4[2], 0.0, 0.0, 0.0],
    [B5[0], B5[1], B5[2], B5[3], 0.0, 0.0],
    [B6[0], B6[1], B6[2], B6[3], B6[4], 0.0],
    [C1, 0.0, C3, C4, C5, C6],
])


def _weighted(row, ks):
    """``row[0] k1 + row[1] k2 + ...``, summed in that order (the terms of
    LAL's sums, and exact zeros)."""
    acc = row[0] * ks[0]
    for j in range(1, len(ks)):
        acc = acc + row[j] * ks[j]
    return acc


def _rkf45(c, y, h, k1):
    """``rkf45_apply``: the new state, the error estimate, the derivative at
    the new state, ``k6`` (for the interpolant) and whether every
    derivative evaluation succeeded.

    The six evaluations of the derivatives are the iterations of a loop
    (one copy of the equations to compile rather than six), with the
    stages ``k1 .. k6`` carried as separate small arrays."""
    stages = jnp.asarray(STAGES)

    def stage(s, carry):
        ks, ok = carry
        k, ok_s = dynamics.derivatives(c, y + h * _weighted(stages[s], ks[:6]))
        ks = tuple(jnp.where(s + 1 == j, k, ks[j]) for j in range(7))
        return ks, ok & ok_s

    ks = (k1,) + tuple(jnp.zeros_like(k1) for _ in range(6))
    ks, ok = jax.lax.fori_loop(0, 6, stage, (ks, jnp.asarray(True)))
    k1, k2, k3, k4, k5, k6, dydt_out = ks
    y_new = y + h * _weighted(STAGES[5], ks[:6])
    yerr = h * (EC[1] * k1 + EC[3] * k3 + EC[4] * k4 + EC[5] * k5 + EC[6] * k6)
    return y_new, yerr, dydt_out, k6, ok


def _hadjust(y, yerr, h):
    """``std_control_hadjust`` with ``a_y = 1``, ``a_dydt = 0``: the new step
    and whether it was decreased."""
    d0 = RELATIVE_TOLERANCE * jnp.abs(y) + ABSOLUTE_TOLERANCE
    r = jnp.abs(yerr) / jnp.abs(d0)
    # GSL_MAX_DBL never selects a NaN
    rmax = jnp.max(jnp.where(jnp.isnan(r), DBL_MIN, jnp.maximum(r, DBL_MIN)))
    decrease = jnp.maximum(0.9 / jnp.power(rmax, 1.0 / 5.0), 0.2)
    increase = jnp.clip(0.9 / jnp.power(rmax, 1.0 / 6.0), 1.0, 5.0)
    h_new = jnp.where(rmax > 1.1, decrease * h, jnp.where(rmax < 0.5, increase * h, h))
    return h_new, rmax > 1.1


def _count_samples(tintp, t, deltat):
    """The number of samples ``tintp + k deltat`` with ``(tintp + k
    deltat)^2 < t^2`` (``k >= 1``), as LAL's interpolation loop adds them."""
    span = (t - tintp) / deltat
    k = jnp.ceil(span) - 1.0
    return jnp.maximum(k, 0.0)


def evolve(c, y0, deltat, max_iterations):
    """``XLALAdaptiveRungeKutta4Hermite`` with the SpinTaylorT4 system and
    ``stop_on_test_only`` from the state ``y0`` at ``t = 0``, sampling
    every ``deltat`` (in units of the total mass; negative to integrate
    backwards), for at most ``max_iterations`` attempted steps.

    Returns a :class:`~.._math.Struct`: ``n_samples`` (the number of samples,
    including the initial one, before LAL's trimming at ``f_end``),
    ``t_last`` (the time of the last sample), the last step (``t0``, ``h``,
    ``y0``, ``y1``, ``k1``, ``k6``) for the Hermite interpolant, the return
    code and ``exhausted`` (the iteration limit was reached).
    """
    c = jax.lax.stop_gradient(c)
    y0 = jax.lax.stop_gradient(y0)
    deltat = jax.lax.stop_gradient(deltat)
    k_init, _ = dynamics.derivatives(c, y0)
    zero = jnp.zeros_like(y0)
    state = Struct(
        iteration=jnp.zeros((), jnp.int64),
        t=jnp.zeros(()),
        y=y0,
        dydt=k_init,
        h=jnp.asarray(deltat, dtype=y0.dtype),
        retries=jnp.asarray(RETRIES),
        prev_domega=jnp.zeros(()),
        tintp=jnp.zeros(()),
        n_samples=jnp.ones(()),
        done=jnp.asarray(False),
        code=jnp.asarray(0),
        last_t0=jnp.zeros(()),
        last_h=jnp.zeros(()),
        last_y0=y0,
        last_y1=y0,
        last_k1=k_init,
        last_k6=zero,
    )

    def cond(s):
        return (~s.done) & (s.iteration < max_iterations)

    def body(s):
        y_new, yerr, dydt_out, k6, ok = _rkf45(c, s.y, s.h, s.dydt)
        # a failed derivative: retry with a ten times smaller step, or stop
        failed = ~ok
        out_of_retries = failed & (s.retries <= 0)
        # the step size control
        t_new = s.t + s.h
        h_adj, decreased = _hadjust(y_new, yerr, s.h)
        reject = ok & decreased & (jnp.abs(h_adj) < jnp.abs(s.h)) & (t_new + h_adj != t_new)
        accept = ok & ~reject
        h_next = jnp.where(decreased, s.h, h_adj)
        # samples in the accepted step
        k = jnp.where(accept, _count_samples(s.tintp, t_new, deltat), 0.0)
        code, prev_domega = dynamics.stopping_test(c, y_new, dydt_out, s.prev_domega)
        stop = accept & (code != 0)
        new = Struct(
            iteration=s.iteration + 1,
            t=jnp.where(accept, t_new, s.t),
            y=jnp.where(accept, y_new, s.y),
            dydt=jnp.where(accept, dydt_out, s.dydt),
            h=jnp.where(failed, s.h / 10.0, jnp.where(reject, h_adj, h_next)),
            retries=jnp.where(failed, s.retries - 1, jnp.where(accept, RETRIES, s.retries)),
            prev_domega=jnp.where(accept, prev_domega, s.prev_domega),
            tintp=s.tintp + k * deltat,
            n_samples=s.n_samples + k,
            done=out_of_retries | stop,
            code=jnp.where(out_of_retries, dynamics.ST_DERIVATIVE_OMEGANONPOS, jnp.where(stop, code, s.code)),
            last_t0=jnp.where(accept, s.t, s.last_t0),
            last_h=jnp.where(accept, s.h, s.last_h),
            last_y0=jnp.where(accept, s.y, s.last_y0),
            last_y1=jnp.where(accept, y_new, s.last_y1),
            last_k1=jnp.where(accept, s.dydt, s.last_k1),
            last_k6=jnp.where(accept, k6, s.last_k6),
        )
        return new

    s = jax.lax.while_loop(cond, body, state)
    return Struct(
        n_samples=s.n_samples,
        t_last=s.tintp,
        t_end=s.t,
        y_end=s.y,
        last_t0=s.last_t0,
        last_h=s.last_h,
        last_y0=s.last_y0,
        last_y1=s.last_y1,
        last_k1=s.last_k1,
        last_k6=s.last_k6,
        code=s.code,
        exhausted=~s.done,
        iterations=s.iteration,
    )


def hermite(t0, h, y0, y1, k1, k6, t):
    """LAL's interpolant within the step ``[t0, t0 + h]`` at ``t``."""
    theta = (t - t0) / h
    i0 = theta * theta * (3.0 - 4.0 * theta) + 1.0
    i1 = -theta * (theta - 1.0)
    i6 = -4.0 * theta * theta * (theta - 1.0)
    iend = theta * theta * (4.0 * theta - 3.0)
    return i0 * y0 + iend * y1 + (h * i1) * k1 + (h * i6) * k6


# ---------------------------------------------------------------------------
# the run with its accepted steps recorded, differentiable
# ---------------------------------------------------------------------------


def _attempt(c, s, deltat):
    """One attempted step of :func:`evolve` (``s`` the state without the
    record) and the row recorded for it, ``(t0, h, y, k1, k6)``. The step
    sizes, times and decisions are constants for the derivatives (the
    derivatives are those of the solution along LAL's steps)."""
    sg = jax.lax.stop_gradient
    h = sg(s.h)
    y_new, yerr, dydt_out, k6, ok = _rkf45(c, s.y, h, s.dydt)
    failed = ~ok
    out_of_retries = failed & (s.retries <= 0)
    t_new = s.t + h
    h_adj, decreased = _hadjust(sg(y_new), sg(yerr), h)
    reject = ok & decreased & (jnp.abs(h_adj) < jnp.abs(h)) & (t_new + h_adj != t_new)
    accept = ok & ~reject
    h_next = jnp.where(decreased, h, h_adj)
    k = jnp.where(accept, _count_samples(s.tintp, t_new, deltat), 0.0)
    code, prev_domega = dynamics.stopping_test(c, sg(y_new), sg(dydt_out), s.prev_domega)
    stop = accept & (code != 0)
    row = jnp.concatenate([jnp.stack([s.t, h]), s.y, s.dydt, k6])
    new = Struct(
        iteration=s.iteration + 1,
        accepted=s.accepted + accept.astype(s.accepted.dtype),
        t=jnp.where(accept, t_new, s.t),
        y=jnp.where(accept, y_new, s.y),
        dydt=jnp.where(accept, dydt_out, s.dydt),
        h=sg(jnp.where(failed, h / 10.0, jnp.where(reject, h_adj, h_next))),
        retries=jnp.where(failed, s.retries - 1, jnp.where(accept, RETRIES, s.retries)),
        prev_domega=sg(jnp.where(accept, prev_domega, s.prev_domega)),
        tintp=s.tintp + k * deltat,
        n_samples=s.n_samples + k,
        done=out_of_retries | stop,
        code=jnp.where(out_of_retries, dynamics.ST_DERIVATIVE_OMEGANONPOS, jnp.where(stop, code, s.code)),
    )
    return new, row


def evolve_recorded(c, y0, deltat, max_iterations, capacity, components=None, ring=0):
    """:func:`evolve` keeping every accepted step (at most ``capacity``):
    their start times ``t0`` and sizes ``h`` (constants for the
    derivatives) and the states ``y``, derivatives ``k1`` and stages ``k6``
    from which LAL interpolates its samples, differentiable (in forward
    mode) with respect to ``c`` and ``y0``: the steps are taken as fixed,
    so the derivatives are those of the solution, not of LAL's choice of
    steps.

    With ``components`` (a number), only the first ``components`` variables
    of ``y``, ``k1`` and ``k6`` are kept for every step, and the whole rows
    of the last ``ring`` steps (:func:`sample_full`).

    Returns a :class:`~.._math.Struct` with ``rows``, the steps ``(t0, h,
    y, k1, k6)`` (``capacity`` rows, see :func:`steps`), ``ring``, the keys
    of the searches among them, ``y_end``, ``dydt_end``, ``accepted``,
    ``n_samples``, ``t_last`` (time of the last sample), ``t_end``,
    ``code`` and ``exhausted`` (out of iterations or capacity).
    """
    deltat = jax.lax.stop_gradient(deltat)
    k_init, _ = dynamics.derivatives(c, y0)
    nv = y0.shape[-1]
    nc = nv if components is None else components
    columns = np.concatenate([[0, 1], 2 + np.arange(nc), 2 + nv + np.arange(nc), 2 + 2 * nv + np.arange(nc)])
    state = Struct(
        iteration=jnp.zeros((), jnp.int64),
        accepted=jnp.zeros((), jnp.int64),
        t=jnp.zeros(()),
        y=y0,
        dydt=k_init,
        h=jnp.asarray(deltat, dtype=y0.dtype),
        retries=jnp.asarray(RETRIES),
        prev_domega=jnp.zeros(()),
        tintp=jnp.zeros(()),
        n_samples=jnp.ones(()),
        done=jnp.asarray(False),
        code=jnp.asarray(0),
    )
    records = (jnp.zeros((capacity, 2 + 3 * nc)),) + ((jnp.zeros((ring, 2 + 3 * nv)),) if ring else ())

    def test(consts, s):
        return (~s.done) & (s.iteration < consts[2]) & (s.accepted < capacity)

    def body(consts, s, records, active):
        c, deltat, _ = consts
        # the attempt in a loop of its own (_math.isolated): its many small
        # operations then run on the calling thread
        new, row = isolated(_attempt, c, s, deltat)
        # written at every attempt: a rejected one is overwritten by the
        # next accepted step (inactive under jax.vmap: written back unchanged)
        i = jnp.minimum(s.accepted, capacity - 1)
        kept = row if components is None else row[columns]
        record = records[0]
        record = jax.lax.dynamic_update_slice(record, jnp.where(active, kept, record[i])[None], (i, jnp.zeros_like(i)))
        if not ring:
            return new, (record,)
        j = s.accepted % ring
        whole = records[1]
        whole = jax.lax.dynamic_update_slice(whole, jnp.where(active, row, whole[j])[None], (j, jnp.zeros_like(j)))
        return new, (record, whole)

    s, records = while_loop(test, body, (c, deltat, max_iterations), state, records)
    record = records[0]
    # the keys of the searches among the steps, once (rather than masking
    # the whole record at every search): the start times (along the
    # direction of the run) and M omega, infinite beyond the steps taken
    valid = jnp.arange(capacity) < jnp.maximum(s.accepted, 1)
    direction = jnp.where(deltat >= 0, 1.0, -1.0)
    return Struct(
        rows=record, ring=records[1] if ring else None, key_t=jnp.where(valid, record[:, 0] * direction, jnp.inf),
        key_omega=jnp.where(valid, record[:, 3], jnp.inf), direction=direction, y_end=s.y, dydt_end=s.dydt,
        accepted=s.accepted, n_samples=s.n_samples, t_last=s.tintp, t_end=s.t, code=s.code, exhausted=~s.done,
        iterations=s.iteration,
    )


def _split_row(rows, nc):
    return rows[..., 0], rows[..., 1], rows[..., 2:2 + nc], rows[..., 2 + nc:2 + 2 * nc], rows[..., 2 + 2 * nc:]


def steps(rec, i):
    """``(t0, h, y, k1, k6)`` of the recorded steps ``i`` (an index array),
    with the variables kept for every step."""
    return _split_row(rec.rows[i], (rec.rows.shape[-1] - 2) // 3)


def omega(rec, i):
    """``M omega`` at the start of the recorded steps ``i``."""
    return rec.rows[i, 3]


def _step_index(rec, t):
    n = jnp.maximum(rec.accepted, 1)
    return jnp.clip(jnp.searchsorted(rec.key_t, t * rec.direction, side="left") - 1, 0, n - 1), n


def sample(rec, t):
    """LAL's samples at times ``t`` (an array) from the recorded steps (the
    variables kept for every step)."""
    cap = rec.rows.shape[0]
    i, n = _step_index(rec, t)
    t0, h, y0, k1, k6 = steps(rec, i)
    y1 = jnp.where((i + 1 < n)[..., None], steps(rec, jnp.minimum(i + 1, cap - 1))[2],
                   rec.y_end[..., :y0.shape[-1]])
    return hermite(t0[..., None], h[..., None], y0, y1, k1, k6, t[..., None])


def sample_full(rec, t):
    """Every variable of LAL's samples at times ``t``, and whether the
    steps holding them are recorded whole (the last steps of a run whose
    record keeps only some of the variables, :func:`evolve_recorded`)."""
    if rec.ring is None:
        return sample(rec, t), jnp.ones(jnp.shape(t), bool)
    ring = rec.ring.shape[0]
    i, n = _step_index(rec, t)
    nv = rec.y_end.shape[-1]
    t0, h, y0, k1, k6 = _split_row(rec.ring[i % ring], nv)
    y1 = jnp.where((i + 1 < n)[..., None], _split_row(rec.ring[(i + 1) % ring], nv)[2], rec.y_end)
    return hermite(t0[..., None], h[..., None], y0, y1, k1, k6, t[..., None]), i >= rec.accepted - ring
