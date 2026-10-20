# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""LAL's adaptive integration of the SpinTaylorT4 equations, step for step,
to find where its output ends.

``XLALSimInspiralSpinTaylorPNEvolveOrbit`` integrates with
``XLALAdaptiveRungeKutta4Hermite`` (``LALAdaptiveRungeKuttaIntegrator.c``):
GSL's ``rkf45`` stepper with the standard step control (``y`` only,
tolerances ``1e-12``), ``gsl_odeiv_evolve_apply``, up to six retries with a
ten times smaller step when the derivative fails, samples every
``delta_t`` from the step's Hermite-like interpolant, and the SpinTaylor
stopping test after every step. The output ends with the samples of the step
at which the stopping test fires, so where it ends depends on the exact
sequence of steps. This module repeats that sequence (after waveformrs's
``spintaylor/integrator.rs``) in a ``jax.lax.while_loop`` and returns the
number of samples and where the integration stopped.

Only these integers and stopping points are used (the samples themselves
come from :mod:`.trajectory`), so the replica is not differentiated: its
inputs are wrapped in ``jax.lax.stop_gradient``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._math import DBL_MIN, Struct
from . import dynamics

__all__ = ["evolve", "ABSOLUTE_TOLERANCE", "RELATIVE_TOLERANCE"]

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


def _rkf45(c, y, h, k1):
    """``rkf45_apply``: the new state, the error estimate, the derivative at
    the new state, ``k6`` (for the interpolant) and whether every
    derivative evaluation succeeded."""
    f = lambda x: dynamics.derivatives(c, x)  # noqa: E731
    k2, ok2 = f(y + AH[0] * h * k1)
    k3, ok3 = f(y + h * (B3[0] * k1 + B3[1] * k2))
    k4, ok4 = f(y + h * (B4[0] * k1 + B4[1] * k2 + B4[2] * k3))
    k5, ok5 = f(y + h * (B5[0] * k1 + B5[1] * k2 + B5[2] * k3 + B5[3] * k4))
    k6, ok6 = f(y + h * (B6[0] * k1 + B6[1] * k2 + B6[2] * k3 + B6[3] * k4 + B6[4] * k5))
    y_new = y + h * (C1 * k1 + C3 * k3 + C4 * k4 + C5 * k5 + C6 * k6)
    dydt_out, ok7 = f(y_new)
    yerr = h * (EC[1] * k1 + EC[3] * k3 + EC[4] * k4 + EC[5] * k5 + EC[6] * k6)
    ok = ok2 & ok3 & ok4 & ok5 & ok6 & ok7
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


def _accepted_step(c, y, k1, h):
    """An accepted RKF45 step of size ``h`` (a constant for the derivative):
    the next state, its derivative and ``k6``."""
    y_new, _, dydt_out, k6, _ = _rkf45(c, y, h, k1)
    return y_new, dydt_out, k6


def _evolve_recording(c, y0, deltat, max_iterations, capacity):
    """:func:`evolve`, recording ``(t0, h, y0, k1, k6)`` of every accepted
    step in arrays of length ``capacity``."""
    k_init, _ = dynamics.derivatives(c, y0)
    zero = jnp.zeros_like(y0)
    record = Struct(
        t0=jnp.zeros(capacity),
        h=jnp.zeros(capacity),
        y=jnp.zeros((capacity,) + y0.shape),
        k1=jnp.zeros((capacity,) + y0.shape),
        k6=jnp.zeros((capacity,) + y0.shape),
    )
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
        record=record,
        last_k6=zero,
    )

    def cond(s):
        return (~s.done) & (s.iteration < max_iterations) & (s.accepted < capacity)

    def body(s):
        y_new, yerr, dydt_out, k6, ok = _rkf45(c, s.y, s.h, s.dydt)
        failed = ~ok
        out_of_retries = failed & (s.retries <= 0)
        t_new = s.t + s.h
        h_adj, decreased = _hadjust(y_new, yerr, s.h)
        reject = ok & decreased & (jnp.abs(h_adj) < jnp.abs(s.h)) & (t_new + h_adj != t_new)
        accept = ok & ~reject
        h_next = jnp.where(decreased, s.h, h_adj)
        k = jnp.where(accept, _count_samples(s.tintp, t_new, deltat), 0.0)
        code, prev_domega = dynamics.stopping_test(c, y_new, dydt_out, s.prev_domega)
        stop = accept & (code != 0)
        # written at every attempt: a rejected one is overwritten by the
        # next accepted step
        i = jnp.minimum(s.accepted, capacity - 1)
        r = s.record
        record = Struct(
            t0=r.t0.at[i].set(s.t),
            h=r.h.at[i].set(s.h),
            y=r.y.at[i].set(s.y),
            k1=r.k1.at[i].set(s.dydt),
            k6=r.k6.at[i].set(k6),
        )
        return Struct(
            iteration=s.iteration + 1,
            accepted=s.accepted + accept.astype(s.accepted.dtype),
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
            record=record,
            last_k6=jnp.where(accept, k6, s.last_k6),
        )

    s = jax.lax.while_loop(cond, body, state)
    return s


@jax.custom_vjp
def _recorded(c, y0, deltat, max_iterations, capacity_marker):
    s = _evolve_recording(c, y0, deltat, max_iterations, capacity_marker.shape[0])
    return _outputs(s)


def _outputs(s):
    r = s.record
    return (r.y, r.k1, r.k6, s.y, s.dydt), (r.t0, r.h, s.accepted, s.n_samples, s.tintp, s.t, s.code,
                                           s.done, s.iteration)


def _recorded_fwd(c, y0, deltat, max_iterations, capacity_marker):
    s = _evolve_recording(jax.lax.stop_gradient(c), y0, deltat, max_iterations, capacity_marker.shape[0])
    out = _outputs(s)
    return out, (c, s.record.y, s.record.k1, s.record.h, s.accepted, y0)


def _recorded_bwd(residuals, cotangents):
    c, ys, k1s, hs, n, y0 = residuals
    (ct_y, ct_k1, ct_k6, ct_last, ct_last_dydt), _ = cotangents
    flat_c, unravel_c = jax.flatten_util.ravel_pytree(c)

    def step(flat_c, y, k1, h):
        return _accepted_step(unravel_c(flat_c), y, k1, h)

    def cond(carry):
        return carry[0] >= 0

    def body(carry):
        i, lam_y, lam_k1, cbar = carry
        y = ys[i]
        k1 = k1s[i]
        _, vjp = jax.vjp(lambda fc, y, k1: step(fc, y, k1, hs[i]), flat_c, y, k1)
        # the step maps (y_i, k1_i) to (y_{i+1}, k1_{i+1}, k6_i)
        dc, dy, dk1 = vjp((lam_y, lam_k1, ct_k6[i]))
        lam_y = dy + ct_y[i]
        lam_k1 = dk1 + ct_k1[i]
        return i - 1, lam_y, lam_k1, cbar + dc

    _, lam_y, lam_k1, cbar = jax.lax.while_loop(
        cond, body, (n - 1, ct_last, ct_last_dydt, jnp.zeros_like(flat_c))
    )
    # k1_0 = f(y0)
    _, vjp0 = jax.vjp(lambda fc, y: dynamics.derivatives(unravel_c(fc), y)[0], flat_c, y0)
    dc0, dy0 = vjp0(lam_k1)
    return unravel_c(cbar + dc0), lam_y + dy0, None, None, None


_recorded.defvjp(_recorded_fwd, _recorded_bwd)


def evolve_recorded(c, y0, deltat, max_iterations, capacity):
    """:func:`evolve` keeping every accepted step (at most ``capacity``):
    their start times ``t0`` and sizes ``h`` (constants for the
    derivatives) and the states ``y``, derivatives ``k1`` and stages ``k6``
    from which LAL interpolates its samples, differentiable with respect to
    ``c`` and ``y0`` (the steps are taken as fixed: the derivatives are
    those of the solution, not of LAL's choice of steps).

    Returns a :class:`~.._math.Struct` with ``t0``, ``h``, ``y``, ``k1``,
    ``k6`` (arrays of length ``capacity``), ``y_end``, ``dydt_end``,
    ``accepted``, ``n_samples``, ``t_last`` (time of the last sample),
    ``t_end``, ``code`` and ``exhausted`` (out of iterations or capacity).
    """
    deltat = jax.lax.stop_gradient(deltat)
    marker = jnp.zeros((capacity, 0))
    (ys, k1s, k6s, y_end, dydt_end), (t0, h, accepted, n_samples, tintp, t_end, code, done, iterations) = _recorded(
        c, y0, deltat, max_iterations, marker
    )
    return Struct(
        t0=t0, h=h, y=ys, k1=k1s, k6=k6s, y_end=y_end, dydt_end=dydt_end, accepted=accepted,
        n_samples=n_samples, t_last=tintp, t_end=t_end, code=code, exhausted=~done, iterations=iterations,
    )


def sample(rec, t):
    """LAL's samples at times ``t`` (an array) from the recorded steps."""
    cap = rec.t0.shape[0]
    n = jnp.maximum(rec.accepted, 1)
    forward = rec.h[0] >= 0
    key = jnp.where(forward, rec.t0, -rec.t0)
    key = jnp.where(jnp.arange(cap) < n, key, jnp.inf)
    query = jnp.where(forward, t, -t)
    i = jnp.clip(jnp.searchsorted(key, query, side="left") - 1, 0, n - 1)
    y0 = rec.y[i]
    k1 = rec.k1[i]
    k6 = rec.k6[i]
    y1 = jnp.where((i + 1 < n)[..., None], rec.y[jnp.minimum(i + 1, cap - 1)], rec.y_end)
    t0 = rec.t0[i][..., None]
    h = rec.h[i][..., None]
    return hermite(t0, h, y0, y1, k1, k6, t[..., None])
