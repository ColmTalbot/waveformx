# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""A dense, differentiable solution of the SpinTaylorT4 equations.

LALSimulation integrates the precession equations with GSL's adaptive
Runge-Kutta-Fehlberg 4(5) method at tolerances of ``1e-12`` and samples the
solution every ``delta_t`` with the method's Hermite-like interpolant. For a
light binary that is millions of steps. Here the same equations are
integrated with the Dormand-Prince 8(5,3) method (:mod:`.._generated.dop853`)
and its seventh-order dense output, with steps of equal increments of
``u = 1 / (M omega)`` (``omega`` the orbital frequency). The precession angle
grows roughly linearly in ``u``, so every step covers about the same angle,
and a few hundred steps resolve a whole inspiral to ``~1e-12``. The solution
is evaluated wherever LAL has samples (:mod:`.angles`).

The step size in time is ``h = du (M omega)^2 / (M domega/dt)`` from the state
at the start of the step: a smooth function of the state, so the solution is
differentiable. The steps run in a ``while_loop`` over the number of steps
actually needed, with the reverse-mode derivative from
:func:`.integrate.fixed_steps`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._generated import dop853
from . import dynamics
from .integrate import fixed_steps

__all__ = ["DENSE_COMPONENTS", "dop853_step", "integrate", "node_times", "evaluate", "evaluate_full"]

#: The components of the state whose dense output is kept for every step:
#: M omega and LNhat.
DENSE_COMPONENTS = (1, 2, 3, 4)
#: The number of polynomial coefficients of the dense output.
N_DENSE = 7


def _rhs(c, y):
    return dynamics.derivatives(c, y)[0]


def _stages(c, y, h, k0):
    """The twelve stages of a DOP853 step, the eighth-order solution and the
    derivative there."""
    stages = [k0]
    for s in range(1, dop853.N_STAGES):
        dy = sum(stages[j] * a for j, a in dop853.A[s]) * h
        stages.append(_rhs(c, y + dy))
    y_new = y + h * sum(stages[j] * b for j, b in dop853.B)
    f_new = _rhs(c, y_new)
    return stages, y_new, f_new


def _dense_coefficients(c, y, h, k0, stages, y_new, f_new, components=None):
    """The seven coefficients ``F`` of the dense output of a DOP853 step
    (three extra stages), for the given components of the state."""
    k = list(stages) + [f_new]
    for s in range(dop853.N_STAGES + 1, dop853.N_STAGES_EXTENDED):
        dy = sum(k[j] * a for j, a in dop853.A[s]) * h
        k.append(_rhs(c, y + dy))
    if components is not None:
        idx = jnp.asarray(components)
        k = [x[..., idx] for x in k]
        y, y_new, k0, f_new = y[..., idx], y_new[..., idx], k0[..., idx], f_new[..., idx]
    dy = y_new - y
    f = [dy, h * k0 - dy, 2.0 * dy - h * (f_new + k0)]
    for row in dop853.D:
        f.append(h * sum(k[j] * d for j, d in row))
    return jnp.stack(f, axis=-2)


def dop853_step(aux, state):
    """One step of the trajectory: ``aux = (coefficients, du)`` with ``du``
    the (signed) increment of ``u = 1 / (M omega)``; ``state = (y, dy/dt)``.
    Returns the next state and ``(h, F)``, the step in time and the dense
    output coefficients of :data:`DENSE_COMPONENTS`."""
    c, du = aux
    y, k0 = state
    omega = y[1]
    h = -du * omega * omega / k0[1]
    stages, y_new, f_new = _stages(c, y, h, k0)
    dense = _dense_coefficients(c, y, h, k0, stages, y_new, f_new, DENSE_COMPONENTS)
    return (y_new, f_new), (h, dense)


def integrate(c, y0, du, n, capacity):
    """``n`` (traced, at most ``capacity``) DOP853 steps of ``du`` in ``u``
    from ``y0`` at ``t = 0`` (``du > 0`` integrates backwards in time).

    Returns a :class:`~.._math.Struct` with the states ``y`` at the start of
    each step, ``dydt`` there, the steps ``h``, the dense-output
    coefficients ``dense`` (of :data:`DENSE_COMPONENTS`), the node times
    ``t`` (``t[0] = 0``, the time at the start of each step), the final state
    and time, and ``n``.
    """
    from .._math import Struct

    k0 = _rhs(c, y0)
    (ys, outs, last) = fixed_steps(dop853_step, (c, du), (y0, k0), n, capacity)
    hs, dense = outs
    valid = jnp.arange(capacity) < n
    hs = jnp.where(valid, hs, 0.0)
    t_end = jnp.cumsum(hs)
    t = t_end - hs
    return Struct(
        y=ys[0], dydt=ys[1], h=hs, dense=dense, t=t, t_end=t_end, y_last=last[0], t_last=t_end[-1], n=n, c=c,
    )


def _dense_eval(dense, y0, x):
    """The dense output ``y0 + F(x)`` at ``x = (t - t0) / h`` (Hairer's
    nested form)."""
    y = jnp.zeros_like(y0)
    for i in range(N_DENSE - 1, -1, -1):
        y = y + dense[..., i, :]
        y = y * (x[..., None] if (N_DENSE - 1 - i) % 2 == 0 else (1.0 - x[..., None]))
    return y + y0


def locate(traj, t, backwards):
    """The index of the step containing each of ``t`` (clipped to the
    steps taken) and the position ``x`` in it."""
    n = jnp.maximum(traj.n, 1)
    # node times are monotonic: decreasing backwards, increasing forwards
    key = -traj.t if backwards else traj.t
    big = jnp.inf
    key = jnp.where(jnp.arange(traj.t.shape[0]) < n, key, big)
    query = -t if backwards else t
    i = jnp.clip(jnp.searchsorted(key, query, side="right") - 1, 0, n - 1)
    h = traj.h[i]
    x = (t - traj.t[i]) / jnp.where(h != 0.0, h, 1.0)
    return i, x


def evaluate(traj, t, backwards):
    """``M omega`` and LNhat (the :data:`DENSE_COMPONENTS`) at times ``t``
    (an array, in units of the total mass), from the dense output."""
    i, x = locate(traj, t, backwards)
    y0 = traj.y[i][..., jnp.asarray(DENSE_COMPONENTS)]
    return _dense_eval(traj.dense[i], y0, x)


def evaluate_full(traj, t, backwards):
    """The whole state at a few times ``t`` (recomputing the stages of the
    steps containing them)."""
    i, x = locate(traj, t, backwards)
    y, k0, h = traj.y[i], traj.dydt[i], traj.h[i]
    c = traj.c

    def one(y, k0, h, x):
        stages, y_new, f_new = _stages(c, y, h, k0)
        dense = _dense_coefficients(c, y, h, k0, stages, y_new, f_new)
        return _dense_eval(dense, y, x)

    return jax.vmap(one)(jnp.atleast_2d(y), jnp.atleast_2d(k0), jnp.atleast_1d(h), jnp.atleast_1d(x)).reshape(
        jnp.shape(t) + (dynamics.NUM_VARIABLES,)
    )
