# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The steps of a dense, differentiable solution of the SpinTaylorT4
equations (:mod:`.lean`).

LALSimulation integrates the precession equations with GSL's adaptive
Runge-Kutta-Fehlberg 4(5) method at tolerances of ``1e-12`` and samples the
solution every ``delta_t`` with the method's Hermite-like interpolant. For a
light binary that is millions of steps. Here the same equations are
integrated with the Dormand-Prince 8(5,3) method (:mod:`.._generated.dop853`)
and its seventh-order dense output, with steps of equal increments of
``u = 1 / (M omega)`` (``omega`` the orbital frequency). The precession angle
grows roughly linearly in ``u``, so every step covers about the same angle,
and a few hundred steps resolve a whole inspiral to ``~1e-12``.

The step size in time is ``h = du (M omega)^2 / (M domega/dt)`` from the state
at the start of the step (:func:`_step_in_time`): a smooth function of the
state, so the solution is differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .._generated import dop853
from . import dynamics
from .._math import isolated_step, real

__all__ = ["DENSE_COMPONENTS", "N_DENSE"]

#: The components of the state whose dense output is kept for every step:
#: M omega and LNhat.
DENSE_COMPONENTS = (1, 2, 3, 4)
#: The number of polynomial coefficients of the dense output.
N_DENSE = 7


def _rhs(c, y):
    return dynamics.derivatives(c, y)[0]


def _tables():
    a = np.zeros((dop853.N_STAGES_EXTENDED, dop853.N_STAGES_EXTENDED))
    for s, row in enumerate(dop853.A):
        for j, value in row:
            a[s, j] = value
    b = np.zeros(dop853.N_STAGES_EXTENDED)
    for j, value in dop853.B:
        b[j] = value
    d = np.zeros((len(dop853.D), dop853.N_STAGES_EXTENDED))
    for i, row in enumerate(dop853.D):
        for j, value in row:
            d[i, j] = value
    return a, b, d


_A, _B, _D = _tables()


def _stages_looped(c, y, h, k0):
    """The twelve stages of a DOP853 step and the extra stages of its dense
    output, in a ``jax.lax.fori_loop`` (one copy of the equations). Row 12 of ``A`` holds the
    weights ``B``: stage 12 is the derivative at the new state. Each
    derivative runs in a loop of its own (:func:`~.._math.isolated_step`), so
    that the loop over the stages (whose stages make a buffer of more than
    512 bytes) runs on the calling thread. Returns the stages (rows) and the
    new state."""
    k = jnp.zeros((dop853.N_STAGES_EXTENDED,) + y.shape, y.dtype).at[0].set(k0)

    def stage(s, k):
        # (the weighted sum of the stages as one fused reduction, and a
        # plain dynamic update: few enough thunks for the loop body to run on
        # the calling thread)
        y_stage = y + h * jnp.sum(jnp.asarray(_A, real())[s][:, None] * k, axis=0)
        k_s = isolated_step(lambda y_stage, c: _rhs(c, y_stage), y_stage, (c,))
        return jax.lax.dynamic_update_slice(k, k_s[None], (s, 0))

    k = jax.lax.fori_loop(1, dop853.N_STAGES_EXTENDED, stage, k)
    return k, y + h * jnp.sum(jnp.asarray(_B, real())[:, None] * k, axis=0)


def _dense_looped(y, h, k, y_new, components=None):
    k0, f_new = k[0], k[dop853.N_STAGES]
    if components is not None:
        idx = jnp.asarray(components)
        k, y, y_new, k0, f_new = k[:, idx], y[idx], y_new[idx], k0[idx], f_new[idx]
    dy = y_new - y
    f = jnp.stack([dy, h * k0 - dy, 2.0 * dy - h * (f_new + k0)])
    # (weighted sums as fused reductions rather than small matrix products,
    # each a call of its own on the CPU)
    return jnp.concatenate([f, h * jnp.sum(jnp.asarray(_D, real())[:, :, None] * k[None], axis=1)], axis=0)


def _step_in_time(du, omega, omega_dot):
    """The step in time that changes ``u = 1 / (M omega)`` by ``du``, from
    the state at its start and the leading-order evolution ``du/dt ~
    u^(-5/3)``: ``h = h0 (3/8) ((1 + x)^(8/3) - 1) / x`` with ``x = du /
    u`` and ``h0 = -du omega^2 / (domega/dt)`` (the steps cover nearly
    ``du`` even where ``du/dt`` changes much within a step)."""
    h0 = -du * omega * omega / omega_dot
    x = du * omega
    small = jnp.abs(x) < 1e-6
    x_safe = jnp.where(small, 1.0, x)
    factor = 0.375 * (jnp.power(1.0 + x_safe, 8.0 / 3.0) - 1.0) / x_safe
    return h0 * jnp.where(small, 1.0 + 5.0 / 6.0 * x, factor)


def _dense_eval(dense, y0, x):
    """The dense output ``y0 + F(x)`` at ``x = (t - t0) / h`` (Hairer's
    nested form)."""
    y = jnp.zeros_like(y0)
    for i in range(N_DENSE - 1, -1, -1):
        y = y + dense[..., i, :]
        y = y * (x[..., None] if (N_DENSE - 1 - i) % 2 == 0 else (1.0 - x[..., None]))
    return y + y0
