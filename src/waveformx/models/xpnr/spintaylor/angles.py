# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Helpers of the SpinTaylor Euler angles (``IMRPhenomX_InspiralAngles_SpinTaylor``
and ``IMRPhenomX_InterpolateAlphaBeta_SpinTaylor`` of
``LALSimIMRPhenomX_precession.c``) shared by the solution
(:mod:`.lean`), the PNR interpolants and the MLX port: the step of the
integration, the Newton step with derivatives that locates a frequency,
and the angles of LNhat in the J frame."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._math import real, safe_div

__all__ = ["STEP_ANGLE", "small_atan2", "small_square"]

#: Precession angle (rad) per step of the DOP853 integration.
STEP_ANGLE = 0.1


def _du_per_step(m1_si, m2_si):
    """The increment of ``u = 1 / (M omega)`` that covers ``STEP_ANGLE`` of
    the leading-order precession of L, ``dalpha/du = 5 (2 + 3 m2 / 2 m1) /
    96`` (smaller near merger and with spin-spin terms; the margin of
    :data:`STEP_ANGLE` covers both)."""
    rate = 5.0 * (2.0 + 1.5 * m2_si / m1_si) / 96.0
    return STEP_ANGLE / rate


def _refined(state, t, omega):
    """One Newton step with derivatives from the root ``t`` of ``state(t)[0]
    = omega``, so that the derivatives are those of the root; the state
    there, its derivative in time and the time."""
    value, slope = jax.jvp(lambda t: state(t)[..., 0], (t,), (jnp.ones_like(t),))
    t = t - safe_div(value - omega, jax.lax.stop_gradient(slope))
    y, dydt = jax.jvp(state, (t,), (jnp.ones_like(t),))
    return y, dydt, t


def _atan2(y, x):
    """``atan2(y, x)``, zero with a zero derivative where both vanish (LNhat
    along J: aligned spins), where ``atan2`` has none."""
    both = (x == 0.0) & (y == 0.0)
    return jnp.arctan2(jnp.where(both, 0.0, y), jnp.where(both, 1.0, x))


#: Below this ``x^2 + y^2``, the derivative of ``atan2(y, x)`` would divide
#: by an underflowing square (float64; :func:`small_square` in float32).
SMALL_SQUARE = 1e-280


def small_square():
    """:data:`SMALL_SQUARE` in the precision of the evaluation (``1e-30`` in
    float32, where ``1e-280`` is zero)."""
    return 1e-30 if real().itemsize == 4 else SMALL_SQUARE


def small_atan2(y, x):
    """:func:`_atan2`, also zero (with a zero derivative) where ``x^2 +
    y^2 < SMALL_SQUARE``."""
    small = x * x + y * y < small_square()
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
