# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Interpolation as LALSimulation does it with GSL: the natural cubic
spline (``gsl_interp_cspline``) and the polynomial through four points
(``gsl_interp_polynomial``).

The natural spline solves a symmetric tridiagonal system whose matrix
depends only on the knots. As in waveformrs, it is factorised once, when the
surrogate data are read (:func:`factorize`, the ``LDL^T`` factors of
``gsl_linalg_solve_symm_tridiag``); fitting a spline to new values then only
needs the two substitution sweeps, done here with ``lax.scan``. The spline is
linear in the values, so differentiable with respect to them, and smooth in
the evaluation point between knots (twice continuously differentiable across
them).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

__all__ = ["Knots", "factorize", "second_derivatives", "locate", "evaluate", "interpolate", "polynomial4"]


class Knots(NamedTuple):
    """Knots ``x`` of natural cubic splines with GSL's factorised system:
    ``h = diff(x)``, ``h_inv = 1 / h``, and the diagonal ``alpha`` and
    sub-diagonal ``gamma`` of its ``LDL^T`` factorisation."""

    x: jnp.ndarray
    h: jnp.ndarray
    h_inv: jnp.ndarray
    alpha: jnp.ndarray
    gamma: jnp.ndarray


def factorize(x) -> Knots:
    """The factorised spline system of the strictly increasing knots ``x``
    (at least four), as ``cspline_init`` and
    ``gsl_linalg_solve_symm_tridiag`` compute it: diagonal ``2 (h_i +
    h_{i+1})``, off-diagonal ``h_{i+1}``."""
    x = np.asarray(x, dtype=float)
    h = np.diff(x)
    diag = 2.0 * (h[1:] + h[:-1])
    off = h[1:-1]
    n = diag.size
    alpha = np.zeros(n)
    gamma = np.zeros(n - 1)
    alpha[0] = diag[0]
    gamma[0] = off[0] / alpha[0]
    for i in range(1, n - 1):
        alpha[i] = diag[i] - off[i - 1] * gamma[i - 1]
        gamma[i] = off[i] / alpha[i]
    alpha[n - 1] = diag[n - 1] - off[n - 2] * gamma[n - 2]
    return Knots(*(jnp.asarray(value) for value in (x, h, 1.0 / h, alpha, gamma)))


def second_derivatives(knots: Knots, y):
    """GSL's ``c`` coefficients (half the second derivatives) of the natural
    cubic spline through ``y`` (``(n, ...)``) on ``knots``: the right-hand
    side ``3 (dy_{i+1} / h_{i+1} - dy_i / h_i)``, the forward and backward
    substitutions with the factors, and ``c_0 = c_{n-1} = 0``."""
    n = knots.x.shape[0]
    shape = y.shape
    y2 = y.reshape(n, -1)
    dy = jnp.diff(y2, axis=0)
    g = 3.0 * (dy[1:] * knots.h_inv[1:, None] - dy[:-1] * knots.h_inv[:-1, None])
    gamma = knots.gamma[:, None]

    def forward(z, xs):
        g_i, gamma_i = xs
        z = g_i - gamma_i * z
        return z, z

    _, z = lax.scan(forward, g[0], (g[1:], gamma), unroll=8)
    z = jnp.concatenate([g[:1], z]) / knots.alpha[:, None]

    def backward(c, xs):
        z_i, gamma_i = xs
        c = z_i - gamma_i * c
        return c, c

    _, c = lax.scan(backward, z[-1], (z[:-1], gamma), reverse=True, unroll=8)
    zeros = jnp.zeros_like(z[:1])
    return jnp.concatenate([zeros, c, z[-1:], zeros]).reshape(shape)


def locate(x, t):
    """The interval ``i`` with ``x[i] <= t < x[i + 1]`` (``n - 2`` at the
    last knot, and clipped outside), as ``gsl_interp_accel_find``."""
    index = jnp.searchsorted(x, t, side="right", method="scan_unrolled") - 1
    return jnp.clip(index, 0, x.shape[0] - 2)


def evaluate(knots: Knots, y, c, index, delx):
    """The spline ``(y, c)`` on ``knots`` in interval ``index`` at
    ``x[index] + delx`` (``coeff_calc`` and ``cspline_eval``); ``y``, ``c``
    of shape ``(n, ...)``, ``index`` and ``delx`` of shape ``(m,)``."""
    expand = (slice(None),) + (None,) * (y.ndim - 1)
    dx = knots.h[index][expand]
    y_i = y[index]
    c_i = c[index]
    c_ip1 = c[index + 1]
    b = (y[index + 1] - y_i) / dx - dx * (c_ip1 + 2.0 * c_i) / 3.0
    d = (c_ip1 - c_i) / (3.0 * dx)
    delx = delx[expand]
    return y_i + delx * (b + delx * (c_i + delx * d))


def interpolate(knots: Knots, y, t):
    """The natural cubic spline through ``y`` on ``knots`` at ``t``."""
    c = second_derivatives(knots, y)
    index = locate(knots.x, t)
    return evaluate(knots, y, c, index, t - knots.x[index])


def polynomial4(x, y, t):
    """The cubic through the four points ``(x, y)`` at ``t`` (Newton's
    divided differences, ``gsl_poly_dd_init`` and ``gsl_poly_dd_eval``);
    ``y`` of shape ``(4, ...)``."""
    dd = [y[0], y[1], y[2], y[3]]
    for j in (3, 2, 1):
        dd[j] = (y[j] - y[j - 1]) / (x[j] - x[j - 1])
    for i in (2, 3):
        for j in range(3, i - 1, -1):
            dd[j] = (dd[j] - dd[j - 1]) / (x[j] - x[j - i])
    value = dd[3]
    for i in (2, 1, 0):
        value = dd[i] + (t - x[i]) * value
    return value
