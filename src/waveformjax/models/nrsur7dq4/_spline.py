# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Interpolation as LALSimulation does it with GSL: the natural cubic
spline (``gsl_interp_cspline``) and the polynomial through four points
(``gsl_interp_polynomial``).

The spline is linear in the values, so it is differentiable with respect to
them, and smooth in the evaluation point between knots (twice continuously
differentiable across them).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve

__all__ = ["second_derivatives", "locate", "evaluate", "interpolate", "polynomial4"]


def second_derivatives(x, y):
    """GSL's ``c`` coefficients (half the second derivatives) of the natural
    cubic spline through ``(x, y)``, ``y`` of shape ``(n, ...)``, as
    ``cspline_init``: the tridiagonal system with diagonal ``2 (h_i +
    h_{i+1})``, off-diagonal ``h_{i+1}`` and right-hand side ``3 (dy_{i+1} /
    h_{i+1} - dy_i / h_i)``, with ``c_0 = c_{n-1} = 0``."""
    n = x.shape[0]
    shape = y.shape
    y2 = y.reshape(n, -1)
    h = jnp.diff(x)
    h_inv = 1.0 / h
    dy = jnp.diff(y2, axis=0)
    g = 3.0 * (dy[1:] * h_inv[1:, None] - dy[:-1] * h_inv[:-1, None])
    diag = 2.0 * (h[1:] + h[:-1])
    off = h[1:-1]
    zero = jnp.zeros((1,), dtype=x.dtype)
    inner = tridiagonal_solve(jnp.concatenate([zero, off]), diag, jnp.concatenate([off, zero]), g)
    zeros = jnp.zeros((1, y2.shape[1]), dtype=inner.dtype)
    return jnp.concatenate([zeros, inner, zeros], axis=0).reshape(shape)


def locate(x, t):
    """The interval ``i`` with ``x[i] <= t < x[i + 1]`` (``n - 2`` at the
    last knot, and clipped outside), as ``gsl_interp_accel_find``."""
    index = jnp.searchsorted(x, t, side="right") - 1
    return jnp.clip(index, 0, x.shape[0] - 2)


def evaluate(x, y, c, index, delx):
    """The spline ``(x, y, c)`` in interval ``index`` at ``x[index] +
    delx`` (``coeff_calc`` and ``cspline_eval``); ``y``, ``c`` of shape
    ``(n, ...)``, ``index`` and ``delx`` of shape ``(m,)``."""
    expand = (slice(None),) + (None,) * (y.ndim - 1)
    dx = (x[index + 1] - x[index])[expand]
    y_i = y[index]
    c_i = c[index]
    c_ip1 = c[index + 1]
    b = (y[index + 1] - y_i) / dx - dx * (c_ip1 + 2.0 * c_i) / 3.0
    d = (c_ip1 - c_i) / (3.0 * dx)
    delx = delx[expand]
    return y_i + delx * (b + delx * (c_i + delx * d))


def interpolate(x, y, t):
    """The natural cubic spline through ``(x, y)`` at ``t`` (inside the
    knots), ``y`` of shape ``(n, ...)``."""
    c = second_derivatives(x, y)
    index = locate(x, t)
    return evaluate(x, y, c, index, t - x[index])


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
