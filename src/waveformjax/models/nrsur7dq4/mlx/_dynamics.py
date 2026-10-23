# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The dynamics surrogate in MLX (as :mod:`.._dynamics`, whose docstring
describes the integration).

Every function takes a batch of binaries along the leading axis: the mass
ratio ``q`` of shape ``(B,)`` and states ``y`` of shape ``(B, 11)``. MLX has
no ``scan``, so the three RK4 steps and the ``n_ds - 4`` Adams-Bashforth
steps are a Python loop, unrolled into the graph; as in JAX, every step
selects the direction and node of each binary with ``where`` and gathers
the fits of that node. The states are put in the order of the nodes by
gathers rather than a scatter.

Indices computed from the parameters (``searchsorted``, ``argmax``) are
cast to int32 (MLX returns uint32) and cut from the gradient
(``stop_gradient``): MLX differentiates through the indices of a gather
otherwise, and fails.
"""

from __future__ import annotations

import mlx.core as mx

from .._dynamics import (
    _A,
    _B,
    _FIT_FIRST,
    _FIT_PARTNER,
    _FIT_SECOND,
    _GROUP,
    _IA,
    _IB,
    _RA,
    _RB,
    _STATE_FIRST,
    _STATE_PARTNER,
    _STATE_SECOND,
    OMEGA_REF_MAX,
    OMEGA_REF_ZERO,
    START_TIME,
)
from ._fits import monomials_at

__all__ = ["index", "safe_norm", "normalize_y", "derivatives", "reference_omegas", "reference_time", "integrate"]

_GROUP_M = mx.array(_GROUP.astype("int32"))
_FIT = (mx.array(_FIT_PARTNER.astype("int32")), mx.array(_FIT_FIRST), mx.array(_FIT_SECOND))
_STATE = (mx.array(_STATE_PARTNER.astype("int32")), mx.array(_STATE_FIRST), mx.array(_STATE_SECOND))
_ASSEMBLE = tuple(mx.array(a) for a in (_A.astype("float32"), _IA.astype("int32"), _RA.astype("int32"),
                                         _B.astype("float32"), _IB.astype("int32"), _RB.astype("int32")))


def index(value):
    """An index computed from the parameters: int32, without gradient."""
    return mx.stop_gradient(value.astype(mx.int32))


def safe_norm(v, axis=-1):
    """``|v|`` with a zero (not NaN) derivative at ``v = 0``."""
    square = mx.sum(v * v, axis=axis)
    positive = square > 0
    return mx.where(positive, mx.sqrt(mx.where(positive, square, 1.0)), 0.0)


def _rotate(v, table, sp, cp, sign):
    """``v`` (``(..., k)``) with the pairs of ``table = (partner, first,
    second)`` rotated by the angle of ``sp``, ``cp`` (``(...)``), as
    :func:`.._dynamics._rotate`."""
    partner, first, second = table
    sp, cp = sp[..., None], cp[..., None]
    c = mx.where(first | second, cp, 1.0)
    s = mx.where(first, -sign * sp, mx.where(second, sign * sp, 0.0))
    return v * c + mx.take(v, partner, axis=-1) * s


def normalize_y(y, norm_a, norm_b):
    """``PrecessingNRSur_normalize_y`` (see :func:`.._dynamics.normalize_y`)
    for ``y`` of shape ``(B, 11)``."""
    square = y * y
    squares = mx.stack(
        [mx.sum(square[:, 0:4], axis=-1), square[:, 4], mx.sum(square[:, 5:8], axis=-1), mx.sum(square[:, 8:11], axis=-1)],
        axis=-1,
    )
    positive = squares > 0
    norms = mx.sqrt(mx.where(positive, squares, 1.0))
    ones = mx.ones_like(norm_a)
    scale = mx.where(positive, mx.stack([ones, ones, norm_a, norm_b], axis=-1) / norms, 0.0)
    # the orbital phase is not normalised
    scale = mx.concatenate([scale[:, :1], ones[:, None], scale[:, 2:]], axis=-1)
    return y * mx.take(scale, _GROUP_M, axis=-1)


def _coorbital_spins(y):
    """The spins of the state in the coorbital frame and ``sin``, ``cos`` of
    the orbital phase."""
    sp = mx.sin(y[..., 4])
    cp = mx.cos(y[..., 4])
    x = _rotate(y, _STATE, sp, cp, -1.0)
    return x[..., 5:8], x[..., 8:11], sp, cp


def _assemble(y, fits, sp, cp):
    """``PrecessingNRSur_assemble_dydt`` (see :func:`.._dynamics._assemble`)."""
    a, ia, ra, b, ib, rb = _ASSEMBLE
    r = _rotate(fits, _FIT, sp, cp, 1.0)
    ye = mx.concatenate([y, mx.ones_like(y[..., :1])], axis=-1)
    return a * mx.take(ye, ia, axis=-1) * mx.take(r, ra, axis=-1) + b * mx.take(ye, ib, axis=-1) * mx.take(r, rb, axis=-1)


def derivatives(data, q, nodes, y):
    """``dy/dt`` at the states ``y`` (``(B, 11)``) with the fits of the
    dynamics nodes ``nodes``: ``(B,)`` (one node per binary, result ``(B,
    11)``) or ``(B, k)`` (result ``(B, k, 11)``)."""
    chi_a, chi_b, sp, cp = _coorbital_spins(y)
    mono = monomials_at(q, chi_a, chi_b, data.monomials)
    coefs = mx.take(data.dynamics, nodes, axis=0)
    if nodes.ndim > 1:
        mono, y, sp, cp = mono[:, None], y[:, None], sp[:, None], cp[:, None]
    fits = (coefs @ mono[..., None])[..., 0]
    return _assemble(y, fits, sp, cp)


def _polynomial4(x, y, t):
    """The cubic through the points ``(x, y)`` at ``t`` (as
    :func:`.._spline.polynomial4`) for ``x`` ``(B, 4)``, ``y`` ``(B, 4,
    k)`` and ``t`` ``(B,)``."""
    x = x[..., None]
    t = t[..., None]
    dd = [y[:, 0], y[:, 1], y[:, 2], y[:, 3]]
    for j in (3, 2, 1):
        dd[j] = (y[:, j] - y[:, j - 1]) / (x[:, j] - x[:, j - 1])
    for i in (2, 3):
        for j in range(3, i - 1, -1):
            dd[j] = (dd[j] - dd[j - 1]) / (x[:, j] - x[:, j - i])
    value = dd[3]
    for i in (2, 1, 0):
        value = dd[i] + (t - x[:, i]) * value
    return value


def derivative_at_time(data, q, t, y, extra=None):
    """``PrecessingNRSur_get_time_deriv`` at the times ``t`` (``(B,)``),
    with the derivative at the node ``extra`` (an int) too if given (see
    :func:`.._dynamics.derivative_at_time`)."""
    n = data.n_ds
    i1 = mx.searchsorted(data.t_ds[1:], t, side="left").astype(mx.int32)
    i0 = mx.clip(i1 - 1, 0, n - 4)
    nodes = i0[:, None] + mx.arange(4, dtype=mx.int32)
    if extra is not None:
        nodes = mx.concatenate([nodes, mx.full((nodes.shape[0], 1), extra, dtype=mx.int32)], axis=1)
    nodes = index(nodes)
    d = derivatives(data, q, nodes, y)
    value = _polynomial4(mx.take(data.t_ds, nodes[:, :4]), d[:, :4], t)
    if extra is None:
        return value
    return value, d[:, 4]


def reference_state(chi_a, chi_b):
    """The ODE state at the reference point: identity quaternion, zero
    orbital phase, the spins ``chi_a``, ``chi_b`` (``(B, 3)``)."""
    zero = mx.zeros_like(chi_a[..., :1])
    return mx.concatenate([zero + 1.0, zero, zero, zero, zero, chi_a, chi_b], axis=-1)


def reference_omegas(data, q, chi_a, chi_b):
    """The orbital-frequency fits at every dynamics node (``(B, n_ds)``) for
    the reference state."""
    chi_a, chi_b, _, _ = _coorbital_spins(reference_state(chi_a, chi_b))
    return monomials_at(q, chi_a, chi_b, data.monomials) @ data.omega_fits


def reference_time(data, omegas, omega_ref):
    """``PrecessingNRSur_get_t_ref`` (see
    :func:`.._dynamics.reference_time`): ``(t_ref, valid)``, each ``(B,)``."""
    above = omegas[:, 1:] > omega_ref[:, None]
    found = mx.any(above, axis=1)
    imax = index(1 + mx.argmax(above, axis=1).astype(mx.int32))
    omega_min = mx.take_along_axis(omegas, (imax - 1)[:, None], axis=1)[:, 0]
    omega_max = mx.take_along_axis(omegas, imax[:, None], axis=1)[:, 0]
    t_min = mx.take(data.t_ds, imax - 1)
    t_max = mx.take(data.t_ds, imax)
    t_ref = (t_min * (omega_max - omega_ref) + t_max * (omega_ref - omega_min)) / (omega_max - omega_min)
    start = mx.abs(omega_ref) < OMEGA_REF_ZERO
    valid = start | (found & (omega_ref <= OMEGA_REF_MAX) & (omega_ref >= omegas[:, 0]))
    return mx.where(start, START_TIME, t_ref), valid


def _initialize_at_node(data, q, t_ref, y_ref, norm_a, norm_b):
    """``PrecessingNRSur_initialize_at_dynamics_node``: an Euler step from
    ``t_ref`` to the nearest node ``i0``."""
    t = data.t_ds
    n = data.n_ds
    imin = index(mx.clip(mx.searchsorted(t, t_ref, side="right").astype(mx.int32) - 1, 0, n - 2))
    tmin = mx.take(t, imin)
    tmax = mx.take(t, imin + 1)
    lower = mx.abs(t_ref - tmin) < mx.abs(tmax - t_ref)
    i0 = index(mx.where(lower, imin, imin + 1))
    dt = mx.where(lower, tmin - t_ref, tmax - t_ref)
    dydt = derivative_at_time(data, q, t_ref, y_ref)
    return i0, normalize_y(y_ref + dt[:, None] * dydt, norm_a, norm_b)


def _rk4_start(data, q, i0, y0, norm_a, norm_b):
    """Three RK4 steps from node ``i0`` (see :func:`.._dynamics._rk4_start`):
    the first node ``i_start`` of the four known nodes, their states and
    derivatives (``(B, 4, 11)``, in increasing order of the nodes)."""
    t = data.t_ds
    n = data.n_ds
    half_nodes = (i0 == 0)[:, None]
    sign = mx.where(i0 > 2, -1, 1).astype(mx.int32)
    y = y0
    ys, k1s = [], []
    for k in range(3):
        node = i0 + sign * k
        following = node + sign
        t_node = mx.take(t, node)
        t_following = mx.take(t, following)
        hs = (t_following - t_node)[:, None]
        t_half = 0.5 * (t_node + t_following)
        k1 = derivatives(data, q, node, y)

        def half(y_mid):
            # at the half node (from node 0) or interpolated in time
            interpolated, at_half_node = derivative_at_time(data, q, t_half, y_mid, extra=n + k)
            return mx.where(half_nodes, at_half_node, interpolated)

        k2 = half(y + (0.5 * hs) * k1)
        k3 = half(y + (0.5 * hs) * k2)
        k4 = derivatives(data, q, following, y + hs * k3)
        y = normalize_y(y + (hs / 6.0) * (((k1 + 2.0 * k2) + 2.0 * k3) + k4), norm_a, norm_b)
        ys.append(y)
        k1s.append(k1)
    d3 = derivatives(data, q, i0 + 3 * sign, ys[2])
    states = mx.stack([y0, ys[0], ys[1], ys[2]], axis=1)
    derivs = mx.stack([k1s[0], k1s[1], k1s[2], d3], axis=1)
    forwards = sign > 0
    states = mx.where(forwards[:, None, None], states, states[:, ::-1])
    derivs = mx.where(forwards[:, None, None], derivs, derivs[:, ::-1])
    i_start = mx.where(forwards, i0, i0 - 3)
    return i_start, states, derivs


def _ab4(data, q, i_start, states, derivs, norm_a, norm_b):
    """``PrecessingNRSur_integrate_AB4`` (see :func:`.._dynamics._ab4`):
    Adams-Bashforth steps backwards from node ``i_start`` to node 0, then
    forwards from node ``i_start + 3`` to the last node, ``n - 4`` steps in
    all. Returns the states at all nodes, ``(B, n_ds, 11)``."""
    n = data.n_ds
    forward, backward = data.ab4_forward, data.ab4_backward
    # the derivatives at the four nodes before the next step of each
    # direction, in increasing order of the nodes
    h_forward = h_backward = derivs
    y = states[:, 0]
    ys = []
    for j in range(n - 4):
        back = j < i_start
        back3 = back[:, None, None]
        node = mx.where(back, i_start - 1 - j, j + 4)
        # the first forward step continues from the RK4 nodes
        y = mx.where((i_start == j)[:, None], states[:, 3], y)
        weights = mx.where(back[:, None], mx.take(backward, node, axis=0), mx.take(forward, node, axis=0))
        h = mx.where(back3, h_backward, h_forward)
        y = normalize_y(y + (weights[:, None, :] @ h)[:, 0], norm_a, norm_b)
        k = derivatives(data, q, node, y)[:, None]
        h = mx.where(back3, mx.concatenate([k, h[:, :3]], axis=1), mx.concatenate([h[:, 1:], k], axis=1))
        h_backward = mx.where(back3, h, h_backward)
        h_forward = mx.where(back3, h_forward, h)
        ys.append(y)
    ys = mx.stack(ys, axis=1)
    # node i is reached by the step i_start - 1 - i (backwards) or i - 4
    # (forwards), or is one of the four RK4 nodes
    i = mx.arange(n, dtype=mx.int32)[None]
    start = i_start[:, None]
    before = i < start
    from_rk4 = (i >= start) & (i < start + 4)
    shape = (ys.shape[0], n, ys.shape[-1])
    step = index(mx.clip(mx.where(before, start - 1 - i, i - 4), 0, n - 5))
    stepped = mx.take_along_axis(ys, mx.broadcast_to(step[..., None], shape), axis=1)
    rk4 = mx.take_along_axis(states, mx.broadcast_to(index(mx.clip(i - start, 0, 3))[..., None], shape), axis=1)
    return mx.where(from_rk4[..., None], rk4, stepped)


def integrate(data, q, chi_a, chi_b, omega_ref, omegas=None):
    """The ODE states at every dynamics node (``(B, n_ds, 11)``) and whether
    each evaluation is valid (``(B,)``), as
    :func:`.._dynamics.integrate`."""
    if omegas is None:
        omegas = reference_omegas(data, q, chi_a, chi_b)
    norm_a = safe_norm(chi_a)
    norm_b = safe_norm(chi_b)
    t_ref, valid = reference_time(data, omegas, omega_ref)
    y_ref = reference_state(chi_a, chi_b)
    i0, y0 = _initialize_at_node(data, q, t_ref, y_ref, norm_a, norm_b)
    i_start, states, derivs = _rk4_start(data, q, i0, y0, norm_a, norm_b)
    return _ab4(data, q, i_start, states, derivs, norm_a, norm_b), valid
