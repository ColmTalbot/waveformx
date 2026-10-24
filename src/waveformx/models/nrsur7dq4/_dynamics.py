# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The dynamics surrogate (``PrecessingNRSur_IntegrateDynamics`` and the
functions it calls).

The ODE state is ``y = [q0, qx, qy, qz, orbphase, chiAx, chiAy, chiAz,
chiBx, chiBy, chiBz]``: the quaternion of the coprecessing frame, the
orbital phase and the spins in the coprecessing frame. Its time derivative
is given by fits at the (non-uniform) dynamics nodes. As in LAL, the state
at the reference frequency is stepped to the nearest node, three RK4 steps
are taken from there (forwards with the fits at the half nodes when that
node is the first, forwards with time-interpolated derivatives when it is
the second or third, and backwards otherwise), and fourth-order
Adams-Bashforth steps then cover the remaining nodes forwards and
backwards.

Everything here is a pure JAX function of fixed shape. The node where the
integration starts depends on the binary, so the integration is one
``lax.scan`` of ``n_ds - 4`` Adams-Bashforth steps that first walks
backwards from the start (while ``j < i_start``) and then forwards, keeping
the last state and derivatives of both directions in its carry: every step
evaluates the fits of one node, as LAL does, whatever the start. The
Adams-Bashforth step to a node uses the four nodes before it (in the
direction of the integration), so its weights depend only on the node and
are tabulated once (:func:`ab4_tables`). The regime choices (the RK4 start,
the reference at the start of the surrogate) are ``jnp.where`` selections,
and the discrete choices (the nearest node, the bracketing nodes of the
reference frequency) only select indices, so gradients flow through the
continuous parts: the reference time, the Euler step to the node and the
integration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ._fits import monomials_at
from ._spline import polynomial4

__all__ = ["N_STATE", "START_TIME", "OMEGA_REF_MAX", "reference_omegas", "reference_time", "integrate", "safe_norm"]

#: Length of the ODE state.
N_STATE = 11
#: The first dynamics time, nudged inside the first node
#: (``NRSUR7DQ4_START_TIME``).
START_TIME = -4299.99999999
#: ``omega_ref`` above which ``PrecessingNRSur_get_t_ref`` refuses.
OMEGA_REF_MAX = 0.201
#: ``|omega_ref|`` below which ``PrecessingNRSur_get_t_ref`` starts at
#: :data:`START_TIME`.
OMEGA_REF_ZERO = 1.0e-10


def safe_norm(v, axis=-1):
    """``|v|`` with a zero (not NaN) derivative at ``v = 0``."""
    square = jnp.sum(v * v, axis=axis)
    positive = square > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.0)), 0.0)


# Index tables that write the small per-state operations below as a few
# vector operations (XLA fuses them into one kernel per step).
# The groups of the state: quaternion, phase, chiA, chiB.
_GROUP = np.array([0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3])
# The pairs of components rotated by the orbital phase in a fit vector
# (Omega_x, Omega_y), (chiA_dot x, y), (chiB_dot x, y), and in a state
# (chiA x, y), (chiB x, y).
_FIT_PARTNER = np.array([0, 2, 1, 4, 3, 5, 7, 6, 8])
_FIT_FIRST = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
_FIT_SECOND = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
_STATE_PARTNER = np.array([0, 1, 2, 3, 4, 6, 5, 7, 9, 8, 10])
_STATE_FIRST = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
_STATE_SECOND = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], dtype=bool)
# dydt[i] = A[i] y[IA[i]] r[RA[i]] + B[i] y[IB[i]] r[RB[i]] with the rotated
# fits r and the state y extended by a one (index 11)
_A = np.array([-0.5, -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
_IA = np.array([1, 3, 3, 1, 11, 11, 11, 11, 11, 11, 11])
_RA = np.array([1, 2, 1, 2, 0, 3, 4, 5, 6, 7, 8])
_B = np.array([-0.5, 0.5, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_IB = np.array([2, 0, 0, 2, 11, 11, 11, 11, 11, 11, 11])
_RB = np.array([2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0])


def _rotate(v, partner, first, second, sp, cp, sign):
    """``v`` with the pairs ``(first, second)`` rotated: ``x cp - sign y
    sp``, ``sign x sp + y cp`` (``sign = 1`` from the coorbital to the
    coprecessing frame, ``-1`` back)."""
    c = jnp.where(first | second, cp, 1.0)
    s = jnp.where(first, -sign * sp, jnp.where(second, sign * sp, 0.0))
    return v * c + v[..., partner] * s


def normalize_y(y, norm_a, norm_b):
    """``PrecessingNRSur_normalize_y``: the quaternion to unit length and
    the spins to their magnitudes at the reference point (zero where a spin
    vanishes, where LAL divides by zero)."""
    squares = jax.ops.segment_sum(y * y, _GROUP, num_segments=4, indices_are_sorted=True)
    positive = squares > 0
    norms = jnp.sqrt(jnp.where(positive, squares, 1.0))
    target = jnp.stack([jnp.ones_like(norm_a), jnp.ones_like(norm_a), norm_a, norm_b])
    scale = jnp.where(positive, target / norms, 0.0)
    scale = scale.at[1].set(1.0)
    return y * scale[_GROUP]


def _coorbital_spins(y):
    """The spins of the state in the coorbital frame
    (``PrecessingNRSur_ds_fit_x``) and ``sin``, ``cos`` of the orbital
    phase."""
    sp = jnp.sin(y[4])
    cp = jnp.cos(y[4])
    x = _rotate(y, _STATE_PARTNER, _STATE_FIRST, _STATE_SECOND, sp, cp, -1.0)
    return x[5:8], x[8:11], sp, cp


def _assemble(y, fits, sp, cp):
    """``PrecessingNRSur_assemble_dydt`` from the nine fits ``(omega,
    Omega_x, Omega_y, chiA_dot (3), chiB_dot (3))`` (last axis): the
    coorbital components rotated to the coprecessing frame, and the
    quaternion derivative ``q Omega / 2``."""
    r = _rotate(fits, _FIT_PARTNER, _FIT_FIRST, _FIT_SECOND, sp, cp, 1.0)
    ye = jnp.concatenate([y, jnp.ones_like(y[:1])])
    a, b = jnp.asarray(_A, y.dtype), jnp.asarray(_B, y.dtype)
    return a * ye[_IA] * r[..., _RA] + b * ye[_IB] * r[..., _RB]


def derivatives(data, q, nodes, y):
    """``dy/dt`` at the state ``y`` with the fits of the dynamics nodes
    ``nodes`` (an index or an array of indices into ``data.dynamics``,
    whose last three entries are the half nodes)
    (``PrecessingNRSur_get_time_deriv_from_index``)."""
    chi_a, chi_b, sp, cp = _coorbital_spins(y)
    mono = monomials_at(q, chi_a, chi_b, data.monomials)
    fits = data.dynamics[nodes] @ mono
    return _assemble(y, fits, sp, cp)


def derivative_at_time(data, q, t, y, extra=None):
    """``PrecessingNRSur_get_time_deriv``: ``dy/dt`` at time ``t`` from the
    cubic through the derivatives at four nodes around ``t``. With
    ``extra`` (a node index), also the derivative with the fits of that
    node, sharing the evaluation of the monomials."""
    n = data.t_ds.shape[0]
    i1 = jnp.searchsorted(data.t_ds[1:], t, side="left")
    i0 = jnp.clip(i1 - 1, 0, n - 4)
    nodes = i0 + jnp.arange(4)
    if extra is not None:
        nodes = jnp.concatenate([nodes, jnp.reshape(extra, (1,))])
    d = derivatives(data, q, nodes, y)
    value = polynomial4(data.t_ds[nodes[:4]], d[:4], t)
    if extra is None:
        return value
    return value, d[4]


def reference_omegas(data, q, chi_a, chi_b):
    """The orbital-frequency fits at every dynamics node for the reference
    state (identity quaternion, zero phase, spins ``chi_a``, ``chi_b``):
    ``PrecessingNRSur_get_omega`` as used by ``get_t_ref`` and
    ``PrecessingNRSur_StartFrequency``."""
    n = data.t_ds.shape[0]
    y0 = reference_state(chi_a, chi_b)
    chi_a, chi_b, _, _ = _coorbital_spins(y0)
    mono = monomials_at(q, chi_a, chi_b, data.monomials)
    return data.dynamics[:n, 0, :] @ mono


def reference_time(data, omegas, omega_ref):
    """``PrecessingNRSur_get_t_ref``: the time at which the orbital
    frequency fit (``omegas``, from :func:`reference_omegas`) equals
    ``omega_ref``, by linear interpolation between the first node above it
    and the one before. Returns ``(t_ref, valid)``; ``valid`` is false where
    LAL fails (``omega_ref`` above 0.201, below the first node, or above
    all nodes). ``|omega_ref| < 1e-10`` selects :data:`START_TIME`."""
    above = omegas[1:] > omega_ref
    found = jnp.any(above)
    imax = 1 + jnp.argmax(above)
    omega_min = omegas[imax - 1]
    omega_max = omegas[imax]
    t_min = data.t_ds[imax - 1]
    t_max = data.t_ds[imax]
    t_ref = (t_min * (omega_max - omega_ref) + t_max * (omega_ref - omega_min)) / (omega_max - omega_min)
    start = jnp.abs(omega_ref) < OMEGA_REF_ZERO
    valid = start | (found & (omega_ref <= OMEGA_REF_MAX) & (omega_ref >= omegas[0]))
    return jnp.where(start, START_TIME, t_ref), valid


def reference_state(chi_a, chi_b):
    """The ODE state at the reference point in LAL's conventions: identity
    quaternion, zero orbital phase, the spins ``chi_a``, ``chi_b``."""
    zero = jnp.zeros_like(chi_a[..., :1])
    return jnp.concatenate([zero + 1.0, zero, zero, zero, zero, chi_a, chi_b], axis=-1)


def _initialize_at_node(data, q, t_ref, y_ref, norm_a, norm_b):
    """``PrecessingNRSur_initialize_at_dynamics_node``: an Euler step from
    ``t_ref`` to the nearest node ``i0``."""
    t = data.t_ds
    n = t.shape[0]
    imin = jnp.clip(jnp.searchsorted(t, t_ref, side="right") - 1, 0, n - 2)
    tmin = t[imin]
    tmax = t[imin + 1]
    lower = jnp.abs(t_ref - tmin) < jnp.abs(tmax - t_ref)
    i0 = jnp.where(lower, imin, imin + 1)
    dt = jnp.where(lower, tmin - t_ref, tmax - t_ref)
    dydt = derivative_at_time(data, q, t_ref, y_ref)
    return i0, normalize_y(y_ref + dt * dydt, norm_a, norm_b)


def _rk4_start(data, q, i0, y0, norm_a, norm_b):
    """Three RK4 steps from node ``i0`` (``PrecessingNRSur_initialize_RK4``
    and ``PrecessingNRSur_initialize_RK4_with_half_nodes``).

    Returns the first node ``i_start`` of the four consecutive nodes now
    known, their states and the derivatives there (in increasing order of
    the nodes)."""
    t = data.t_ds
    n = t.shape[0]
    half_nodes = i0 == 0
    sign = jnp.where(i0 > 2, -1, 1)

    def step(y, k):
        node = i0 + sign * k
        following = node + sign
        hs = t[following] - t[node]
        t_half = 0.5 * (t[node] + t[following])
        k1 = derivatives(data, q, node, y)

        def half(y_mid):
            # at the half node (from node 0) or interpolated in time
            interpolated, at_half_node = derivative_at_time(data, q, t_half, y_mid, extra=n + k)
            return jnp.where(half_nodes, at_half_node, interpolated)

        k2 = half(y + (0.5 * hs) * k1)
        k3 = half(y + (0.5 * hs) * k2)
        k4 = derivatives(data, q, following, y + hs * k3)
        following_y = y + (hs / 6.0) * (((k1 + 2.0 * k2) + 2.0 * k3) + k4)
        following_y = normalize_y(following_y, norm_a, norm_b)
        return following_y, (following_y, k1)

    _, (ys, k1s) = lax.scan(step, y0, jnp.arange(3))
    d3 = derivatives(data, q, i0 + 3 * sign, ys[2])
    states = jnp.stack([y0, ys[0], ys[1], ys[2]])
    derivs = jnp.stack([k1s[0], k1s[1], k1s[2], d3])
    forwards = sign > 0
    states = jnp.where(forwards, states, states[::-1])
    derivs = jnp.where(forwards, derivs, derivs[::-1])
    i_start = jnp.where(forwards, i0, i0 - 3)
    return i_start, states, derivs


def _ab4_weights(t12, t23, t34, t45):
    """The weights of ``k1..k4`` in ``NRSur_ab4_dy``, the 4th-order
    Adams-Bashforth increment from the derivatives at four nodes separated
    by ``t12, t23, t34`` to the next node at ``t45`` (``dy = sum_j w_j
    k_j``; LAL sums the terms of each power of ``t45`` first, which differs
    by rounding)."""
    t13 = t12 + t23
    t14 = t13 + t34
    t24 = t23 + t34
    denom1 = t12 * t13 * t14
    denom2 = t12 * t23 * t24
    denom3 = t13 * t23 * t34
    dx0 = t45
    dx1 = t45 * t45 / 2.0
    dx2 = t45 * t45 * t45 / 3.0
    dx3 = dx1 * dx1
    a1 = -1.0 * t24 * t34 / denom1
    a2 = t14 * t34 / denom2
    a3 = -1.0 * t14 * t24 / denom3
    a4 = -(a1 + a2 + a3)
    b1 = -1.0 * (t24 + t34) / denom1
    b2 = (t14 + t34) / denom2
    b3 = -1.0 * (t14 + t24) / denom3
    b4 = -(b1 + b2 + b3)
    c1 = -1.0 / denom1
    c2 = 1.0 / denom2
    c3 = -1.0 / denom3
    c4 = -(c1 + c2 + c3)
    return jnp.stack(
        [
            a1 * dx1 + b1 * dx2 + c1 * dx3,
            a2 * dx1 + b2 * dx2 + c2 * dx3,
            a3 * dx1 + b3 * dx2 + c3 * dx3,
            dx0 + a4 * dx1 + b4 * dx2 + c4 * dx3,
        ],
        axis=-1,
    )


def ab4_tables(t):
    """The Adams-Bashforth weights of the steps to every node, ``(n, 4)``
    each, applied to the derivatives at the four nodes before it in
    increasing order of the nodes: forwards to node ``i`` from nodes ``i -
    4 .. i - 1`` and backwards to node ``i`` from nodes ``i + 1 .. i + 4``.
    The history of a step depends only on its node, so the weights do not
    depend on where the integration starts. Rows without four nodes before
    them are unused."""
    n = t.shape[0]
    pad = jnp.full((4,), jnp.nan, dtype=t.dtype)
    # forwards: k1..k4 at i-4..i-1, steps t[i-3]-t[i-4], ..., t[i]-t[i-1]
    tp = jnp.concatenate([pad, t])
    i = jnp.arange(n) + 4
    forward = _ab4_weights(tp[i - 3] - tp[i - 4], tp[i - 2] - tp[i - 3], tp[i - 1] - tp[i - 2], tp[i] - tp[i - 1])
    # backwards: k1..k4 at i+4..i+1, steps t[i+3]-t[i+4], ..., t[i]-t[i+1]
    tp = jnp.concatenate([t, pad])
    i = jnp.arange(n)
    backward = _ab4_weights(tp[i + 3] - tp[i + 4], tp[i + 2] - tp[i + 3], tp[i + 1] - tp[i + 2], tp[i] - tp[i + 1])
    return forward, backward[:, ::-1]


def _ab4(data, q, i_start, states, derivs, norm_a, norm_b):
    """``PrecessingNRSur_integrate_AB4``: Adams-Bashforth steps backwards
    from node ``i_start`` to node 0, then forwards from node ``i_start +
    3`` to the last node, in one ``lax.scan`` of ``n - 4`` steps. The carry
    holds the last state and the derivatives at the four nodes before the
    next step, for each direction (index 0 forwards, 1 backwards, in
    increasing order of the nodes). Returns the states at all nodes."""
    t = data.t_ds
    n = t.shape[0]
    forward, backward = ab4_tables(t)
    history = jnp.stack([derivs, derivs])

    def step(carry, j):
        y, history = carry
        back = j < i_start
        direction = back.astype(jnp.int32)
        node = jnp.where(back, i_start - 1 - j, j + 4)
        # the first forward step continues from the RK4 nodes
        y = jnp.where(j == i_start, states[3], y)
        weights = jnp.where(back, backward[node], forward[node])
        h = history[direction]
        y = normalize_y(y + weights @ h, norm_a, norm_b)
        k = derivatives(data, q, node, y)
        h = jnp.where(back, jnp.concatenate([k[None], h[:3]]), jnp.concatenate([h[1:], k[None]]))
        return (y, history.at[direction].set(h)), (node, y)

    _, (nodes, ys) = lax.scan(step, (states[0], history), jnp.arange(n - 4))
    out = jnp.zeros((n, states.shape[-1]), dtype=states.dtype).at[i_start + jnp.arange(4)].set(states)
    return out.at[nodes].set(ys)


def integrate(data, q, chi_a, chi_b, omega_ref, omegas=None):
    """The ODE state at every dynamics node (``(n_ds, 11)``) for mass ratio
    ``q >= 1`` and spins ``chi_a``, ``chi_b`` given in the coorbital frame
    at the dimensionless orbital frequency ``omega_ref``, where the
    quaternion is the identity and the orbital phase zero (LAL's waveform
    conventions). ``omegas`` are the :func:`reference_omegas` (computed if
    not given). Returns ``(states, valid)``."""
    if omegas is None:
        omegas = reference_omegas(data, q, chi_a, chi_b)
    norm_a = safe_norm(chi_a)
    norm_b = safe_norm(chi_b)
    t_ref, valid = reference_time(data, omegas, omega_ref)
    y_ref = reference_state(chi_a, chi_b)
    i0, y0 = _initialize_at_node(data, q, t_ref, y_ref, norm_a, norm_b)
    i_start, states, derivs = _rk4_start(data, q, i0, y0, norm_a, norm_b)
    return _ab4(data, q, i_start, states, derivs, norm_a, norm_b), valid
