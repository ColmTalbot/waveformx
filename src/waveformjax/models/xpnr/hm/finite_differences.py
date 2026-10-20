# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The numerical derivatives of IMRPhenomXHM, with exact tangents.

LAL fixes the intermediate amplitude with derivatives of the inspiral and
ringdown amplitudes taken by fourth-order finite differences with a step of
1e-9 (``IMRPhenomXHM_Inspiral_Amp_NDAnsatz``, ``IMRPhenomXHM_RD_Amp_NDAnsatz``)
and the intermediate phase of the (3,2) mode with the first and second
derivatives of the rotated ringdown phase by central differences with a
step of 1e-7. The values here are these finite differences, as in LAL. Their
derivatives with respect to the parameters of the binary, however, would be
finite differences of derivatives, whose rounding errors the small steps
amplify (by up to ~1e14 for the second derivative of a phase of ~1e4 rad):
the tangents (``jax.jvp``, and so ``jax.grad``) are instead those of the exact
derivatives, obtained by differentiating the ansatz with respect to the
frequency.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from .._math import PI, fma

__all__ = ["value_and_nd_derivative", "phase_derivatives"]

# step of IMRPhenomXHM_Inspiral_Amp_NDAnsatz / IMRPhenomXHM_RD_Amp_NDAnsatz
ND_STEP = 10e-10
# step of the (3,2) ringdown phase derivatives
PHASE_STEP = 0.0000001


def value_and_nd_derivative(f, at):
    """``at(f)`` and LAL's fourth-order finite difference of ``at`` at ``f``
    with step 1e-9. ``at`` maps an array of frequencies to values (it is
    evaluated on five frequencies at once); the tangents are exact."""
    f = jnp.asarray(f, jnp.float64)
    fun, consts = jax.closure_convert(at, jnp.zeros(5))
    return _nd(fun, f, *consts)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _nd(fun, f, *consts):
    df = ND_STEP
    values = fun(jnp.stack([f, fma(2.0, df, f), f + df, f - df, fma(-2.0, df, f)]), *consts)
    fun2r, funr, funl, fun2l = values[1], values[2], values[3], values[4]
    return values[0], (fma(-8.0, funl, fma(8.0, funr, -fun2r)) + fun2l) / (12.0 * df)


@_nd.defjvp
def _nd_jvp(fun, primals, tangents):
    out = _nd(fun, *primals)

    def exact(f, *consts):
        def value(x):
            return fun(jnp.stack([x] * 5), *consts)[0]

        return jax.jvp(value, (f,), (jnp.ones_like(f),))

    _, tangent = jax.jvp(exact, primals, tangents)
    return out, tangent


def phase_derivatives(fcut, phase_of):
    """The phase, its first and its second derivative at ``fcut`` as
    ``IMRPhenomXHM_GetPhaseCoefficients`` computes them for the (3,2) ringdown:
    ``phase_of`` (an array of frequencies to phases in ``(-pi, pi]``, the
    ``atan2`` of the rotated ringdown) at ``fcut - 1e-7``, ``fcut``, ``fcut +
    1e-7``, reduced modulo ``2 pi`` to ``(-2 pi, 0]``, and central
    differences; the tangents are exact."""
    fcut = jnp.asarray(fcut, jnp.float64)
    fun, consts = jax.closure_convert(phase_of, jnp.zeros(3))
    return _phase(fun, fcut, *consts)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _phase(fun, fcut, *consts):
    ffs = jnp.stack([fma(float(i - 1), PHASE_STEP, fcut) for i in range(3)])
    args = jnp.fmod(fun(ffs, *consts), 2.0 * PI)
    # all the points on the same branch
    args = jnp.where(args > 0.0, args - 2.0 * PI, args)
    fstep_m1 = 1.0 / PHASE_STEP
    dphi = 0.5 * fstep_m1 * (args[2] - args[0])
    d2phi = fstep_m1 * fstep_m1 * (fma(-2.0, args[1], args[2]) + args[0])
    return args[1], dphi, d2phi


@_phase.defjvp
def _phase_jvp(fun, primals, tangents):
    out = _phase(fun, *primals)

    def exact(fcut, *consts):
        def phase(x):
            return fun(jnp.stack([x] * 3), *consts)[0]

        def derivative(x):
            return jax.jvp(phase, (x,), (jnp.ones_like(x),))[1]

        _, second = jax.jvp(derivative, (fcut,), (jnp.ones_like(fcut),))
        return phase(fcut), derivative(fcut), second

    _, tangent = jax.jvp(exact, primals, tangents)
    return out, tangent
