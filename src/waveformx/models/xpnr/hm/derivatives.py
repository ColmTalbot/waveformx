# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The derivatives in frequency that fix the intermediate regions of the
higher modes, exact.

LAL fixes the intermediate amplitude with derivatives of the inspiral and
ringdown amplitudes taken by fourth-order finite differences with a step of
1e-9 (``IMRPhenomXHM_Inspiral_Amp_NDAnsatz``, ``IMRPhenomXHM_RD_Amp_NDAnsatz``)
and the intermediate phase of the (3,2) mode with the first and second
derivatives of the rotated ringdown phase by central differences with a
step of 1e-7. Here they are the exact derivatives of the ansatzes (forward
mode): LAL's finite differences of the second derivative of the (3,2) phase
change by ~1e-4 for a change of the binary in the last bit, which makes the
(3,2) mode noisy at that level, and their derivatives with respect to the
binary would amplify rounding errors by up to ~1e14.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._math import PI, real

__all__ = ["value_and_derivative", "phase_derivatives"]


def value_and_derivative(f, at):
    """``at(f)`` and its derivative at ``f``; ``at`` maps an array of
    frequencies to values."""
    f = jnp.asarray(f, real())
    return jax.jvp(lambda x: at(x[None])[0], (f,), (jnp.ones_like(f),))


def phase_derivatives(fcut, phase_of):
    """The phase, its first and its second derivative at ``fcut`` as
    ``IMRPhenomXHM_GetPhaseCoefficients`` needs them for the (3,2) ringdown:
    ``phase_of`` (an array of frequencies to phases in ``(-pi, pi]``, the
    ``atan2`` of the rotated ringdown) at ``fcut``, reduced modulo ``2 pi``
    to ``(-2 pi, 0]``, and its exact derivatives."""
    fcut = jnp.asarray(fcut, real())

    def phase(x):
        return phase_of(x[None])[0]

    def derivative(x):
        return jax.jvp(phase, (x,), (jnp.ones_like(x),))[1]

    value = jnp.fmod(phase(fcut), 2.0 * PI)
    value = jnp.where(value > 0.0, value - 2.0 * PI, value)
    return value, derivative(fcut), jax.jvp(derivative, (fcut,), (jnp.ones_like(fcut),))[1]
