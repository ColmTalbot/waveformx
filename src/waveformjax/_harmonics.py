# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Spin-weighted spherical harmonics (LAL's conventions) and the
polarizations from frequency-domain modes."""

from __future__ import annotations

from math import factorial, sqrt

import jax.numpy as jnp
import numpy as np

__all__ = ["wigner_small_d", "spin_weighted_spherical_harmonic", "polarizations_from_modes"]


def wigner_small_d(ell: int, mp: int, m: int, beta):
    """Wigner's ``d^l_{m' m}(beta)`` (the convention of LAL's
    ``XLALWignerdMatrix``), differentiable in ``beta``."""
    cos_half = jnp.cos(beta / 2)
    sin_half = jnp.sin(beta / 2)
    total = 0.0
    prefactor = sqrt(factorial(ell + mp) * factorial(ell - mp) * factorial(ell + m) * factorial(ell - m))
    for k in range(max(0, m - mp), min(ell + m, ell - mp) + 1):
        coefficient = (-1) ** (k - m + mp) * prefactor / (
            factorial(ell + m - k) * factorial(k) * factorial(ell - k - mp) * factorial(k - m + mp)
        )
        total = total + coefficient * cos_half ** (2 * ell - 2 * k + m - mp) * sin_half ** (2 * k - m + mp)
    return total


def spin_weighted_spherical_harmonic(theta, phi, ell: int, m: int, s: int = -2):
    """``{}_sY_{lm}(theta, phi)`` as ``XLALSpinWeightedSphericalHarmonic``:
    ``(-1)^s sqrt((2l + 1) / 4 pi) d^l_{m,-s}(theta) e^{i m phi}``."""
    norm = (-1) ** s * np.sqrt((2 * ell + 1) / (4 * np.pi))
    return norm * wigner_small_d(ell, m, -s, theta) * jnp.exp(1j * m * phi)


def polarizations_from_modes(modes):
    """``(h_+, h_x)`` on the non-negative frequencies of a
    :class:`~waveformjax.FrequencyModes` in LAL's two-sided layout.

    With ``h_lm(f)`` and ``h_lm(-f)`` from the two halves of each mode and
    ``Y = {}_{-2}Y_{lm}(theta, phi)``::

        h_+ = 1/2 sum (h_lm(f) Y + conj(h_lm(-f) Y))
        h_x = i/2 sum (h_lm(f) Y - conj(h_lm(-f) Y))

    rotated by twice the polarization angle, as waveformrs's
    ``FrequencyModes::polarizations``.
    """
    frequencies = jnp.asarray(modes.frequencies)
    n2 = frequencies.shape[0]
    n = int(np.sum(np.asarray(frequencies) >= 0))
    hplus = jnp.zeros(n, dtype=jnp.complex128)
    hcross = jnp.zeros(n, dtype=jnp.complex128)
    for (ell, m), h in modes.modes.items():
        h = jnp.asarray(h)
        positive = h[n2 - n :]
        negative = h[:n][::-1]
        y = spin_weighted_spherical_harmonic(modes.theta, modes.phi, ell, m)
        a = positive * y
        b = jnp.conj(negative) * jnp.conj(y)
        hplus = hplus + 0.5 * (a + b)
        hcross = hcross + 0.5j * (a - b)
    zeta = modes.polarization_angle
    cos, sin = jnp.cos(2 * zeta), jnp.sin(2 * zeta)
    return hplus * cos + hcross * sin, hcross * cos - hplus * sin
