# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Rotation of the modes from the coorbital to the inertial frame
(``TransformModesCoorbitalToInertial`` and ``WignerDMatrices_Compute`` in
``LALSimNRSurrogateUtilities.c``).

The coorbital modes are first rotated to the coprecessing frame by the
orbital phase, ``h_copr^{l,m} = exp(-i m phi) h_coorb^{l,m}``, then to the
inertial frame with the Wigner D-matrices of the coprecessing quaternion,
``h^{l,m} = sum_m' D^l_{m,m'} h_copr^{l,m'}``.

LAL writes the D-matrices with ``ra = q0 - i q3``, ``rb = -q2 - i q1`` as::

    D^l_{m,m'} = c ra^(m+m') rb^(m-m') |ra|^(2(l-m))
                 sum_rho c_rho (|rb|^2 / |ra|^2)^rho

with negative powers of ``ra``, ``rb`` where ``m + m' < 0`` or ``m < m'``,
and special values where ``|ra|`` or ``|rb|`` vanish (e.g. for spins along
the orbital angular momentum, where the frame does not precess). The same
sum is evaluated here with the negative powers cancelled against the powers
of ``|ra|^2`` and ``|rb|^2``::

    D^l_{m,m'} = Z(m + m', m - m') sum_rho c c_rho |ra|^(2 alpha) |rb|^(2 beta)

where ``Z(a, b)`` is ``ra^a`` (``conj(ra)^-a`` for ``a < 0``) times ``rb^b``
(``conj(rb)^-b`` for ``b < 0``) and all exponents are non-negative. This is
algebraically identical, a polynomial in the quaternion, so it needs no
special cases and is differentiable everywhere, including the non-precessing
limit ``rb = 0``. (Where ``|rb|^2 < 1e-24`` LAL sets ``D^l_{m,m} = 1``
rather than ``ra^(2m)`` for ``m < 0``; the two differ by ``O(|rb|^2)``.)

The sums are written out element by element in real arithmetic: XLA fuses
them into a few passes over the time samples, without storing the 155
elements of the D-matrices.
"""

from __future__ import annotations

from math import factorial, sqrt
from typing import Dict, List, Tuple

import jax.numpy as jnp

from ._data import LMAX, MODES

__all__ = ["wigner_terms", "coorbital_to_coprecessing", "coprecessing_to_inertial"]


def _binomial(n: int, k: int) -> float:
    """LAL's ``binomial`` (``factorial_ratio(n, k) / factorial(n - k)``)."""
    ratio = 1.0
    for i in range(k + 1, n + 1):
        ratio *= i
    return ratio / factorial(n - k) if n - k >= 0 else ratio


def wigner_terms() -> Dict[Tuple[int, int, int], List[Tuple[float, int, int]]]:
    """For every ``(l, m, m')``: the terms ``(c c_rho, alpha, beta)`` of the
    sum over ``rho`` (``WignerDMatrices_Compute``, ``wigner_coef`` and
    ``binomial``)."""
    terms = {}
    for ell in range(2, LMAX + 1):
        for m in range(-ell, ell + 1):
            for mp in range(-ell, ell + 1):
                norm = sqrt(factorial(ell + m) * factorial(ell - m) / (factorial(ell + mp) * factorial(ell - mp)))
                rho_min = mp - m if mp > m else 0
                rho_max = ell - m if mp > -m else ell + mp
                element = []
                for rho in range(rho_min, rho_max + 1):
                    value = _binomial(ell + mp, rho) * _binomial(ell - mp, ell - rho - m)
                    coef = norm * (-value if rho % 2 == 1 else value)
                    alpha = ell - m - rho if m + mp >= 0 else ell + mp - rho
                    beta = rho if m - mp >= 0 else m - mp + rho
                    element.append((coef, alpha, beta))
                terms[ell, m, mp] = element
    return terms


_TERMS = wigner_terms()


class _Powers:
    """Integer powers of an array, computed when first needed by repeated
    multiplication (as ``RealPowers_Compute``)."""

    def __init__(self, x):
        self.values = [None, x]

    def __getitem__(self, k: int):
        while len(self.values) <= k:
            self.values.append(self.values[-1] * self.values[1])
        return self.values[k]


class _ComplexPowers:
    """``z^k`` for ``k > 0`` and ``conj(z)^-k`` for ``k < 0`` of ``z = x +
    i y``, as ``(re, im)``, computed when first needed (as
    ``ComplexPowers_Compute`` for the positive powers)."""

    def __init__(self, x, y):
        self.values = [None, (x, y)]

    def __getitem__(self, k: int):
        while len(self.values) <= abs(k):
            (a, b), (x, y) = self.values[-1], self.values[1]
            self.values.append((a * x - b * y, a * y + b * x))
        re, im = self.values[abs(k)]
        return (re, im) if k >= 0 else (re, -im)


def coorbital_to_coprecessing(re, im, phase):
    """``h_copr^{l,m} = exp(-i m phi) h_coorb^{l,m}`` with ``cos(m phi)``,
    ``sin(m phi)``; ``re``, ``im`` of shape ``(n_modes, n)``."""
    cos = {k: jnp.cos(k * phase) for k in range(1, LMAX + 1)}
    sin = {k: jnp.sin(k * phase) for k in range(1, LMAX + 1)}
    out_re, out_im = [], []
    for i, (_, m) in enumerate(MODES):
        if m == 0:
            out_re.append(re[i])
            out_im.append(im[i])
            continue
        c = cos[abs(m)]
        s = sin[abs(m)] if m > 0 else -sin[abs(m)]
        out_re.append(re[i] * c + im[i] * s)
        out_im.append(-(re[i] * s) + im[i] * c)
    return jnp.stack(out_re), jnp.stack(out_im)


def coprecessing_to_inertial(re, im, quat):
    """``h^{l,m} = sum_m' D^l_{m,m'}(quat) h_copr^{l,m'}`` for the modes of
    :data:`MODES`: ``re``, ``im`` of shape ``(n_modes, n)``, ``quat`` of
    shape ``(4, n)``; returns the real and imaginary parts."""
    ra = _ComplexPowers(quat[0], -quat[3])
    rb = _ComplexPowers(-quat[2], -quat[1])
    ra_abs = _Powers(quat[0] * quat[0] + quat[3] * quat[3])
    rb_abs = _Powers(quat[2] * quat[2] + quat[1] * quat[1])
    z = {}
    out_re, out_im = [], []
    index = {mode: i for i, mode in enumerate(MODES)}
    for ell in range(2, LMAX + 1):
        for m in range(-ell, ell + 1):
            acc_re = acc_im = None
            for mp in range(-ell, ell + 1):
                key = (m + mp, m - mp)
                if key not in z:
                    if m + mp == 0 and m - mp == 0:
                        z[key] = (1.0, 0.0)
                    elif m + mp == 0:
                        z[key] = rb[m - mp]
                    elif m - mp == 0:
                        z[key] = ra[m + mp]
                    else:
                        (ar, ai), (br, bi) = ra[m + mp], rb[m - mp]
                        z[key] = (ar * br - ai * bi, ar * bi + ai * br)
                zr, zi = z[key]
                total = None
                for coef, alpha, beta in _TERMS[ell, m, mp]:
                    term = coef
                    if alpha:
                        term = term * ra_abs[alpha]
                    if beta:
                        term = term * rb_abs[beta]
                    total = term if total is None else total + term
                d_re = zr * total
                d_im = None if isinstance(zi, float) else zi * total
                h_re, h_im = re[index[ell, mp]], im[index[ell, mp]]
                if d_im is None:
                    term_re = d_re * h_re
                    term_im = d_re * h_im
                else:
                    term_re = d_re * h_re - d_im * h_im
                    term_im = d_re * h_im + d_im * h_re
                acc_re = term_re if acc_re is None else acc_re + term_re
                acc_im = term_im if acc_im is None else acc_im + term_im
            out_re.append(acc_re)
            out_im.append(acc_im)
    return jnp.stack(out_re), jnp.stack(out_im)
