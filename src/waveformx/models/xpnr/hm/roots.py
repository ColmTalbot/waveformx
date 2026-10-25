# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The zero-crossing tests of the intermediate amplitude of IMRPhenomXHM
(release 122019): ``CrossZeroP2`` ... ``CrossZeroP5`` of
``LALSimIMRPhenomXHM_intermediate.c``, after waveformrs's
``imrphenomx/hm_amp_122019.rs`` and ``imrphenomx/poly_roots.rs``.

They only decide the order of the intermediate polynomial (a discrete
choice), so their inputs are wrapped in ``jax.lax.stop_gradient`` by the
callers and they return booleans. ``CrossZeroP3`` and ``CrossZeroP4`` use
the closed-form roots in complex arithmetic as LAL does; for
``CrossZeroP5``, where LAL takes the roots of GSL's
``gsl_poly_complex_solve`` (and reads them with shifted indices, so that its
test depends on the order in which GSL finds them), :func:`changes_sign`
tests the signs of the quintic.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from .._generated import hm_amp_122019 as gen
from .._math import fma, real

__all__ = ["cross_zero_p2", "cross_zero_p3", "cross_zero_p4", "changes_sign"]

# pow(10., -15): imaginary parts below this make a root real
ROOT_THRESHOLD = 1e-15

def _inside(ftest, fmin, fmax):
    """``InsideInterval``."""
    return (ftest >= fmin) & (ftest <= fmax)


def cross_zero_p2(a, fstart, fend):
    """``CrossZeroP2``: whether ``a0 + a1 f + a2 f^2`` vanishes in
    ``[fstart, fend]``."""
    a0, a1, a2 = a[0], a[1], a[2]
    discriminant = gen.p2_discriminant(a0, a1, a2)
    root = jnp.sqrt(jnp.where(discriminant < 0.0, 0.0, discriminant))
    f1 = (-a1 - root) / (2.0 * a2)
    f2 = (-a1 + root) / (2.0 * a2)
    return (discriminant >= 0.0) & (_inside(f1, fstart, fend) | _inside(f2, fstart, fend))


def _complex(re, im):
    return lax.complex(jnp.asarray(re, real()), jnp.asarray(im, real()))


def _cpow_one_third(z):
    """``cpow(z, 1/3)`` as ``cexp(1/3 clog(z))``."""
    return jnp.exp(jnp.log(z) * (1.0 / 3.0))


def _real_root_inside(f, fstart, fend):
    return (jnp.abs(jnp.imag(f)) < ROOT_THRESHOLD) & _inside(jnp.real(f), fstart, fend)


def cross_zero_p3(a, fstart, fend):
    """``CrossZeroP3``: whether ``a0 + ... + a3 f^3`` vanishes in
    ``[fstart, fend]``, from Cardano's formula in complex arithmetic (a root
    is real when its imaginary part is below 1e-15)."""
    a0, a1, a2, a3 = a[0], a[1], a[2], a[3]
    q1 = -a2 / (3.0 * a3)
    q2 = gen.p3_q2(a1, a2, a3)
    q3 = gen.p3_q3(a0, a1, a2, a3)
    discri = gen.p3_discri(q2, q3)
    root = jnp.sqrt(_complex(discri, 0.0))
    onethirdpower = _cpow_one_third(_complex(q3 + jnp.real(root), jnp.imag(root)))
    twothird = 2.0 ** (1.0 / 3.0)
    sqrt3 = 3.0 ** 0.5
    z1 = _complex(1.0, sqrt3)
    z2 = _complex(1.0, -sqrt3)
    f1 = q1 - q2 * twothird / (3.0 * a3 * onethirdpower) + onethirdpower / (3.0 * a3 * twothird)
    den = 3.0 * twothird * twothird * a3 * onethirdpower
    f2 = q1 + q2 * z1 / den - onethirdpower * z2 / (6.0 * twothird * a3)
    f3 = q1 + q2 * z2 / den - onethirdpower * z1 / (6.0 * twothird * a3)
    return (_real_root_inside(f1, fstart, fend) | _real_root_inside(f2, fstart, fend)
            | _real_root_inside(f3, fstart, fend))


def cross_zero_p4(a, fstart, fend):
    """``CrossZeroP4``: whether ``a0 + ... + a4 f^4`` vanishes in
    ``[fstart, fend]``, from Ferrari's formula in complex arithmetic.

    Like LAL, ``sqrt(q1 + frac1 + frac2)`` in the denominator is the real
    square root of the real part (C's ``sqrt`` of a complex argument)."""
    a0, a1, a2, a3, a4 = a[0], a[1], a[2], a[3], a[4]
    q0 = -a3 / (4.0 * a4)
    q1 = gen.p4_q1(a2, a3, a4)
    q1b = 2.0 * q1
    q2 = gen.p4_q2(a0, a1, a2, a3, a4)
    q3 = gen.p4_q3(a0, a1, a2, a3, a4)
    squareroot = jnp.sqrt(_complex(gen.p4_sqrt_arg(q2, q3), 0.0))
    onethird = _cpow_one_third(_complex(q3 + jnp.real(squareroot), jnp.imag(squareroot)))
    twothird = 2.0 ** (1.0 / 3.0)
    frac1 = twothird * q2 / (3.0 * a4 * onethird)
    frac2 = onethird / (3.0 * twothird * a4)
    s = q1 + frac1 + frac2
    bigdenom = 4.0 * jnp.sqrt(jnp.real(s))
    bignum = gen.p4_bignum(a1, a2, a3, a4)
    sqrt_s = jnp.sqrt(s)
    ratio = bignum / bigdenom
    base = q1b - frac1 - frac2
    minus = jnp.sqrt(base - ratio)
    plus = jnp.sqrt(base + ratio)
    left = q0 - 0.5 * sqrt_s
    right = q0 + 0.5 * sqrt_s
    roots = (left - 0.5 * minus, left + 0.5 * minus, right - 0.5 * plus, right + 0.5 * plus)
    result = False
    for f in roots:
        result = result | _real_root_inside(f, fstart, fend)
    return result


# --------------------------------------------------------------------------
# CrossZeroP5
# --------------------------------------------------------------------------

#: The points at which :func:`changes_sign` evaluates the polynomial.
SIGN_POINTS = 257


def changes_sign(a, fstart, fend):
    """Whether ``a0 + a1 f + ... + a5 f^5`` vanishes in ``[fstart, fend]``,
    from its signs at :data:`SIGN_POINTS` equally spaced points
    (``CrossZeroP5``, without GSL's root finder): the same answer as the
    roots except for pairs of roots closer than the spacing (a polynomial
    that barely touches zero, where the roots' answer depends on rounding
    too)."""
    x = jnp.linspace(0.0, 1.0, SIGN_POINTS, dtype=real())
    f = fstart + (fend - fstart) * x
    value = a[5]
    for k in range(4, -1, -1):
        value = fma(value, f, a[k])
    positive = value > 0.0
    return jnp.any(positive) & jnp.any(~positive)
