# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The zero-crossing tests of the intermediate amplitude of IMRPhenomXHM
(release 122019): ``CrossZeroP2`` ... ``CrossZeroP5`` of
``LALSimIMRPhenomXHM_intermediate.c``, after waveformrs's
``imrphenomx/hm_amp_122019.rs`` and ``imrphenomx/poly_roots.rs``.

They only decide the order of the intermediate polynomial (a discrete
choice), so their inputs are wrapped in ``jax.lax.stop_gradient`` by the
callers and they return booleans. ``CrossZeroP3`` and ``CrossZeroP4`` use
the closed-form roots in complex arithmetic as LAL does; ``CrossZeroP5``
uses the roots of GSL's ``gsl_poly_complex_solve`` (``poly/zsolve.c``: the
eigenvalues of the balanced companion matrix from the Francis double-shift
QR algorithm), ported step by step (in ``jax.lax.while_loop`` loops ending
where GSL's loops end, with masked iterations under ``jax.vmap``; the row
and column operations of a QR step on whole rows and columns).

Unlike LAL, :func:`cross_zero_p5` tests every root with its own real and
imaginary parts: LAL reads shifted entries of GSL's packed roots, so that
its test depends on the order in which GSL finds the roots (and misses
some roots), which rounding can change.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from .._generated import hm_amp_122019 as gen
from .._math import dbl_epsilon, fma, real

__all__ = ["cross_zero_p2", "cross_zero_p3", "cross_zero_p4", "cross_zero_p5", "complex_solve_quintic"]

# pow(10., -15): imaginary parts below this make a root real
ROOT_THRESHOLD = 1e-15

# the most sweeps of balance_companion_matrix, iterations of the doubling
# loops inside a sweep, and passes of qr_companion (each finds one or two
# roots or makes one QR step); GSL loops until convergence (at most 120 QR
# steps per root)
BALANCE_SWEEPS = 32
BALANCE_DOUBLINGS = 64
QR_STEPS = 100


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
# gsl_poly_complex_solve for a quintic
# --------------------------------------------------------------------------

def _balance(m, nc):
    """``balance_companion_matrix`` (``m`` is the ``nc x nc`` companion
    matrix, only its sub-diagonal and last column are non-zero)."""

    def doublings(col_norm, f, g, up):
        def more(state):
            col_norm, _, count = state
            return ((col_norm < g) if up else (col_norm > g)) & (count < BALANCE_DOUBLINGS)

        def body(state):
            col_norm, f, count = state
            if up:
                return col_norm * 4.0, f * 2.0, count + 1
            return col_norm / 4.0, f / 2.0, count + 1

        col_norm, f, _ = lax.while_loop(more, body, (col_norm, jnp.asarray(f, real()), jnp.asarray(0)))
        return col_norm, f

    def sweep(state):
        m, not_converged, count = state
        changed = jnp.asarray(False)
        for i in range(nc):
            if i != nc - 1:
                col_norm = jnp.abs(m[i + 1, i])
            else:
                col_norm = 0.0
                for j in range(nc - 1):
                    col_norm = col_norm + jnp.abs(m[j, nc - 1])
            if i == 0:
                row_norm = jnp.abs(m[0, nc - 1])
            elif i == nc - 1:
                row_norm = jnp.abs(m[i, i - 1])
            else:
                row_norm = jnp.abs(m[i, i - 1]) + jnp.abs(m[i, nc - 1])
            skip = (col_norm == 0.0) | (row_norm == 0.0)
            s = col_norm + row_norm
            col_norm, f = doublings(col_norm, 1.0, row_norm / 2.0, True)
            col_norm, f = doublings(col_norm, f, row_norm * 2.0, False)
            do = not_converged & ~skip & ((row_norm + col_norm) < 0.95 * s * f)
            changed = changed | do
            g = 1.0 / f
            if i == 0:
                m = m.at[0, nc - 1].set(jnp.where(do, m[0, nc - 1] * g, m[0, nc - 1]))
            else:
                m = m.at[i, i - 1].set(jnp.where(do, m[i, i - 1] * g, m[i, i - 1]))
                m = m.at[i, nc - 1].set(jnp.where(do, m[i, nc - 1] * g, m[i, nc - 1]))
            if i == nc - 1:
                for j in range(nc):
                    m = m.at[j, i].set(jnp.where(do, m[j, i] * f, m[j, i]))
            else:
                m = m.at[i + 1, i].set(jnp.where(do, m[i + 1, i] * f, m[i + 1, i]))
        return m, changed, count + 1

    # (a sweep that changes nothing ends the loop: GSL's "not_converged")
    m, _, _ = lax.while_loop(lambda state: state[1] & (state[2] < BALANCE_SWEEPS), sweep,
                             (m, jnp.asarray(True), jnp.asarray(0)))
    return m


def _qr_companion(h, nc):
    """``qr_companion``: the eigenvalues of the upper Hessenberg matrix
    ``h``, packed as GSL stores them (``[re0, im0, re1, im1, ...]``, each
    root at ``n - 1`` as it converges at the bottom of the active block),
    and whether the iteration converged."""

    def H(h, i, j):
        # 1-based, static indices
        return h[i - 1, j - 1]

    def Hd(h, i, j):
        # 1-based, traced indices (clipped into the matrix)
        return h[jnp.clip(i - 1, 0, nc - 1), jnp.clip(j - 1, 0, nc - 1)]

    def set_root(z, k, re, im, cond):
        k = jnp.clip(k, 0, nc - 1)
        z = z.at[2 * k].set(jnp.where(cond, re, z[2 * k]))
        return z.at[2 * k + 1].set(jnp.where(cond, im, z[2 * k + 1]))

    # 1-based row and column numbers
    index = jnp.arange(1, nc + 1)

    def step(state):
        h, t, n, its, z, failed, count = state
        active = (n > 0) & ~failed

        # look for a small sub-diagonal element (the largest e <= n)
        e = jnp.asarray(1)
        for k in range(2, nc + 1):
            small = jnp.abs(H(h, k, k - 1)) <= dbl_epsilon() * (jnp.abs(H(h, k - 1, k - 1)) + jnp.abs(H(h, k, k)))
            e = jnp.where((k <= n) & small, k, e)
        x = Hd(h, n, n)
        one_root = active & (e == n)
        z = set_root(z, n - 1, x + t, 0.0, one_root)

        y = Hd(h, n - 1, n - 1)
        w = Hd(h, n - 1, n) * Hd(h, n, n - 1)
        two_roots = active & (e == n - 1)
        p = (y - x) / 2.0
        q = fma(p, p, w)
        yy = jnp.sqrt(jnp.abs(q))
        xx = x + t
        real_pair = q > 0.0
        yr = jnp.where(p < 0.0, -yy, yy) + p
        yr_safe = jnp.where(yr == 0.0, 1.0, yr)
        z = set_root(z, n - 1, jnp.where(real_pair, xx - w / yr_safe, xx + p), jnp.where(real_pair, 0.0, -yy), two_roots)
        z = set_root(z, n - 2, jnp.where(real_pair, xx + yr, xx + p), jnp.where(real_pair, 0.0, yy), two_roots)

        iterate = active & ~(e == n) & ~(e == n - 1)
        fail_now = iterate & (its == 120)
        iterate = iterate & ~fail_now
        exceptional = iterate & (its % 10 == 0) & (its > 0)
        t_new = jnp.where(exceptional, t + x, t)
        diag = jnp.arange(1, nc + 1) <= n
        h_shift = h - jnp.diag(jnp.where(diag, x, 0.0))
        h_it = jnp.where(exceptional, h_shift, h)
        s = jnp.abs(Hd(h_it, n, n - 1)) + jnp.abs(Hd(h_it, n - 1, n - 2))
        y_it = jnp.where(exceptional, 0.75 * s, y)
        x_it = jnp.where(exceptional, y_it, x)
        w_it = jnp.where(exceptional, -0.4375 * s * s, w)

        # the start m of the double QR sweep, with its p, q, r
        m = jnp.asarray(1)
        pm, qm, rm = 0.0, 0.0, 0.0
        for m_ in range(1, nc - 1):
            zz = H(h_it, m_, m_)
            r_ = x_it - zz
            s0 = y_it - zz
            p_ = H(h_it, m_, m_ + 1) + fma(r_, s0, -w_it) / H(h_it, m_ + 1, m_)
            q_ = H(h_it, m_ + 1, m_ + 1) - zz - r_ - s0
            r_ = H(h_it, m_ + 2, m_ + 1)
            s_ = jnp.abs(p_) + jnp.abs(q_) + jnp.abs(r_)
            p_, q_, r_ = p_ / s_, q_ / s_, r_ / s_
            if m_ == 1:
                stop = jnp.asarray(True)
            else:
                a1 = jnp.abs(H(h_it, m_, m_ - 1))
                a2 = jnp.abs(H(h_it, m_ - 1, m_ - 1))
                a3 = jnp.abs(H(h_it, m_ + 1, m_ + 1))
                stop = a1 * (jnp.abs(q_) + jnp.abs(r_)) <= dbl_epsilon() * jnp.abs(p_) * (a2 + a3)
            stop = (m_ == e) | stop
            take = (m_ >= e) & (m_ <= n - 2) & stop
            m = jnp.where(take, m_, m)
            pm = jnp.where(take, p_, pm)
            qm = jnp.where(take, q_, qm)
            rm = jnp.where(take, r_, rm)

        hq = h_it
        for i in range(3, nc + 1):
            zero = (i >= m + 2) & (i <= n)
            hq = hq.at[i - 1, i - 3].set(jnp.where(zero, 0.0, hq[i - 1, i - 3]))
        for i in range(4, nc + 1):
            zero = (i >= m + 3) & (i <= n)
            hq = hq.at[i - 1, i - 4].set(jnp.where(zero, 0.0, hq[i - 1, i - 4]))

        # double QR step (p, q, r of the first step come from the search of m,
        # the others from the matrix): a loop over k = 1 .. nc - 1, with the
        # rows and columns k - 1, k, k + 1 (0-based; k + 1 clipped, and
        # masked out, at the last k) read and written at a traced k
        def sweep(k, hq):
            in_sweep = (k >= m) & (k <= n - 1)
            has_third = k + 2 <= nc
            notlast = has_third & (k != n - 1)
            first = k == m
            later = k > 1
            pk = jnp.where(later, Hd(hq, k, k - 1), 0.0)
            qk = jnp.where(later, Hd(hq, k + 1, k - 1), 0.0)
            rk = jnp.where(later & notlast, Hd(hq, k + 2, k - 1), 0.0)
            xk = jnp.abs(pk) + jnp.abs(qk) + jnp.abs(rk)
            xk_safe = jnp.where(xk == 0.0, 1.0, xk)
            p = jnp.where(first, pm, pk / xk_safe)
            q = jnp.where(first, qm, qk / xk_safe)
            r = jnp.where(first, rm, rk / xk_safe)
            do = in_sweep & (first | (xk != 0.0))
            sk = jnp.sqrt(fma(r, r, fma(p, p, q * q)))
            sk = jnp.where(p < 0.0, -sk, sk)
            sub = jnp.where(first, jnp.where(e != m, -pk, pk), -sk * xk)
            row, col = jnp.clip(k - 1, 0, nc - 1), jnp.clip(k - 2, 0, nc - 1)
            hq = hq.at[row, col].set(jnp.where(later & do, sub, hq[row, col]))
            p = p + sk
            sk_safe = jnp.where(sk == 0.0, 1.0, sk)
            p_safe = jnp.where(p == 0.0, 1.0, p)
            xr = p / sk_safe
            yr_ = q / sk_safe
            zr = r / sk_safe
            q = q / p_safe
            r = r / p_safe
            k0, k1, k2 = k - 1, k, jnp.minimum(k + 1, nc - 1)
            # row modifications (each column from its own old values: done
            # for the columns j = k .. n at once)
            on = do & (index >= k) & (index <= n)
            r0, r1, r2 = hq[k0], hq[k1], hq[k2]
            pj = jnp.where(notlast, fma(r, r2, fma(q, r1, r0)), fma(q, r1, r0))
            hq = hq.at[k2].set(jnp.where(on & notlast, fma(-pj, zr, r2), r2))
            hq = hq.at[k1].set(jnp.where(on, fma(-pj, yr_, r1), r1))
            hq = hq.at[k0].set(jnp.where(on, fma(-pj, xr, r0), r0))
            # column modifications (the rows i = e .. min(k + 3, n) at once)
            on = do & (index >= e) & (index <= jnp.minimum(k + 3, n))
            c0, c1, c2 = hq[:, k0], hq[:, k1], hq[:, k2]
            pi_ = jnp.where(notlast, fma(zr, c2, fma(xr, c0, yr_ * c1)), fma(xr, c0, yr_ * c1))
            hq = hq.at[:, k2].set(jnp.where(on & notlast, fma(-pi_, r, c2), c2))
            hq = hq.at[:, k1].set(jnp.where(on, fma(-pi_, q, c1), c1))
            return hq.at[:, k0].set(jnp.where(on, c0 - pi_, c0))

        hq = lax.fori_loop(1, nc, sweep, hq)

        h = jnp.where(iterate, hq, h)
        t = jnp.where(iterate, t_new, t)
        its = jnp.where(iterate, its + 1, jnp.where(one_root | two_roots, 0, its))
        n = n - jnp.where(one_root, 1, 0) - jnp.where(two_roots, 2, 0)
        failed = failed | fail_now
        return h, t, n, its, z, failed, count + 1

    def active(state):
        _, _, n, _, _, failed, count = state
        return (n > 0) & ~failed & (count < QR_STEPS)

    state = (h, jnp.asarray(0.0), jnp.asarray(nc), jnp.asarray(0), jnp.zeros(2 * nc, real()), jnp.asarray(False),
             jnp.asarray(0))
    h, t, n, its, z, failed, _ = lax.while_loop(active, step, state)
    return z, ~failed & (n == 0)


def complex_solve_quintic(a):
    """``gsl_poly_complex_solve`` for ``a[0] + a[1] x + ... + a[5] x^5``:
    the packed roots ``[re0, im0, ..., re4, im4]`` (in GSL's order, but for
    rounding) and whether GSL succeeds (leading coefficient non-zero and the
    QR iteration converging)."""
    a = [jnp.asarray(c, real()) for c in a]
    nc = 5
    lead = jnp.where(a[nc] == 0.0, 1.0, a[nc])
    m = jnp.zeros((nc, nc), real())
    for i in range(1, nc):
        m = m.at[i, i - 1].set(1.0)
    for i in range(nc):
        m = m.at[i, nc - 1].set(-a[i] / lead)
    m = _balance(m, nc)
    z, ok = _qr_companion(m, nc)
    return z, ok & (a[nc] != 0.0)


def cross_zero_p5(a, fstart, fend):
    """``CrossZeroP5``: whether the quintic crosses zero in ``[fstart,
    fend]``, from the roots of ``gsl_poly_complex_solve``: whether one of
    them is real (imaginary part below ``1e-15``) and inside the interval.

    LAL tests the roots with shifted indices into the packed array (``z[4]``
    for the imaginary part of the third root, ``z[5]`` and ``z[7]`` for the
    fourth and fifth); here each root is tested with its own parts, which
    does not depend on the order of the roots. Returns ``(crosses, ok)``
    with ``ok`` False where GSL fails (LAL aborts)."""
    z, ok = complex_solve_quintic(a)
    crosses = jnp.any((jnp.abs(z[1::2]) < ROOT_THRESHOLD) & _inside(z[0::2], fstart, fend))
    return crosses, ok
