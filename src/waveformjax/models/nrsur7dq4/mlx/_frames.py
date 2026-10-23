# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Rotation of the modes from the coorbital to the inertial frame in MLX
(as :mod:`.._frames`, whose docstring derives the sums). The modes have
shape ``(B, n_modes, n)``, the phase ``(B, n)`` and the quaternion ``(B, n,
4)``."""

from __future__ import annotations

import mlx.core as mx

from .._data import LMAX, MODES
from .._frames import _TERMS, _ComplexPowers, _Powers

__all__ = ["coorbital_to_coprecessing", "coprecessing_to_inertial"]


def coorbital_to_coprecessing(re, im, phase):
    """``h_copr^{l,m} = exp(-i m phi) h_coorb^{l,m}``."""
    cos = {k: mx.cos(k * phase) for k in range(1, LMAX + 1)}
    sin = {k: mx.sin(k * phase) for k in range(1, LMAX + 1)}
    out_re, out_im = [], []
    for i, (_, m) in enumerate(MODES):
        if m == 0:
            out_re.append(re[:, i])
            out_im.append(im[:, i])
            continue
        c = cos[abs(m)]
        s = sin[abs(m)] if m > 0 else -sin[abs(m)]
        out_re.append(re[:, i] * c + im[:, i] * s)
        out_im.append(-(re[:, i] * s) + im[:, i] * c)
    return mx.stack(out_re, axis=1), mx.stack(out_im, axis=1)


def coprecessing_to_inertial(re, im, quat):
    """``h^{l,m} = sum_m' D^l_{m,m'}(quat) h_copr^{l,m'}`` (see
    :func:`.._frames.coprecessing_to_inertial`)."""
    q0, q1, q2, q3 = (quat[..., k] for k in range(4))
    ra = _ComplexPowers(q0, -q3)
    rb = _ComplexPowers(-q2, -q1)
    ra_abs = _Powers(q0 * q0 + q3 * q3)
    rb_abs = _Powers(q2 * q2 + q1 * q1)
    z = {}
    abs_products = {}
    out_re, out_im = [], []
    position = {mode: i for i, mode in enumerate(MODES)}
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
                    if (alpha, beta) not in abs_products:
                        if alpha and beta:
                            abs_products[alpha, beta] = ra_abs[alpha] * rb_abs[beta]
                        else:
                            abs_products[alpha, beta] = ra_abs[alpha] if alpha else (rb_abs[beta] if beta else None)
                    product = abs_products[alpha, beta]
                    term = coef if product is None else coef * product
                    total = term if total is None else total + term
                d_re = zr * total
                d_im = None if isinstance(zi, float) else zi * total
                h_re, h_im = re[:, position[ell, mp]], im[:, position[ell, mp]]
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
    return mx.stack(out_re, axis=1), mx.stack(out_im, axis=1)
