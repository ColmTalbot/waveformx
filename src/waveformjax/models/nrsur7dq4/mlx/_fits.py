# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The parameter-space fits of NRSur7dq4 in MLX (as :mod:`.._fits`, whose
docstring describes them). The functions broadcast over leading axes."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .._fits import Q_FIT_OFFSET, Q_FIT_SLOPE

__all__ = ["q_fit", "fit_parameters", "power_table", "monomials", "monomials_at", "fold_mass_ratio", "spin_monomials"]


def q_fit(q):
    """``NRSUR7DQ4_Q_FIT_OFFSET + NRSUR7DQ4_Q_FIT_SLOPE log q``."""
    return Q_FIT_OFFSET + Q_FIT_SLOPE * mx.log(q)


def fit_parameters(q, chi_a, chi_b):
    """The six spin parameters of the fits (``(..., 6)``) from the
    coorbital-frame spins ``chi_a``, ``chi_b`` (``(..., 3)``), for mass
    ratio ``q`` (of shape ``(...)``)."""
    chi1z = chi_a[..., 2]
    chi2z = chi_b[..., 2]
    eta = q / (1.0 + q) / (1.0 + q)
    chi_wt_avg = (q * chi1z + chi2z) / (1.0 + q)
    chi_hat = (chi_wt_avg - 38.0 * eta / 113.0 * (chi1z + chi2z)) / (1.0 - 76.0 * eta / 113.0)
    chi_asym = (chi1z - chi2z) / 2.0
    return mx.stack([chi_a[..., 0], chi_a[..., 1], chi_hat, chi_b[..., 0], chi_b[..., 1], chi_asym], axis=-1)


def power_table(qf, spins):
    """The table of powers ``x_j^k`` at ``4 j + k`` (``(..., 28)``)."""
    qf = mx.broadcast_to(qf, spins.shape[:-1])
    ones = mx.ones_like(qf)
    q_part = mx.stack([ones, qf, qf * qf, qf * qf * qf], axis=-1)
    spin_part = mx.stack([mx.ones_like(spins), spins, spins * spins, mx.zeros_like(spins)], axis=-1)
    return mx.concatenate([q_part, spin_part.reshape(*spins.shape[:-1], 24)], axis=-1)


def monomials(table, factors):
    """The monomials (``(..., n_monomials)``) from a table of powers and
    their factors (``(n_monomials, 5)`` indices), multiplied in LAL's
    order."""
    values = mx.take(table, factors, axis=-1)
    product = values[..., 0]
    for i in range(1, factors.shape[-1]):
        product = product * values[..., i]
    return product


def monomials_at(q, chi_a, chi_b, factors):
    """The monomials of the dynamics fits at mass ratio ``q`` and
    coorbital-frame spins ``chi_a``, ``chi_b``."""
    return monomials(power_table(q_fit(q), fit_parameters(q, chi_a, chi_b)), factors)


def fold_mass_ratio(coefs, qf):
    """``sum_k coefs[..., k, :] qf^k`` for ``coefs`` of shape ``(n, 4,
    n_spin)`` and ``qf`` broadcasting against ``(..., n, n_spin)``."""
    return coefs[:, 0] + qf * coefs[:, 1] + (qf * qf) * coefs[:, 2] + (qf * qf * qf) * coefs[:, 3]


def spin_monomials(spins, orders):
    """The monomials of the six spin parameters ``spins`` (``(..., 6)``) for
    the static ``orders`` (``(n_spin, 6)``, ``k_j <= 2``), multiplied in the
    order of ``j``: ``(..., n_spin)``."""
    orders = np.asarray(orders)
    product = None
    for j in range(orders.shape[1]):
        k = orders[:, j]
        if not np.any(k):
            continue
        x = spins[..., j : j + 1]
        factor = mx.where(mx.array(k == 1), x, mx.where(mx.array(k == 2), x * x, 1.0))
        product = factor if product is None else product * factor
    if product is None:
        return mx.ones((*spins.shape[:-1], orders.shape[0]), dtype=spins.dtype)
    return product
