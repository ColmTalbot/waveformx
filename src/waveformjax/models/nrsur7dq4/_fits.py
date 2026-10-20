# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The parameter-space fits of NRSur7dq4 (``NRSur7dq4_eval_fit``).

A fit is ``sum_i c_i prod_j B_j(k_ij; x_j)``, with ``B_j`` monomials of
order ``k_ij`` in the fit parameters ``x = (q_fit, chi1x, chi1y, chiHat,
chi2x, chi2y, chi_a)``, where ``q_fit`` is an affine function of ``log q``
and ``chiHat``, ``chi_a`` are the effective spins of
``NRSur7dq4_effective_spins``. LAL tabulates the powers ``x_j^k`` (``k <=
3`` for ``q_fit``, ``k <= 2`` for the spins) for every fit; here the table
is built once per parameter point, the monomials of all fits are evaluated
from it at once (:func:`monomials`) and every fit is a dot product with its
dense row of coefficients (see :mod:`._data`).

The fits at the empirical nodes of the waveform all have the same mass
ratio, so its powers are folded into their coefficients once
(:func:`fold_mass_ratio`) and only the monomials of the spins are evaluated
at each node (:func:`spin_monomials`, with selects rather than gathers,
which vectorise well). The sums are in a different order than LAL's, which
changes the fits by rounding only.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = [
    "Q_FIT_SLOPE",
    "Q_FIT_OFFSET",
    "q_fit",
    "fit_parameters",
    "power_table",
    "monomials",
    "monomials_at",
    "fold_mass_ratio",
    "spin_monomials",
]

# These rescale log(q) from [-0.01, log(4.01)] to [-1, 1].
Q_FIT_SLOPE = 1.4298059216576398
Q_FIT_OFFSET = -0.9857019407834238


def q_fit(q):
    """``NRSUR7DQ4_Q_FIT_OFFSET + NRSUR7DQ4_Q_FIT_SLOPE log q``."""
    return Q_FIT_OFFSET + Q_FIT_SLOPE * jnp.log(q)


def fit_parameters(q, chi_a, chi_b):
    """The six spin parameters ``(chi1x, chi1y, chiHat, chi2x, chi2y,
    chi_a)`` of the fits from the coorbital-frame spins ``chi_a``,
    ``chi_b`` (``(..., 3)``) of the heavier and lighter black holes, for
    mass ratio ``q >= 1``."""
    chi1z = chi_a[..., 2]
    chi2z = chi_b[..., 2]
    eta = q / (1.0 + q) / (1.0 + q)
    chi_wt_avg = (q * chi1z + chi2z) / (1.0 + q)
    chi_hat = (chi_wt_avg - 38.0 * eta / 113.0 * (chi1z + chi2z)) / (1.0 - 76.0 * eta / 113.0)
    chi_asym = (chi1z - chi2z) / 2.0
    return jnp.stack([chi_a[..., 0], chi_a[..., 1], chi_hat, chi_b[..., 0], chi_b[..., 1], chi_asym], axis=-1)


def power_table(qf, spins):
    """The table of powers, ``(..., 28)``: ``x_j^k`` at ``4 j + k`` for the
    mass-ratio parameter ``qf`` (``j = 0``, ``k <= 3``) and the six spin
    parameters ``spins`` (``(..., 6)``, ``k <= 2``; the unused ``k = 3``
    entries are zero). ``ipow`` multiplies from the left, as here."""
    qf = jnp.broadcast_to(qf, spins.shape[:-1])
    ones = jnp.ones_like(qf)
    q_part = jnp.stack([ones, qf, qf * qf, qf * qf * qf], axis=-1)
    spin_part = jnp.stack([jnp.ones_like(spins), spins, spins * spins, jnp.zeros_like(spins)], axis=-1)
    return jnp.concatenate([q_part, spin_part.reshape(spins.shape[:-1] + (24,))], axis=-1)


def monomials(table, factors):
    """The monomials ``prod_j x_j^{k_j}`` (``(..., n_monomials)``) from a
    table of powers (``(..., 28)``) and their factors (``(n_monomials,
    5)`` indices into the table, see :class:`._data.NRSurData`), multiplied
    in LAL's order."""
    values = jnp.take(table, factors, axis=-1)
    product = values[..., 0]
    for i in range(1, factors.shape[-1]):
        product = product * values[..., i]
    return product


def monomials_at(q, chi_a, chi_b, factors):
    """The monomials of the fits at mass ratio ``q`` and coorbital-frame
    spins ``chi_a``, ``chi_b`` (``(..., 3)``)."""
    return monomials(power_table(q_fit(q), fit_parameters(q, chi_a, chi_b)), factors)


def fold_mass_ratio(coefs, qf):
    """The coefficients of the spin monomials at the mass-ratio parameter
    ``qf``: ``sum_k coefs[..., k, :] qf^k`` for ``coefs`` of shape ``(...,
    4, n_spin)``."""
    return coefs[..., 0, :] + qf * coefs[..., 1, :] + (qf * qf) * coefs[..., 2, :] + (qf * qf * qf) * coefs[..., 3, :]


def spin_monomials(spins, orders):
    """The monomials ``prod_j p_j^{k_j}`` of the six spin parameters
    ``spins`` (``(..., 6)``) for the static ``orders`` (``(n_spin, 6)``,
    ``k_j <= 2``), multiplied in the order of ``j``: ``(..., n_spin)``."""
    orders = np.asarray(orders)
    product = None
    for j in range(orders.shape[1]):
        k = orders[:, j]
        if not np.any(k):
            continue
        x = spins[..., j : j + 1]
        factor = jnp.where(k == 1, x, jnp.where(k == 2, x * x, 1.0))
        product = factor if product is None else product * factor
    if product is None:
        return jnp.ones(spins.shape[:-1] + (orders.shape[0],), dtype=spins.dtype)
    return product
