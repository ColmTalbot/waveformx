# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The NNLO post-Newtonian Euler angles of IMRPhenomXP/XPHM/XPNR
(``PhenomXPrecVersion`` 101-104, the single-spin angles of IMRPhenomPv2):
the coefficients of ``IMRPhenomXGetAndSetPrecessionVariables``
(``pPrec->alpha1 ... alpha5``, ``epsilon1 ... epsilon5``), their evaluation
(``IMRPhenomX_PN_Euler_alpha_NNLO``, ``IMRPhenomX_PN_Euler_epsilon_NNLO``) and
the opening angle of ``IMRPhenomXPTwistUp22``, after waveformrs's
``imrphenomx/precession.rs``.

The orbital frequency is ``omega = pi M f`` for the co-precessing mode
``m' = 2`` (``2 pi M f / m'`` for the mode ``m'``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._math import PI, Struct, fenced, real
from .msa import acos_safe

__all__ = ["coefficients", "nnlo_coefficients", "evaluate", "nnlo_evaluate", "nnlo_angles", "nnlo_cos_beta",
           "nnlo_beta"]


@fenced
def coefficients(eta, delta, q, chi_eff, chip, m1, *m1_powers):
    """:func:`nnlo_coefficients` with the signature of waveformrs's
    ``nnlo_coefficients`` (the powers ``m1_2, m1_3, m1_4, m1_5, m1_6, m1_8``
    are accepted and recomputed from ``m1`` as LAL computes them)."""
    return nnlo_coefficients(eta, delta, q, chi_eff, chip, m1)


def nnlo_coefficients(eta, delta, q, chi_eff, chip, m1):
    """The NNLO Euler-angle coefficients of
    ``IMRPhenomXGetAndSetPrecessionVariables`` (versions 101-104), written as
    in LAL: ``eta`` and ``delta`` of the waveform struct, ``q = m1 / m2 >= 1``,
    the effective aligned spin ``chi_eff``, ``chi_p`` and the mass ``m1`` of
    the larger body in units of the total mass (``m1_SI / Mtot_SI``).

    Returns a :class:`~._math.Struct` with ``alpha1, alpha2, alpha3,
    alpha4l, alpha5, epsilon1, ..., epsilon5``."""
    eta2 = eta * eta
    eta3 = eta * eta2
    eta4 = eta * eta3
    eta5 = eta * eta4
    eta6 = eta * eta5
    delta2 = delta * delta
    delta3 = delta * delta2
    m1_2 = m1 * m1
    m1_3 = m1 * m1_2
    m1_4 = m1 * m1_3
    m1_5 = m1 * m1_4
    m1_6 = m1 * m1_5
    m1_7 = m1 * m1_6
    m1_8 = m1 * m1_7
    chip2 = chip * chip
    chil = (1.0 + q) * (chi_eff / q)
    chil2 = chil * chil
    pi = PI

    alpha1 = -35 / 192. + (5 * delta) / (64. * m1)
    alpha2 = ((15 * chil * delta * m1) / 128. - (35 * chil * m1_2) / 128.) / eta
    alpha3 = (-5515 / 3072. + eta * (-515 / 384. - (15 * delta2) / (256. * m1_2) + (175 * delta) / (256. * m1))
              + (4555 * delta) / (7168. * m1)
              + ((15 * chip2 * delta * m1_3) / 128. - (35 * chip2 * m1_4) / 128.) / eta2)
    # the term proportional to log(omega)
    alpha4l = ((5 * chil * delta2) / 16. - (5 * chil * delta * m1) / 3. + (2545 * chil * m1_2) / 1152.
               + ((-2035 * chil * delta * m1) / 21504. + (2995 * chil * m1_2) / 9216.) / eta
               + ((5 * chil * chip2 * delta * m1_5) / 128. - (35 * chil * chip2 * m1_6) / 384.) / eta3
               - (35 * pi) / 48. + (5 * delta * pi) / (16. * m1))
    alpha5 = (5 * (-190512 * delta3 * eta6 + 2268 * delta2 * eta3 * m1 * (eta2 * (323 + 784 * eta)
                   + 336 * (25 * chil2 + chip2) * m1_4) + 7 * m1_3 * (8024297 * eta4 + 857412 * eta5
                   + 3080448 * eta6 + 143640 * chip2 * eta2 * m1_4
                   - 127008 * chip2 * (-4 * chil2 + chip2) * m1_8
                   + 6048 * eta3 * ((2632 * chil2 + 115 * chip2) * m1_4 - 672 * chil * m1_2 * pi))
                   + 3 * delta * m1_2 * (-5579177 * eta4 + 80136 * eta5 - 3845520 * eta6
                   + 146664 * chip2 * eta2 * m1_4 + 127008 * chip2 * (-4 * chil2 + chip2) * m1_8
                   - 42336 * eta3 * ((726 * chil2 + 29 * chip2) * m1_4
                   - 96 * chil * m1_2 * pi)))) / (6.5028096e7 * eta4 * m1_3)

    epsilon1 = -35 / 192. + (5 * delta) / (64. * m1)
    epsilon2 = ((15 * chil * delta * m1) / 128. - (35 * chil * m1_2) / 128.) / eta
    epsilon3 = (-5515 / 3072. + eta * (-515 / 384. - (15 * delta2) / (256. * m1_2) + (175 * delta) / (256. * m1))
                + (4555 * delta) / (7168. * m1))
    epsilon4l = ((5 * chil * delta2) / 16. - (5 * chil * delta * m1) / 3. + (2545 * chil * m1_2) / 1152.
                 + ((-2035 * chil * delta * m1) / 21504. + (2995 * chil * m1_2) / 9216.) / eta - (35 * pi) / 48.
                 + (5 * delta * pi) / (16. * m1))
    epsilon5 = (5 * (-190512 * delta3 * eta3 + 2268 * delta2 * m1 * (eta2 * (323 + 784 * eta)
                     + 8400 * chil2 * m1_4)
                     - 3 * delta * m1_2 * (eta * (5579177 + 504 * eta * (-159 + 7630 * eta))
                     + 254016 * chil * m1_2 * (121 * chil * m1_2 - 16 * pi))
                     + 7 * m1_3 * (eta * (8024297 + 36 * eta * (23817 + 85568 * eta))
                     + 338688 * chil * m1_2 * (47 * chil * m1_2 - 12 * pi)))) / (6.5028096e7 * eta * m1_3)

    return Struct(alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4l=alpha4l, alpha5=alpha5,
                  epsilon1=epsilon1, epsilon2=epsilon2, epsilon3=epsilon3, epsilon4l=epsilon4l,
                  epsilon5=epsilon5)


def evaluate(c, omega, logomega, omega_cbrt, omega_cbrt2):
    """The NNLO sums ``alpha1 / omega + alpha2 / omega^(2/3) + alpha3 /
    omega^(1/3) + alpha4l log(omega) + alpha5 omega^(1/3)`` and the same for
    ``epsilon``, before the reference offsets (``alpha_offset``,
    ``epsilon_offset``, which LAL subtracts as the last term of the sums),
    given the powers of the orbital frequency ``omega``. Returns ``(alpha,
    epsilon)``."""
    alpha = (c.alpha1 / omega + c.alpha2 / omega_cbrt2 + c.alpha3 / omega_cbrt + c.alpha4l * logomega
             + c.alpha5 * omega_cbrt)
    epsilon = (c.epsilon1 / omega + c.epsilon2 / omega_cbrt2 + c.epsilon3 / omega_cbrt + c.epsilon4l * logomega
               + c.epsilon5 * omega_cbrt)
    return alpha, epsilon


def nnlo_evaluate(c, omega):
    """:func:`evaluate` at the orbital frequency ``omega``."""
    omega = jnp.asarray(omega, dtype=real())
    omega_cbrt = jnp.cbrt(omega)
    return evaluate(c, omega, jnp.log(omega), omega_cbrt, omega_cbrt * omega_cbrt)


def nnlo_angles(c, omega, alpha_offset, epsilon_offset):
    """``IMRPhenomX_PN_Euler_alpha_NNLO`` and
    ``IMRPhenomX_PN_Euler_epsilon_NNLO``: ``(alpha, epsilon)`` at the orbital
    frequency ``omega`` with the reference offsets subtracted."""
    alpha, epsilon = nnlo_evaluate(c, omega)
    return alpha - alpha_offset, epsilon - epsilon_offset


def nnlo_cos_beta(l, sperp, sl):
    """``cos(beta)`` of the NNLO angles (``IMRPhenomXPTwistUp22``,
    ``XLALSimIMRPhenomXPPNAngles``): ``sign(L + SL) / sqrt(1 + s^2)`` with
    ``s = Sperp / (L + SL)``, for the orbital angular momentum ``l`` (the
    value of the PN ansatz, units of ``M^2``), the in-plane spin ``sperp =
    chi_p m1^2`` and the aligned spin ``sl``."""
    l_total = l + sl
    s = sperp / l_total
    s2 = s * s
    return jnp.copysign(1.0, l_total) / jnp.sqrt(1.0 + s2)


def nnlo_beta(l, sperp, sl):
    """``acos`` of :func:`nnlo_cos_beta`. The value is LAL's ``acos(cos(beta))``
    (which loses accuracy for small angles); the derivative is that of the
    equivalent, well-conditioned ``atan2(sperp, L + SL)`` (finite for aligned
    spins)."""
    value = acos_safe(nnlo_cos_beta(l, sperp, sl))
    smooth = jnp.arctan2(sperp, l + sl)
    return smooth + jax.lax.stop_gradient(value - smooth)
