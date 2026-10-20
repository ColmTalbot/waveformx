# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Multiple-scale-analysis (MSA) Euler angles of IMRPhenomXP/XPHM/XPNR
(``PhenomXPrecVersion`` 223; Chatziioannou et al. 2017, arXiv:1703.03967, as
implemented in ``LALSimInspiralFDPrecAngles``), after waveformrs's
``imrphenomx/msa.rs`` and the MSA routines of ``LALSimIMRPhenomX_precession.c``.

* :func:`msa_system` is ``IMRPhenomX_Initialize_MSA_System`` (version 223,
  ``ExpansionOrder`` 5): a :class:`~._math.Struct` of the precession-averaged
  spin couplings, the PN coefficients of ``d omega / dt``, the phase of the
  spin-magnitude oscillations, the coefficients of ``phi_z`` and ``zeta`` and
  their integration constants. ``system.msa_error`` is LAL's ``MSA_ERROR``
  (``|Omega_z^(5)| > 1000``) as a traced boolean; like LAL the system is
  completed (with finite values) when it is set.
* :func:`phi_zeta_costhetal` is ``IMRPhenomX_Return_phi_zeta_costhetaL_MSA``:
  ``phi_z`` (the Euler angle ``alpha`` before the reference offset), ``zeta``
  (``epsilon = -gamma`` before the offset) and ``cos(theta_L)`` (``cos(beta)``)
  at the orbital velocities ``v`` (any array shape).
* :func:`cos_theta_l_with` is ``cos(theta_L)`` with the 3PN orbital angular
  momentum of another precession version (as LAL evaluates it when the
  precession struct does not carry version 222 or 223).

The Jacobi elliptic function ``sn`` and the incomplete elliptic integral of the
first kind are ports of GSL's ``gsl_sf_elljac_e`` and ``gsl_sf_ellint_F``
(GSL 2.8) with a fixed number of iterations; the iterations that GSL does not
do are masked, so the results are those of GSL.

All the functions work on traced values. The discrete switches of LAL (the
degenerate roots of the spin-magnitude cubic, the constant spin magnitude
``|S_+^2 - S_-^2| < 1e-5``, the branches of ``psi_0``, LAL's NaN checks) are
``jnp.where`` selections between branches that are evaluated with safe
arguments, so that gradients stay finite (also at zero in-plane spins, equal
masses or a single spin). Where LAL divides by an exact zero (isolated
points of the parameter space) the values here are finite instead of LAL's
``inf``/``NaN``.

Unlike LAL, which stores the roots of the cubic at each frequency in the
shared precession struct (and races on them in its OpenMP loop), everything
here is local: the results agree with single-threaded LAL.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._math import DBL_EPSILON, DBL_MIN, PI, Struct, cross, dot

__all__ = [
    "msa_system", "system", "phi_zeta_costhetal", "cos_theta_l_with", "s_norm", "psi_terms", "msa_error",
    "final_spin", "cos_theta_lj", "roots", "j_norm", "l_norm_3pn", "psi_of_v", "elljac_sn", "ellint_rf",
    "ellint_kcomp", "ellint_f", "gsl_hypot", "gsl_sign", "acos_safe",
]

TWOPI = 2.0 * PI
GSL_SQRT_DBL_EPSILON = 1.0 / 67108864.0  # 2^-26
GSL_DBL_MAX = 1.7976931348623157e308
# floor of the arguments of square roots and logarithms (keeps their
# derivatives finite at zero; changes values only below 1e-300)
TINY = 1e-300
# iterations of the AGM in sn (GSL's N = 16 stores at most 15 steps) and of
# Carlson's duplication in R_F (GSL stops at a relative spread of 1e-3, which
# takes at most ~15 steps for arguments allowed by GSL, x + y >= 5 DBL_MIN)
ELLJAC_N = 16
ELLINT_RF_ITERATIONS = 40


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _sqrt(x):
    """``sqrt(x)`` for ``x >= 0`` with a finite derivative at zero."""
    return jnp.sqrt(jnp.maximum(x, TINY))


def _sqrt_abs(x):
    """``sqrt(fabs(x))`` with a finite derivative at zero."""
    return jnp.sqrt(jnp.maximum(jnp.abs(x), TINY))


def _log_abs(x):
    """``log(fabs(x))`` with a finite derivative at zero."""
    return jnp.log(jnp.maximum(jnp.abs(x), TINY))


def _div(numerator, denominator):
    """``numerator / denominator``, with ``1`` in place of a vanishing
    denominator (LAL divides by zero only at isolated points)."""
    return numerator / jnp.where(denominator == 0.0, 1.0, denominator)


def _nan_to_zero(x):
    """LAL's ``if (x != x) x = 0``."""
    return jnp.where(jnp.isnan(x), 0.0, x)


def acos_safe(x):
    """``acos`` of ``x`` clamped to ``[-1, 1]``, with zero derivative where
    the clamp is active (rather than an infinite one at ``|x| = 1``)."""
    inside = jnp.abs(x) < 1.0
    return jnp.where(inside, jnp.arccos(jnp.where(inside, x, 0.0)), jnp.where(x > 0.0, 0.0, PI))


def gsl_sign(x):
    """``GSL_SIGN``: ``1`` for ``x >= 0``, else ``-1``."""
    return jnp.where(x >= 0.0, 1.0, -1.0)


def gsl_hypot(x, y):
    """``gsl_hypot(x, y)`` (GSL's ``sys/hypot.c``)."""
    xabs = jnp.abs(x)
    yabs = jnp.abs(y)
    small = jnp.minimum(xabs, yabs)
    large = jnp.maximum(xabs, yabs)
    u = small / jnp.where(large == 0.0, 1.0, large)
    value = jnp.where(small == 0.0, large, large * jnp.sqrt(1.0 + u * u))
    return jnp.where(jnp.isinf(x) | jnp.isinf(y), jnp.inf, value)


# ---------------------------------------------------------------------------
# GSL special functions
# ---------------------------------------------------------------------------


def elljac_sn(u, m):
    """The Jacobi elliptic function ``sn(u | m)`` of ``gsl_sf_elljac_e``
    (GSL 2.8: the descending Landen/AGM recursion of Bulirsch, evaluated at
    ``K - u`` where ``|sin(u mu_n)| >= |cos(u mu_n)|``). NaN where GSL signals
    a domain error (``|m| > 1``).

    The AGM runs a fixed 15 steps (GSL's maximum) with the steps after
    convergence masked, so that the result is that of GSL."""
    u, m = jnp.broadcast_arrays(jnp.asarray(u, dtype=float), jnp.asarray(m, dtype=float))
    domain = jnp.abs(m) > 1.0
    small = jnp.abs(m) < 2.0 * DBL_EPSILON
    near_one = jnp.abs(m - 1.0) < 2.0 * DBL_EPSILON
    general = ~(domain | small | near_one)
    m_safe = jnp.where(general, m, 0.5)

    sqrt_1mm = jnp.sqrt(1.0 - m_safe)

    def converging(mu, nu):
        return jnp.abs(mu - nu) > 4.0 * DBL_EPSILON * jnp.abs(mu + nu)

    def agm_step(carry, _):
        mu, nu, n, active = carry
        mu_next = jnp.where(active, 0.5 * (mu + nu), mu)
        nu_next = jnp.where(active, jnp.sqrt(mu * nu), nu)
        n = n + active.astype(jnp.int32)
        active = active & converging(mu_next, nu_next)
        return (mu_next, nu_next, n, active), (mu_next, nu_next)

    mu0 = jnp.ones_like(m_safe)
    n0 = jnp.zeros(m_safe.shape, dtype=jnp.int32)
    (mu_n, _, n, _), (mus, nus) = jax.lax.scan(
        agm_step, (mu0, sqrt_1mm, n0, converging(mu0, sqrt_1mm)), None, length=ELLJAC_N - 1)
    # mu_k, nu_k for k = 0 ... 15 (frozen after convergence, so mu_15 = mu_n)
    mus = jnp.concatenate([mu0[None], mus])
    nus = jnp.concatenate([sqrt_1mm[None], nus])

    sin_umu = jnp.sin(u * mu_n)
    cos_umu = jnp.cos(u * mu_n)
    use_sin = jnp.abs(sin_umu) < jnp.abs(cos_umu)
    # the larger of |sin| and |cos| is the denominator (never zero)
    t = jnp.where(use_sin, sin_umu, cos_umu) / jnp.where(use_sin, cos_umu, sin_umu)

    # Bulirsch's descending recursion from c_n = mu_n t, d_n = 1:
    # c_k = d_{k+1} c_{k+1}, d_k = (r + nu_k) / (r + mu_k), r = c_{k+1}^2 / mu_{k+1}
    def descend(carry, x):
        c, d = carry
        k, mu_k, nu_k, mu_k1 = x
        step = k < n
        r = (c * c) / mu_k1
        d_k = (r + nu_k) / (r + mu_k)
        return (jnp.where(step, d * c, c), jnp.where(step, d_k, d)), None

    ks = jnp.arange(ELLJAC_N - 1)
    (c, d), _ = jax.lax.scan(descend, (mu_n * t, jnp.ones_like(t)), (ks, mus[:-1], nus[:-1], mus[1:]), reverse=True)
    hypot = gsl_hypot(1.0, c)
    dn = sqrt_1mm / d
    cn = dn * gsl_sign(cos_umu) / hypot
    sn_from_cos = cn * c / sqrt_1mm
    sn_from_sin = gsl_sign(sin_umu) / hypot
    sn = jnp.where(use_sin, sn_from_cos, sn_from_sin)

    sn = jnp.where(near_one, jnp.tanh(u), sn)
    sn = jnp.where(small, jnp.sin(u), sn)
    return jnp.where(domain, jnp.nan, sn)


def _max3(a, b, c):
    """GSL's ``locMAX3``."""
    bc = jnp.where(b > c, b, c)
    return jnp.where(a > bc, a, bc)


def ellint_rf(x, y, z):
    """Carlson's ``R_F(x, y, z)``, ``gsl_sf_ellint_RF_e`` with
    ``GSL_PREC_DOUBLE`` (NaN for GSL's domain errors). The duplication runs
    a fixed number of steps, masked after GSL's stopping criterion."""
    x, y, z = jnp.broadcast_arrays(*(jnp.asarray(a, dtype=float) for a in (x, y, z)))
    lolim = 5.0 * DBL_MIN
    uplim = 0.2 * GSL_DBL_MAX
    errtol = 0.001
    domain = (x < 0.0) | (y < 0.0) | (z < 0.0)
    domain = domain | (x + y < lolim) | (x + z < lolim) | (y + z < lolim)
    domain = domain | (_max3(x, y, z) >= uplim)
    xn = jnp.where(domain, 1.0, x)
    yn = jnp.where(domain, 1.0, y)
    zn = jnp.where(domain, 1.0, z)

    def deviations(xn, yn, zn):
        mu = (xn + yn + zn) / 3.0
        return mu, 2.0 - (mu + xn) / mu, 2.0 - (mu + yn) / mu, 2.0 - (mu + zn) / mu

    def spread(xndev, yndev, zndev):
        return _max3(jnp.abs(xndev), jnp.abs(yndev), jnp.abs(zndev))

    def duplicate(_, carry):
        xn, yn, zn, mu, xndev, yndev, zndev, done = carry
        # (floored at DBL_MIN, below GSL's lower limit of the arguments, for a
        # finite derivative at x = 0)
        xnroot = jnp.sqrt(jnp.maximum(xn, DBL_MIN))
        ynroot = jnp.sqrt(jnp.maximum(yn, DBL_MIN))
        znroot = jnp.sqrt(jnp.maximum(zn, DBL_MIN))
        lamda = xnroot * (ynroot + znroot) + ynroot * znroot
        new = (xn + lamda) * 0.25, (yn + lamda) * 0.25, (zn + lamda) * 0.25
        new = new + deviations(*new)
        old = (xn, yn, zn, mu, xndev, yndev, zndev)
        xn, yn, zn, mu, xndev, yndev, zndev = (jnp.where(done, a, b) for a, b in zip(old, new))
        return xn, yn, zn, mu, xndev, yndev, zndev, done | (spread(xndev, yndev, zndev) < errtol)

    mu, xndev, yndev, zndev = deviations(xn, yn, zn)
    done = spread(xndev, yndev, zndev) < errtol
    carry = jax.lax.fori_loop(0, ELLINT_RF_ITERATIONS, duplicate, (xn, yn, zn, mu, xndev, yndev, zndev, done))
    mu, xndev, yndev, zndev = carry[3:7]
    c1 = 1.0 / 24.0
    c2 = 3.0 / 44.0
    c3 = 1.0 / 14.0
    e2 = xndev * yndev - zndev * zndev
    e3 = xndev * yndev * zndev
    s = 1.0 + (c1 * e2 - 0.1 - c2 * e3) * e2 + c3 * e3
    value = s / jnp.sqrt(mu)
    return jnp.where(domain, jnp.nan, value)


def ellint_kcomp(k):
    """The complete elliptic integral of the first kind ``K(k)``,
    ``gsl_sf_ellint_Kcomp_e`` with ``GSL_PREC_DOUBLE`` (modulus ``k``; GSL
    2.8 evaluates ``1 - k^2`` as ``(1 - k)(1 + k)``)."""
    k = jnp.asarray(k, dtype=float)
    k2 = k * k
    domain = k2 >= 1.0
    near_one = (k2 >= 1.0 - GSL_SQRT_DBL_EPSILON) & ~domain
    # Abramowitz & Stegun 17.3.34
    y = jnp.where(near_one, (1.0 - k) * (1.0 + k), 0.5)
    a = (1.38629436112, 0.09666344259, 0.03590092383)
    b = (0.5, 0.12498593597, 0.06880248576)
    ta = a[0] + y * (a[1] + y * a[2])
    tb = -jnp.log(y) * (b[0] + y * (b[1] + y * b[2]))
    regular = ~(domain | near_one)
    rf = ellint_rf(0.0, jnp.where(regular, (1.0 - k) * (1.0 + k), 0.5), 1.0)
    value = jnp.where(near_one, ta + tb, rf)
    return jnp.where(domain, jnp.nan, value)


def ellint_f(phi, k):
    """The incomplete elliptic integral of the first kind ``F(phi, k)``,
    ``gsl_sf_ellint_F(phi, k, GSL_PREC_DOUBLE)`` (Legendre form, modulus
    ``k``), with GSL's reduction of ``phi`` to ``(-pi/2, pi/2]``."""
    phi, k = jnp.broadcast_arrays(jnp.asarray(phi, dtype=float), jnp.asarray(k, dtype=float))
    nc = jnp.floor(phi / PI + 0.5)
    phi = phi - nc * PI
    sin_phi = jnp.sin(phi)
    sin2_phi = sin_phi * sin_phi
    x = 1.0 - sin2_phi
    y = 1.0 - k * k * sin2_phi
    val = sin_phi * ellint_rf(x, y, 1.0)
    reduced = nc != 0.0
    kcomp = ellint_kcomp(jnp.where(reduced, k, 0.0))
    return jnp.where(reduced, 2.0 * nc * kcomp + val, val)


# ---------------------------------------------------------------------------
# the MSA system
# ---------------------------------------------------------------------------

# PN coefficients of d omega / dt and of L (LALSimInspiralFDPrecAngles)
_DOMEGADT_NS = (96. / 5., -1486. / 35., -264. / 5., 384. * PI / 5., 34103. / 945., 13661. / 105.,
                944. / 15., PI * (-4159. / 35.), PI * (-2268. / 5.))
_DOMEGADT_SO = (-904. / 5., -120., -62638. / 105., 4636. / 5., -6472. / 35., 3372. / 5.)
_DOMEGADT_SS = (-494. / 5., -1442. / 5., -233. / 5., -719. / 5.)
_L_NONSPIN = (3. / 2., 1. / 6., 27. / 8., -19. / 8., 1. / 24., 135. / 16., -6889 / 144. + 41. / 24. * PI * PI,
              31. / 24., 7. / 1296.)
_L_SPINORBIT = (-14. / 6., -3. / 2., -11. / 2., 133. / 72., -33. / 8., 7. / 4.)


def _vpowers(v):
    """Powers of the orbital velocity shared by the MSA functions."""
    v = jnp.asarray(v, dtype=float)
    v2 = v * v
    v3 = v * v2
    invv = 1.0 / v
    invv2 = invv * invv
    return Struct(v=v, v2=v2, v3=v3, v4=v2 * v2, v6=v3 * v3, invv=invv, invv2=invv2, invv3=invv * invv2,
                  logv=jnp.log(v))


def msa_system(eta, m1, m2, m1_2, m2_2, chi1, chi2, pi_gm, f_ref):
    """``IMRPhenomX_Initialize_MSA_System`` for ``PhenomXPrecVersion`` 223 and
    ``ExpansionOrder`` 5 (waveformrs's ``MsaSystem::new_unchecked``).

    Parameters (waveformrs's ``MsaInputs``):

    * ``eta``: symmetric mass ratio (``pWF->eta``);
    * ``m1``, ``m2``: the component masses of the waveform struct in units of
      the total mass, ``m1 >= m2`` (``pWF->m1``, ``pWF->m2``);
    * ``m1_2``, ``m2_2``: ``(m1_SI / Mtot_SI)^2`` and ``(m2_SI / Mtot_SI)^2`` of
      the precession struct (the normalisation of the spin norms);
    * ``chi1``, ``chi2``: the dimensionless spins as 3-sequences ``(x, y, z)``
      in the L frame at the reference frequency;
    * ``pi_gm``: ``pi G M / c^3`` in seconds (``pPrec->piGM``);
    * ``f_ref``: the reference frequency in Hz.

    Returns a :class:`~._math.Struct`; ``msa_error`` is LAL's ``MSA_ERROR``
    (``|Omega_z^(5)| > 1000``, after which LAL uses the NNLO angles of
    version 102). The system is complete and finite either way.
    """
    eta = jnp.asarray(eta, dtype=float)
    chi1 = [jnp.asarray(c, dtype=float) for c in chi1]
    chi2 = [jnp.asarray(c, dtype=float) for c in chi2]
    eta2 = eta * eta
    eta3 = eta * eta2
    eta4 = eta * eta3
    inveta = 1.0 / eta
    inveta2 = 1.0 / eta2
    inveta3 = 1.0 / eta3
    inveta4 = 1.0 / eta4
    sqrt_inveta = 1.0 / jnp.sqrt(eta)

    # pPrec->S1_norm_2 = (|chi1| m1^2)^2 (squared directly: finite derivative
    # at zero spin; the same to rounding)
    s1_norm_2 = dot(chi1, chi1) * (m1_2 * m1_2)
    s2_norm_2 = dot(chi2, chi2) * (m2_2 * m2_2)

    # Chatziioannou et al use q = m2 / m1 < 1
    q = m2 / m1
    delta_qq = (1.0 - q) / (1.0 + q)

    # dimensionful spins: eta / q = m1^2, eta q = m2^2
    s1v = [c * eta / q for c in chi1]
    s2v = [c * eta * q for c in chi2]

    v_0 = jnp.cbrt(pi_gm * f_ref)
    v_0_2 = v_0 * v_0
    l_0 = [jnp.zeros_like(v_0), jnp.zeros_like(v_0), eta / v_0]

    dot_s1l = s1v[2]
    dot_s2l = s2v[2]
    dot_s1s2 = dot(s1v, s2v)

    # degenerate roots of the cubic (IMRPhenomX_Return_Roots_MSA): a spin
    # (anti-)aligned with L to rounding (LAL's dotS1Ln = S1.L / |S1| == +-1,
    # tested as |S1| == |S1.L|: XLA evaluates x / sqrt(y) as x * rsqrt(y),
    # which is not exactly +-1 for aligned spins; the barrier keeps XLA from
    # folding constant factors of the components into their squares)
    def aligned(sv):
        sv = jax.lax.optimization_barrier([jax.lax.stop_gradient(c) for c in sv])
        return jnp.sqrt(dot(sv, sv)) == jnp.abs(sv[2])

    s1_aligned = aligned(s1v)
    s2_aligned = aligned(s2v)
    degenerate_roots = s1_aligned | s2_aligned | (s1_norm_2 == 0.0) | (s2_norm_2 == 0.0)

    # IMRPhenomX_Get_PN_beta, _sigma, _tau
    def pn_beta(a, b):
        return dot_s1l * (a + b * q) + dot_s2l * (a + b / q)

    def pn_sigma(a, b):
        return inveta * (a * dot_s1s2 - b * dot_s1l * dot_s2l)

    def pn_tau(a, b):
        return (q * ((s1_norm_2 * a) - b * dot_s1l * dot_s1l) + (a * s2_norm_2 - b * dot_s2l * dot_s2l) / q) / eta

    constants_l = (
        _L_NONSPIN[0] + eta * _L_NONSPIN[1],
        pn_beta(_L_SPINORBIT[0], _L_SPINORBIT[1]),
        _L_NONSPIN[2] + eta * _L_NONSPIN[3] + eta * eta * _L_NONSPIN[4],
        pn_beta(_L_SPINORBIT[2] + _L_SPINORBIT[3] * eta, _L_SPINORBIT[4] + _L_SPINORBIT[5] * eta),
        _L_NONSPIN[5] + _L_NONSPIN[6] * eta + _L_NONSPIN[7] * eta * eta + _L_NONSPIN[8] * eta * eta * eta,
    )

    # effective spin
    seff = (1.0 + q) * dot_s1l + (1 + (1.0 / q)) * dot_s2l

    # total spin and angular momentum at the reference frequency
    s_0 = [a + b for a, b in zip(s1v, s2v)]
    j_0 = [a + b for a, b in zip(l_0, s_0)]
    s_0_norm_2 = dot(s_0, s_0)
    l_0_norm = eta / v_0
    j_0_norm = jnp.sqrt(dot(j_0, j_0))

    system = Struct(
        eta=eta, eta2=eta2, eta3=eta3, eta4=eta4, inveta=inveta, inveta2=inveta2, inveta3=inveta3,
        inveta4=inveta4, sqrt_inveta=sqrt_inveta, qq=q, delta_qq=delta_qq, s1_norm_2=s1_norm_2,
        s2_norm_2=s2_norm_2, s_0_norm_2=s_0_norm_2, degenerate_roots=degenerate_roots, seff=seff,
        constants_l=constants_l,
    )

    s32, smi2, spl2 = roots(system, l_0_norm, j_0_norm)

    # Eq. 45, from the initial conditions
    sav2 = 0.5 * (spl2 + smi2)
    sav = _sqrt(sav2)
    system.sav2 = sav2
    system.sav = sav
    system.inv_sav2 = 1.0 / sav2
    system.inv_sav = 1.0 / sav

    # c_1, Eq. 41
    c_1 = 0.5 * (j_0_norm * j_0_norm - l_0_norm * l_0_norm - sav2) / l_0_norm * eta
    c1_2 = c_1 * c_1
    system.c1 = c_1
    system.c1_over_eta = c_1 / eta

    # precession-averaged spin couplings, Eqs. A9 - A14 (S1.L, S2.L)
    # LAL's (1 - q^2) + 1e-16, written so that XLA cannot reassociate it into
    # (1 + 1e-16) - q^2 = 1 - q^2, which vanishes at equal masses
    one_m_q2 = 1.0 - q * q
    omq2 = jnp.where(one_m_q2 == 0.0, 1e-16, one_m_q2 + 1e-16)
    system.s1l_pav = (c_1 * (1.0 + q) - q * eta * seff) / (eta * omq2)
    system.s2l_pav = -q * (c_1 * (1.0 + q) - eta * seff) / (eta * omq2)
    # their sum (for the final spin) without the cancellation of the two
    # O(1 / (1 - q^2)) terms near equal masses (0 at q = 1, as in LAL)
    system.sl_pav = c_1 / eta * jnp.where(one_m_q2 == 0.0, 0.0, ((1.0 + q) * (1.0 - q)) / omq2)

    # version 223: PN coefficients from the initial spin couplings
    a0 = eta * _DOMEGADT_NS[0]
    a2 = eta * (_DOMEGADT_NS[1] + eta * (_DOMEGADT_NS[2]))
    a3 = eta * (_DOMEGADT_NS[3] + pn_beta(_DOMEGADT_SO[0], _DOMEGADT_SO[1]))
    a4 = eta * (_DOMEGADT_NS[4] + eta * (_DOMEGADT_NS[5] + eta * (_DOMEGADT_NS[6]))
                + pn_sigma(_DOMEGADT_SS[0], _DOMEGADT_SS[1]) + pn_tau(_DOMEGADT_SS[2], _DOMEGADT_SS[3]))
    a0_2 = a0 * a0
    a0_3 = a0_2 * a0
    a2_2 = a2 * a2

    # Appendix A of arXiv:1703.03967 (g5 only enters the n = 5 terms, which
    # ExpansionOrder 5 drops)
    g0 = 1. / a0
    g2 = -(a2 / a0_2)
    g3 = -(a3 / a0_2)
    g4 = -(a4 * a0 - a2_2) / a0_3
    system.g0 = g0

    delta2 = delta_qq * delta_qq
    # Eq. C1; LAL divides by delta^2 = 0 at equal masses (psi1 = +-inf, and
    # the phase psi(v) = 0 * inf = NaN, which LAL never uses there because the
    # spin magnitude is then constant): 0 here
    equal_masses = delta2 == 0.0
    system.psi1 = jnp.where(
        equal_masses, 0.0, 3.0 * (2.0 * eta2 * seff - c_1) / jnp.where(equal_masses, 1.0, eta * delta2))

    c_1_over_nu = system.c1_over_eta
    c_1_over_nu_2 = c_1_over_nu * c_1_over_nu
    one_p_q_sq = (1. + q) * (1. + q)
    seff_2 = seff * seff
    q_2 = q * q
    one_m_q_sq = (1. - q) * (1. - q)
    one_m_q2_2 = (1. - q_2) * (1. - q_2)
    one_m_q_4 = one_m_q_sq * one_m_q_sq

    # Delta as in LALSimInspiralFDPrecAngles
    del1 = 4. * c_1_over_nu_2 * one_p_q_sq
    del2 = 8. * c_1_over_nu * q * (1. + q) * seff
    del3 = 4. * (one_m_q2_2 * s1_norm_2 - q_2 * seff_2)
    del4 = 4. * c_1_over_nu_2 * q_2 * one_p_q_sq
    del5 = 8. * c_1_over_nu * q_2 * (1. + q) * seff
    del6 = 4. * (one_m_q2_2 * s2_norm_2 - q_2 * seff_2)
    big_delta = _sqrt_abs((del1 - del2 - del3) * (del4 - del5 - del6))

    # Eq. C2 as in LALSimInspiralFDPrecAngles (infinite at equal masses in
    # LAL; 0 here, see psi1)
    u1 = 3. * g2 / g0
    u2 = 0.75 * one_p_q_sq / jnp.where(equal_masses, 1.0, one_m_q_4)
    u3 = -20. * c_1_over_nu_2 * q_2 * one_p_q_sq
    u4 = 2. * one_m_q2_2 * (q * (2. + q) * s1_norm_2 + (1. + 2. * q) * s2_norm_2 - 2. * q * sav2)
    u5 = 2. * q_2 * (7. + 6. * q + 7. * q_2) * 2. * c_1_over_nu * seff
    u6 = 2. * q_2 * (3. + 4. * q + 3. * q_2) * seff_2
    u7 = q * big_delta
    system.psi2 = jnp.where(equal_masses, 0.0, u1 + u2 * (u3 + u4 + u5 - u6 + u7))

    # Appendix D
    rm = spl2 - smi2
    rm_2 = rm * rm
    cp = spl2 * eta2 - c1_2
    cm = smi2 * eta2 - c1_2
    sqrt_cpcm = _sqrt_abs(cp * cm)
    a1dd = 0.5 + 0.75 / eta
    a2dd = -0.75 * seff / eta
    d2rmsq = (cp - sqrt_cpcm) / eta2
    d4rmsq = -0.5 * rm * sqrt_cpcm / eta2 - cp / eta4 * (sqrt_cpcm - cp)
    s0m = s1_norm_2 - s2_norm_2
    aw = (-3. * (1. + q) / q * (2. * (1. + q) * eta2 * seff * c_1 - (1. + q) * c1_2 + (1. - q) * eta2 * s0m))
    cw = 3. / 32. / eta * rm_2
    dw = 4.0 * cp - 4.0 * d2rmsq * eta2
    hw = -2.0 * (2.0 * d2rmsq - rm) * c_1
    fw = rm * d2rmsq - d4rmsq - 0.25 * rm_2
    add = _div(aw, dw)
    hdd = _div(hw, dw)
    cdd = _div(cw, dw)
    fdd = _div(fw, dw)
    gw = 3. / 16. / eta2 / eta * rm_2 * (c_1 - eta2 * seff)
    gdd = _div(gw, dw)
    hdd_2 = hdd * hdd
    addfdd = add * fdd
    addfddhdd = addfdd * hdd
    addhdd_2 = add * hdd_2

    # Eqs. D10 - D15
    omegaz0 = a1dd + add
    omegaz1 = a2dd - add * seff - add * hdd
    omegaz2 = add * hdd * seff + cdd - add * fdd + add * hdd_2
    omegaz3 = (addfdd - cdd - addhdd_2) * (seff + hdd) + addfddhdd
    omegaz4 = (cdd + addhdd_2 - 2.0 * addfdd) * (hdd * seff + hdd_2 - fdd) - add * fdd * fdd
    omegaz5 = ((cdd - addfdd + addhdd_2) * fdd * (seff + 2.0 * hdd)
               - (cdd + addhdd_2 - 2.0 * addfdd) * hdd_2 * (seff + hdd) - addfdd * fdd * hdd)
    # MSA_ERROR: "|Omegaz5| is larger than expected and may be pathological"
    # (dw = 0 makes LAL's coefficients infinite or NaN)
    system.msa_error = (jnp.abs(omegaz5) > 1000.0) | (dw == 0.0)

    # Eqs. D16 - D21; ExpansionOrder 5 drops the n = 5 terms
    zero = jnp.zeros_like(omegaz0)
    system.omegaz_coeff = (
        3.0 * g0 * omegaz0,
        3.0 * g0 * omegaz1,
        3.0 * (g0 * omegaz2 + g2 * omegaz0),
        3.0 * (g0 * omegaz3 + g2 * omegaz1 + g3 * omegaz0),
        3.0 * (g0 * omegaz4 + g2 * omegaz2 + g3 * omegaz1 + g4 * omegaz0),
        zero,
    )

    # Appendix E/F
    c1oveta2 = c_1 / eta2
    omegazeta0 = omegaz0
    omegazeta1 = omegaz1 + omegaz0 * c1oveta2
    omegazeta2 = omegaz2 + omegaz1 * c1oveta2
    omegazeta3 = omegaz3 + omegaz2 * c1oveta2 + gdd
    omegazeta4 = omegaz4 + omegaz3 * c1oveta2 - gdd * seff - gdd * hdd
    system.omegazeta_coeff = (
        -g0 * omegazeta0,
        -1.5 * g0 * omegazeta1,
        -3.0 * (g0 * omegazeta2 + g2 * omegazeta0),
        3.0 * (g0 * omegazeta3 + g2 * omegazeta1 + g3 * omegazeta0),
        3.0 * (g0 * omegazeta4 + g2 * omegazeta2 + g3 * omegazeta1 + g4 * omegazeta0),
        zero,
    )

    # psi0 (zero for a constant spin magnitude)
    oscillating = jnp.abs(smi2 - spl2) >= 1.0e-5
    span = jnp.where(oscillating, smi2 - spl2, -1.0)
    mm = _sqrt(span / jnp.where(oscillating, s32 - spl2, -2.0))
    tmp_b = (s_0_norm_2 - spl2) / span
    volume_element = dot(cross(l_0, s1v), s2v)
    vol_sign = jnp.sign(volume_element)
    system.psi0 = jnp.zeros_like(c_1)
    psi_of_v0 = psi_of_v(system, v_0, v_0_2)
    inside = (tmp_b >= 0.0) & (tmp_b <= 1.0)
    above = (tmp_b > 1.0) & ((tmp_b - 1.) < 0.00001)
    below = (tmp_b < 0.0) & (tmp_b > -0.00001)
    # LAL's else branch also takes NaN
    inside = inside | jnp.isnan(tmp_b)
    sqrt_b = jnp.where(above, 1.0, jnp.where(below, 0.0, _sqrt(jnp.where(inside, tmp_b, 0.25))))
    sqrt_b = jnp.where(jnp.isnan(tmp_b), jnp.nan, sqrt_b)
    arg = vol_sign * sqrt_b
    phi = _asin_safe(arg)
    psi0 = ellint_f(phi, mm) - psi_of_v0
    system.psi0 = jnp.where(oscillating & (inside | above | below), psi0, 0.0)

    # integration constants phi_z,0 and zeta_0
    p0 = _vpowers(v_0)
    roots_0 = (s32, smi2, spl2)
    phiz_corr, zeta_corr = _corrections(system, p0, l_0_norm, j_0_norm, roots_0)
    system.phiz_0 = jnp.zeros_like(c_1)
    system.zeta_0 = jnp.zeros_like(c_1)
    phiz_0 = _phiz(system, p0, j_0_norm)
    zeta_0 = _zeta(system, p0)
    system.phiz_0 = -phiz_0 - phiz_corr
    system.zeta_0 = -zeta_0 - zeta_corr
    return system


def system(inputs):
    """:func:`msa_system` from a :class:`~._math.Struct` with the fields of
    waveformrs's ``MsaInputs``: ``eta, m1, m2, m1_2, m2_2, chi1, chi2, pi_gm,
    f_ref``."""
    return msa_system(inputs.eta, inputs.m1, inputs.m2, inputs.m1_2, inputs.m2_2, inputs.chi1, inputs.chi2,
                      inputs.pi_gm, inputs.f_ref)


def _asin_safe(x):
    """``asin(x)`` for ``|x| <= 1`` with zero derivative at ``|x| = 1``."""
    inside = jnp.abs(x) < 1.0
    return jnp.where(inside, jnp.arcsin(jnp.where(inside, x, 0.0)), jnp.sign(x) * (0.5 * PI))


def msa_error(system):
    """``MSA_ERROR`` of ``IMRPhenomX_Initialize_MSA_System`` (a boolean
    array): LAL then uses the NNLO angles (version 102)."""
    return system.msa_error


def psi_terms(system):
    """``(g0, delta_qq, psi1, psi2)`` of the precession struct (used by the
    PNR angles to estimate the period of the two-spin oscillations). At equal
    masses LAL's ``psi1``, ``psi2`` are infinite (and its estimate NaN); they
    are 0 here, so ``delta_qq == 0`` marks that case."""
    return system.g0, system.delta_qq, system.psi1, system.psi2


def psi_of_v(system, v, v2, psi0=None):
    """``IMRPhenomX_psiofv``: the phase of the spin-magnitude oscillations,
    Eq. 51 (with ``psi0`` of the system by default)."""
    if psi0 is None:
        psi0 = system.psi0
    return psi0 - 0.75 * system.g0 * system.delta_qq * (1.0 + system.psi1 * v + system.psi2 * v2) / (v2 * v)


def j_norm(system, l_norm):
    """``IMRPhenomX_JNorm_MSA``, Eq. 41."""
    return _sqrt(l_norm * l_norm + (2.0 * l_norm * system.c1_over_eta) + system.sav2)


def l_norm_3pn(system, v, v2, l_norm):
    """``IMRPhenomX_L_norm_3PN_of_v``: the 3PN orbital angular momentum of
    ``LALSimInspiralFDPrecAngles``."""
    c = system.constants_l
    return l_norm * (1. + v2 * (c[0] + v * c[1] + v2 * (c[2] + v * c[3] + v2 * (c[4]))))


def roots(system, l_norm, j_norm):
    """``IMRPhenomX_Return_Roots_MSA``: the roots ``(S_3^2, S_-^2, S_+^2)``
    of the cubic for the total spin magnitude, Eq. 21 (with
    ``IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA``)."""
    jn2 = j_norm * j_norm
    ln2 = l_norm * l_norm
    s1 = system.s1_norm_2
    s2 = system.s2_norm_2
    q = system.qq
    eta = system.eta
    seff = system.seff
    delta = system.delta_qq
    delta_sq = delta * delta
    j2ml2 = jn2 - ln2
    j2ml2sq = j2ml2 * j2ml2

    # Eqs. B2 - B4
    b = (ln2 + s1) * q + 2.0 * l_norm * seff - 2.0 * jn2 - s1 - s2 + (ln2 + s2) / q
    c = (j2ml2sq - 2.0 * l_norm * seff * j2ml2 - 2.0 * ((1.0 - q) / q) * ln2 * (s1 - q * s2)
         + 4.0 * eta * ln2 * seff * seff - 2.0 * delta * (s1 - s2) * seff * l_norm
         + 2.0 * ((1.0 - q) / q) * (q * s1 - s2) * jn2)
    d = (((1.0 - q) / q) * (s2 - q * s1) * j2ml2sq
         + delta_sq * (s1 - s2) * (s1 - s2) * ln2 / eta
         + 2.0 * delta * l_norm * seff * (s1 - s2) * j2ml2)

    b2 = b * b
    b3 = b2 * b
    bc = b * c
    p = c - b2 / 3
    qc = (2.0 / 27.0) * b3 - bc / 3.0 + d
    # LAL takes the degenerate roots where sqrt(-p / 3) or theta is NaN: p > 0,
    # or p = qc = 0 (for p = 0 and qc != 0 the three roots are -B/3)
    valid = (p < 0.0) | ((p == 0.0) & (qc != 0.0))
    negative = p < 0.0
    sqrtarg = jnp.where(negative, _sqrt(-p / 3.0), 0.0)
    acosarg = jnp.where(negative, 1.5 * qc / jnp.where(negative, p, -1.0) / jnp.where(negative, sqrtarg, 1.0), 1.0)
    theta = acos_safe(acosarg) / 3.0
    cos_theta = jnp.cos(theta)
    tmp1 = 2.0 * sqrtarg * jnp.cos(theta - 2.0 * TWOPI / 3.0) - b / 3.0
    tmp2 = 2.0 * sqrtarg * jnp.cos(theta - TWOPI / 3.0) - b / 3.0
    tmp3 = 2.0 * sqrtarg * cos_theta - b / 3.0
    tmp4 = jnp.maximum(jnp.maximum(tmp1, tmp2), tmp3)
    tmp5 = jnp.minimum(jnp.minimum(tmp1, tmp2), tmp3)
    tmp6 = jnp.where(((tmp4 - tmp3) > 0.0) & ((tmp5 - tmp3) < 0.0), tmp3,
                     jnp.where(((tmp4 - tmp1) > 0.0) & ((tmp5 - tmp1) < 0.0), tmp1, tmp2))

    degenerate = system.degenerate_roots | ~valid
    # numerical perturbation to keep alpha from diverging
    smi2_degenerate = system.s_0_norm_2
    s32 = jnp.where(degenerate, 0.0, tmp5)
    smi2 = jnp.where(degenerate, smi2_degenerate, jnp.abs(tmp6))
    spl2 = jnp.where(degenerate, smi2_degenerate + 1e-9, jnp.abs(tmp4))
    return s32, smi2, spl2


def s_norm(system, v, roots_):
    """``IMRPhenomX_Return_SNorm_MSA``: the total spin magnitude, Eq. 23,
    at the orbital velocity ``v`` given the roots of the cubic."""
    s32, smi2, spl2 = roots_
    v = jnp.asarray(v, dtype=float)
    oscillating = ~(jnp.abs(smi2 - spl2) < 1.0e-5)
    # Eq. 25
    m = (smi2 - spl2) / jnp.where(oscillating, s32 - spl2, 1.0)
    psi = psi_of_v(system, v, v * v)
    sn = elljac_sn(jnp.where(oscillating, psi, 0.0), jnp.where(oscillating, m, 0.5))
    sn = jnp.where(oscillating, sn, 0.0)
    return _sqrt(spl2 + (smi2 - spl2) * sn * sn)


def cos_theta_lj(l_norm, j_norm, s_norm_):
    """``IMRPhenomX_costhetaLJ``: ``cos`` of the angle between L and J from
    the norms of L, J and S, clamped to ``[-1, 1]``."""
    cos = 0.5 * (j_norm * j_norm + l_norm * l_norm - s_norm_ * s_norm_) / (l_norm * j_norm)
    return jnp.clip(cos, -1.0, 1.0)


def _phiz_sum(system, p, j_norm_):
    """``IMRPhenomX_Return_phiz_MSA``, Eq. 66, without the integration
    constant."""
    eta = system.eta
    invv = p.invv
    invv2 = invv * invv
    l_newt = eta / p.v
    c1 = system.c1
    c12 = c1 * c1
    sav2 = system.sav2
    sav = system.sav
    inv_sav = system.inv_sav
    inv_sav2 = system.inv_sav2
    v = p.v

    # Eqs. D27, D28
    log1 = _log_abs(c1 + j_norm_ * eta + eta * l_newt)
    log2 = _log_abs(c1 + j_norm_ * sav * v + sav2 * v)

    # Eqs. D22 - D27
    phiz_0_coeff = ((j_norm_ * system.inveta4)
                    * (0.5 * c12 - c1 * system.eta2 * invv / 6.0 - sav2 * system.eta2 / 3.0
                       - system.eta4 * invv2 / 3.0)
                    - (c1 * 0.5 * system.inveta) * (c12 * system.inveta4 - sav2 * system.inveta2) * log1)
    phiz_1_coeff = (-0.5 * j_norm_ * system.inveta2 * (c1 + eta * l_newt)
                    + 0.5 * system.inveta3 * (c12 - system.eta2 * sav2) * log1)
    phiz_2_coeff = -j_norm_ + sav * log2 - c1 * log1 * system.inveta
    phiz_3_coeff = j_norm_ * v - eta * log1 + c1 * log2 * inv_sav
    phiz_4_coeff = ((0.5 * j_norm_ * inv_sav2 * v) * (c1 + v * sav2)
                    - (0.5 * inv_sav2 * inv_sav) * (c12 - system.eta2 * sav2) * log2)
    phiz_5_coeff = (-j_norm_ * v * ((0.5 * c12 * inv_sav2 * inv_sav2) - (c1 * v * inv_sav2 / 6.0) - v * v / 3.0
                                    - system.eta2 * inv_sav2 / 3.0)
                    + (0.5 * c1 * inv_sav2 * inv_sav2 * inv_sav) * (c12 - system.eta2 * sav2) * log2)

    o = system.omegaz_coeff
    return (phiz_0_coeff * o[0] + phiz_1_coeff * o[1] + phiz_2_coeff * o[2] + phiz_3_coeff * o[3]
            + phiz_4_coeff * o[4] + phiz_5_coeff * o[5])


def _phiz(system, p, j_norm_):
    """``IMRPhenomX_Return_phiz_MSA`` (with LAL's NaN check)."""
    return _nan_to_zero(_phiz_sum(system, p, j_norm_) + system.phiz_0)


def _zeta(system, p):
    """``IMRPhenomX_Return_zeta_MSA``, Eq. F5 (with LAL's NaN check)."""
    o = system.omegazeta_coeff
    zeta = system.eta * (o[0] * p.invv3 + o[1] * p.invv2 + o[2] * p.invv + o[3] * p.logv + o[4] * p.v
                         + o[5] * p.v2) + system.zeta_0
    return _nan_to_zero(zeta)


def _corrections(system, p, l_norm, j_norm_, roots_, paper=False):
    """``IMRPhenomX_Return_MSA_Corrections_MSA`` where
    ``|S_+^2 - S_-^2| > 1e-5`` (zero elsewhere): the first MSA corrections to
    ``phi_z`` (Eq. 67) and ``zeta`` (Eq. F19).

    ``paper`` (a boolean, possibly traced) selects the expressions LAL uses
    when the precession struct does not carry version 222 or 223 (Eqs. 67
    and F19 of arXiv:1703.03967) over those of ``LALSimInspiralFDPrecAngles``
    (version 223)."""
    s32, smi2, spl2 = roots_
    has_corrections = jnp.abs(smi2 - spl2) > 1.0e-5
    # safe roots where the corrections are not evaluated
    s32 = jnp.where(has_corrections, s32, -1.0)
    smi2 = jnp.where(has_corrections, smi2, 0.5)
    spl2 = jnp.where(has_corrections, spl2, 1.0)

    eta = system.eta
    seff = system.seff
    v1, v2, v3, v4, v6 = p.v, p.v2, p.v3, p.v4, p.v6
    spl2_m_smi2 = spl2 - smi2
    spl = _sqrt(spl2)
    j_norm2 = j_norm_ * j_norm_
    l_norm2 = l_norm * l_norm
    j2_m_spl2 = j_norm2 - spl2

    # Eqs. B6 - B8 (IMRPhenomX_Return_Constants_c_MSA, versions != 220)
    c0 = j_norm_ * (0.75 * (1.0 - seff * v1) * v2
                    * (system.eta3 + 4.0 * system.eta3 * seff * v1
                       - 2.0 * eta * (j2_m_spl2 + 2.0 * (system.s1_norm_2 - system.s2_norm_2) * system.delta_qq) * v2
                       - 4.0 * eta * seff * j2_m_spl2 * v3 + j2_m_spl2 * j2_m_spl2 * v4 * system.inveta))
    c2 = j_norm_ * (-1.5 * eta * (spl2 - smi2) * (1.0 + 2.0 * seff * v1 - j2_m_spl2 * v2 * system.inveta2)
                    * (1.0 - seff * v1) * v4)
    c4 = j_norm_ * (0.75 * system.inveta * (spl2 - smi2) * (spl2 - smi2) * (1.0 - seff * v1) * v6)

    # Eqs. B9 - B11 (IMRPhenomX_Return_Constants_d_MSA)
    d0 = -(j_norm2 - (l_norm + spl) * (l_norm + spl)) * ((j_norm2 - (l_norm - spl) * (l_norm - spl)))
    d2 = -2.0 * (spl2 - smi2) * (j_norm2 + l_norm2 - spl2)
    d4 = -(spl2 - smi2) * (spl2 - smi2)

    two_d0 = 2.0 * d0
    # Eq. B20
    sd = _sqrt_abs(d2 * d2 - 4.0 * d0 * d4)
    # Eqs. F20, F21
    a_theta_l = 0.5 * ((j_norm_ / l_norm) + (l_norm / j_norm_) - (spl2 / (j_norm_ * l_norm)))
    b_theta_l = 0.5 * spl2_m_smi2 / (j_norm_ * l_norm)
    # Eqs. B16, B17
    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd
    nc = _div(nc_num, nc_denom)
    nd = _div(nc_denom, two_d0)
    sqrt_nc = _sqrt_abs(nc)
    sqrt_nd = _sqrt_abs(nd)

    # phase of S and its derivative (IMRPhenomX_Return_Psi_MSA, _Psi_dot_MSA)
    psi = (-0.75 * system.g0 * system.delta_qq * (1.0 + system.psi1 * v1 + system.psi2 * v2) / (v2 * v1)
           + system.psi0)
    a_coeff = -1.5 * (v2 * v2 * v2) * (1.0 - v1 * seff) * system.sqrt_inveta
    psi_dot = 0.5 * a_coeff * _sqrt(spl2 - s32)
    tan_psi = jnp.tan(psi)
    atan_psi = jnp.arctan(tan_psi)

    # Eq. B19 denominator
    c2den = 2.0 * d0 * sd * (d0 + d2 + d4)

    # Eq. 67 as in LALSimInspiralFDPrecAngles
    cphi_num = (c4 * d0 * ((2 * d0 + d2) + sd) - c2 * d0 * ((d2 + 2. * d4) - sd)
                - c0 * ((2 * d0 * d4) - (d2 + d4) * (d2 - sd)))
    psi_dot_safe = jnp.where(psi_dot == 0.0, 1.0, psi_dot)
    cphi_fd = (jnp.abs(_div(cphi_num, c2den)) * (_div(sqrt_nc, nc - 1.) * (atan_psi - jnp.arctan(sqrt_nc * tan_psi)))
               / psi_dot_safe)
    dphi_num = (-c4 * d0 * ((2 * d0 + d2) - sd) + c2 * d0 * ((d2 + 2. * d4) + sd)
                - c0 * (-(2 * d0 * d4) + (d2 + d4) * (d2 + sd)))
    dphi_fd = (_div(jnp.abs(dphi_num), c2den) * (_div(sqrt_nd, nd - 1.) * (atan_psi - jnp.arctan(sqrt_nd * tan_psi)))
               / psi_dot_safe)
    cphi_term = jnp.where(nc == 1.0, 0.0, cphi_fd)
    dphi_term = jnp.where(nd == 1.0, 0.0, dphi_fd)
    phiz_msa = cphi_term + dphi_term
    # Eq. F19 as in LALSimInspiralFDPrecAngles
    zeta_msa = (a_theta_l * phiz_msa
                + 2. * b_theta_l * d0 * (_div(cphi_term, sd - d2) - _div(dphi_term, sd + d2)))

    if paper is not False:
        # Eqs. B18, B19, 67 and F19 of arXiv:1703.03967
        c_1 = -0.5 * (_div(c0, d0) - 2.0 * _div(c0 + c2 + c4, nc_num))
        c2num = c0 * (-2.0 * d0 * d4 + d2 * d2 + d2 * d4) - c2 * d0 * (d2 + 2.0 * d4) + c4 * d0 * (two_d0 + d2)
        c_2 = _div(c2num, c2den)
        c_phi = c_1 + c_2
        d_phi = c_1 - c_2
        cphi_paper = (((c_phi / psi_dot_safe) * _div(sqrt_nc, nc - 1.0))
                      * jnp.arctan(((1.0 - sqrt_nc) * tan_psi) / (1.0 + (sqrt_nc * tan_psi * tan_psi))))
        dphi_paper = (((d_phi / psi_dot_safe) * _div(sqrt_nd, nd - 1.0))
                      * jnp.arctan(((1.0 - sqrt_nd) * tan_psi) / (1.0 + (sqrt_nd * tan_psi * tan_psi))))
        phiz_paper = jnp.where(nc == 1.0, 0.0, cphi_paper) + jnp.where(nd == 1.0, 0.0, dphi_paper)
        zeta_paper = (((a_theta_l * (c_phi + d_phi))
                       + (2.0 * d0 * b_theta_l) * (_div(c_phi, sd - d2) - _div(d_phi, sd + d2))) / psi_dot_safe)
        phiz_msa = jnp.where(paper, phiz_paper, phiz_msa)
        zeta_msa = jnp.where(paper, zeta_paper, zeta_msa)

    phiz_msa = jnp.where(has_corrections, _nan_to_zero(phiz_msa), 0.0)
    zeta_msa = jnp.where(has_corrections, _nan_to_zero(zeta_msa), 0.0)
    return phiz_msa, zeta_msa


def _state(system, v):
    """The per-frequency quantities of
    ``IMRPhenomX_Return_phi_zeta_costhetaL_MSA``."""
    p = _vpowers(v)
    l_norm = system.eta / p.v
    jn = j_norm(system, l_norm)
    roots_ = roots(system, l_norm, jn)
    return p, l_norm, jn, roots_, s_norm(system, p.v, roots_)


def phi_zeta_costhetal(system, v, paper_corrections=False, l=None):
    """``IMRPhenomX_Return_phi_zeta_costhetaL_MSA``: a :class:`~._math.Struct`
    with ``phiz``, ``zeta`` and ``cos_theta_l`` at the orbital velocities
    ``v`` (``v = (pi M f)^(1/3)`` for the co-precessing mode ``m' = 2``; XPHM
    evaluates the mode ``m'`` at ``v = (2 pi M f / m')^(1/3)``), as arrays of
    the shape of ``v``.

    ``phi_z`` is the Euler angle ``alpha`` and ``zeta`` the angle ``epsilon =
    -gamma``, both before the reference offsets; ``cos(theta_L)`` is
    ``cos(beta)``.

    By default these are LAL's angles for version 223. ``paper_corrections``
    (a possibly traced boolean) selects LAL's MSA corrections for the versions
    other than 222 and 223 and ``l`` (an orbital angular momentum of
    :mod:`.precession`, used where ``paper_corrections`` is true) the 3PN
    orbital angular momentum of ``cos(theta_L)``: what LAL evaluates when the
    MSA system failed and the precession struct carries version 102
    (``XLALSimIMRPhenomXPMSAAngles`` does this).
    """
    p, l_norm, jn, roots_, sn = _state(system, v)
    phiz_corr, zeta_corr = _corrections(system, p, l_norm, jn, roots_, paper=paper_corrections)
    phiz = _phiz(system, p, jn) + phiz_corr
    zeta = _zeta(system, p) + zeta_corr
    l_norm3 = l_norm_3pn(system, p.v, p.v2, l_norm)
    if l is not None:
        from .precession import l_ansatz

        l_norm3 = jnp.where(paper_corrections, l_ansatz(l, p.v, l_norm), l_norm3)
    cos_theta_l = cos_theta_lj(l_norm3, j_norm(system, l_norm3), sn)
    return Struct(phiz=phiz, zeta=zeta, cos_theta_l=cos_theta_l)


def cos_theta_l_with(system, v, l):
    """``cos(theta_L)`` of :func:`phi_zeta_costhetal` as LAL evaluates it when
    the precession struct does not carry version 222 or 223 (as in the PNR
    angles): the 3PN orbital angular momentum is ``XLALSimIMRPhenomXLPNAnsatz``
    with the coefficients ``l`` (a :mod:`.precession` orbital angular
    momentum) instead of the MSA expansion."""
    from .precession import l_ansatz

    p, l_norm, jn, roots_, sn = _state(system, v)
    l_norm3 = l_ansatz(l, p.v, l_norm)
    return cos_theta_lj(l_norm3, j_norm(system, l_norm3), sn)


def final_spin(system, af_parallel, m, m1_2, m2_2, chi1l, chi2l):
    """The final spin of ``IMRPhenomX_SetPrecessingRemnantParams`` for the
    MSA angles (``PhenomXPFinalSpinMod`` 3 with the default
    ``PhenomXPTransPrecessionMethod`` 1): from the precession-averaged
    couplings and the aligned-spin final spin ``af_parallel``. ``m`` is the
    total mass, ``m1_2``, ``m2_2`` the squared masses of the precession
    struct (units of the total mass) and ``chi1l``, ``chi2l`` the aligned
    spins.

    The sum ``S1L_pav + S2L_pav`` is evaluated without LAL's cancellation of
    two terms that diverge at equal masses (same values to rounding; it
    vanishes at ``q = 1`` exactly, as in LAL)."""
    l_final = m * m * af_parallel - m1_2 * chi1l - m2_2 * chi2l
    sign = jnp.copysign(1.0, af_parallel)
    return sign * _sqrt(system.sav2 + l_final * l_final + 2.0 * l_final * system.sl_pav) / (m * m)
