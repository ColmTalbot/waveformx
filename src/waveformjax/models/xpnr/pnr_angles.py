# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The model-independent parts of the PNR precession angles of
IMRPhenomXPNR (Hamilton et al. 2021, arXiv:2107.08876), after waveformrs's
``imrphenomx/pnr_angles.rs`` (``LALSimIMRPhenomX_PNR_alpha.c``,
``LALSimIMRPhenomX_PNR_beta.c`` and ``LALSimIMRPhenomX_PNR_internals.c`` of
LALSimulation 6.2.0):

- the merger-ringdown (MR) expressions of ``alpha`` and ``beta`` and their
  connection to the inspiral (C1 interpolation of ``alpha``, rescaling of
  the inspiral ``beta``, connection frequencies of ``beta`` from the
  turning points and the inflection point of the MR ``beta``);
- the "waveform" ``beta`` of the inspiral (Eq. 41 of arXiv:2107.08876);
- the windows that blend the PNR and PN angles outside the calibration
  region and switch off the co-precessing deviations;
- the linear frequency map of the angles of the higher modes, and the
  Boole's-rule step of the integral of ``gamma``.

Frequencies ``mf`` are in units of the total mass, angles in radians. The
parameters of the MR angles are :class:`~._math.Struct` records made by
:func:`alpha_params` and :func:`beta_params` (LAL's
``IMRPhenomX_PNR_alpha_parameters`` and ``IMRPhenomX_PNR_beta_parameters``
in snake case).

Every function works on traced (and batched) values. LAL's ``if``\\ s are
``jnp.where`` selections. The connection frequencies of ``beta`` involve
many decisions on values that LAL may compute as NaN (complex roots of the
inflection cubic, square roots of negative numbers, ...), which LAL's
comparisons then treat as false. They are evaluated twice: once with LAL's
arithmetic (NaNs included, under ``stop_gradient``) to take the decisions,
and once with finite stand-ins for the NaNs, following the recorded
decisions, to give the values and their gradients. The values are the same
wherever LAL's are finite, and the gradients are finite.

Arithmetic: :func:`~._math.fma` is ``a * b + c``, which XLA contracts into
fused multiply-adds when it compiles (as clang does in LAL's arm64 build,
though not always for the same pairs). This is rounding-level everywhere but
in the inflection points of the MR ``beta``: they are roots of a cubic from
the trigonometric formula, ``acos`` of a value within a few ulps of 1 in
typical cases (an error ``eps`` there moves the roots by ``sqrt(eps)``) and a
cancellation of terms ~1e3 times larger than the roots: they agree with LAL
to ~1e-8 (relative), which moves the connection frequencies of ``beta`` by as
much (a mismatch of ~1e-16). LAL's complex arithmetic, which has no fused
multiply-adds, is kept from being contracted (``_rounded``).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax

from ._math import NAN, PI, Struct, fma, fnms, is_lean, safe_sqrt, tiny, real

__all__ = [
    "PNR_Q_WINDOW_LOWER", "PNR_CHI_WINDOW_LOWER", "PNR_Q_WINDOW_UPPER", "PNR_CHI_WINDOW_UPPER",
    "PNR_DMF", "MAX_TOL_ATAN",
    "alpha_params", "beta_params",
    "alpha_interpolation_0", "alpha_interpolation_1", "alpha_interpolation_2",
    "alpha_interpolation_3", "intermediate_alpha", "mr_alpha",
    "mr_beta", "mr_dbeta", "mr_ddbeta", "mr_dddbeta",
    "beta_rescaling_1", "beta_rescaling_2", "rescale_beta",
    "st_beta_coeffs_0", "st_beta_coeffs_1", "st_beta_coeffs_2", "st_beta_coeffs_3",
    "beta_connection", "arctan_window", "attach_mr_beta", "attach_weight",
    "inflection_cubic", "two_inflection_points", "three_inflection_points",
    "single_inflection_point", "beta_connection_frequencies",
    "chi_calc", "pn_waveform_beta",
    "angles_window", "coprec_window",
    "linear_frequency_slope", "linear_frequency_map",
    "linear_frequency_map_transition_frequencies",
    "boole_step", "alphadot_cosbeta", "sqrt_one_minus_square",
]

#: Calibration region of the PNR angles: ``q <= 8.5`` and ``chi <= 0.85``.
PNR_Q_WINDOW_LOWER = 8.5
PNR_CHI_WINDOW_LOWER = 0.85
#: End of the transition to the PN angles: ``q <= 12`` and ``chi <= 1.2``.
PNR_Q_WINDOW_UPPER = 12.0
PNR_CHI_WINDOW_UPPER = 1.2
#: Frequency step of the finite differences of the connections (``dMf``).
PNR_DMF = 0.0005
#: ``MAX_TOL_ATAN`` of ``LALSimIMRPhenomX_precession.h``.
MAX_TOL_ATAN = 1.0e-15

# ---------------------------------------------------------------------------
# parameter structs
# ---------------------------------------------------------------------------

_ALPHA_FIELDS = (
    "a1", "a2", "a3", "a4", "mf_alpha_lower", "mf_alpha_upper", "alpha_lower", "alpha_upper",
    "derivative_alpha_lower", "derivative_alpha_upper", "alpha_interp", "alpha_mr_offset",
)
_BETA_FIELDS = (
    "b0", "b1", "b2", "b3", "b4", "b5", "mf_beta_lower", "mf_beta_upper", "beta_lower",
    "beta_upper", "derivative_beta_lower", "derivative_beta_upper", "beta_rescale_1",
    "beta_rescale_2", "beta_interp", "beta_freeze", "attach_weight",
)


def _params(names, interp, fields):
    unknown = set(fields) - set(names)
    if unknown:
        raise TypeError(f"unknown fields {sorted(unknown)}")
    shape = jnp.broadcast_shapes(*(jnp.shape(x) for x in jax.tree_util.tree_leaves(fields)))
    zero = jnp.zeros(shape, real()) if shape else 0.0
    values = {name: zero for name in names}
    values[interp] = (zero,) * 4
    values.update(fields)
    return Struct(**values)


def alpha_params(**fields):
    """``IMRPhenomX_PNR_alpha_parameters``: the coefficients ``a1`` ... ``a4``
    of the MR ``alpha`` (Eq. 48 of arXiv:2107.08876), the connection
    frequencies ``mf_alpha_lower``, ``mf_alpha_upper`` with the values
    and derivatives of ``alpha`` there, the coefficients ``alpha_interp``
    (a 4-tuple: LAL's ``alpha_interp_0`` ... ``_3``) of the intermediate
    ``alpha`` (Eq. 51) and ``alpha_mr_offset``. Unset fields are zeros of
    the broadcast shape of the given ones (so that a batch can be mapped
    over with ``jax.vmap``)."""
    return _params(_ALPHA_FIELDS, "alpha_interp", fields)


def beta_params(**fields):
    """``IMRPhenomX_PNR_beta_parameters``: the coefficients ``b0`` ... ``b5``
    of the MR ``beta`` (Eq. 49 of arXiv:2107.08876), the connection
    frequencies ``mf_beta_lower``, ``mf_beta_upper`` with the values and
    derivatives of ``beta`` there, the rescaling ``beta_rescale_1``,
    ``beta_rescale_2`` of the inspiral ``beta`` (Eqs. 53-55) and the cubic
    ``beta_interp`` (a 4-tuple: LAL's ``beta_interp_0`` ... ``_3``)
    connecting the SpinTaylor and PNR ``beta`` (version 330), and
    ``beta_freeze``, the weight of ``beta`` held at its value at
    ``mf_beta_upper`` above it (1 as in LAL; between 0 and 1 for the smooth
    connection, see :func:`beta_connection_frequencies`). Unset fields are
    zeros as in :func:`alpha_params` (``beta_freeze``: ones)."""
    out = _params(_BETA_FIELDS, "beta_interp", fields)
    if "beta_freeze" not in fields:
        out.beta_freeze = out.beta_freeze + 1.0
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ratio(numerator, denominator):
    """``numerator / denominator``, and 0 where the denominator vanishes.

    The interpolation and rescaling coefficients of LAL divide by zero at
    the ``Mf = 100`` sentinels of the connection frequencies (or for a
    vanishing ``beta``), giving NaN, which its callers replace by 0, or
    +-inf (from the rounding residues of the numerators), which then goes
    unused (the interval between the connection frequencies is empty)."""
    ok = denominator != 0
    return jnp.where(ok, numerator / jnp.where(ok, denominator, 1.0), 0.0)


def _rounded(*values):
    """The values, rounded: an ``optimization_barrier`` keeps XLA (which
    contracts ``a * b + c`` into fused multiply-adds when it compiles) from
    fusing a product into the sum or difference that uses it (not in the
    lean model)."""
    if not is_lean():
        values = lax.optimization_barrier(values)
    return values[0] if len(values) == 1 else values


def _atan2_tol(a, b, tol):
    """``XLALSimIMRPhenomXatan2tol``: ``atan2(a, b)``, 0 where both
    arguments are below ``tol`` in magnitude (zero gradient there)."""
    small = (jnp.abs(a) < tol) & (jnp.abs(b) < tol)
    return jnp.where(small, 0.0, jnp.arctan2(jnp.where(small, 1.0, a), jnp.where(small, 1.0, b)))


class _Branches:
    """The discrete decisions of an evaluation with LAL's arithmetic,
    recorded in order, replayed by the differentiable evaluation (see the
    module docstring)."""

    def __init__(self, recorded=None):
        self.replay = recorded is not None
        self.recorded = [] if recorded is None else list(recorded)
        self.index = 0

    def __call__(self, condition):
        if self.replay:
            condition = self.recorded[self.index]
            self.index += 1
            return condition
        self.recorded.append(condition)
        return condition


def _raw_and_safe(function, *args):
    """``function(*args, safe, branch)`` with LAL's arithmetic (``safe =
    False``, NaNs included, no gradient) for the decisions and with finite
    stand-ins (``safe = True``) for the values: LAL's value where it is not
    finite, the differentiable one elsewhere.

    The arguments pass an ``optimization_barrier``: XLA would otherwise
    duplicate their (expensive) computation, e.g. the fits of the
    coefficients, into the many kernels of both evaluations and their
    derivatives, which multiplies the compilation time."""
    args = lax.optimization_barrier(args)
    branches = _Branches()
    raw = function(*jax.tree_util.tree_map(lax.stop_gradient, args), safe=False, branch=branches)
    safe = function(*args, safe=True, branch=_Branches(branches.recorded))
    return jax.tree_util.tree_map(lambda r, s: jnp.where(jnp.isfinite(r), s, r), raw, safe)


def _sqrt(x, safe):
    """``sqrt``; with ``safe``, 0 (zero gradient) for negative arguments and
    a zero gradient at 0."""
    if not safe:
        return jnp.sqrt(x)
    positive = x > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), jnp.where(x == 0, x, 0.0))


def _div(numerator, denominator, safe):
    """``numerator / denominator``; with ``safe``, ``numerator`` where the
    denominator vanishes."""
    if not safe:
        return numerator / denominator
    return numerator / jnp.where(denominator != 0, denominator, 1.0)


def _acos(x, safe):
    """``acos``; with ``safe``, 0 or pi (zero gradient) outside ``(-1, 1)``."""
    if not safe:
        return jnp.arccos(x)
    inside = jnp.abs(x) < 1
    return jnp.where(inside, jnp.arccos(jnp.where(inside, x, 0.0)), jnp.where(x > 0, 0.0, PI))


# complex numbers as (re, im) pairs with C's double complex arithmetic as
# compiled in LAL: no fused multiply-adds, signed zeros included (they
# choose the branch of csqrt)


def _cmul(x, y):
    rr, ii, ri, ir = _rounded(x[0] * y[0], x[1] * y[1], x[0] * y[1], x[1] * y[0])
    return (rr - ii, ri + ir)


def _cscale(x, s):
    """``real * complex``, componentwise."""
    return _rounded(s * x[0], s * x[1])


def _cadd(x, y):
    return (x[0] + y[0], x[1] + y[1])


def _csub(x, y):
    return (x[0] - y[0], x[1] - y[1])


def _cdiv(x, y, safe):
    """``complex / complex``: compiler-rt's ``__divdc3`` (its scaling by
    ``logb`` of the denominator is exact and only matters for overflow; the
    NaN recovery branches are not needed here)."""
    (a, b), (c, d) = x, y
    denom = fma(c, c, d * d)
    if safe:
        denom = jnp.where(denom != 0, denom, 1.0)
    return (fma(a, c, b * d) / denom, fma(b, c, -(a * d)) / denom)


def _csqrt(x, safe):
    """C99 ``csqrt`` for an argument with vanishing imaginary part (whose
    sign gives the sign of the imaginary part of the root of a negative
    number)."""
    re, im = x
    negative = re < 0
    if safe:
        positive = re > 0
        root = jnp.where(positive, jnp.sqrt(jnp.where(positive, re, 1.0)), re)
        root_neg = jnp.sqrt(jnp.where(negative, -re, 1.0))
        return (jnp.where(negative, 0.0, root), jnp.where(negative, jnp.copysign(root_neg, im), im))
    out_re = jnp.where(negative, 0.0, jnp.sqrt(re))
    out_im = jnp.where(negative, jnp.copysign(jnp.sqrt(-re), im), im)
    return (out_re, jnp.where(jnp.isnan(re), re, out_im))


# ---------------------------------------------------------------------------
# alpha
# ---------------------------------------------------------------------------


def alpha_interpolation_0(mf1, mf2, alpha1, alpha2, dalpha1, dalpha2):
    """``IMRPhenomX_PNR_alpha_interpolation_0``: the coefficient of ``Mf^2``
    of the intermediate ``alpha`` (Eq. 52 of arXiv:2107.08876) matching
    ``alpha1``, ``dalpha1`` at ``mf1`` and ``alpha2``, ``dalpha2`` at
    ``mf2``. 0 for ``mf1 == mf2`` (the ``Mf = 100`` sentinel), where LAL
    divides by zero (see :func:`_ratio`)."""
    d = (mf2 - mf1) * (mf2 - mf1) * (mf2 - mf1)
    n = fma(2.0, fma(mf1, alpha1, -(mf2 * alpha2)), -((mf1 - mf2) * (fma(mf1, dalpha1, mf2 * dalpha2) + (alpha1 + alpha2))))
    return _ratio(n, d)


def alpha_interpolation_1(mf1, mf2, alpha1, alpha2, dalpha1, dalpha2):
    """``IMRPhenomX_PNR_alpha_interpolation_1``: the coefficient of ``Mf``
    (see :func:`alpha_interpolation_0`)."""
    d = (mf2 - mf1) * (mf2 - mf1) * (mf2 - mf1)
    n = fma(3.0 * (mf1 + mf2), fma(mf2, alpha2, -(mf1 * alpha1)), (mf1 - mf2) * fma(fma(2.0, mf2, mf1), fma(mf1, dalpha1, alpha1), fma(2.0, mf1, mf2) * fma(mf2, dalpha2, alpha2)))
    return _ratio(n, d)


def alpha_interpolation_2(mf1, mf2, alpha1, alpha2, dalpha1, dalpha2):
    """``IMRPhenomX_PNR_alpha_interpolation_2``: the constant term (see
    :func:`alpha_interpolation_0`)."""
    d = (mf2 - mf1) * (mf2 - mf1) * (mf2 - mf1)
    n = fma(6.0 * mf1 * mf2, fma(mf1, alpha1, -(mf2 * alpha2)), -((mf1 - mf2) * fma(mf2 * fma(2.0, mf1, mf2), fma(mf1, dalpha1, alpha1), mf1 * fma(2.0, mf2, mf1) * fma(mf2, dalpha2, alpha2))))
    return _ratio(n, d)


def alpha_interpolation_3(mf1, mf2, alpha1, alpha2, dalpha1, dalpha2):
    """``IMRPhenomX_PNR_alpha_interpolation_3``: the coefficient of ``1 /
    Mf`` (see :func:`alpha_interpolation_0`)."""
    d = (mf2 - mf1) * (mf2 - mf1) * (mf2 - mf1)
    n = fma(mf1 * mf2 * (mf1 - mf2), fma(mf2, fma(mf1, dalpha1, alpha1), mf1 * fma(mf2, dalpha2, alpha2)), fma(mf1 * mf2 * mf2 * fma(-3.0, mf1, mf2), alpha1, -(mf1 * mf1 * mf2 * fma(-3.0, mf2, mf1) * alpha2)))
    return _ratio(n, d)


def intermediate_alpha(mf, p):
    """``IMRPhenomX_PNR_intermediate_alpha_expression``, Eq. 51 of
    arXiv:2107.08876: ``a0 Mf^2 + a1 Mf + a2 + a3 / Mf`` with ``(a0, a1, a2,
    a3) = p.alpha_interp``."""
    a0, a1, a2, a3 = p.alpha_interp
    return fma(a0 * mf, mf, a1 * mf) + a2 + a3 / mf


def mr_alpha(mf, p):
    """``IMRPhenomX_PNR_MR_alpha_expression``, Eq. 48 of arXiv:2107.08876:
    ``-(A1 / Mf + A2 sqrt(A3) / (A3 + (Mf - A4)^2))``."""
    return -(p.a1 / mf + (p.a2 * jnp.sqrt(p.a3)) / fma(mf - p.a4, mf - p.a4, p.a3))


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------


def mr_beta(mf, p):
    """``IMRPhenomX_PNR_MR_beta_expression``, Eq. 49 of arXiv:2107.08876:
    ``B0 + (B1 + B2 Mf + B3 Mf^2) / (1 + B4 (Mf + B5)^2)``."""
    return p.b0 + fma(p.b3 * mf, mf, fma(p.b2, mf, p.b1)) / fma(p.b4 * (mf + p.b5), mf + p.b5, 1.0)


def mr_dbeta(mf, p):
    """``IMRPhenomX_PNR_MR_dbeta_expression``: the derivative of
    :func:`mr_beta` with respect to ``Mf``.

    The derivatives of the MR ``beta`` only enter the connection
    frequencies; they are written with LAL's fused multiply-adds
    (:func:`~._math.fma`)."""
    den = fma(p.b4 * (mf + p.b5), mf + p.b5, 1.0)
    return (fma(fma(2.0 * p.b3 * p.b4, p.b5, -(p.b2 * p.b4)) * mf, mf, fma(2.0 * p.b3 * p.b4 * p.b5, p.b5, fma(2.0, p.b3, -(2.0 * p.b1 * p.b4))) * mf) + fma(p.b2 * p.b4 * p.b5, p.b5, fma(-(2.0 * p.b1 * p.b4), p.b5, p.b2))) / (den * den)


def mr_ddbeta(mf, p):
    """``IMRPhenomX_PNR_MR_ddbeta_expression``: the second derivative of
    :func:`mr_beta`."""
    b1, b2, b3, b4, b5 = p.b1, p.b2, p.b3, p.b4, p.b5
    a = fma(b2 * b4, b4, -(2.0 * b3 * b4 * b4 * b5))
    b = fma(-(3.0 * b3 * b4 * b4 * b5), b5, fma(-3.0 * b3, b4, 3.0 * b1 * b4 * b4))
    c = fma(-(3.0 * b2 * b4 * b4 * b5), b5, fma(-3.0 * b2, b4, 6.0 * b1 * b4 * b4 * b5))
    d = fma(b3 * b4 * b4 * b5 * b5 * b5, b5, fma(-(2.0 * b2 * b4 * b4 * b5 * b5), b5, fma(3.0 * b1 * b4 * b4 * b5, b5, fma(2.0 * b3 * b4 * b5, b5, fma(-(2.0 * b2 * b4), b5, fma(-b1, b4, b3))))))
    return 2.0 * (fma(c, mf, fma(a * mf * mf, mf, b * mf * mf)) + d) / fma(b4 * (b5 + mf), b5 + mf, 1.0) ** 3


def mr_dddbeta(mf, p):
    """``IMRPhenomX_PNR_MR_dddbeta_expression``: the third derivative of
    :func:`mr_beta`."""
    b1, b2, b3, b4, b5 = p.b1, p.b2, p.b3, p.b4, p.b5
    a = fma(-b2 * b4, b4, 2.0 * b3 * b4 * b4 * b5)
    b = fma(4.0 * b3 * b4 * b4 * b5, b5, fma(4.0 * b3, b4, -(4.0 * b1 * b4 * b4)))
    c = fma(6.0 * b2 * b4 * b4 * b5, b5, fma(6.0 * b2, b4, -(12.0 * b1 * b4 * b4 * b5)))
    d = fma(-(4.0 * b3 * b4 * b4 * b5 * b5 * b5), b5, fma(8.0 * b2 * b4 * b4 * b5 * b5, b5, fma(-(12.0 * b1 * b4 * b4 * b5), b5, fma(-(8.0 * b3 * b4 * b5), b5, fma(8.0 * b2 * b4, b5, fma(-4.0, b3, 4.0 * b1 * b4))))))
    e = fma(-(2.0 * b3 * b4 * b4 * b5 * b5 * b5 * b5), b5, fma(3.0 * b2 * b4 * b4 * b5 * b5 * b5, b5, fma(-(4.0 * b1 * b4 * b4 * b5 * b5), b5, fma(-(4.0 * b3 * b4 * b5 * b5), b5, fma(2.0 * b2 * b4 * b5, b5, fma(4.0 * b1 * b4, b5, fma(-(2.0 * b3), b5, -b2)))))))
    return 6.0 * b4 * (fma(d, mf, fma(c * mf, mf, fma(a * mf * mf * mf, mf, b * mf * mf * mf))) + e) / fma(b4 * (b5 + mf), b5 + mf, 1.0) ** 4


def beta_rescaling_1(mf, beta1, beta2, dbeta1, dbeta2):
    """``IMRPhenomX_PNR_beta_rescaling_1``, ``b1`` of Eq. 54 of
    arXiv:2107.08876, from the inspiral ``beta1``, ``dbeta1`` and the MR
    ``beta2``, ``dbeta2`` at ``mf``. 0 where LAL divides by zero
    (``beta1 Mf = 0``; its callers replace the NaN by 0)."""
    d = beta1 * beta1 * mf
    n = fma(-2.0 * beta1, beta2 - beta1, fma(beta1, dbeta2, -(beta2 * dbeta1)) * mf)
    return _ratio(-n, d)


def beta_rescaling_2(mf, beta1, beta2, dbeta1, dbeta2):
    """``IMRPhenomX_PNR_beta_rescaling_2``, ``b2`` of Eq. 55 of
    arXiv:2107.08876 (see :func:`beta_rescaling_1`)."""
    d = beta1 * mf * (beta1 * mf)
    n = fma(beta1, beta2 - beta1, -(fma(beta1, dbeta2, -(beta2 * dbeta1)) * mf))
    return _ratio(-n, d)


def rescale_beta(mf, p):
    """``IMRPhenomX_PNR_rescale_beta_expression``, Eq. 53 of
    arXiv:2107.08876: ``1 + b1 Mf + b2 Mf^2``."""
    return fma(p.beta_rescale_2 * mf, mf, fma(p.beta_rescale_1, mf, 1.0))


def st_beta_coeffs_0(mf1, mf2, beta1, beta2, dbeta1, dbeta2):
    """``IMRPhenomX_ST_PNR_beta_coeffs_0``: the constant term of the cubic
    connecting the SpinTaylor ``beta1``, ``dbeta1`` at ``mf1`` to the PNR
    ``beta2``, ``dbeta2`` at ``mf2`` (version 330). 0 for ``mf1 == mf2``,
    where LAL divides by zero."""
    mf1_2 = mf1 * mf1
    mf1_3 = mf1_2 * mf1
    mf2_2 = mf2 * mf2
    mf2_3 = mf2_2 * mf2
    n = fma(-(dbeta1 * mf1), mf2_3, fma(beta1, mf2_3, fma(-(dbeta2 * mf1_2), mf2_2, fma(dbeta1 * mf1_2, mf2_2, fma(-(3.0 * beta1 * mf1), mf2_2, fma(dbeta2 * mf1_3, mf2, fma(-beta2, mf1_3, 3.0 * beta2 * mf1_2 * mf2)))))))
    d = (mf1 - mf2) * (mf1 - mf2) * (mf1 - mf2)
    return _ratio(-n, d)


def st_beta_coeffs_1(mf1, mf2, beta1, beta2, dbeta1, dbeta2):
    """``IMRPhenomX_ST_PNR_beta_coeffs_1``: the coefficient of ``Mf`` (see
    :func:`st_beta_coeffs_0`)."""
    mf1_2 = mf1 * mf1
    mf1_3 = mf1_2 * mf1
    mf2_2 = mf2 * mf2
    mf2_3 = mf2_2 * mf2
    n = fma(dbeta1, mf2_3, fma(2.0 * dbeta2 * mf1, mf2_2, fma(dbeta1 * mf1, mf2_2, fma(-(dbeta2 * mf1_2), mf2, fma(-(2.0 * dbeta1 * mf1_2), mf2, fma(-(6.0 * beta2 * mf1), mf2, fma(-dbeta2, mf1_3, 6.0 * beta1 * mf1 * mf2)))))))
    d = (mf1 - mf2) * (mf1 - mf2) * (mf1 - mf2)
    return _ratio(-n, d)


def st_beta_coeffs_2(mf1, mf2, beta1, beta2, dbeta1, dbeta2):
    """``IMRPhenomX_ST_PNR_beta_coeffs_2``: the coefficient of ``Mf^2`` (see
    :func:`st_beta_coeffs_0`)."""
    mf1_2 = mf1 * mf1
    mf2_2 = mf2 * mf2
    n = fma(-fma(2.0, dbeta1, dbeta2), mf2_2, fma((dbeta1 - dbeta2) * mf1, mf2, fma(-3.0 * (beta1 - beta2), mf1 + mf2, fma(2.0, dbeta2, dbeta1) * mf1_2)))
    d = (mf1 - mf2) * (mf1 - mf2) * (mf1 - mf2)
    return _ratio(-n, d)


def st_beta_coeffs_3(mf1, mf2, beta1, beta2, dbeta1, dbeta2):
    """``IMRPhenomX_ST_PNR_beta_coeffs_3``: the coefficient of ``Mf^3`` (see
    :func:`st_beta_coeffs_0`)."""
    n = fma(2.0, beta1 - beta2, -((dbeta1 + dbeta2) * (mf1 - mf2)))
    d = (mf1 - mf2) * (mf1 - mf2) * (mf1 - mf2)
    return _ratio(-n, d)


def beta_connection(mf, p):
    """``beta_connection`` of ``LALSimIMRPhenomX_precession.c``: the cubic
    ``b0 + b1 Mf + b2 Mf^2 + b3 Mf^3`` with ``(b0, b1, b2, b3) =
    p.beta_interp`` between the SpinTaylor and the PNR ``beta``."""
    b0, b1, b2, b3 = p.beta_interp
    return fma(b3 * mf * mf, mf, fma(b2 * mf, mf, fma(b1, mf, b0)))


def arctan_window(beta):
    """``IMRPhenomX_PNR_arctan_window``, Eq. 62 of arXiv:2107.08876: keeps
    ``beta`` away from 0 and pi within 0.01 of them; ``beta`` elsewhere.

    LAL's constants are kept (its ``pi / 2`` is two ulps above
    ``FRAC_PI_2``); ``pow(beta - pi / 2, 1 / p)`` of a negative base with
    the even exponent 500 is ``|beta - pi / 2|^500``. LAL's ``atan2(|beta -
    pi / 2|^500, 7.3e97)`` is computed as ``atan((|beta - pi / 2| / 7.3e97^(1 /
    500))^500)``, whose power stays below 1.5 (both of LAL's overflow in
    float32)."""
    window_border = 0.01
    inside = (beta <= window_border) | (beta >= PI - window_border)
    p = 0.002
    pi_by_2 = 1.570796326794897
    pi_by_2_1mp = 1.569378278348018
    pi_by_2_1oq = 7.308338225719002e97
    windowed = jnp.where(inside, beta, 0.0)
    sign = jnp.where(windowed < pi_by_2, -1.0, 1.0)
    ratio = (jnp.abs(windowed - pi_by_2) / pi_by_2_1oq ** p) ** (1.0 / p)
    value = fma(sign * pi_by_2_1mp, jnp.arctan(ratio) ** p, pi_by_2)
    return jnp.where(inside, value, beta)


def attach_mr_beta(p):
    """``IMRPhenomX_PNR_AttachMRBeta``: whether the MR ``beta`` is attached
    (a boolean): not when ``Mf_beta_lower < 0.009``, ``Mf_beta_lower ==
    100`` (no connection), ``beta_upper <= 0`` or ``beta_upper > 5 (B0 +
    0.1)``."""
    return jnp.logical_not(
        (p.mf_beta_lower < 0.009) | (p.mf_beta_lower == 100.0) | (p.beta_upper <= 0.0)
        | (p.beta_upper > 5.0 * (p.b0 + 0.1))
    )


#: The half-width (rad) of the band across which the smooth connection
#: blends the upper bound of :func:`attach_mr_beta` on ``beta_upper``, and
#: that of its sign relative to ``|beta_upper| + |beta_lower|``.
BETA_SOFT_ANGLE = 0.02
BETA_SOFT_SIGN = 0.05


def attach_weight(p, soft=None):
    """The weight of the ``beta`` with the MR ``beta`` attached against the
    one without: :func:`attach_mr_beta` (0 or 1), or with ``soft`` (a
    half-width in ``Mf``), its tests blended across bands of half-width
    ``soft`` in ``Mf_beta_lower``, :data:`BETA_SOFT_ANGLE` in the upper
    bound on ``beta_upper`` and :data:`BETA_SOFT_SIGN` in its sign relative
    to ``|beta_upper| + |beta_lower|`` (0 where there is no connection)."""
    attach = attach_mr_beta(p)
    if soft is None:
        return jnp.where(attach, 1.0, 0.0)
    # the sign of beta_upper relative to the size of beta at the connection
    # (which scales with the spin): a band that does not catch small spins
    size = jnp.abs(p.beta_upper) + jnp.abs(p.beta_lower)
    w = _ramp(p.mf_beta_lower - 0.009, soft) * _ramp(_div(p.beta_upper, size, True), BETA_SOFT_SIGN)
    w = w * _ramp(5.0 * (p.b0 + 0.1) - p.beta_upper, BETA_SOFT_ANGLE)
    return jnp.where(p.mf_beta_lower == 100.0, 0.0, w)


def inflection_cubic(p):
    """The coefficients ``(a, b, c, d)`` of the cubic ``a x^3 + b x^2 + c x +
    d`` whose roots are the inflection points of the MR ``beta`` (Eq. 68 of
    arXiv:2107.08876), with LAL's fused multiply-adds (:func:`~._math.fma`)."""
    b1, b2, b3, b4, b5 = p.b1, p.b2, p.b3, p.b4, p.b5
    f = fma
    a = 2.0 * f(b2 * b4, b4, -(2.0 * b3 * b4 * b4 * b5))
    b = 6.0 * f(-(b3 * b4 * b4 * b5), b5, f(-b3, b4, b1 * b4 * b4))
    c = 6.0 * f(-(b2 * b4 * b4 * b5), b5, f(-b2, b4, 2.0 * b1 * b4 * b4 * b5))
    d = 2.0 * f(b3 * b4 * b4 * b5 * b5 * b5, b5, f(-(2.0 * b2 * b4 * b4 * b5 * b5), b5, f(3.0 * b1 * b4 * b4 * b5, b5, f(2.0 * b3 * b4 * b5, b5, f(-(2.0 * b2 * b4), b5, f(-b1, b4, b3))))))
    return a, b, c, d


def _two_roots(b, c, d, safe):
    """``IMRPhenomX_PNR_two_inflection_points`` as (re, im) pairs."""
    w = _csqrt((fma(c, c, -(4.0 * b * d)), 0.0), safe)
    k = _div(1.0, 2.0 * b, safe)
    return [_cscale((-c + w[0], w[1]), k), _cscale((-c - w[0], -w[1]), k)]


def _three_roots(a, b, c, d, safe):
    """``IMRPhenomX_PNR_three_inflection_points`` as (re, im) pairs: the
    trigonometric formula for the roots of the depressed cubic ``t^3 + r t +
    s`` in LAL's complex arithmetic, with the implicit conversions of the
    complex arguments of ``acos`` and ``cos`` to their real parts."""
    ca, cb, cc, cd = (a, 0.0), (b, 0.0), (c, 0.0), (d, 0.0)
    three_a = _cscale(ca, 3.0)
    # r = (3ac - b^2) / (3a^2), s = (2b^3 - 9abc + 27a^2 d) / (27a^3)
    r = _cdiv(_csub(_cmul(three_a, cc), _cmul(cb, cb)), _cmul(three_a, ca), safe)
    s_num = _cadd(
        _csub(_cmul(_cmul(_cscale(cb, 2.0), cb), cb), _cmul(_cmul(_cscale(ca, 9.0), cb), cc)),
        _cmul(_cmul(_cscale(ca, 27.0), ca), cd),
    )
    s = _cdiv(s_num, _cmul(_cmul(_cscale(ca, 27.0), ca), ca), safe)
    # phi = acos(Re[(3 s / (2 r)) csqrt(-3 / r)])
    q1 = _cdiv(_cscale(s, 3.0), _cscale(r, 2.0), safe)
    w1 = _csqrt(_cdiv((-3.0, 0.0), r, safe), safe)
    phi = _acos(_cmul(q1, w1)[0], safe)
    # 2 csqrt(-r / 3) cos((phi - 2 n pi) / 3) - b / (3a)
    two_w2 = _cscale(_csqrt((-r[0] / 3.0, -r[1] / 3.0), safe), 2.0)
    q = _cdiv(cb, three_a, safe)
    roots = []
    for n in range(3):
        cosv = jnp.cos((phi - 2.0 * n * PI) / 3.0)
        re, im = _cscale(two_w2, cosv)
        roots.append((re - q[0], im - q[1]))
    return roots


def _ramp(s, soft):
    """The weight (0 to 1, C2) of a decision ``s > 0`` blended across ``|s| <
    soft``."""
    t = jnp.clip(0.5 + 0.5 * s / soft, 0.0, 1.0)
    return t * t * t * fma(t, fma(6.0, t, -15.0), 10.0)


def _select(branch, condition, s, a, b, soft=None):
    """``a`` where LAL's ``condition`` (recorded by ``branch``) holds, else
    ``b``; with ``soft``, blended across ``|s| < soft`` of the continuous
    ``s`` whose sign is the condition, where both are finite (LAL's choice
    where one is not: a candidate LAL does not compute there)."""
    condition = branch(condition)
    hard = jnp.where(condition, a, b)
    if soft is None:
        return hard
    w = _ramp(s, soft)
    blend = jnp.where(w >= 1.0, a, jnp.where(w <= 0.0, b, fma(w, a - b, b)))
    return jnp.where(jnp.isfinite(a) & jnp.isfinite(b), blend, hard)


def _bracketed_roots(a, b, c, d, iterations=110):
    """The three real roots of ``a x^3 + b x^2 + c x + d`` in descending
    order (the order of :func:`_three_roots`), robust for small ``a``: its
    stationary points (the roots of ``3 a x^2 + 2 b x + c``, from the stable
    form of the quadratic formula) separate them; each is found by bisection
    in its interval (the outer ones bounded by Cauchy's bound), then one
    Newton step gives the derivatives of the root. NaN where the cubic has
    no stationary points (one real root)."""
    sg = lax.stop_gradient

    def cubic(x):
        return fma(fma(fma(a, x, b), x, c), x, d)

    qa, qb, qc = 3.0 * a, 2.0 * b, c
    disc = fma(qb, qb, -4.0 * qa * qc)
    ok = disc > 0.0
    root_disc = jnp.sqrt(jnp.where(ok, disc, 1.0))
    q = -0.5 * (qb + jnp.where(qb >= 0.0, root_disc, -root_disc))
    s1 = q / jnp.where(qa != 0.0, qa, 1.0)
    s1 = jnp.where(qa != 0.0, s1, jnp.inf * jnp.sign(-q))
    s2 = qc / jnp.where(q != 0.0, q, 1.0)
    lo_s, hi_s = jnp.minimum(s1, s2), jnp.maximum(s1, s2)
    bound = 1.0 + jnp.maximum(jnp.maximum(jnp.abs(b), jnp.abs(c)), jnp.abs(d)) / jnp.maximum(jnp.abs(a), tiny())
    bound = jnp.maximum(bound, jnp.maximum(jnp.abs(jnp.where(jnp.isfinite(lo_s), lo_s, 0.0)),
                                           jnp.abs(jnp.where(jnp.isfinite(hi_s), hi_s, 0.0))) + 1.0)
    intervals = [(jnp.where(jnp.isfinite(hi_s), hi_s, -bound), jnp.full_like(a, bound)),
                 (jnp.where(jnp.isfinite(lo_s), lo_s, -bound), jnp.where(jnp.isfinite(hi_s), hi_s, bound)),
                 (-jnp.full_like(a, bound), jnp.where(jnp.isfinite(lo_s), lo_s, bound))]
    out = []
    for lo, hi in intervals:
        lo, hi = sg(lo), sg(hi)
        rising = sg(cubic(hi)) >= sg(cubic(lo))

        def bisect(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            below = (sg(cubic(mid)) < 0.0) == rising
            return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

        lo, hi = lax.fori_loop(0, iterations, bisect, (lo, hi))
        x = sg(0.5 * (lo + hi))
        slope = fma(fma(3.0 * a, x, 2.0 * b), x, c)
        x = x - cubic(x) / jnp.where(slope != 0.0, slope, 1.0)
        out.append(jnp.where(ok, x, jnp.nan))
    return out


def _single_inflection_point(p, safe, branch, soft=None):
    a, b, c, d = inflection_cubic(p)
    small_a = branch(jnp.abs(a) < 1e-10)
    small_b = branch(jnp.abs(b) < 2e-10)
    # the stand-ins keep the cases that are not selected finite
    if safe:
        c_one = jnp.where(small_a & small_b, c, 1.0)
        b_two = jnp.where(small_b, 1.0, b)
        a_three = jnp.where(small_a, 1.0, a)
    else:
        c_one, b_two, a_three = c, b, a
    # a = b = 0: the single inflection point
    f_one = _div(-d, c_one, safe)
    # a = 0: the inflection point with negative gradient
    roots = _two_roots(b_two, c, d, safe)
    negative = [branch(mr_dbeta(root[0], p) < 0.0) for root in roots]
    f_two = jnp.where(negative[1], roots[1][0], jnp.where(negative[0], roots[0][0], 0.0))
    # three roots: the real one if only one is (a substantial imaginary part,
    # sic: without fabs, marks a complex root), else Eq. 69
    roots = _three_roots(a_three, b, c, d, safe)
    if soft is not None:
        # the trigonometric formula loses the roots where a is small next to
        # the other coefficients (for b / a ~ 1e7 only ~1e-2 of them is
        # left, and LAL's choice jumps between them): the real roots in its
        # (descending) order, bracketed by the stationary points
        stable = _bracketed_roots(a, b, c, d)
        roots = [(jnp.where(jnp.isfinite(x), x, root[0]), root[1]) for x, root in zip(stable, roots)]
    real = [branch(root[1] < 1e-10) for root in roots]
    count = sum(jnp.where(x, 1, 0) for x in real)
    f_temp = jnp.where(real[2], roots[2][0], jnp.where(real[1], roots[1][0], jnp.where(real[0], roots[0][0], 0.0)))
    a_negative = branch(a < 0.0)
    s69 = _div(b, 3.0 * a_three, safe) - (p.b5 / 2.0 - (2141.0 / 90988.0))
    eq69 = _select(branch, s69 > 0.0, s69, roots[0][0], roots[2][0], soft)
    f_three = jnp.where(count == 1, f_temp, jnp.where(a_negative, roots[1][0], eq69))
    return jnp.where(small_a, jnp.where(small_b, f_one, f_two), f_three)


def two_inflection_points(p):
    """``IMRPhenomX_PNR_two_inflection_points``: the two roots (complex) of
    the inflection cubic when its cubic term vanishes."""
    _, b, c, d = inflection_cubic(p)
    def parts(b, c, d, safe, branch):
        return tuple(x for root in _two_roots(b, c, d, safe) for x in root)

    values = _raw_and_safe(parts, b, c, d)
    return tuple(lax.complex(values[2 * n], values[2 * n + 1]) for n in range(2))


def three_inflection_points(p):
    """``IMRPhenomX_PNR_three_inflection_points``: the three roots (complex)
    of the inflection cubic (Eq. 68 of arXiv:2107.08876), from the
    trigonometric formula in LAL's complex arithmetic: NaN where the
    argument of ``acos`` leaves ``[-1, 1]`` (complex roots; for the MR
    ``beta`` the three roots are real and this only happens by rounding at
    double roots)."""
    a, b, c, d = inflection_cubic(p)

    def parts(a, b, c, d, safe, branch):
        return tuple(x for root in _three_roots(a, b, c, d, safe) for x in root)

    values = _raw_and_safe(parts, a, b, c, d)
    return tuple(lax.complex(values[2 * n], values[2 * n + 1]) for n in range(3))


def single_inflection_point(p):
    """``IMRPhenomX_PNR_single_inflection_point``: the inflection frequency
    of the MR ``beta`` used for its connection (Eq. 69 of
    arXiv:2107.08876): ``-d / c`` when the cubic and quadratic terms of the
    inflection cubic vanish (``|a| < 1e-10``, ``|b| < 2e-10``), the root
    with negative gradient of ``beta`` when the cubic term vanishes, else
    the real root of three if only one is real, the central root for ``a <
    0`` and the first or last root by Eq. 69. NaN where LAL's is (complex
    roots of the trigonometric formula, see
    :func:`three_inflection_points`, or ``-d / c = 0 / 0``)."""
    return _raw_and_safe(_single_inflection_point, p)


def _beta_connection_frequencies(p, safe, branch, soft=None):
    b1, b2, b3, b4, b5 = p.b1, p.b2, p.b3, p.b4, p.b5
    mf_inflection = _single_inflection_point(p, safe, branch, soft)
    dbeta_inflection = mr_dbeta(mf_inflection, p)

    # the two extrema of beta
    k = fma(b3 * b4 * b5, b5, fma(-b1, b4, b3))
    root_term = fma(b4 * fma(-(2.0 * b3), b5, b2), fma(b2 * b4 * b5, b5, fma(-(2.0 * b1 * b4), b5, b2)), k * k)
    den = b4 * fma(-(2.0 * b3), b5, b2)
    sqrt_root_term = _sqrt(root_term, safe)
    mf_plus = _div(k + sqrt_root_term, den, safe)
    mf_minus = _div(k - sqrt_root_term, den, safe)
    # classified by the sign of the second derivative
    plus_is_minimum = branch(mr_ddbeta(mf_plus, p) > 0.0)
    mf_at_minimum = jnp.where(plus_is_minimum, mf_plus, mf_minus)
    mf_at_maximum = jnp.where(plus_is_minimum, mf_minus, mf_plus)
    ddbeta = mr_ddbeta(mf_at_maximum, p)
    dddbeta = mr_dddbeta(mf_at_maximum, p)

    # Eqs. 56 and 57: the step from the maximum
    sign = jnp.where(branch(dbeta_inflection > 0.0), 1.0, -1.0)
    chosen_dbeta = sign * (dbeta_inflection / 100.0 * (dbeta_inflection / 100.0)) * 25.0
    root = _sqrt(fma(ddbeta, ddbeta, 2.0 * dddbeta * chosen_dbeta), safe)
    inverse = _div(1.0, dddbeta, safe)
    d1 = inverse * (-ddbeta + root)
    d2 = inverse * (-ddbeta - root)
    # (LAL's choice also for the smooth connection: the step from the
    # maximum scales with the size of beta, which leaves no band of fixed
    # width that separates its choices only where they change)
    d1_positive = branch(d1 > 0.0)
    d2_positive = branch(d2 > 0.0)
    d1_smaller = branch(d1 < d2)
    delta = jnp.where(d1_positive, jnp.where(d2_positive, jnp.where(d1_smaller, d1, d2), d1), d2)

    # Eq. 58, and the three shapes of Fig. 10 of arXiv:2107.08876
    mf_low = _select(branch, mf_inflection >= 0.06, mf_inflection - 0.06, mf_inflection - 0.03,
                     3.0 * mf_inflection / 5.0, soft)
    s_turnover = jnp.maximum(mf_at_minimum - mf_at_maximum, mf_inflection - mf_at_maximum)
    s_above_low = mf_at_maximum - mf_low
    after_maximum = _select(branch, mf_at_maximum >= mf_low, s_above_low, mf_at_maximum + delta, mf_low, soft)
    before_minimum = _select(branch, mf_at_minimum > 0.06, mf_at_minimum - 0.06, mf_at_minimum - 0.03,
                             3.0 * mf_at_minimum / 5.0, soft)
    turnover = (mf_at_minimum > mf_at_maximum) | (mf_inflection > mf_at_maximum)
    mf_im = _select(branch, turnover, s_turnover, after_maximum, before_minimum, soft)
    if soft is not None:
        # where the coefficient of the cubic of the inflection points changes
        # sign, the maximum of beta leaves to -inf and comes back from +inf:
        # the shape of beta does not change in any waveform's range, but
        # LAL's turnover does. Beyond |Mf| = 1 / FAR_MAXIMUM the two are
        # blended in 1 / Mf of the maximum (continuous through infinity)
        inverse = _div(1.0, mf_at_maximum, True)
        far = jnp.abs(inverse) < FAR_MAXIMUM
        blended = fma(_ramp(-inverse, FAR_MAXIMUM), mf_low - before_minimum, before_minimum)
        mf_im = jnp.where(far & jnp.isfinite(blended), blended, mf_im)
    freeze = branch(mf_at_minimum > mf_inflection)
    if soft is None:
        mf_mr = jnp.where(freeze, mf_at_minimum, 100.0)
        weight = jnp.ones_like(mf_mr)
    else:
        # beta held from its minimum with a weight, rather than not at all
        # (mf_beta_upper = 100) below the inflection point
        finite = jnp.isfinite(mf_at_minimum)
        mf_mr = jnp.where(finite, mf_at_minimum, 100.0)
        weight = jnp.where(finite, _ramp(mf_at_minimum - mf_inflection, soft), 0.0)
    # failsafe: no MR beta
    fail = branch((mf_im < 0.0) | jnp.isnan(mf_im))
    return jnp.where(fail, 100.0, mf_im), jnp.where(fail, 100.0, mf_mr), weight


#: The half-width (in ``Mf``) of the bands across which the smooth
#: connection blends LAL's choices.
BETA_SOFT_WIDTH = 0.002
#: The smooth connection blends the turnover of ``beta`` where its maximum
#: lies beyond ``|Mf| = 1 / FAR_MAXIMUM`` (see :func:`_beta_connection_frequencies`).
FAR_MAXIMUM = 0.1


def beta_connection_frequencies(p, soft=None):
    """``IMRPhenomX_PNR_BetaConnectionFrequencies``: ``p`` with the lower
    and upper connection frequencies of ``beta`` (``mf_beta_lower``,
    ``mf_beta_upper``; Sec. 8B of arXiv:2107.08876) from the inflection
    point and the extrema of the MR ``beta``. Both are 100 (no MR ``beta``)
    where the lower one is negative or NaN; the upper one is 100 where the
    minimum of ``beta`` is below the inflection point.

    With ``soft`` (a half-width in ``Mf``, e.g. :data:`BETA_SOFT_WIDTH`),
    LAL's choices between candidate frequencies (the rules at ``Mf =
    0.06``, the turnover of ``beta``, the root of Eq. 69) are blended across bands of that half-width of the quantities
    they compare, so that the frequencies are continuous; and instead of
    the upper frequency switching to 100 where the minimum of ``beta``
    falls below its inflection point, the weight ``beta_freeze`` of
    ``beta`` held from the minimum goes smoothly from 1 to 0."""
    lower, upper, freeze = _raw_and_safe(functools.partial(_beta_connection_frequencies, soft=soft), p)
    return p.replace(mf_beta_lower=lower, mf_beta_upper=upper, beta_freeze=freeze)


# ---------------------------------------------------------------------------
# "waveform" beta of the inspiral
# ---------------------------------------------------------------------------


def chi_calc(m1, l, j0, l0, chi_parr, beta):  # noqa: E741
    """``IMRPhenomX_PNR_chi_calc``: the effective spin magnitude from the
    frequency-dependent in-plane spin ``(J0 - (L0 - L)) sin(beta) / m1^2``
    and ``chi_parr`` (Eqs. 47 and 18 of arXiv:2107.08876)."""
    s_perp = (j0 - (l0 - l)) * jnp.sin(beta)
    chi_p = s_perp / (m1 * m1)
    # (a finite derivative for non-spinning binaries)
    return safe_sqrt(fma(chi_parr, chi_parr, chi_p * chi_p), tiny())


def pn_waveform_beta(mf, iota, m1, m2, chi, costheta):
    """``IMRPhenomX_PNR_PNWaveformBeta``, Eq. 41 of arXiv:2107.08876: the
    opening angle of the optimal emission direction from the dynamical
    opening angle ``iota`` at ``mf``, for masses ``m1``, ``m2`` (in units
    of the total mass) and the effective single spin ``chi`` at polar
    angle ``acos(costheta)``: ``2 atan2(N, 2 D)`` of the ratio ``N / D``
    of the PN expansions of ``h21`` and ``h22``."""
    m = m1 + m2
    nu = (m1 * m2) / (m * m)
    delta = (m1 - m2) / m
    theta = jnp.arccos(costheta)
    w_orb = PI * mf
    v = w_orb ** (1.0 / 3.0)
    v2 = v * v
    v3 = v * v2
    cos_iota = jnp.cos(iota)
    sin_iota = jnp.sin(iota)
    cos_half_iota = jnp.cos(iota / 2.0)
    sin_half_iota = jnp.sin(iota / 2.0)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    # numerator expansion of h21/h22
    n0 = 84.0 * sin_iota
    n2 = 2.0 * fma(55.0, nu, -107.0) * sin_iota
    n3 = fma(-7.0 * (fma(5.0, nu, 6.0 * delta) + 6.0) * chi * fma(2.0, cos_iota, -1.0), sin_theta, 56.0 * fma(-((1.0 + delta - nu) * chi), cos_theta, 3.0 * PI) * sin_iota)
    n = fma(n3, v3, fma(n2, v2, n0)) / cos_half_iota
    # denominator expansion of h21/h22
    d0 = 84.0 * cos_half_iota
    d2 = 2.0 * fma(55.0, nu, -107.0) * cos_half_iota
    d3 = fma(56.0 * fma((nu - 1.0 - delta) * chi, cos_theta, 3.0 * PI), cos_half_iota, 14.0 * fma(5.0, nu, fma(6.0, delta, 6.0)) * chi * sin_theta * sin_half_iota)
    d = fma(d3, v3, fma(d2, v2, d0))
    return 2.0 * _atan2_tol(n, 2.0 * d, MAX_TOL_ATAN)


# ---------------------------------------------------------------------------
# windows and frequency maps
# ---------------------------------------------------------------------------


def _cosine_window(argument):
    """``0.5 cos(pi x) + 0.5`` for ``0 < x <= 1``, 0 above and 1 below."""
    inside = (argument > 0.0) & (argument <= 1.0)
    value = fma(0.5, jnp.cos(jnp.where(inside, argument, 0.0) * PI), 0.5)
    return jnp.where(inside, value, jnp.where(argument > 1.0, 0.0, 1.0))


def angles_window(q, chi_single_spin):
    """``IMRPhenomX_PNR_AnglesWindow``: the weight of the PNR angles
    against the PN angles, 1 in the calibration region (``q <= 8.5``, ``chi
    <= 0.85``), 0 beyond ``q = 12`` or ``chi = 1.2``, cosine tapers in
    between (product of the two)."""
    window_q_argument = (q - 8.5) / 3.5
    window_chi_argument = (chi_single_spin - 0.85) / 0.35
    return _cosine_window(window_q_argument) * _cosine_window(window_chi_argument)


def coprec_window(q):
    """``IMRPhenomX_PNR_CoprecWindow``: the weight of the co-precessing
    deviations, switched off by a cosine taper between ``q = 10`` and ``q =
    20``."""
    return _cosine_window((q - 10.0) / 10.0)


def linear_frequency_slope(emm, mf_lower, mf_upper, mf_rd_22, mf_rd_lm):
    """``IMRPhenomX_PNR_LinearFrequencySlope``: the slope of the linear part
    of the frequency map (LAL fails for ``mf_lower > mf_upper``)."""
    return (mf_upper - (mf_rd_lm - mf_rd_22) - 2.0 * mf_lower / emm) / (mf_upper - mf_lower)


def linear_frequency_map(mf, ell, emm, mf_lower, mf_upper, mf_rd_22, mf_rd_lm, inspiral):
    """``IMRPhenomX_PNR_LinearFrequencyMap``: the frequency of the (2,2)
    angles used for the mode ``(ell, emm)`` at ``mf``: ``mf`` for the
    (2,2) mode; the inspiral scaling ``2 Mf / emm`` below ``mf_lower`` or
    everywhere if ``inspiral``; the shift by the difference ``mf_rd_lm -
    mf_rd_22`` of the ringdown frequencies above ``mf_upper``; linear in
    between.

    Returns ``(value, error)``: where LAL fails (``mf_lower > mf_upper``
    when the transition is needed) the value is NaN and ``error`` is
    true."""
    is22 = jnp.logical_and(ell == 2, emm == 2)
    spa = jnp.logical_or(mf <= mf_lower, inspiral)
    error = jnp.logical_not(is22 | spa | (mf_lower <= mf_upper))
    width = mf_upper - mf_lower
    slope = (mf_upper - (mf_rd_lm - mf_rd_22) - 2.0 * mf_lower / emm) / jnp.where(width != 0, width, 1.0)
    linear = fma(slope, mf - mf_lower, 2.0 * mf_lower / emm)
    value = jnp.where(mf > mf_upper, mf - (mf_rd_lm - mf_rd_22), linear)
    value = jnp.where(error, NAN, value)
    value = jnp.where(spa, 2.0 * mf / emm, value)
    value = jnp.where(is22, mf, value)
    return value, error


def linear_frequency_map_transition_frequencies(emmprime, mf_rd_22, mf_rd_lm, pnr_hm_mflow, pnr_hm_mfhigh):
    """``IMRPhenomX_PNR_LinearFrequencyMapTransitionFrequencies``: the
    transition frequencies ``(Mf_low, Mf_high)`` of the map for ``m' =
    emmprime``, from ``PNR_HM_Mflow`` and ``PNR_HM_Mfhigh`` of the
    precession struct; ``Mf_high`` falls back to ``PNR_HM_Mfhigh`` where it
    misbehaves (negative, or below half of it for a ringdown frequency below
    that of the (2,2) mode)."""
    mf_low = 0.65 * pnr_hm_mflow * emmprime / 2.0
    mf_high = 1.1 * (pnr_hm_mfhigh + (mf_rd_lm - mf_rd_22))
    fallback = (mf_high < 0.0) | ((mf_rd_lm - mf_rd_22 < 0.0) & (mf_high < pnr_hm_mfhigh / 2.0))
    return mf_low, jnp.where(fallback, pnr_hm_mfhigh, mf_high)


def boole_step(gamma_prev, t1, t2, f):
    """One step of the integral of ``gamma`` in
    ``IMRPhenomX_PNR_GeneratePNRGamma(_FromInterpolants)``: ``gamma_prev``
    plus the integral of ``f`` over ``[t1, t2]`` by Boole's rule. ``f`` is
    vectorised: it is called once, on the five nodes stacked along a new
    leading axis (``t1`` and ``t2`` may be arrays of intervals); in LAL it
    is :func:`alphadot_cosbeta` of the splines of ``alpha`` and ``beta``."""
    t1 = jnp.asarray(t1)
    t2 = jnp.asarray(t2)
    nodes = jnp.stack([t1, fma(3.0, t2, t1) / 4.0, (t1 + t2) / 2.0, fma(3.0, t1, t2) / 4.0, t2])
    f1, fa, fm, fb, f2 = f(nodes)
    total = fma(7.0, f2, fma(32.0, fb, fma(12.0, fm, fma(7.0, f1, 32.0 * fa))))
    return fma(1.0 / 90.0 * (t2 - t1), total, gamma_prev)


def alphadot_cosbeta(alphadot, beta):
    """``IMRPhenomX_PNR_alphadot_cosbeta``: the integrand ``-alpha' cos(beta)``
    of ``gamma`` (Eq. A7 of arXiv:2107.08876) from the derivative of
    ``alpha`` and ``beta`` at a frequency."""
    return -1.0 * alphadot * jnp.cos(beta)


def sqrt_one_minus_square(x):
    """``sqrt(1 - x^2)`` (e.g. a sine from a cosine), with a finite
    derivative at ``|x| = 1`` (aligned spins)."""
    return safe_sqrt(fnms(x, x, 1.0), tiny())
