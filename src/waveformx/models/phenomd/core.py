# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomD as a table of coefficients per binary and a short formula per
frequency, shared by the JAX model (:mod:`.model`) and the MLX one
(:mod:`.mlx`), as IMRPhenomXAS (:mod:`..xas.core`).

LAL's IMRPhenomD (``IMRPhenomDGenerateFD``) is, in each of its three
regions of the phase and of the amplitude, a sum of powers of ``Mf`` (and
of ``log Mf``), an arctangent (the merger-ringdown phase), a polynomial
(the intermediate amplitude) and a Lorentzian times an exponential (the
merger-ringdown amplitude). :func:`table` writes those sums in powers of the
frequency ``f`` in Hz, with the powers of the binary's ``M`` folded into the
coefficients, together with LAL's linear time shift (``t0``, which puts the
peak of the amplitude at zero time) and reference phase. Each frequency
costs one sum of at most 12 terms for the phase, one polynomial for the
amplitude, an arctangent, an exponential and a sine and cosine
(:func:`evaluate`). The leading term of the inspiral phase is computed in
float64 whatever the precision, as in :func:`..xas.core.phase`.

The differences from LAL: the masses are in solar masses (no round trip
through kg), the binary is not reordered (the fits are symmetric under the
exchange of the bodies), the intermediate amplitude is solved for in powers
of ``Mf / Mf_peak`` (better conditioned in float32), and the quasi-normal
mode frequencies are closed-form fits in the final spin
(:func:`.fits.qnm`) rather than LAL's spline through a table. The strain
is multiplied by ``amp_scale`` (``1e21`` unless ``scaled_strain`` is
false).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..taylorf2 import pn
from ..xas.core import K0, reduce
from ..xpnr._math import MPC_SI, MRSUN_SI, MTSUN_SI, PI, real
from . import fits

__all__ = ["BASIS", "LAYOUT", "SIZE", "OFFSETS", "REGION_TERMS", "table", "unpack", "frequency_powers",
           "phase_basis", "phase", "amplitude", "evaluate", "coefficients", "narrow", "F_PHASE_INS", "F_AMP_INS"]

#: The end of the inspiral phase and of the inspiral amplitude (``Mf``).
F_PHASE_INS = 0.018
F_AMP_INS = 0.014

#: The powers ``(a, l)`` of the basis ``f^a (log f)^l`` of the phase (the
#: leading term of the inspiral, ``f^(-5/3)``, is apart: ``big``).
BASIS = (
    (-1, 0), (-2 / 3, 0), (-1 / 3, 0), (0, 1), (0, 0), (1 / 3, 0), (1 / 3, 1), (2 / 3, 0), (1, 0), (4 / 3, 0),
    (5 / 3, 0), (2, 0), (-3, 0), (3 / 4, 0),
)
_INDEX = {key: i for i, key in enumerate(BASIS)}
CONST = _INDEX[(0, 0)]

#: The terms of :data:`BASIS` each region uses (but for the constant): the
#: inspiral, the intermediate and the merger-ringdown.
REGION_TERMS = (
    tuple(i for i, key in enumerate(BASIS) if key not in ((0, 0), (-3, 0), (3 / 4, 0))),
    (_INDEX[(1, 0)], _INDEX[(-3, 0)], _INDEX[(0, 1)]),
    (_INDEX[(1, 0)], _INDEX[(-1, 0)], _INDEX[(3 / 4, 0)]),
)

#: The fields of a table: ``phase`` (the coefficients of :data:`BASIS` in
#: the three regions), ``big``, ``big_lo`` (the leading term, as in
#: :mod:`..xas.core`), ``atan_a``, ``atan_w``, ``atan_shift`` (the
#: merger-ringdown's ``a atan(w (f - shift))``), ``amp_ins`` (the inspiral
#: amplitude ``f^(-7/6) sum_k c_k f^(k/3)``), ``amp_int`` (the intermediate
#: ``f^(-7/6) sum_j d_j y^j``, ``y = amp_int_s f``), ``amp_rd_*`` (the
#: merger-ringdown ``n f^(-7/6) exp(-gr u) / (u^2 + 1)``, ``u = s (f -
#: shift)``), the boundaries of the regions (Hz) and the factors ``kp``,
#: ``kc`` of the polarizations (NaN where LAL fails).
LAYOUT = (
    ("phase", (3, len(BASIS))),
    ("big", (2,)),
    ("big_lo", (2,)),
    ("atan_a", ()),
    ("atan_w", ()),
    ("atan_shift", ()),
    ("amp_ins", (10,)),
    ("amp_int", (5,)),
    ("amp_int_s", ()),
    ("amp_rd_n", ()),
    ("amp_rd_gr", ()),
    ("amp_rd_s", ()),
    ("amp_rd_shift", ()),
    ("f_phase_ins", ()),
    ("f_phase_mrd", ()),
    ("f_amp_ins", ()),
    ("f_amp_mrd", ()),
    ("kp", ()),
    ("kc", ()),
)


def _offsets():
    out, start = {}, 0
    for name, shape in LAYOUT:
        size = int(np.prod(shape)) if shape else 1
        out[name] = (start, shape)
        start += size
    return out, start


OFFSETS, SIZE = _offsets()

#: The amplitude's prefactor is ``K_AMP (m1 m2)^(1/2) M^(-1/6) / D``
#: (masses in solar masses, ``D`` in Mpc): LAL's ``amp0`` (``2 sqrt(5 /
#: (64 pi)) M^2 / D``) times the ``amp0`` of its amplitude (``sqrt(2 eta /
#: 3) pi^(-1/6)``) and ``M^(-7/6)`` of ``Mf^(-7/6)``.
K_AMP = 2.0 * float(np.sqrt(5.0 / (64.0 * np.pi))) * MRSUN_SI * MTSUN_SI / MPC_SI \
    * float(np.sqrt(2.0 / 3.0) * np.pi ** (-1.0 / 6.0)) * MTSUN_SI ** (-7.0 / 6.0)


def unpack(flat):
    """The fields of a flat table (``(..., SIZE)``) as a dict of arrays."""
    out = {}
    for name, (start, shape) in OFFSETS.items():
        size = int(np.prod(shape)) if shape else 1
        out[name] = flat[..., start:start + size].reshape(flat.shape[:-1] + tuple(shape))
    return out


# --- the model in Mf (one binary) --------------------------------------------


def _amplitude_pn(eta, delta, chi1, chi2):
    """The coefficients of ``v^k`` (``v = (pi Mf)^(1/3)``, ``k < 7``) of the
    inspiral amplitude (``init_amp_ins_prefactors``)."""
    eta2 = eta * eta
    eta3 = eta2 * eta
    chi12, chi22 = chi1 * chi1, chi2 * chi2
    d1 = 1.0 + delta
    v = [1.0, 0.0, (-969.0 + 1804.0 * eta) / 672.0,
         (chi1 * (81.0 * d1 - 44.0 * eta) + chi2 * (81.0 - 81.0 * delta - 44.0 * eta)) / 48.0,
         (-27312085.0 - 10287648.0 * chi22 - 10287648.0 * chi12 * d1 + 10287648.0 * chi22 * delta
          + 24.0 * (-1975055.0 + 857304.0 * chi12 - 994896.0 * chi1 * chi2 + 857304.0 * chi22) * eta
          + 35371056.0 * eta2) / 8.128512e6,
         (chi2 * (-285197.0 * (-1.0 + delta) + 4.0 * (-91902.0 + 1579.0 * delta) * eta - 35632.0 * eta2)
          + chi1 * (285197.0 * d1 - 4.0 * (91902.0 + 1579.0 * delta) * eta - 35632.0 * eta2)
          + 42840.0 * (-1.0 + 4.0 * eta) * PI) / 32256.0,
         -(-336.0 * (-3248849057.0 + 2943675504.0 * chi12 - 3339284256.0 * chi1 * chi2 + 2943675504.0 * chi22) * eta2
           - 324322727232.0 * eta3
           - 7.0 * (-177520268561.0 + 107414046432.0 * chi22 + 107414046432.0 * chi12 * d1
                    - 107414046432.0 * chi22 * delta + 11087290368.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta) * PI)
           + 12.0 * eta * (-545384828789.0 - 176491177632.0 * chi1 * chi2 + 202603761360.0 * chi22
                           + 77616.0 * chi12 * (2610335.0 + 995766.0 * delta) - 77287373856.0 * chi22 * delta
                           + 5841690624.0 * (chi1 + chi2) * PI + 21384760320.0 * PI * PI)) / 6.0085960704e10]
    return v


def coefficients(x1, x2, chi1, chi2):
    """The coefficients of IMRPhenomD in ``Mf`` (``Struct``-like dict):
    the inspiral phasing ``v``, ``l`` (of ``v^(k - 5)`` and ``v^(k - 5) log
    v``, ``v = (pi Mf)^(1/3)``, ``k <= 11``), ``beta``, ``alpha``, the
    connection coefficients ``c_int``, ``c_mrd``, the time shift ``t0``,
    the inspiral amplitude ``rho`` (of ``v^k``, ``k <= 9``), ``delta`` (the
    intermediate amplitude in powers of ``Mf / f_peak``), ``gamma``, the
    frequencies ``f_ring``, ``f_damp``, ``f_peak`` and the final spin, for
    the masses ``x1``, ``x2`` relative to the total."""
    eta = x1 * x2
    delta = x1 - x2
    xi = fits.chi_pn(eta, delta, chi1, chi2) - 1.0
    fit = {name: fits.phenomenological(eta, xi, c) for name, c in fits.COEFFICIENTS.items()}
    af = fits.final_spin(eta, delta, chi1, chi2)
    ring, damp = fits.qnm(af)
    remnant = 1.0 - fits.radiated_energy(eta, delta, chi1, chi2)
    f_ring, f_damp = ring / remnant, damp / remnant

    # the inspiral phase: TaylorF2 but for its 3PN spin-spin term, and the
    # pseudo-PN terms of the fits
    v, logs = pn.phasing(x1, x2, chi1, chi2)
    v = list(v) + [0.0] * 4
    v[5] = v[5] - PI / 4.0
    v[6] = v[6] - pn.spin_spin_3pn(eta, x1, x2, chi1, chi2) * v[0]
    v[8] = fit["sigma1"] / eta / PI
    v[9] = fit["sigma2"] * 0.75 / eta / PI ** (4.0 / 3.0)
    v[10] = fit["sigma3"] * 0.6 / eta / PI ** (5.0 / 3.0)
    v[11] = fit["sigma4"] * 0.5 / eta / PI ** 2
    beta = [fit["beta1"], fit["beta2"], fit["beta3"]]
    alpha = [fit[f"alpha{i}"] for i in range(1, 6)]

    def ins(mf):
        u = jnp.cbrt(PI * mf)
        log_u = jnp.log(u)
        total = sum(c * u ** (k - 5) for k, c in enumerate(v) if k != 1)
        return total + sum(c * u ** (k - 5) * log_u for k, c in logs.items())

    def dins(mf):
        u = jnp.cbrt(PI * mf)
        log_u = jnp.log(u)
        total = sum((k - 5) / 3.0 * c * u ** (k - 5) for k, c in enumerate(v) if k != 1)
        total = total + sum(c * u ** (k - 5) * ((k - 5) / 3.0 * log_u + 1.0 / 3.0) for k, c in logs.items())
        return total / mf

    def inter(mf):
        return (beta[0] * mf - beta[2] / 3.0 / mf ** 3 + beta[1] * jnp.log(mf)) / eta

    def dinter(mf):
        return (beta[0] + beta[2] / mf ** 4 + beta[1] / mf) / eta

    def mrd(mf):
        return (alpha[0] * mf - alpha[1] / mf + 4.0 / 3.0 * alpha[2] * mf ** 0.75
                + alpha[3] * jnp.arctan((mf - alpha[4] * f_ring) / f_damp)) / eta

    def dmrd(mf):
        y = (mf - alpha[4] * f_ring) / f_damp
        return (alpha[0] + alpha[1] / mf ** 2 + alpha[2] / mf ** 0.25 + alpha[3] / (f_damp * (1.0 + y * y))) / eta

    f1, f2 = F_PHASE_INS, 0.5 * f_ring
    c2_int = dins(f1) - dinter(f1)
    c1_int = ins(f1) - inter(f1) - c2_int * f1
    joined = inter(f2) + c1_int + c2_int * f2
    c2_mrd = c2_int + dinter(f2) - dmrd(f2)
    c1_mrd = joined - mrd(f2) - c2_mrd * f2

    # the amplitude
    rho = _amplitude_pn(eta, delta, chi1, chi2) + [fit["rho1"] / PI ** (7.0 / 3.0), fit["rho2"] / PI ** (8.0 / 3.0),
                                                     fit["rho3"] / PI ** 3]
    gamma = [fit["gamma1"], fit["gamma2"], fit["gamma3"]]
    g1, g2, g3 = gamma
    # (LAL drops the square root for gamma2 > 1; its argument is kept
    # positive there, so that the branch not taken has finite derivatives)
    below = g2 <= 1.0
    root = jnp.sqrt(jnp.where(below, 1.0 - g2 * g2, 1.0))
    f_peak = jnp.abs(jnp.where(below, f_ring + f_damp * (root - 1.0) * g3 / g2, f_ring - f_damp * g3 / g2))
    t0 = dmrd(f_peak)

    def amp_ins(mf):
        u = jnp.cbrt(PI * mf)
        return sum(c * u ** k for k, c in enumerate(rho) if k != 1)

    def damp_ins(mf):
        u = jnp.cbrt(PI * mf)
        return sum(k / 3.0 * c * u ** k for k, c in enumerate(rho) if k > 1) / mf

    sigma = f_damp * g3

    def amp_mrd(mf):
        d = mf - f_ring
        return g1 * sigma / (d * d + sigma * sigma) * jnp.exp(-g2 * d / sigma)

    def damp_mrd(mf):
        d = mf - f_ring
        s2 = d * d + sigma * sigma
        return (-2.0 * d * sigma * g1 / s2 - g2 * g1) / (jnp.exp(g2 * d / sigma) * s2)

    # the intermediate amplitude sum_j delta_j y^j, y = Mf / f_peak: the
    # values at F_AMP_INS, halfway and f_peak, the derivatives at the ends
    y1 = F_AMP_INS / f_peak
    y2 = 0.5 * (1.0 + y1)
    rows = jnp.stack([jnp.stack([jnp.ones_like(y1), y1, y1 ** 2, y1 ** 3, y1 ** 4]),
                      jnp.stack([jnp.ones_like(y2), y2, y2 ** 2, y2 ** 3, y2 ** 4]),
                      jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0], real()),
                      jnp.stack([jnp.zeros_like(y1), jnp.ones_like(y1), 2.0 * y1, 3.0 * y1 ** 2, 4.0 * y1 ** 3]),
                      jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0], real())])
    values = jnp.stack([amp_ins(F_AMP_INS), fit["collocation"], amp_mrd(f_peak), damp_ins(F_AMP_INS) * f_peak,
                        damp_mrd(f_peak) * f_peak])
    delta_int = jnp.linalg.solve(rows, values)
    return dict(v=v, l=logs, beta=beta, alpha=alpha, c_int=(c1_int, c2_int), c_mrd=(c1_mrd, c2_mrd), t0=t0,
                rho=rho, delta=delta_int, gamma=gamma, f_ring=f_ring, f_damp=f_damp, f_peak=f_peak, final_spin=af,
                eta=eta)


# --- the table ---------------------------------------------------------------


def table(params, f_min, amp_scale=1.0):
    """The table of a binary (a flat array of :data:`SIZE` values) from
    ``params``: ``mass_1``, ``mass_2`` (solar masses, either order),
    ``chi_1``, ``chi_2`` (aligned spins), ``luminosity_distance`` (Mpc),
    ``inclination``, ``phi_ref``, ``f_ref`` (Hz, zero for ``f_min``), and
    the first frequency ``f_min``, as
    ``XLALSimInspiralChooseFDWaveformSequence``."""
    m1, m2, chi1, chi2, distance, inclination, phi_ref, f_ref = params
    total = m1 + m2
    c = coefficients(m1 / total, m2 / total, chi1, chi2)
    eta = c["eta"]
    s = total * MTSUN_SI
    log_s = jnp.log(s)
    zero = jnp.zeros((), real())

    # -Psi (h = A exp(-i Psi)) and the time shift t0 Mf, in powers of f
    rows = [[zero] * len(BASIS) for _ in range(3)]

    def add(r, a, l, value):
        rows[r][_INDEX[(a, l)]] = rows[r][_INDEX[(a, l)]] + value

    log_pis = jnp.log(PI * s)
    # the powers of (pi M)^(1/3) by products (more accurate than exp and log
    # in the fast arithmetic of the MLX set-up)
    u = jnp.cbrt(PI * s)
    u_pow = [1.0, u]
    for _ in range(8):
        u_pow.append(u_pow[-1] * u)
    u_inv = 1.0 / u
    u_pow_inv = {1: u_inv, 2: u_inv * u_inv, 3: u_inv * u_inv * u_inv}
    for k, coefficient in enumerate(c["v"]):
        if k < 2:
            continue
        a = (k - 5) / 3.0
        scale = u_pow[k - 5] if k >= 5 else u_pow_inv[5 - k]
        if k in c["l"]:
            coefficient = coefficient + c["l"][k] * log_pis / 3.0
            add(0, a, 1, -c["l"][k] / 3.0 * scale)
        add(0, a, 0, -coefficient * scale)
    b1, b2, b3 = c["beta"]
    add(1, 1, 0, -(b1 / eta + c["c_int"][1]) * s)
    add(1, -3, 0, b3 / (3.0 * eta) / s ** 3)
    add(1, 0, 1, -b2 / eta)
    add(1, 0, 0, -b2 / eta * log_s - c["c_int"][0])
    a1, a2, a3, a4, a5 = c["alpha"]
    add(2, 1, 0, -(a1 / eta + c["c_mrd"][1]) * s)
    add(2, -1, 0, a2 / eta / s)
    add(2, 3 / 4, 0, -4.0 / 3.0 * a3 / eta * jnp.sqrt(s) * jnp.sqrt(jnp.sqrt(s)))
    add(2, 0, 0, -c["c_mrd"][0])
    for r in range(3):
        add(r, 1, 0, c["t0"] * s)
    rows = jnp.stack([jnp.stack([jnp.asarray(x, real()) for x in row]) for row in rows])
    f_damp, f_ring = c["f_damp"], c["f_ring"]
    fields = dict(phase=rows, atan_a=-a4 / eta, atan_w=s / f_damp, atan_shift=a5 * f_ring / s,
                  f_phase_ins=F_PHASE_INS / s, f_phase_mrd=0.5 * f_ring / s)

    # the reference phase: phi(f_ref) = 2 phi_ref (f_ref = 0 is f_min)
    f_ref = jnp.where(f_ref == 0.0, f_min, f_ref)
    rest, inspiral = _phase_rest(fields, frequency_powers(jnp.asarray(f_ref, real())))
    rows = rows + (2.0 * phi_ref - rest[..., 0]) * jnp.asarray(np.eye(len(BASIS))[CONST], real())

    # the leading term in float64 (see ..xas.core.phase)
    m1d, m2d = (jnp.asarray(m, jnp.float64) for m in (m1, m2))
    t0 = K0 * jnp.cbrt(m1d + m2d) / (m1d * m2d)
    f_ref64 = jnp.asarray(f_ref, jnp.float64)
    anchor = jnp.where(inspiral[..., 0], t0 * jnp.exp(-5.0 / 3.0 * jnp.log(f_ref64)), 0.0)
    big64 = jnp.stack([t0, anchor])
    big = big64.astype(real())
    big_lo = jax.lax.stop_gradient(big64 - big.astype(jnp.float64)).astype(real())

    # the amplitude
    prefactor = K_AMP * amp_scale * jnp.sqrt(m1 * m2) * jnp.exp(-jnp.log(total) / 6.0) / distance
    amp_ins = [prefactor * x * u_pow[k] for k, x in enumerate(c["rho"])]
    g1, g2, g3 = c["gamma"]
    width = g3 * f_damp
    error = c["final_spin"] > 1.0
    cos_i = jnp.cos(inclination)
    fields = dict(
        phase=rows, big=big, big_lo=big_lo, **{k: fields[k] for k in ("atan_a", "atan_w", "atan_shift")},
        amp_ins=jnp.stack([jnp.asarray(x, real()) for x in amp_ins]), amp_int=prefactor * c["delta"],
        amp_int_s=s / c["f_peak"], amp_rd_n=prefactor * g1 / width, amp_rd_gr=g2, amp_rd_s=s / width,
        amp_rd_shift=f_ring / s, f_phase_ins=fields["f_phase_ins"], f_phase_mrd=fields["f_phase_mrd"],
        f_amp_ins=F_AMP_INS / s, f_amp_mrd=c["f_peak"] / s,
        kp=jnp.where(error, jnp.nan, 0.5 * (1.0 + cos_i * cos_i)), kc=jnp.where(error, jnp.nan, -cos_i),
    )
    return jnp.concatenate([jnp.asarray(fields[name], real()).reshape(-1) for name, _ in LAYOUT])


def narrow(flat, dtype):
    """A table (flat, float64) in ``dtype``, the part of the leading term
    of the phase that ``dtype`` lacks in ``big_lo`` (as :func:`table` makes
    it in that precision)."""
    out = flat.astype(dtype)
    if np.dtype(dtype) == np.float64:
        return out
    start = OFFSETS["big"][0]
    lo_start = OFFSETS["big_lo"][0]
    big64 = flat[..., start:start + 2] + flat[..., lo_start:lo_start + 2]
    big = big64.astype(dtype)
    lo = jax.lax.stop_gradient(big64 - big.astype(jnp.float64)).astype(dtype)
    return out.at[..., start:start + 2].set(big).at[..., lo_start:lo_start + 2].set(lo)

# --- the waveform at the frequencies -----------------------------------------


def frequency_powers(f):
    """The powers of the frequencies (Hz) that :func:`evaluate` reads:
    ``f``, ``f^(1/3)``, ``1/f``, ``log f``, ``f^(-7/6)``, ``f^(3/4)`` and,
    in float64, ``f^(-5/3)``. The cube root as in
    :func:`..xas.core.frequency_powers`."""
    m1 = 1.0 / f
    log_f = jnp.log(f)
    t = jnp.exp(log_f / 3.0)
    t = t * (4.0 - t * t * t * m1) / 3.0
    root = jnp.sqrt(f)
    f64 = jnp.asarray(f, jnp.float64)
    return f, t, m1, log_f, m1 / jnp.sqrt(t), root * jnp.sqrt(root), jnp.exp(-5.0 / 3.0 * jnp.log(f64))


def phase_basis(fp):
    """:data:`BASIS` at the frequencies (from :func:`frequency_powers`): a
    list of arrays (``None`` for the constant)."""
    f, t, m1, log_f, _, f34, _ = fp
    t2 = t * t
    mt1 = t2 * m1
    return [m1, mt1 * mt1, mt1, log_f, None, t, t * log_f, t2, f, f * t, f * t2, f * f, m1 * m1 * m1, f34]


def _col(fields, name, *index):
    value = fields[name]
    return value[(Ellipsis,) + index + (None,)] if index else value[..., None]


def _phase_rest(fields, fp):
    """The phase but for the leading term of the inspiral (and ``-A``), and
    whether the frequencies are in the inspiral."""
    f = fp[0]
    inspiral = f < _col(fields, "f_phase_ins")
    mrd = f >= _col(fields, "f_phase_mrd")
    basis = phase_basis(fp)
    sums = []
    for r, terms in enumerate(REGION_TERMS):
        total = _col(fields, "phase", r, CONST)
        for k in terms:
            total = total + _col(fields, "phase", r, k) * basis[k]
        sums.append(total)
    sums[2] = sums[2] + _col(fields, "atan_a") * jnp.arctan(_col(fields, "atan_w") * (f - _col(fields, "atan_shift")))
    return jnp.where(inspiral, sums[0], jnp.where(mrd, sums[2], sums[1])), inspiral


def phase(fields, fp):
    """The phase ``phi`` (``h = A exp(i phi)``) at the frequencies, reduced
    to ``[-pi, pi]``, with the leading term of the inspiral and the sum in
    float64 as :func:`..xas.core.phase`."""
    rest, inspiral = _phase_rest(fields, fp)
    t0 = (_col(fields, "big", 0).astype(jnp.float64)
          + jax.lax.stop_gradient(_col(fields, "big_lo", 0)).astype(jnp.float64))
    anchor = (_col(fields, "big", 1).astype(jnp.float64)
              + jax.lax.stop_gradient(_col(fields, "big_lo", 1)).astype(jnp.float64))
    phi = jnp.where(inspiral, t0 * fp[6], 0.0) + rest.astype(jnp.float64) - anchor
    return reduce(phi).astype(rest.dtype)


def amplitude(fields, fp):
    """The amplitude ``A`` at the frequencies."""
    f, t, _, _, m76 = fp[:5]

    def col(name, *index):
        return _col(fields, name, *index)

    poly = col("amp_ins", 9)
    for k in range(8, -1, -1):
        poly = poly * t + col("amp_ins", k)
    y = col("amp_int_s") * f
    inter = col("amp_int", 4)
    for j in range(3, -1, -1):
        inter = inter * y + col("amp_int", j)
    u = col("amp_rd_s") * (f - col("amp_rd_shift"))
    mrd = col("amp_rd_n") * jnp.exp(-col("amp_rd_gr") * u) / (u * u + 1.0)
    amp = jnp.where(f < col("f_amp_ins"), poly, jnp.where(f >= col("f_amp_mrd"), mrd, inter))
    return m76 * amp


def evaluate(fields, fp):
    """``(h_+, h_x)`` at the frequencies (powers ``fp`` from
    :func:`frequency_powers`) for a table (:func:`unpack`; one binary, or a
    batch with a trailing axis for the frequencies to broadcast against)."""
    amp = amplitude(fields, fp)
    phi = phase(fields, fp)
    c, s = jnp.cos(phi), jnp.sin(phi)
    ap = _col(fields, "kp") * amp
    ac = _col(fields, "kc") * amp
    return jax.lax.complex(ap * c, ap * s), jax.lax.complex(-ac * s, ac * c)
