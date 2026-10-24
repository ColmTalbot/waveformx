# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXAS as a table of coefficients per binary and a short formula
per frequency, shared by the JAX model (:mod:`.model`) and the MLX one
(:mod:`.mlx`).

LAL's (2,2) mode (:mod:`..xpnr.xas`) is, in each of its three regions, a
sum of powers of ``Mf`` (and their products with ``log Mf``), an arctangent
and, for the amplitude, a rational function and an exponential.
:func:`table` writes those sums in powers of the frequency ``f`` in Hz,
with the powers of the binary's ``M`` folded into the coefficients (``(M
f)^a = M^a f^a``, ``log M f = log M + log f``), together with LAL's linear
time shift and reference phase. The powers of ``f`` are then those of the
frequencies alone, shared by every binary of a batch, and each frequency
costs one sum of at most 13 terms for the phase, one polynomial for the
amplitude, an arctangent, an exponential and a sine and cosine
(:func:`evaluate`). The leading term of the inspiral phase, about ``10^4``
radians for binary neutron stars, is computed in float64 whatever the
precision (:func:`phase`), so that float32 keeps the phase to its own
rounding.

float32 and float64 run the same arithmetic, which differs from LAL's by
rounding only: masses in solar masses rather than kg, and the QNM fits in
powers of ``1 - a_f`` (:func:`qnm_22`).

The strain is multiplied by ``amp_scale`` (``1e21`` unless
``scaled_strain`` is false) and the masses are in solar masses, so that
the coefficients stay within the range of float32 (the coefficient of
``f^3`` in the inspiral amplitude is about ``1e-26 M^(3 - 7/6)`` without
the scale).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..xpnr._generated import qnm as _qnm
from ..xpnr._math import MTSUN_SI, PI, Struct, lean, powers, real
from ..xpnr.amp22 import amp_coefficients
from ..xpnr.internals import errors, waveform_struct
from ..xpnr.phase22 import phase_coefficients, time_shift_22

__all__ = ["LAYOUT", "SIZE", "PHASE_BASIS", "table", "unpack", "frequency_powers", "phase", "evaluate", "YLM", "K0"]

#: The powers ``(a, l)`` of the basis ``f^a (log f)^l`` of the phase.
PHASE_BASIS = (
    (-5 / 3, 0), (-1, 0), (-2 / 3, 0), (-1 / 3, 0), (0, 1), (0, 0), (1 / 3, 0), (1 / 3, 1), (2 / 3, 0), (1, 0),
    (1, 1), (4 / 3, 0), (5 / 3, 0), (2, 0), (-2, 0), (-3, 0),
)
_BASIS_INDEX = {key: i for i, key in enumerate(PHASE_BASIS)}

#: The fields of a table and their shapes, in the order of the flat table:
#: ``phase`` (the coefficients of :data:`PHASE_BASIS` in the inspiral,
#: intermediate and ringdown, but for the leading term), ``big`` (that
#: term's coefficient ``T0`` of ``f^(-5/3)`` in the inspiral and its value
#: ``A`` at the reference frequency) and ``big_lo`` (what they lack of
#: their float64 values, for the extended arithmetic of
#: :func:`phase`), ``atan_a``, ``atan_w`` (the arctangent
#: ``a atan(w (f - shift))`` of each region), ``amp_ins`` (the inspiral
#: amplitude ``f^(-7/6) sum_k c_k f^(k/3)``), ``amp_int_n``, ``amp_int_d``
#: (the intermediate amplitude ``n / sum_j d_j f^j``), ``amp_rd_*`` (the
#: ringdown amplitude ``n f^(-7/6) exp(-gr u) / (u^2 + g)``, ``u = s (f -
#: shift)``), the boundaries of the regions and the cutoff (Hz), and the
#: factors ``kp``, ``kc`` of the polarizations (NaN where LAL fails).
LAYOUT = (
    ("phase", (3, len(PHASE_BASIS))),
    ("big", (2,)),
    ("big_lo", (2,)),
    ("atan_a", (3,)),
    ("atan_w", (3,)),
    ("shift", ()),
    ("amp_ins", (10,)),
    ("amp_int_n", ()),
    ("amp_int_d", (5,)),
    ("amp_rd_n", ()),
    ("amp_rd_gr", ()),
    ("amp_rd_s", ()),
    ("amp_rd_g", ()),
    ("f_phase_in", ()),
    ("f_phase_im", ()),
    ("f_amp_in", ()),
    ("f_amp_rd", ()),
    ("f_cut", ()),
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

#: ``2 sqrt(5 / (64 pi))``: the spherical harmonic of LAL's polarizations
#: of IMRPhenomXAS (``h_+ = -(1 + cos^2 i) / 2 Y h22``, ``h_x = i cos i Y
#: h22``).
YLM = 2.0 * float(np.sqrt(5.0 / (64.0 * np.pi)))


class _Poly:
    """A polynomial with float64 coefficients (from the lowest power), for
    evaluating the generated fits symbolically (they are rational functions
    written with ``+``, ``-``, ``*``, ``/`` and :func:`..xpnr._math.fma`)."""

    def __init__(self, c):
        self.c = np.trim_zeros(np.asarray(c, dtype=float), "b") if np.any(c) else np.zeros(1)

    @staticmethod
    def of(x):
        return x if isinstance(x, (_Poly, _Ratio)) else _Poly([float(x)])

    def __add__(self, other):
        if isinstance(other, _Ratio):
            return other + self
        return _Poly(np.polynomial.polynomial.polyadd(self.c, _Poly.of(other).c))

    __radd__ = __add__

    def __neg__(self):
        return _Poly(-self.c)

    def __sub__(self, other):
        return self + (-_Poly.of(other))

    def __rsub__(self, other):
        return _Poly.of(other) - self

    def __mul__(self, other):
        if isinstance(other, _Ratio):
            return other * self
        return _Poly(np.polynomial.polynomial.polymul(self.c, _Poly.of(other).c))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return _Ratio(self, _Poly.of(other))

    def __rtruediv__(self, other):
        return _Ratio(_Poly.of(other), self)


class _Ratio:
    """A ratio of :class:`_Poly`."""

    def __init__(self, num, den):
        self.num, self.den = num, den

    def __add__(self, other):
        other = other if isinstance(other, _Ratio) else _Ratio(_Poly.of(other), _Poly([1.0]))
        return _Ratio(self.num * other.den + other.num * self.den, self.den * other.den)

    __radd__ = __add__

    def __neg__(self):
        return _Ratio(-self.num, self.den)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, _Ratio) else _Ratio(_Poly.of(other), _Poly([1.0]))
        return _Ratio(self.num * other.num, self.den * other.den)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, _Ratio) else _Ratio(_Poly.of(other), _Poly([1.0]))
        return _Ratio(self.num * other.den, self.den * other.num)


def _about_one(poly):
    """The coefficients of ``poly(a)`` as a polynomial in ``u = 1 - a``."""
    out = np.zeros(len(poly.c))
    for k, c in enumerate(poly.c):
        out[:k + 1] += c * np.polynomial.polynomial.polypow([1.0, -1.0], k)
    return tuple(float(x) for x in out)


def about_one(fit):
    """A fit of the final spin (a ratio of polynomials in ``a``, from
    :mod:`..xpnr._generated`) as a function of ``a`` evaluated in powers of
    ``u = 1 - a``, coefficients re-expanded in float64 (the fits nearly
    cancel near ``a = 1``, which float32 cannot resolve in powers of
    ``a``)."""
    ratio = fit(_Poly([0.0, 1.0]))
    if not isinstance(ratio, _Ratio):
        ratio = _Ratio(_Poly.of(ratio), _Poly([1.0]))
    num, den = _about_one(ratio.num), _about_one(ratio.den)

    def evaluate(a):
        u = 1.0 - a
        return _horner(num, u) / _horner(den, u)

    evaluate.__doc__ = f"``{fit.__name__}`` in powers of ``1 - a`` (:func:`about_one`)."
    return evaluate


_FRING = about_one(_qnm.fring22)
_FDAMP = about_one(_qnm.fdamp22)


def _horner(coefficients, x):
    total = coefficients[-1]
    for c in coefficients[-2::-1]:
        total = total * x + c
    return total


def qnm_22(a_final):
    """``fRING`` and ``fDAMP`` of the (2,2) mode (in units of the final
    mass) for the final spin ``a_final``: LAL's fits
    (``evaluate_QNMfit_fring22``, ``..._fdamp22``) evaluated in ``1 -
    a_final``. Their numerators and denominators both nearly vanish at
    ``a_final = 1``; in powers of ``a_final`` they lose a part in 10^4 of
    their value to rounding in float32 at ``a_final = 0.9`` and one in 100
    near 1, in powers of ``1 - a_final`` they do not."""
    return _FRING(a_final), _FDAMP(a_final)


#: ``T0 = K0 M^(1/3) / (m1 m2)``: the coefficient of ``f^(-5/3)`` in the
#: inspiral phase, ``-3 / (128 eta) (pi M f)^(-5/3)``, with the masses in
#: solar masses.
K0 = -3.0 / 128.0 * float(np.pi * MTSUN_SI) ** (-5.0 / 3.0)


def unpack(flat):
    """The fields of a flat table (``(..., SIZE)``) as a dict of arrays."""
    out = {}
    for name, (start, shape) in OFFSETS.items():
        size = int(np.prod(shape)) if shape else 1
        out[name] = flat[..., start:start + size].reshape(flat.shape[:-1] + tuple(shape))
    return out


def _add_terms(row, terms, s, log_s):
    """Add ``c x^a (log x)^l`` (``x = s f``) to the coefficients ``row`` of
    :data:`PHASE_BASIS` (a list)."""
    for c, a, l in terms:
        scaled = c * s[a]
        row[_BASIS_INDEX[(a, 0)]] = row[_BASIS_INDEX[(a, 0)]] + (scaled * log_s if l else scaled)
        if l:
            row[_BASIS_INDEX[(a, 1)]] = row[_BASIS_INDEX[(a, 1)]] + scaled


def table(params, f_min, amp_scale=1.0):
    """The table of a binary (a flat array of :data:`SIZE` values) from
    ``params``: ``mass_1``, ``mass_2`` (solar masses), ``chi_1``, ``chi_2``
    (aligned spins), ``luminosity_distance`` (Mpc), ``inclination``,
    ``phi_ref``, ``f_ref`` (Hz, zero for ``f_min``), and the first frequency
    ``f_min``, as ``XLALSimInspiralChooseFDWaveformSequence``.

    The same arithmetic in float32 and float64: the masses stay in solar
    masses (rather than LAL's round trip through kg) and the QNM fits are
    those of :func:`qnm_22`."""
    with lean(True):
        return _table(params, f_min, amp_scale)


def waveform_variables(params, f_min, amp_scale):
    """``IMRPhenomXSetWaveformVariables`` as :func:`table` makes it (in
    :func:`..xpnr._math.lean` mode: masses in solar masses)."""
    m1, m2, chi1, chi2, distance, inclination, phi_ref, f_ref = params
    wf = waveform_struct(m1, m2, chi1, chi2, 0.0, f_ref, phi_ref, f_min, 0.0, distance, inclination, amp_scale)
    f_ring, f_damp = qnm_22(wf.a_final)
    wf.f_ring = f_ring / wf.m_final
    wf.f_damp = f_damp / wf.m_final
    return wf


def _table(params, f_min, amp_scale):
    m1, m2, _, _, _, inclination, _, _ = params
    return table_of(waveform_variables(params, f_min, amp_scale), m1, m2, inclination)[0]


def table_of(wf, m1, m2, inclination):
    """The table (flat) of the waveform variables ``wf``
    (:func:`waveform_variables`) of the binary of masses ``m1``, ``m2``
    (either order) and the coefficients it came from (``amp``, ``phase``,
    ``linb``: :mod:`..xpnr.amp22`, :mod:`..xpnr.phase22`)."""
    amp = amp_coefficients(wf)
    phase = phase_coefficients(wf)
    linb = time_shift_22(phase, wf)
    inveta = 1.0 / wf.eta

    s = wf.m_sec
    p = powers(s)
    s_pow = {-5 / 3: p.m_five_thirds, -1: p.m_one, -2 / 3: p.m_two_thirds, -1 / 3: p.m_one_third, 0: 1.0,
             1 / 3: p.one_third, 2 / 3: p.two_thirds, 1: s, 4 / 3: p.four_thirds, 5 / 3: p.five_thirds, 2: p.two,
             -2: p.m_two, -3: p.m_three}
    zero = jnp.zeros((), real())
    ph = phase
    norm = inveta * ph.phi_norm
    common = [(linb, 1, 0)]
    regions = [
        [(norm * ph.phi2, -1, 0), (norm * ph.phi3, -2 / 3, 0),
         (norm * ph.phi4, -1 / 3, 0), (norm * ph.phi5l, 0, 1), (norm * ph.phi6, 1 / 3, 0),
         (norm * ph.phi6l, 1 / 3, 1), (norm * ph.phi7, 2 / 3, 0), (norm * (ph.phi8 + ph.sigma1), 1, 0),
         (norm * ph.phi8l, 1, 1), (norm * ph.sigma2, 4 / 3, 0), (norm * ph.sigma3, 5 / 3, 0),
         (norm * ph.sigma4, 2, 0)],
        [(inveta * (ph.b0 + ph.c2_int), 1, 0), (inveta * ph.b1, 0, 1), (-inveta * ph.b2, -1, 0),
         (-inveta * ph.b3 / 2.0, -2, 0), (-inveta * ph.b4 / 3.0, -3, 0), (inveta * ph.c1_int, 0, 0)],
        [(inveta * (ph.c0 + ph.c2_mrd), 1, 0), (inveta * 1.5 * ph.c1, 2 / 3, 0), (-inveta * ph.c2, -1, 0),
         (-inveta * ph.c4_ov3, -3, 0), (inveta * ph.c1_mrd, 0, 0)],
    ]
    rows = []
    for terms in regions:
        row = [zero] * len(PHASE_BASIS)
        _add_terms(row, terms + common, s_pow, p.log)
        rows.append(jnp.stack([jnp.asarray(c, real()) for c in row]))
    f_damp, f_ring = wf.f_damp, wf.f_ring
    atan_a = [zero, inveta * 2.0 * ph.c_l_gr / f_damp, inveta * ph.c_l_ov_fda]
    atan_w = [zero, s / (2.0 * f_damp), s / f_damp]
    shift = f_ring / s

    prefactor = wf.amp0 * wf.amp_norm
    scale = prefactor * p.m_seven_sixths
    third = {k: v for k, v in zip(range(10), [1.0, p.one_third, p.two_thirds, s, p.four_thirds, p.five_thirds,
                                              p.two, p.two * p.one_third, p.two * p.two_thirds, p.three])}
    ins = [1.0, 0.0, amp.pn_two_thirds, amp.pn_three_thirds, amp.pn_four_thirds, amp.pn_five_thirds,
           amp.pn_six_thirds, amp.rho1, amp.rho2, amp.rho3]
    amp_ins = [scale * c * third[k] for k, c in enumerate(ins)]
    s_int = [1.0, s, p.two, p.three, p.four]
    amp_int_d = [d * sk for d, sk in zip([amp.delta0, amp.delta1, amp.delta2, amp.delta3, amp.delta4], s_int)]

    error = errors(wf)
    pfac = 0.5 * (1.0 + jnp.cos(inclination) ** 2)
    cfac = jnp.cos(inclination)
    kp = jnp.where(error, jnp.nan, -YLM * pfac)
    kc = jnp.where(error, jnp.nan, YLM * cfac)

    # the leading term in float64 (see phase)
    m1d, m2d = (jnp.asarray(m, jnp.float64) for m in (m1, m2))
    t0 = K0 * jnp.cbrt(m1d + m2d) / (m1d * m2d)
    rows = jnp.stack(rows)
    fields = dict(
        phase=rows, atan_a=jnp.stack([jnp.asarray(v, real()) for v in atan_a]),
        atan_w=jnp.stack([jnp.asarray(v, real()) for v in atan_w]), shift=shift,
        f_phase_in=ph.f_phase_match_in / s, f_phase_im=ph.f_phase_match_im / s,
    )
    # the reference phase: phi(f_ref) = 2 phi0 + pi / 4 (LAL's phifRef), the
    # leading term's part A apart
    f_ref = jnp.asarray(wf.f_ref, real())
    rest, past_inspiral = _phase_rest(fields, frequency_powers(f_ref))
    shift_constant = 2.0 * wf.phi0 + PI / 4.0 - rest[..., 0]
    constant = np.zeros(len(PHASE_BASIS))
    constant[_BASIS_INDEX[(0, 0)]] = 1.0
    rows = rows + shift_constant * jnp.asarray(constant, real())
    f_ref64 = jnp.asarray(f_ref, jnp.float64)
    anchor = jnp.where(past_inspiral[..., 0], 0.0, t0 * jnp.exp(-5.0 / 3.0 * jnp.log(f_ref64)))
    big64 = jnp.stack([t0, anchor])
    big = big64.astype(real())
    big_lo = jax.lax.stop_gradient(big64 - big.astype(jnp.float64)).astype(real())

    fields = dict(
        phase=rows, big=big, big_lo=big_lo,
        atan_a=atan_a, atan_w=atan_w, shift=shift,
        amp_ins=amp_ins, amp_int_n=prefactor, amp_int_d=amp_int_d,
        amp_rd_n=scale * amp.gamma_d13, amp_rd_gr=amp.gamma_r, amp_rd_s=s, amp_rd_g=amp.gamma_d2,
        f_phase_in=ph.f_phase_match_in / s, f_phase_im=ph.f_phase_match_im / s,
        f_amp_in=amp.f_amp_match_in / s, f_amp_rd=amp.f_amp_rd_min / s,
        f_cut=wf.f_max_prime, kp=kp, kc=kc,
    )
    flat = []
    for name, shape in LAYOUT:
        value = fields[name]
        value = jnp.stack([jnp.asarray(v, real()) for v in value]) if isinstance(value, list) else value
        flat.append(jnp.asarray(value, real()).reshape(-1))
    return jnp.concatenate(flat), Struct(amp=amp, phase=phase, linb=linb)


def frequency_powers(f):
    """The powers of the frequencies (Hz) that :func:`evaluate` reads:
    ``f``, ``f^(1/3)``, ``1/f``, ``log f``, ``f^(-7/6)`` and, in float64,
    ``f^(-5/3)``. The cube root is ``exp(log f / 3)`` with a Newton step
    (``log f`` is needed anyway; XLA's ``cbrt`` costs more than both)."""
    m1 = 1.0 / f
    log_f = jnp.log(f)
    t = jnp.exp(log_f / 3.0)
    t = t * (4.0 - t * t * t * m1) / 3.0
    f64 = jnp.asarray(f, jnp.float64)
    return f, t, m1, log_f, m1 / jnp.sqrt(t), jnp.exp(-5.0 / 3.0 * jnp.log(f64))


def phase_basis(fp):
    """:data:`PHASE_BASIS` at the frequencies (from
    :func:`frequency_powers`): a list of arrays."""
    f, t, m1, log_f = fp[:4]
    t2 = t * t
    mt1 = t2 * m1
    mt2 = mt1 * mt1
    m2 = m1 * m1
    return [m1 * mt2, m1, mt2, mt1, log_f, None, t, t * log_f, t2, f, f * log_f, f * t, f * t2, f * f, m2, m2 * m1]


#: The terms of :data:`PHASE_BASIS` each region uses (the others are zero
#: in its row of the table): the inspiral (but for its leading term,
#: ``big``), the intermediate and the ringdown.
REGION_TERMS = (
    tuple(range(1, 14)),
    (_BASIS_INDEX[(1, 0)], _BASIS_INDEX[(0, 1)], _BASIS_INDEX[(-1, 0)], _BASIS_INDEX[(-2, 0)],
     _BASIS_INDEX[(-3, 0)], _BASIS_INDEX[(0, 0)]),
    (_BASIS_INDEX[(1, 0)], _BASIS_INDEX[(2 / 3, 0)], _BASIS_INDEX[(-1, 0)], _BASIS_INDEX[(-3, 0)],
     _BASIS_INDEX[(0, 0)]),
)


def _col(fields, name, *index):
    value = fields[name]
    return value[(Ellipsis,) + index + (None,)] if index else value[..., None]


def _phase_rest(fields, fp):
    """The phase but for the leading term of the inspiral (``-A`` too),
    and whether the frequencies are past the inspiral."""
    f = fp[0]
    past_inspiral = f >= _col(fields, "f_phase_in")
    ringdown = f > _col(fields, "f_phase_im")
    basis = phase_basis(fp)
    sums = []
    for r, terms in enumerate(REGION_TERMS):
        total = _col(fields, "phase", r, _BASIS_INDEX[(0, 0)])
        for k in terms:
            if basis[k] is not None:
                total = total + _col(fields, "phase", r, k) * basis[k]
        sums.append(total)
    rest = jnp.where(ringdown, sums[2], jnp.where(past_inspiral, sums[1], sums[0]))
    y = f - _col(fields, "shift")
    a = jnp.where(ringdown, _col(fields, "atan_a", 2), _col(fields, "atan_a", 1))
    w = jnp.where(ringdown, _col(fields, "atan_w", 2), _col(fields, "atan_w", 1))
    rest = rest + jnp.where(past_inspiral, a * jnp.arctan(w * y), 0.0)
    return rest, past_inspiral


def phase(fields, fp, anchor=True):
    """The phase at the frequencies, reduced to ``[-pi, pi]``.

    The leading term of the inspiral, ``T0 f^(-5/3) - A`` (about ``10^4``
    radians from 20 Hz for a binary neutron star, where a float32 ulp is
    ``10^-3`` radians), and the sum with the rest are in float64 (the other
    terms are at most ``10^3`` radians), so that the phase of a float32
    evaluation is as accurate as its rounding to float32 allows. The
    derivatives are those of the working precision. Without ``anchor`` the
    phase lacks ``-A`` (the higher modes of IMRPhenomXHM add theirs)."""
    rest, past_inspiral = _phase_rest(fields, fp)
    t0 = (_col(fields, "big", 0).astype(jnp.float64)
          + jax.lax.stop_gradient(_col(fields, "big_lo", 0)).astype(jnp.float64))
    phi = jnp.where(past_inspiral, 0.0, t0 * fp[5]) + rest.astype(jnp.float64)
    if anchor:
        phi = phi - (_col(fields, "big", 1).astype(jnp.float64)
                     + jax.lax.stop_gradient(_col(fields, "big_lo", 1)).astype(jnp.float64))
    return reduce(phi).astype(rest.dtype)


def reduce(phi):
    """``phi`` (float64) reduced to ``[-pi, pi]``."""
    turns = jax.lax.stop_gradient(jnp.round(phi / (2.0 * np.pi)))
    return phi - turns * (2.0 * np.pi)


def amplitude(fields, fp):
    """The amplitude of the (2,2) mode at the frequencies (zero above the
    cutoff), ``h22 = amplitude exp(i phase)``."""
    f, t, m1, log_f, m76, _ = fp

    def col(name, *index):
        return _col(fields, name, *index)

    in_amp = f >= col("f_amp_in")
    rd_amp = f > col("f_amp_rd")
    y = f - col("shift")
    poly = col("amp_ins", 9)
    for k in range(8, -1, -1):
        poly = poly * t + col("amp_ins", k)
    amp_ins = m76 * poly
    den = col("amp_int_d", 4)
    for j in range(3, -1, -1):
        den = den * f + col("amp_int_d", j)
    amp_int = col("amp_int_n") / den
    u = col("amp_rd_s") * y
    amp_rd = col("amp_rd_n") * m76 * jnp.exp(-col("amp_rd_gr") * u) / (u * u + col("amp_rd_g"))
    amp = jnp.where(rd_amp, amp_rd, jnp.where(in_amp, amp_int, amp_ins))
    return jnp.where(f <= col("f_cut"), amp, 0.0)


def evaluate(fields, fp):
    """``(h_+, h_x)`` at the frequencies (powers ``fp`` from
    :func:`frequency_powers`) for a table (:func:`unpack`; the fields of
    one binary, or of a batch with a trailing axis for the frequencies to
    broadcast against). One elementwise expression: the three regions are
    evaluated at every frequency and selected."""
    amp = amplitude(fields, fp)
    phi = phase(fields, fp)
    c, s = jnp.cos(phi), jnp.sin(phi)
    ap = _col(fields, "kp") * amp
    ac = _col(fields, "kc") * amp
    return jax.lax.complex(ap * c, ap * s), jax.lax.complex(-ac * s, ac * c)
