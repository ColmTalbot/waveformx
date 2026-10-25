# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""LALSimulation's generic waveform conditioning, as ``SimInspiralTD`` and
``SimInspiralFD`` use it (``LALSimInspiral.c`` and
``LALSimInspiralGeneratorConditioning.c`` of LALSimulation 6.2.0, lal's
``ButterworthTimeSeries.c``, ``BilinearTransform.c``, ``IIRFilter*.c`` and
``TimeFreqFFT.c``), following waveformrs's ``src/conditioning.rs``.

Two kinds of code:

* On the host, in Python floats: the bounds on the duration of the signal
  (``XLALSimInspiralChirpTimeBound`` and its relatives), the lengths, shifts
  and taper widths derived from them, the GPS epochs (``LIGOTimeGPS``,
  rounded to nanoseconds) and the search of
  ``XLALSimInspiralREAL8WaveTaper`` for the end of its taper. The arithmetic
  is LAL's: C's ``pow``, ``cbrt`` and ``round``, and the fused multiply-adds
  of clang's arm64 build (exactly, through rational arithmetic), so that the
  lengths and epochs are LAL's.
* In JAX, pure and differentiable: the same bounds where the waveform
  depends on them continuously (the starting frequencies given to the model
  and the cutoff of a high-pass filter), the coefficients of LAL's
  Butterworth filter and the filter itself (a ``jax.lax.scan`` of LAL's
  recursion per second-order section), and the tapers. The positions and
  widths of the tapers are integers computed on the host and passed as
  traced values, so that the arrays have fixed (power-of-two) lengths and
  one compilation serves many binaries.

The bounds are written once for both, with the arithmetic of
:data:`HOST` (Python floats) or :data:`TRACED` (``jax.numpy``).
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "HOST", "TRACED", "EXTRA_TIME_FRACTION", "EXTRA_CYCLES",
    "c_round", "c_int_difference", "next_power_of_two_above", "nyquist_frequency",
    "chirp_time_bound", "merge_time_bound", "ringdown_time_bound", "final_black_hole_spin_bound",
    "chirp_start_frequency_bound", "kerr_isco_frequency", "schwarzschild_isco_frequency",
    "merge_and_ringdown_time", "GpsTime", "wave_taper_start", "apply_wave_taper_start",
    "hann_taper_start", "hann_taper_end", "high_pass", "butterworth_high_pass", "taper_frequency_band",
    "time_shift", "trailing_zeros",
]

# LAL's constants (lal 7.7)
PI = math.pi
TWO_PI = 2.0 * PI
MTSUN_SI = 4.925490947641266978197229498498379006e-6

#: Fraction of the chirp time added before ``f_min`` for tapering.
EXTRA_TIME_FRACTION = 0.1
#: Cycles at ``f_min`` added as padding.
EXTRA_CYCLES = 3.0
#: The attenuation and order of the high-pass filters of the conditioning.
FILTER_ATTENUATION = 0.99
FILTER_ORDER = 8
#: Shortest taper of ``XLALSimInspiralTDConditionStage2``.
MIN_TAPER_SAMPLES = 4
#: Local extrema closer than this to the end of the signal are ignored
#: (``LALSIMULATION_RINGING_EXTENT``).
RINGING_EXTENT = 19


# ---------------------------------------------------------------------------
# arithmetic of the host and of JAX
# ---------------------------------------------------------------------------


def _fma_exact(a: float, b: float, c: float) -> float:
    """``a * b + c`` rounded once (C's ``fma``)."""
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return a * b + c
    result = float(Fraction(a) * Fraction(b) + Fraction(c))
    return result if result != 0.0 else a * b + c  # the sign of zero


class _Host:
    """Python floats with the arithmetic of LAL's arm64 build."""

    fma = staticmethod(_fma_exact)
    cbrt = staticmethod(math.cbrt)
    pow = staticmethod(math.pow)
    abs = staticmethod(abs)
    minimum = staticmethod(min)
    maximum = staticmethod(max)

    @staticmethod
    def where(condition, a, b):
        return a if condition else b


class _Traced:
    """``jax.numpy``: ``fma`` is ``a * b + c`` (which XLA may or may not
    contract), so the values agree with LAL to rounding."""

    @staticmethod
    def fma(a, b, c):
        return a * b + c

    cbrt = staticmethod(jnp.cbrt)
    pow = staticmethod(jnp.power)
    abs = staticmethod(jnp.abs)
    minimum = staticmethod(jnp.minimum)
    maximum = staticmethod(jnp.maximum)
    where = staticmethod(jnp.where)


HOST = _Host()
TRACED = _Traced()


def c_round(x: float) -> int:
    """C's ``round``: to the nearest integer, halves away from zero."""
    if x < 0:
        return -c_round(-x)
    floor = math.floor(x)
    return int(floor) + (1 if x - floor >= 0.5 else 0)


def c_int_difference(a: int, b: int) -> int:
    """``(int)(a - b)`` for C's ``size_t`` operands: wrapped to 32 bits."""
    d = (a - b) & 0xFFFFFFFF
    return d - (1 << 32) if d >= 1 << 31 else d


def next_power_of_two_above(x: float) -> int:
    """``ldexp(1, e)`` with ``e`` from ``frexp(x, &e)``: the smallest power
    of two above ``x`` (twice ``x`` if it is a power of two already), as LAL
    rounds lengths."""
    return 1 << math.frexp(x)[1] if x > 0 else 1


def nyquist_frequency(f_max: float, delta_f: float) -> float:
    """The Nyquist frequency of ``SimInspiralFD``: ``f_max``, or ``f_max /
    delta_f`` rounded up to a power of two times ``delta_f`` if it is not
    one."""
    if delta_f != 0.0:
        n = c_round(f_max / delta_f)
        if n & (n - 1):
            return next_power_of_two_above(float(n)) * delta_f
    return f_max


# ---------------------------------------------------------------------------
# bounds on the duration of the signal (masses in solar masses; LAL's are
# in kg)
# ---------------------------------------------------------------------------


def chirp_time_bound(f_start, mass_1, mass_2, spin_1, spin_2, m=HOST):
    """``XLALSimInspiralChirpTimeBound``: an upper bound on the time from the
    gravitational-wave frequency ``f_start`` to coalescence."""
    total = mass_1 + mass_2
    mu = mass_1 * mass_2 / total
    eta = mu / total
    chi = m.abs(m.where(m.abs(spin_1) > m.abs(spin_2), spin_1, spin_2))
    c0 = m.abs(-5.0 * (total * MTSUN_SI) / (256.0 * eta))
    c2 = m.fma(11.0 / 3.0, eta, 7.43 / 2.52)
    c3 = (226.0 / 15.0) * chi
    c4 = m.fma(61.7 / 7.2 * eta, eta, m.fma(54.29 / 5.04, eta, 30.58673 / 5.08032))
    v = m.cbrt(PI * MTSUN_SI * total * f_start)
    inner = m.fma(m.fma(c4, v, c3), v, c2)
    return c0 * m.pow(v, -8.0) * m.fma(inner * v, v, 1.0)


def merge_time_bound(mass_1, mass_2, m=HOST):
    """``XLALSimInspiralMergeTimeBound``: one orbit at ``r = 9 M`` with
    speed ``c / 3``."""
    r = 9.0 * (mass_1 + mass_2) * MTSUN_SI
    v = 1.0 / 3.0
    return TWO_PI * r / v


def ringdown_time_bound(total_mass, final_spin, m=HOST):
    """``XLALSimInspiralRingdownTimeBound``: 11 e-folds of the (2,2)
    quasinormal mode (Berti, Cardoso & Will 2006), the mass in solar
    masses."""
    omega = m.fma(-1.1568, m.pow(1.0 - final_spin, 0.1292), 1.5251) / (total_mass * MTSUN_SI)
    q = m.fma(1.4187, m.pow(1.0 - final_spin, -0.4990), 0.7000)
    tau = 2.0 * q / omega
    return 11.0 * tau


def final_black_hole_spin_bound(spin_1z, spin_2z, m=HOST):
    """``XLALSimInspiralFinalBlackHoleSpinBound``."""
    s = m.fma(0.15, spin_1z + spin_2z, 0.686)
    s = m.maximum(s, m.abs(spin_1z))
    s = m.maximum(s, m.abs(spin_2z))
    return m.minimum(s, 0.998)


def chirp_start_frequency_bound(t_chirp, mass_1, mass_2, m=HOST):
    """``XLALSimInspiralChirpStartFrequencyBound``: a lower bound on the
    frequency at which the Newtonian chirp lasts ``t_chirp``."""
    total = mass_1 + mass_2
    mu = mass_1 * mass_2 / total
    eta = mu / total
    c0 = 1.0 / (8.0 * PI * (total * MTSUN_SI))
    return c0 * m.pow(5.0 * total * MTSUN_SI / (eta * t_chirp), 3.0 / 8.0)


def kerr_isco_frequency(mass_1, mass_2, m=HOST):
    """The lowest Kerr ISCO frequency, ``1 / (9^(3/2) pi M)``."""
    return 1.0 / (math.pow(9.0, 1.5) * PI * (mass_1 + mass_2) * MTSUN_SI)


def schwarzschild_isco_frequency(mass_1, mass_2, m=HOST):
    """The Schwarzschild ISCO frequency, ``1 / (6^(3/2) pi M)``."""
    return 1.0 / (math.pow(6.0, 1.5) * PI * (mass_1 + mass_2) * MTSUN_SI)


def merge_and_ringdown_time(mass_1, mass_2, spin_1z, spin_2z, m=HOST):
    """``t_merge``: the merger bound plus the ringdown bound of the final
    spin bound."""
    s = final_black_hole_spin_bound(spin_1z, spin_2z, m)
    return merge_time_bound(mass_1, mass_2, m) + ringdown_time_bound(mass_1 + mass_2, s, m)


# ---------------------------------------------------------------------------
# GPS times
# ---------------------------------------------------------------------------

_BILLION = 1_000_000_000


class GpsTime:
    """``LIGOTimeGPS`` as integer nanoseconds."""

    __slots__ = ("nanoseconds",)

    def __init__(self, nanoseconds: int = 0):
        self.nanoseconds = int(nanoseconds)

    @staticmethod
    def from_seconds(t: float) -> "GpsTime":
        """``XLALGPSSetREAL8``: ``floor(t)`` seconds and
        ``nearbyint((t - floor(t)) 1e9)`` nanoseconds."""
        seconds = math.floor(t)
        return GpsTime(_BILLION * seconds + round((t - seconds) * 1e9))

    def add_seconds(self, dt: float) -> "GpsTime":
        """``XLALGPSAdd``."""
        return GpsTime(self.nanoseconds + GpsTime.from_seconds(dt).nanoseconds)

    def seconds(self) -> float:
        """``XLALGPSGetREAL8``."""
        whole = abs(self.nanoseconds) // _BILLION * (1 if self.nanoseconds >= 0 else -1)
        return float(whole) + float(self.nanoseconds - _BILLION * whole) / 1e9


# ---------------------------------------------------------------------------
# tapers
# ---------------------------------------------------------------------------


def wave_taper_start(data) -> Optional[Tuple[int, int]]:
    """The search of ``XLALSimInspiralREAL8WaveTaper(data,
    LAL_SIM_INSPIRAL_TAPER_START)`` on the host: ``(start, n)`` with
    ``start`` the first non-zero sample and ``n`` the samples to the second
    local maximum of ``|h|`` more than 19 samples in (or to the middle of the
    signal); ``None`` where LAL does nothing (no signal, or fewer than three
    samples)."""
    data = np.asarray(data, dtype=float)
    nonzero = np.flatnonzero(data != 0.0)
    if nonzero.size == 0:
        return None
    start, end = int(nonzero[0]), int(nonzero[-1])
    if end - start <= 1:
        return None
    a = np.abs(data).tolist()
    length = len(a)
    mid = (start + end) // 2
    flag, i, n = 0, start + 1, 0
    while flag < 2 and i != mid and i + 1 < length:
        if a[i] >= a[i - 1] and a[i] >= a[i + 1]:
            if a[i] == a[i + 1]:
                i += 1
            if i - start > RINGING_EXTENT:
                flag += 1
            n = i - start
        i += 1
    if flag < 2:
        n = mid - start
    return start, n


def apply_wave_taper_start(x, start, n, active):
    """The taper of ``XLALSimInspiralREAL8WaveTaper`` at the start of ``x``
    (last axis), from :func:`wave_taper_start`: zero at ``start`` and ``1 /
    (exp(z) + 1)`` for the next ``n - 2`` samples; nothing unless
    ``active``."""
    rel = jnp.arange(x.shape[-1]) - start
    inside = (rel >= 1) & (rel < n - 1)
    real_n = jnp.asarray(n, jnp.float64)
    real_i = jnp.where(inside, rel, 1).astype(jnp.float64)
    z = (real_n - 1.0) / real_i + (real_n - 1.0) / jnp.where(inside, real_i - (real_n - 1.0), -1.0)
    sigma = 1.0 / (jnp.exp(z) + 1.0)
    factor = jnp.where(inside, sigma, jnp.where(rel == 0, 0.0, 1.0))
    return x * jnp.where(active, factor, 1.0)


def _hann(j, n):
    """``0.5 - 0.5 cos(pi j / n)``."""
    return 0.5 - 0.5 * jnp.cos(j * PI / n)


def hann_taper_start(x, first, length, n):
    """Multiply the samples ``j = 0 .. min(n, length) - 1`` of the series
    starting at index ``first`` of ``x`` by ``0.5 - 0.5 cos(pi j / n)``."""
    j = jnp.arange(x.shape[-1]) - first
    inside = (j >= 0) & (j < jnp.minimum(n, length))
    n_safe = jnp.maximum(n, 1).astype(jnp.float64)
    return x * jnp.where(inside, _hann(j.astype(jnp.float64), n_safe), 1.0)


def hann_taper_end(x, first, length, n):
    """Multiply the samples ``length - j``, ``j = 1 .. n - 1``, of the
    series starting at ``first`` by ``0.5 - 0.5 cos(pi j / n)`` (the end
    taper of ``XLALSimInspiralTDConditionStage2``)."""
    j = first + length - jnp.arange(x.shape[-1])
    inside = (j >= 1) & (j < n) & (j <= length)
    n_safe = jnp.maximum(n, 1).astype(jnp.float64)
    return x * jnp.where(inside, _hann(j.astype(jnp.float64), n_safe), 1.0)


def taper_frequency_band(h, k0, k1):
    """The frequency-domain taper of ``SimInspiralFD``: zero below ``k0``,
    ``0.5 - 0.5 cos(pi (k - k0) / (k1 - k0))`` for ``k0 <= k < k1``, and
    zero at the last (Nyquist) sample."""
    k = jnp.arange(h.shape[-1])
    width = jnp.where(k1 > k0, k1 - k0, 1).astype(jnp.float64)
    w = jnp.where(k < k0, 0.0, jnp.where(k < k1, 0.5 - 0.5 * jnp.cos(PI * (k - k0).astype(jnp.float64) / width), 1.0))
    w = w.at[-1].set(0.0)
    return h * w


def time_shift(h, delta_f, t_shift):
    """Multiply sample ``k`` by ``cexp(2 pi i k delta_f t_shift)``."""
    phase = TWO_PI * jnp.arange(h.shape[-1], dtype=jnp.float64) * delta_f * t_shift
    return h * (jnp.cos(phase) + 1j * jnp.sin(phase))


def trailing_zeros(x, first, length):
    """The number of trailing samples of the series ``x[..., first:first +
    length]`` that vanish in every row (``XLALSimInspiralTDConditionStage1``)."""
    index = jnp.arange(x.shape[-1])
    nonzero = jnp.any(x != 0.0, axis=0) & (index >= first) & (index < first + length)
    last = jnp.max(jnp.where(nonzero, index, first - 1))
    return first + length - 1 - last


# ---------------------------------------------------------------------------
# LAL's Butterworth high-pass filter
# ---------------------------------------------------------------------------

#: ``(1 / sqrt(a) - 1)^(0.5 / n)`` for the attenuation and order of the
#: conditioning filters (``XLALParsePassBandParamStruc``).
_WC_FACTOR = math.pow(1.0 / math.sqrt(FILTER_ATTENUATION) - 1.0, 0.5 / FILTER_ORDER)


def _cmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def butterworth_high_pass(delta_t, frequency):
    """The second-order sections ``(d0, d1, d2, r1, r2)`` of
    ``XLALHighPassREAL8TimeSeries(series, frequency, 0.99, 8)`` for a series
    sampled every ``delta_t``, and whether LAL filters at all (``0 <
    frequency delta_t < 0.5``): ``XLALButterworthREAL8TimeSeries``, the
    bilinear transform of ``XLALWToZCOMPLEX16ZPGFilter`` and
    ``XLALCreateREAL8IIRFilter``, for traced ``frequency``.

    Each section has zeros at ``w = 0`` (``z = 1``, gain factors ``-i``) and
    the poles ``w = +-ar + i ai``; LAL keeps the ``z``-plane pole with
    positive imaginary part ``x + i y`` (``r1 = 2 x``, ``r2 = -(x^2 +
    y^2)``) and the real part ``g`` of the gain (``d = (g, -2 g, g)``)."""
    w2 = delta_t * frequency
    valid = (w2 > 0.0) & (w2 < 0.5)
    wc = jnp.tan(PI * jnp.where(valid, w2, 0.25)) * _WC_FACTOR
    sections = []
    for i in range(FILTER_ORDER // 2):
        theta = PI * (i + 0.5) / FILTER_ORDER
        ar = wc * math.cos(theta)
        ai = wc * math.sin(theta)
        first = jnp.abs(1.0 + ai) > jnp.abs(ar)
        # the pole w = ar + i ai
        ratio_1 = -ar / (1.0 + ai)
        denom_1 = -ratio_1 * ar + (1.0 + ai)
        ratio_2 = -(1.0 + ai) / ar
        denom_2 = ratio_2 * (1.0 + ai) - ar
        ratio = jnp.where(first, ratio_1, ratio_2)
        denom = jnp.where(first, denom_1, denom_2)
        x = jnp.where(first, (ratio_1 * ar + (1.0 - ai)) / denom_1, ((1.0 - ai) * ratio_2 + ar) / denom_2)
        y = jnp.where(first, (-ratio_1 * (1.0 - ai) + ar) / denom_1, ((ar * ratio_2 - 1.0) + ai) / denom_2)
        g_pole = (ratio / denom, 1.0 / denom)
        # the pole w = -ar + i ai has the gain factor (-ratio / denom, 1 / denom)
        g_mirror = (-ratio / denom, 1.0 / denom)
        g_zero = (jnp.zeros_like(ar), -jnp.ones_like(ar))
        # LAL multiplies the factors sorted by modulus, smallest with largest
        small = g_pole[0] * g_pole[0] + g_pole[1] * g_pole[1] < 1.0
        one = (jnp.ones_like(ar), jnp.zeros_like(ar))
        # sorted (pole, mirror, zero, zero): pole * zero, then mirror * zero
        gain_a = _cmul(_cmul(one, _cmul(g_pole, g_zero)), _cmul(g_mirror, g_zero))
        # sorted (zero, zero, pole, mirror): zero * mirror, then zero * pole
        gain_b = _cmul(_cmul(one, _cmul(g_zero, g_mirror)), _cmul(g_zero, g_pole))
        g = jnp.where(small, gain_a[0], gain_b[0])
        modulus = x * x + y * y
        sections.append((g, -2.0 * g, g, 2.0 * x, -modulus))
    return sections, valid


def _section(x, coefficients, reverse):
    """``XLALIIRFilterREAL8Vector`` (``reverse`` false) or
    ``XLALIIRFilterReverseREAL8Vector`` of a second-order section from zero
    history, on the rows of ``x`` at once."""
    d0, d1, d2, r1, r2 = coefficients

    def step(carry, value):
        t0, t1 = carry
        t2 = t1
        w = r2 * t2 + value
        t1 = t0
        w = r1 * t1 + w
        return (w, t1), d2 * t2 + (d1 * t1 + d0 * w)

    zero = jnp.zeros(x.shape[:-1], x.dtype)
    _, y = jax.lax.scan(step, (zero, zero), jnp.moveaxis(x, -1, 0), reverse=reverse)
    return jnp.moveaxis(y, 0, -1)


def high_pass(x, delta_t, frequency, first=None, length=None):
    """``XLALHighPassREAL8TimeSeries(series, frequency, 0.99, 8)`` on the
    rows of ``x`` (last axis): the sections forwards and backwards, one
    after the other. With ``first`` and ``length`` the series is
    ``x[..., first:first + length]`` (traced), the rest of ``x`` being zero:
    it is zeroed before every pass, so that each pass starts from zero
    history at the ends of the series."""
    sections, valid = butterworth_high_pass(delta_t, frequency)
    if first is None:
        inside = None
    else:
        index = jnp.arange(x.shape[-1])
        inside = (index >= first) & (index < first + length)
    y = x
    for coefficients in sections:
        for reverse in (False, True):
            if inside is not None:
                y = jnp.where(inside, y, 0.0)
            y = _section(y, coefficients, reverse)
    if inside is not None:
        y = jnp.where(inside, y, 0.0)
    return jnp.where(valid, y, x)
