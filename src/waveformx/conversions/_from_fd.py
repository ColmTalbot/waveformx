# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``SimInspiralFD`` and ``SimInspiralTD`` for a frequency-domain model
(IMRPhenomXPNR), following waveformrs's ``src/conversions/mod.rs``
(``conditioned_fd``, ``td_from_fd``, ``conditioned_td``).

In LALSimulation 6.2.0 IMRPhenomXPNR has a time-domain generator,
``XLALSimInspiralTDFromFD``, so ``SimInspiralTD`` conditions twice: it asks
``SimInspiralChooseTDWaveform`` for the waveform from a lower frequency
``f_start`` (where the chirp is longer by ``0.1 t_chirp + t_merge + 3 /
f_min``), which transforms the model from the frequency domain (starting
lower again, tapered, shifted, high-pass filtered at ``f_start``, cropped
and tapered at both ends), and then conditions that in the time domain
(``XLALSimInspiralTDConditionStage1`` and ``Stage2``: tapered over the extra
time, high-pass filtered at ``f_min`` and tapered at both ends again).

:func:`fd_plan` and :func:`td_plan` compute on the host, from the concrete
binary, LAL's lengths, shifts, taper widths and epochs. The kernels compute
the waveform from a traced binary with these fixed, in arrays whose lengths
depend only on the length of the Fourier transform (a power of two), so that
one compilation serves many binaries:

* the model is evaluated on one array for every binary of that length:
  IMRPhenomXPNR smooths its antisymmetric amplitude ratio with a moving
  average along the array of frequencies (81 wide, leaving the first 40 and
  the last 41 values unchanged), which LAL takes over its band ``i_start <=
  k < i_stop``. The array starts with the band, continues with frequencies
  just above its last one and ends with the band's last 82 frequencies
  again, so that the first ``L - 41`` frequencies of the band (``L`` its
  length) are averaged as in LAL and the last 41, unaveraged, are taken
  from the end (:func:`model_frequencies`); bands shorter than
  :data:`SHORT_BAND` are evaluated on their own;
* the time series lives in a buffer of the length of the transform, with
  the samples outside LAL's series zero;
* the starting frequencies given to the model and the cutoff of the first
  high-pass filter are computed from the traced binary, so that the
  gradients include their dependence on the masses and spins.
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import _conditioning as c
from ._conditioning import HOST, TRACED

__all__ = ["FdPlan", "TdPlan", "fd_plan", "td_plan", "conditioned_fd", "time_domain"]


class Params(NamedTuple):
    """The masses (solar masses) and aligned spins used by the bounds."""

    mass_1: float
    mass_2: float
    spin_1z: float
    spin_2z: float


def params_of(binary, m=HOST) -> Params:
    if m is HOST:
        return Params(float(binary.mass_1), float(binary.mass_2), float(np.asarray(binary.spin_1)[2]),
                      float(np.asarray(binary.spin_2)[2]))
    return Params(binary.mass_1, binary.mass_2, binary.spin_1[2], binary.spin_2[2])


def _merge(p: Params, m):
    return c.merge_and_ringdown_time(p.mass_1, p.mass_2, p.spin_1z, p.spin_2z, m)


def _chirp(p: Params, f, m):
    return c.chirp_time_bound(f, p.mass_1, p.mass_2, p.spin_1z, p.spin_2z, m)


def _start(p: Params, t, m):
    return c.chirp_start_frequency_bound(t, p.mass_1, p.mass_2, m)


def model_start_frequency(p: Params, f_min, m):
    """The starting frequency ``conditioned_fd`` gives the model: where the
    Newtonian chirp lasts ``1.1 t_chirp(min(f_min, f_kerr))``."""
    f_low = m.minimum(f_min, c.kerr_isco_frequency(p.mass_1, p.mass_2, m))
    return _start(p, (1.0 + c.EXTRA_TIME_FRACTION) * _chirp(p, f_low, m), m), f_low


def td_start_frequency(p: Params, f_min, m):
    """The starting frequency ``conditioned_td`` asks the time-domain
    generator for, with ``min(f_min, f_kerr)``, ``t_chirp`` and ``t_extra``."""
    f_low = m.minimum(f_min, c.kerr_isco_frequency(p.mass_1, p.mass_2, m))
    t_chirp = _chirp(p, f_low, m)
    t_extra = c.EXTRA_CYCLES / f_low
    f_start = _start(p, m.fma(1.0 + c.EXTRA_TIME_FRACTION, t_chirp, _merge(p, m)) + t_extra, m)
    return f_start, f_low, t_chirp, t_extra


# ---------------------------------------------------------------------------
# plans (host)
# ---------------------------------------------------------------------------


class FdPlan(NamedTuple):
    """``conditioned_fd`` for one binary: the series of ``n`` samples spaced
    by ``delta_f`` (static ``n``), the model's band ``i_start <= k <
    i_stop`` and starting frequency ``f_start``, the taper ``k0 <= k < k1``,
    the shift of ``shift`` samples of ``delta_t`` and the epoch."""

    n: int
    delta_f: float
    delta_t: float
    f_max: float
    f_start: float
    i_start: int
    i_stop: int
    k0: int
    k1: int
    shift: int
    epoch: c.GpsTime


def fd_plan(p: Params, delta_f: float, f_min: float, f_max: float, cutoff) -> FdPlan:
    """The plan of ``conditioned_fd(model, delta_f, f_min, f_max)``.
    ``cutoff(f_max)`` is the model's highest frequency ``f_max'`` (LAL's
    ``f_max_prime``): the model's series has ``n = 2^ceil(log2(floor(max(f_max,
    f_max') / delta_f))) + 1`` samples (IMRPhenomX's uniform grid)."""
    f_nyquist = c.nyquist_frequency(f_max, delta_f)
    delta_t = 0.5 / f_nyquist
    f_low = min(f_min, c.kerr_isco_frequency(p.mass_1, p.mass_2))
    t_chirp = _chirp(p, f_low, HOST)
    t_merge = _merge(p, HOST)
    t_extra = c.EXTRA_CYCLES / f_low
    f_start = _start(p, (1.0 + c.EXTRA_TIME_FRACTION) * t_chirp, HOST)
    t_chirp = _chirp(p, f_start, HOST)
    chirplen = c.next_power_of_two_above(float(c.c_round(HOST.fma(2.0, t_extra, t_chirp + t_merge) / delta_t)))
    if delta_f == 0.0:
        delta_f = 1.0 / (chirplen * delta_t)
    # the uniform grid of IMRPhenomX
    f_max_prime = cutoff(f_max)
    i_start = int(f_start / delta_f)
    i_stop = int(f_max_prime / delta_f) + 1
    last = f_max if f_max_prime < f_max else f_max_prime
    n = int(2.0 ** math.ceil(math.log2(max(int(last / delta_f), 1)))) + 1
    k0 = c.c_round(f_start / delta_f)
    k1 = c.c_round(f_low / delta_f)
    shift = c.c_round(t_merge / delta_t)
    epoch = c.GpsTime.from_seconds(-1.0 / delta_f).add_seconds(shift * delta_t)
    return FdPlan(n, delta_f, delta_t, f_max, f_start, i_start, i_stop, k0, k1, shift, epoch)


class TdPlan(NamedTuple):
    """``SimInspiralTD`` (``conditioned_td`` of ``td_from_fd``) for one
    binary: the inner frequency-domain plan, the transform of ``n_fft``
    samples (static) shifted by ``shift`` samples, the series ``first <= j <
    first + length`` of the buffer sampled every ``delta_t`` (LAL's
    ``1 / (delta_f n_fft)``), the taper widths of the two stages and the
    epoch. The output is ``length`` samples less the trailing zeros found
    before the second high-pass filter."""

    fd: FdPlan
    n_fft: int
    delta_t: float
    requested_delta_t: float
    f_min: float
    shift: int
    first: int
    length: int
    inner_end_taper: int
    inner_start_taper: int
    stage_1_taper: int
    end_taper: int
    start_taper: int
    epoch: c.GpsTime


def td_plan(p: Params, delta_t: float, f_min: float, cutoff) -> TdPlan:
    """The plan of ``SimInspiralTD(..., delta_t, f_min)``."""
    # conditioned_td
    f_start, f_low, t_chirp, t_extra = td_start_frequency(p, f_min, HOST)
    # td_from_fd(model, delta_t, f_start)
    f_max = 0.5 / delta_t
    f_low_i = min(f_start, c.kerr_isco_frequency(p.mass_1, p.mass_2))
    t_chirp_i = _chirp(p, f_low_i, HOST)
    t_merge = _merge(p, HOST)
    t_extra_i = c.EXTRA_CYCLES / f_low_i
    fd = fd_plan(p, 0.0, f_start, f_max, cutoff)
    shift = c.c_round(t_extra_i / delta_t)
    t_shift = shift * delta_t
    epoch = fd.epoch.add_seconds(t_shift)
    n_fft = 2 * (fd.n - 1)
    series_delta_t = 1.0 / (fd.delta_f * n_fft)
    f_start_i = _start(p, (1.0 + c.EXTRA_TIME_FRACTION) * t_chirp_i, HOST)
    t_chirp_i = _chirp(p, f_start_i, HOST)
    length = c.c_round((t_chirp_i + t_merge) / delta_t)
    end = int(n_fft - c.c_round(t_shift / delta_t))
    first = c.c_int_difference(end, length)
    if first < 0 or first + length > n_fft:
        raise ValueError("the conditioned waveform does not fit into its transform")
    epoch = epoch.add_seconds(first * series_delta_t)
    f_isco = c.schwarzschild_isco_frequency(p.mass_1, p.mass_2)

    def width(f):
        return max(c.c_round(1.0 / (f * series_delta_t)), c.MIN_TAPER_SAMPLES)

    inner = length >= 2 * c.MIN_TAPER_SAMPLES
    return TdPlan(
        fd=fd, n_fft=n_fft, delta_t=series_delta_t, requested_delta_t=delta_t, f_min=f_min, shift=shift,
        first=first, length=length,
        inner_end_taper=width(f_isco) if inner else 0, inner_start_taper=width(f_low_i) if inner else 0,
        stage_1_taper=c.c_round(HOST.fma(c.EXTRA_TIME_FRACTION, t_chirp, t_extra) / series_delta_t),
        end_taper=width(f_isco), start_taper=width(f_low), epoch=epoch,
    )


def _dynamic(plan: NamedTuple) -> dict:
    """The numbers of a plan as numpy scalars (traced by ``jax.jit``)."""
    out = {}
    for name, value in plan._asdict().items():
        if isinstance(value, bool) or name in ("n", "n_fft"):
            continue
        if isinstance(value, int):
            out[name] = np.int64(value)
        elif isinstance(value, float):
            out[name] = np.float64(value)
    return out


# ---------------------------------------------------------------------------
# kernels (traced)
# ---------------------------------------------------------------------------


#: Frequencies at the end of the model's array: a copy of the end of the
#: band whose last 41 values the moving average leaves unchanged.
TAIL = 82
#: Bands shorter than this are evaluated on an array of their own length.
SHORT_BAND = 2 * TAIL


def model_frequencies(i_start, i_stop, delta_f, n):
    """The frequencies at which the model is evaluated for the band
    ``i_start <= k < i_stop`` of a series of ``n`` samples: the band, then
    frequencies increasing from its last one by ``delta_f / (1024 n)``
    (within the model's range, so that every window of the moving average
    spans a positive frequency interval), then the band's last
    :data:`TAIL` frequencies again (``n + TAIL`` in all), and the index of
    the value of each ``k`` of the series in that array."""
    j = jnp.arange(n)
    length = i_stop - i_start
    above = (j - length + 1).astype(jnp.float64) * (delta_f / (1024.0 * n))
    head = jnp.where(j < length, (i_start + j) * delta_f, (i_stop - 1) * delta_f + above)
    tail = (i_stop - TAIL + jnp.arange(TAIL)) * delta_f
    index = jnp.where(j < i_stop - TAIL // 2, j - i_start, n + j - (i_stop - TAIL))
    return jnp.concatenate([head, tail]), jnp.clip(index, 0, n + TAIL - 1)


def _conditioned_fd(binary, q, f_min, n, band, model, options):
    """``conditioned_fd`` from a traced binary with the numbers ``q`` of an
    :class:`FdPlan` (:func:`_dynamic`): the model from the traced starting
    frequency, tapered and shifted. ``band`` is the length of a short band
    evaluated on its own (static), else ``None``."""
    p = params_of(binary, TRACED)
    f_start, _ = model_start_frequency(p, f_min, TRACED)
    i_start, i_stop, delta_f = q["i_start"], q["i_stop"], q["delta_f"]
    k = jnp.arange(n)
    if band is None:
        frequencies, index = model_frequencies(i_start, i_stop, delta_f, n)
    else:
        frequencies = (i_start + jnp.arange(band)) * delta_f
        index = jnp.clip(k - i_start, 0, band - 1)
    hp, hc = model.uniform(binary, frequencies, delta_f, f_start, q["f_max"], options)
    inside = (k >= i_start) & (k < i_stop)
    h = jnp.where(inside, jnp.stack([hp, hc])[:, index], 0.0)
    h = c.taper_frequency_band(h, q["k0"], q["k1"])
    return c.time_shift(h, delta_f, q["shift"] * q["delta_t"])


def _short_band(plan: FdPlan):
    """The static band length of :func:`_conditioned_fd`."""
    length = min(plan.i_stop, plan.n) - plan.i_start
    return length if length < SHORT_BAND else None


@functools.partial(jax.jit, static_argnames=("n", "band", "model", "options"))
def _fd_kernel(binary, q, f_min, *, n, band, model, options):
    h = _conditioned_fd(binary, q, f_min, n, band, model, options)
    return h[0], h[1]


def conditioned_fd(binary, plan: FdPlan, f_min, model, options):
    """``SimInspiralFD`` of a frequency-domain model: ``(h_+, h_x)`` on ``k
    plan.delta_f``, ``k < plan.n``."""
    return _fd_kernel(binary, _dynamic(plan), f_min, n=plan.n, band=_short_band(plan), model=model, options=options)


@functools.partial(jax.jit, static_argnames=("n_fft", "band", "inner", "model", "options"))
def _td_kernel(binary, q, qfd, *, n_fft, band, inner, model, options):
    """``SimInspiralTD`` in a buffer of ``n_fft`` samples, and the number
    of trailing zeros removed by the first stage of conditioning."""
    p = params_of(binary, TRACED)
    f_min = q["f_min"]
    f_start, f_low, _, _ = td_start_frequency(p, f_min, TRACED)
    # td_from_fd(model, delta_t, f_start)
    h = _conditioned_fd(binary, qfd, f_start, n_fft // 2 + 1, band, model, options)
    h = c.time_shift(h, qfd["delta_f"], q["shift"] * q["requested_delta_t"])
    x = jnp.fft.irfft(h, n_fft, axis=-1) * n_fft * qfd["delta_f"]
    delta_t = q["delta_t"]
    x = c.high_pass(x, delta_t, f_start)
    first, length = q["first"], q["length"]
    index = jnp.arange(n_fft)
    x = jnp.where((index >= first) & (index < first + length), x, 0.0)
    if inner:
        # XLALSimInspiralTDConditionStage2
        x = c.hann_taper_end(x, first, length, q["inner_end_taper"])
        x = c.hann_taper_start(x, first, length, q["inner_start_taper"])
    # XLALSimInspiralTDConditionStage1 and Stage2 of conditioned_td
    nzeros = c.trailing_zeros(x, first, length)
    x = c.hann_taper_start(x, first, length, q["stage_1_taper"])
    x = c.high_pass(x, delta_t, f_min, first, length)
    length = length - nzeros
    x = jnp.where(index < first + length, x, 0.0)
    tapered = c.hann_taper_start(c.hann_taper_end(x, first, length, q["end_taper"]), first, length,
                                 q["start_taper"])
    x = jnp.where(length >= 2 * c.MIN_TAPER_SAMPLES, tapered, x)
    return x[0], x[1], nzeros


def time_domain(binary, plan: TdPlan, model, options):
    """``SimInspiralTD``: the buffer ``(h_+, h_x)`` of ``plan.n_fft``
    samples (the series is ``[plan.first, plan.first + plan.length -
    nzeros)``) and ``nzeros``."""
    return _td_kernel(binary, _dynamic(plan), _dynamic(plan.fd), n_fft=plan.n_fft, band=_short_band(plan.fd),
                      inner=plan.length >= 2 * c.MIN_TAPER_SAMPLES, model=model, options=options)
