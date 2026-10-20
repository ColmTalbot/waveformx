# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``SimInspiralTD`` and ``SimInspiralFD`` for a time-domain model that
starts at the requested frequency (NRSur7dq4, which allows ``f_min = 0``),
following waveformrs's ``src/conversions/mod.rs`` (``conditioned_td`` and
``fd_from_td``).

LAL does not lower the starting frequency of such a model: it tapers the
waveform of ``SimInspiralChooseTDWaveform`` at the start with
``XLALSimInspiralREAL8WaveTaper`` (a Planck-like taper up to the second
extremum of ``|h|``; ``generate_conditioned_td_waveform_from_td_fallback``)
and, for ``SimInspiralFD``, pads it with zeros at the start (or cuts it) to
a power of two and Fourier transforms it (``XLALREAL8TimeFreqFFT``).

Where the taper ends depends on the samples, so it is found on the host
(:func:`taper_plan`) and passed to the kernels as traced integers, with the
length of the padded series static.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import _conditioning as c

__all__ = ["TaperPlan", "taper_plan", "FdPlan", "fd_plan", "tapered", "frequency_domain"]


class TaperPlan(NamedTuple):
    """The start tapers of ``h_+`` and ``h_x`` (:func:`~._conditioning.wave_taper_start`):
    ``(start, n, active)`` of each."""

    start_plus: int
    n_plus: int
    active_plus: bool
    start_cross: int
    n_cross: int
    active_cross: bool


def taper_plan(hplus, hcross) -> TaperPlan:
    """The tapers of the concrete polarizations."""
    out = []
    for h in (hplus, hcross):
        found = c.wave_taper_start(h)
        out.extend((0, 0, False) if found is None else (found[0], found[1], True))
    return TaperPlan(*out)


class FdPlan(NamedTuple):
    """``fd_from_td`` for a waveform of ``length`` samples sampled every
    ``delta_t``: ``chirplen`` samples (static) from index ``first`` of the
    waveform (zeros before its start), and the epoch."""

    length: int
    chirplen: int
    first: int
    delta_t: float
    delta_f: float
    epoch: c.GpsTime


def sampling(delta_f: float, f_max: float) -> float:
    """The sampling interval ``0.5 / f_nyquist`` of ``SimInspiralFD``."""
    return 0.5 / c.nyquist_frequency(f_max, delta_f)


def fd_plan(length: int, delta_t: float, epoch: float, delta_f: float, f_max: float) -> FdPlan:
    """The plan of ``fd_from_td`` for a conditioned waveform of ``length``
    samples from ``epoch`` (s)."""
    f_nyquist = c.nyquist_frequency(f_max, delta_f)
    if delta_f == 0.0:
        chirplen = c.next_power_of_two_above(float(length))
    else:
        chirplen = int(2.0 * f_nyquist / delta_f)
    if chirplen == 0:
        raise ValueError("the frequency series would be empty")
    first = c.c_int_difference(length, chirplen)
    epoch = c.GpsTime.from_seconds(epoch).add_seconds(first * delta_t)
    return FdPlan(length, chirplen, first, delta_t, 1.0 / (delta_t * chirplen), epoch)


def _taper(x, t: dict):
    return jnp.stack([
        c.apply_wave_taper_start(x[0], t["start_plus"], t["n_plus"], t["active_plus"]),
        c.apply_wave_taper_start(x[1], t["start_cross"], t["n_cross"], t["active_cross"]),
    ])


def _numbers(plan: NamedTuple) -> dict:
    """The numbers of a plan as numpy scalars (traced by ``jax.jit``)."""
    out = {}
    for name, value in plan._asdict().items():
        if name == "chirplen":
            continue
        if isinstance(value, (bool, np.bool_)):
            out[name] = np.bool_(value)
        elif isinstance(value, int):
            out[name] = np.int64(value)
        elif isinstance(value, float):
            out[name] = np.float64(value)
    return out


@jax.jit
def _tapered(hplus, hcross, t):
    x = _taper(jnp.stack([hplus, hcross]), t)
    return x[0], x[1]


def tapered(hplus, hcross, plan: TaperPlan):
    """``SimInspiralTD``: the polarizations tapered at the start."""
    return _tapered(hplus, hcross, _numbers(plan))


@functools.partial(jax.jit, static_argnames=("chirplen",))
def _fd_kernel(hplus, hcross, t, q, *, chirplen):
    x = _taper(jnp.stack([hplus, hcross]), t)
    source = q["first"] + jnp.arange(chirplen)
    inside = (source >= 0) & (source < q["length"])
    y = jnp.where(inside, jnp.take(x, jnp.clip(source, 0, x.shape[-1] - 1), axis=-1), 0.0)
    h = jnp.fft.rfft(y, axis=-1) * q["delta_t"]
    return h[0], h[1]


def frequency_domain(hplus, hcross, taper: TaperPlan, plan: FdPlan):
    """``SimInspiralFD``: the polarizations (samples ``0 .. plan.length -
    1`` of ``hplus`` and ``hcross``; later ones are ignored) tapered, padded
    and transformed."""
    return _fd_kernel(hplus, hcross, _numbers(taper), _numbers(plan), chirplen=plan.chirplen)
