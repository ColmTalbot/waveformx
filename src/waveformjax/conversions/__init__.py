# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Conditioned waveforms and the conversions between the time and frequency
domains: LALSimulation 6.2.0's ``SimInspiralTD`` and ``SimInspiralFD``
(``LALSimInspiralGeneratorConditioning.c``, ``LALSimInspiral.c``), following
waveformrs's ``src/conversions/`` and ``src/conditioning.rs``.

=====================  ===============  ==========================================
function               approximant      LAL
=====================  ===============  ==========================================
:func:`simulate_td`    IMRPhenomXPNR    the frequency-domain model in the time
                                        domain (``XLALSimInspiralTDFromFD``),
                                        conditioned again in the time domain
:func:`simulate_td`    NRSur7dq4        the time-domain model tapered at the start
:func:`simulate_fd`    IMRPhenomXPNR    the frequency-domain model from a lower
                                        frequency, tapered and shifted in time
:func:`simulate_fd`    NRSur7dq4        the time-domain model tapered, padded to a
                                        power of two and Fourier transformed
=====================  ===============  ==========================================

The lengths, epochs, frequency bounds, shifts and taper widths depend on the
binary; they are computed on the host as LAL computes them (lengths and
epochs exactly). The waveform itself is computed by compiled JAX functions
whose array lengths are fixed (the length of the Fourier transform), so that
one compilation serves many binaries.

:func:`simulate_td_function` and :func:`simulate_fd_function` give the
conversion as a pure function of a :class:`~waveformjax.Binary` for use with
``jax.jit``, ``jax.vmap`` and ``jax.grad``: the lengths, shifts, tapers and
epoch are those of a reference binary, while the starting frequencies given
to the model and the cutoff of the high-pass filters that move continuously
with the masses and spins follow the binary.

Accuracy: lengths and epochs are LAL's. The high-pass filter, whose sections
have poles near ``z = 1``, amplifies rounding differences of its input (from
the Fourier transforms, which are not FFTW's) by ``~1e4``, so the
time-domain waveforms agree with LAL to ``~1e-12`` of their peak; the
frequency-domain ones to rounding.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .._binary import Binary
from .._results import FrequencySeriesPolarizations, TimeSeriesPolarizations
from ..exceptions import InvalidParameterError, ModelError, UnsupportedError
from . import _from_fd, _from_td
from ._models import GENERATORS

__all__ = ["simulate_td", "simulate_fd", "simulate_td_function", "simulate_fd_function", "SimulationFunction"]


def _generator(approximant: str):
    try:
        return GENERATORS[approximant]
    except (KeyError, TypeError):
        raise UnsupportedError(f"{approximant} has no conversions in waveformjax") from None


def _arrays(binary) -> Binary:
    return Binary(*(np.asarray(value, dtype=np.float64) for value in binary))


def _check_finite(approximant, hplus, hcross):
    if not (np.all(np.isfinite(np.asarray(hplus))) and np.all(np.isfinite(np.asarray(hcross)))):
        raise ModelError(f"{approximant} could not be evaluated for this binary")


def _check_td(delta_t: float, f_min: float, positive_f_min: bool) -> None:
    if not (np.isfinite(delta_t) and delta_t > 0):
        raise InvalidParameterError("delta_t must be positive")
    if not np.isfinite(f_min) or (f_min <= 0 if positive_f_min else f_min < 0):
        raise InvalidParameterError(f"f_min must be {'positive' if positive_f_min else 'non-negative'} for this model")


def _check_fd(delta_f: float, f_min: float, f_max: float, positive_f_min: bool) -> None:
    if not (np.isfinite(delta_f) and delta_f >= 0):
        raise InvalidParameterError("delta_f must be non-negative")
    if not (np.isfinite(f_max) and f_max > 0):
        raise InvalidParameterError("f_max must be positive")
    if not np.isfinite(f_min) or (f_min <= 0 if positive_f_min else f_min < 0):
        raise InvalidParameterError(f"f_min must be {'positive' if positive_f_min else 'non-negative'} for this model")


class SimulationFunction:
    """A conversion as a pure function ``f(binary) -> (hplus, hcross)`` of a
    :class:`~waveformjax.Binary` (of float64 arrays), from
    :func:`simulate_td_function` or :func:`simulate_fd_function`.

    The arrays have the length of the ``reference`` result (a
    :class:`~waveformjax.TimeSeriesPolarizations` or
    :class:`~waveformjax.FrequencySeriesPolarizations`, whose ``epoch`` and
    sampling the outputs share); at the reference binary they are its
    arrays, to rounding."""

    def __init__(self, function: Callable, reference: Any):
        self._function = function
        self.reference = reference

    @property
    def epoch(self) -> float:
        return self.reference.epoch

    def __len__(self) -> int:
        return int(self.reference.hplus.shape[-1])

    def __call__(self, binary):
        return self._function(binary)


# ---------------------------------------------------------------------------
# the conversions, with what the pure functions need
# ---------------------------------------------------------------------------


def _td(approximant: str, binary: Binary, options, delta_t: float, f_min: float):
    """``(result, function)``: :func:`simulate_td` and the pure function
    with its plan."""
    model = _generator(approximant)
    if model.domain == "frequency":
        _check_td(delta_t, f_min, True)
        cutoff = model.cutoff(binary)
        try:
            plan = _from_fd.td_plan(_from_fd.params_of(binary), delta_t, f_min, cutoff)
        except ValueError as error:
            raise ModelError(str(error)) from None
        model.validate(binary, plan.fd.f_start, cutoff(plan.fd.f_max), options)
        q, qfd = _from_fd._dynamic(plan), _from_fd._dynamic(plan.fd)
        inner = plan.length >= 2 * _from_fd.c.MIN_TAPER_SAMPLES

        band = _from_fd._short_band(plan.fd)

        def buffer(binary):
            return _from_fd._td_kernel(binary, q, qfd, n_fft=plan.n_fft, band=band, inner=inner, model=model,
                                       options=options)

        hp, hc, nzeros = buffer(_arrays(binary))
        first, stop = plan.first, plan.first + plan.length - int(nzeros)
        result = TimeSeriesPolarizations(hp[first:stop], hc[first:stop], plan.epoch.seconds(), plan.delta_t)

        def function(binary):
            hp, hc, _ = buffer(binary)
            return hp[first:stop], hc[first:stop]

        return result, function
    _check_td(delta_t, f_min, False)
    grid = model.setup(binary, delta_t, f_min, options)
    hp, hc = model.evaluate(grid, options)
    taper = _from_td._numbers(_from_td.taper_plan(np.asarray(hp)[:grid.n], np.asarray(hc)[:grid.n]))
    hp, hc = _from_td._tapered(hp, hc, taper)
    result = TimeSeriesPolarizations(hp[:grid.n], hc[:grid.n], grid.epoch, delta_t)
    times = jnp.asarray(model.times(grid))

    def function(binary):
        hp, hc = model.at(_reference_frequency(binary, f_min), times, options)
        return _from_td._tapered(hp, hc, taper)

    return result, function


def _fd(approximant: str, binary: Binary, options, delta_f: float, f_min: float, f_max: float):
    """``(result, function)``: :func:`simulate_fd` and the pure function
    with its plan."""
    model = _generator(approximant)
    if model.domain == "frequency":
        _check_fd(delta_f, f_min, f_max, True)
        cutoff = model.cutoff(binary)
        plan = _from_fd.fd_plan(_from_fd.params_of(binary), delta_f, f_min, f_max, cutoff)
        model.validate(binary, plan.f_start, cutoff(f_max), options)
        q = _from_fd._dynamic(plan)

        band = _from_fd._short_band(plan)

        def function(binary):
            return _from_fd._fd_kernel(binary, q, f_min, n=plan.n, band=band, model=model, options=options)

        hp, hc = function(_arrays(binary))
        return FrequencySeriesPolarizations(hp, hc, plan.epoch.seconds(), plan.delta_f), function
    _check_fd(delta_f, f_min, f_max, False)
    delta_t = _from_td.sampling(delta_f, f_max)
    grid = model.setup(binary, delta_t, f_min, options)
    hp, hc = model.evaluate(grid, options)
    taper = _from_td._numbers(_from_td.taper_plan(np.asarray(hp)[:grid.n], np.asarray(hc)[:grid.n]))
    try:
        plan = _from_td.fd_plan(grid.n, delta_t, grid.epoch, delta_f, f_max)
    except ValueError as error:
        raise InvalidParameterError(str(error)) from None
    q = _from_td._numbers(plan)
    hp, hc = _from_td._fd_kernel(hp, hc, taper, q, chirplen=plan.chirplen)
    result = FrequencySeriesPolarizations(hp, hc, plan.epoch.seconds(), plan.delta_f)
    times = jnp.asarray(model.times(grid))

    def function(binary):
        hp, hc = model.at(_reference_frequency(binary, f_min), times, options)
        return _from_td._fd_kernel(hp, hc, taper, q, chirplen=plan.chirplen)

    return result, function


def _reference_frequency(binary, f_min):
    """``f_ref = 0`` as NRSur7dq4 reads it: ``f_min``."""
    f_ref = jnp.asarray(binary.f_ref, jnp.float64)
    return binary._replace(f_ref=jnp.where(f_ref == 0.0, f_min, f_ref))


# ---------------------------------------------------------------------------
# public functions
# ---------------------------------------------------------------------------


def simulate_td(approximant: str, binary: Binary, options, delta_t: float, f_min: float) -> TimeSeriesPolarizations:
    """``SimInspiralTD(..., delta_t, f_min, binary.f_ref, approximant)``:
    the conditioned time-domain waveform, with LAL's length, epoch and
    sampling interval."""
    result, _ = _td(approximant, binary, options, float(delta_t), float(f_min))
    _check_finite(approximant, result.hplus, result.hcross)
    return result


def simulate_fd(
    approximant: str, binary: Binary, options, delta_f: float, f_min: float, f_max: float
) -> FrequencySeriesPolarizations:
    """``SimInspiralFD(..., delta_f, f_min, f_max, binary.f_ref,
    approximant)``: the conditioned frequency-domain waveform on ``k
    delta_f`` (``delta_f = 0`` chooses it), with the epoch of its time
    series."""
    result, _ = _fd(approximant, binary, options, float(delta_f), float(f_min), float(f_max))
    _check_finite(approximant, result.hplus, result.hcross)
    return result


def simulate_td_function(approximant: str, binary: Binary, options, delta_t: float, f_min: float) -> SimulationFunction:
    """:func:`simulate_td` as a pure function of the binary, with the
    lengths, shifts, tapers and epoch of ``binary`` (see
    :class:`SimulationFunction`)."""
    result, function = _td(approximant, binary, options, float(delta_t), float(f_min))
    _check_finite(approximant, result.hplus, result.hcross)
    return SimulationFunction(function, result)


def simulate_fd_function(
    approximant: str, binary: Binary, options, delta_f: float, f_min: float, f_max: float
) -> SimulationFunction:
    """:func:`simulate_fd` as a pure function of the binary, with the
    lengths, shifts, tapers and epoch of ``binary`` (see
    :class:`SimulationFunction`)."""
    result, function = _fd(approximant, binary, options, float(delta_f), float(f_min), float(f_max))
    _check_finite(approximant, result.hplus, result.hcross)
    return SimulationFunction(function, result)
