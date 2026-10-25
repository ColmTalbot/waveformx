# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``SimInspiralTD`` and ``SimInspiralFD`` for NRSur7dq4 in MLX, as
:mod:`waveformx.conversions` gives them for NRSur7dq4
(:mod:`waveformx.conversions._from_td`): the waveform on LAL's grid,
tapered at the start (``XLALSimInspiralREAL8WaveTaper``) and, for
``SimInspiralFD``, padded with zeros at the start (or cut) to a power of
two and Fourier transformed, in float32.

LAL does not high-pass filter this model, so nothing here is sequential.
The lengths, epochs and the end of the taper are found on the host by the
functions of the JAX conversions (:func:`~waveformx.conversions._from_td.taper_plan`
and :func:`~waveformx.conversions._from_td.fd_plan`); the taper is then a
fixed array of factors (computed in float64 by
:func:`~waveformx.conversions._conditioning.apply_wave_taper_start`) and
the transform MLX's. The end of the taper is the second local maximum of
``|h|``, found here in the float32 samples, so it can differ from LAL's by
a sample where two samples are equal to float32 rounding.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np

from ....conversions import SimulationFunction, _check_fd, _check_finite, _check_td
from ....conversions import _conditioning as c
from ....conversions import _from_td
from ...._results import FrequencySeriesPolarizations, TimeSeriesPolarizations
from ....exceptions import InvalidParameterError
from .. import _driver as _jax_driver
from .._waveform import strain_scale
from . import _waveform
from ._data import NRSurData, default_data
from ._driver import on_grid

__all__ = ["simulate_td", "simulate_fd", "simulate_td_function", "simulate_fd_function"]

NAME = "NRSur7dq4"


def _taper_factors(plan: _from_td.TaperPlan, n: int) -> mx.array:
    """The factors of the start tapers of ``h_+`` and ``h_x`` (``(2,
    n)``, float32)."""
    ones = np.ones(n)
    factors = [
        c.apply_wave_taper_start(ones, plan.start_plus, plan.n_plus, plan.active_plus),
        c.apply_wave_taper_start(ones, plan.start_cross, plan.n_cross, plan.active_cross),
    ]
    return mx.array(np.asarray(factors, dtype=np.float32))


def _padded(x, plan: _from_td.FdPlan):
    """The ``plan.chirplen`` samples of ``x`` (last axis, ``plan.length``
    samples) from index ``plan.first``, zero outside it."""
    lo = max(plan.first, 0)
    hi = min(plan.length, plan.first + plan.chirplen)
    x = x[..., lo:hi]
    left = lo - plan.first
    widths = [(0, 0)] * (x.ndim - 1) + [(left, plan.chirplen - left - (hi - lo))]
    return mx.pad(x, widths)


def _transform(h, plan: _from_td.FdPlan):
    """``XLALREAL8TimeFreqFFT`` of the padded series ``h`` (last axis)."""
    return mx.fft.rfft(_padded(h, plan), axis=-1) * plan.delta_t


def _setup(binary, options, delta_t: float, f_min: float, data: NRSurData):
    """LAL's grid, the polarizations on it and their tapers."""
    grid = _jax_driver.setup(binary, delta_t, f_min, options)
    hp, hc = on_grid(data, grid, options)
    mx.eval(hp, hc)
    plan = _from_td.taper_plan(np.array(hp), np.array(hc))
    return grid, mx.stack([hp, hc]), _taper_factors(plan, grid.n)


def _function(grid, options, f_min: float, data: NRSurData, after):
    """The pure function of the binary: the polarizations at the times of
    ``grid``, with ``f_ref = 0`` read as ``f_min`` (unless the reference is
    mass-scaled), passed (stacked, ``(...,
    2, n)``) to ``after``."""
    times = mx.array((grid.m_tot_sec * (grid.t0 + grid.dt * np.arange(grid.n))).astype(np.float32))
    scale = strain_scale(options)

    def function(binary):
        binary, has_batch = _waveform.batched(binary)
        if options.reference_mf is None:
            binary = binary._replace(f_ref=mx.where(binary.f_ref == 0, f_min, binary.f_ref))
        hp, hc = _waveform.polarizations_at(data, binary, times, options.mode_array, scale, options.reference_mf)
        out = after(mx.stack([hp, hc], axis=1))
        return (out[:, 0], out[:, 1]) if has_batch else (out[0, 0], out[0, 1])

    return function


def _td(binary, options, delta_t: float, f_min: float, data: Optional[NRSurData]):
    data = default_data() if data is None else data
    _check_td(delta_t, f_min, False)
    grid, h, taper = _setup(binary, options, delta_t, f_min, data)
    h = h * taper
    result = TimeSeriesPolarizations(h[0], h[1], grid.epoch, delta_t)
    _check_finite(NAME, result.hplus, result.hcross)
    return result, _function(grid, options, f_min, data, lambda h: h * taper)


def _fd(binary, options, delta_f: float, f_min: float, f_max: float, data: Optional[NRSurData]):
    data = default_data() if data is None else data
    _check_fd(delta_f, f_min, f_max, False)
    delta_t = _from_td.sampling(delta_f, f_max)
    grid, h, taper = _setup(binary, options, delta_t, f_min, data)
    try:
        plan = _from_td.fd_plan(grid.n, delta_t, grid.epoch, delta_f, f_max)
    except ValueError as error:
        raise InvalidParameterError(str(error)) from None
    h = _transform(h * taper, plan)
    result = FrequencySeriesPolarizations(h[0], h[1], plan.epoch.seconds(), plan.delta_f)
    _check_finite(NAME, result.hplus, result.hcross)
    return result, _function(grid, options, f_min, data, lambda h: _transform(h * taper, plan))


def simulate_td(binary, delta_t: float, f_min: float, options, data: Optional[NRSurData] = None):
    """``SimInspiralTD``: a :class:`~waveformx.TimeSeriesPolarizations`
    of float32 MLX arrays, with LAL's length, epoch and sampling
    interval."""
    return _td(binary, options, float(delta_t), float(f_min), data)[0]


def simulate_fd(binary, delta_f: float, f_min: float, f_max: float, options, data: Optional[NRSurData] = None):
    """``SimInspiralFD``: a
    :class:`~waveformx.FrequencySeriesPolarizations` of complex64 MLX
    arrays on ``k delta_f`` (``delta_f = 0`` chooses it), with the epoch of
    its time series."""
    return _fd(binary, options, float(delta_f), float(f_min), float(f_max), data)[0]


def simulate_td_function(binary, delta_t: float, f_min: float, options,
                         data: Optional[NRSurData] = None) -> SimulationFunction:
    """:func:`simulate_td` as a pure function of the binary, with the
    length, tapers and epoch of ``binary`` (a
    :class:`~waveformx.conversions.SimulationFunction`). It takes one
    binary or a batch, as :func:`~waveformx.models.nrsur7dq4.mlx.polarizations_at`,
    and can be differentiated with ``mx.grad``."""
    result, function = _td(binary, options, float(delta_t), float(f_min), data)
    return SimulationFunction(function, result)


def simulate_fd_function(binary, delta_f: float, f_min: float, f_max: float, options,
                         data: Optional[NRSurData] = None) -> SimulationFunction:
    """:func:`simulate_fd` as a pure function of the binary (see
    :func:`simulate_td_function`)."""
    result, function = _fd(binary, options, float(delta_f), float(f_min), float(f_max), data)
    return SimulationFunction(function, result)
