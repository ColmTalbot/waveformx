# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 in MLX, for Apple GPUs: the functions of
:mod:`waveformjax.models.nrsur7dq4` in float32, which MLX evaluates on the
GPU by default.

* The pure functions ``polarizations_at`` and ``modes_at``, at any times.
* LAL's waveforms on its grid (``SimInspiralChooseTDWaveform`` and
  ``SimInspiralChooseTDModes``): ``td_polarizations``, ``td_modes`` and
  ``batch_td_polarizations``, whose lengths, epochs and errors are LAL's
  (computed on the host in float64, as the JAX driver computes them).
* The conditioned waveforms (``SimInspiralTD`` and ``SimInspiralFD``):
  ``simulate_td``, ``simulate_fd`` and their pure functions
  ``simulate_td_function`` and ``simulate_fd_function``.

It needs the optional dependency ``mlx`` (``pip install waveformjax[mlx]``)
and the same data file as the JAX implementation, which reads it.

=================  ==========================================================
module             contents
=================  ==========================================================
``_data``          the data as float32 MLX arrays, with the spline
                   operators that replace JAX's ``lax.scan``
``_fits``          the parameter-space fits
``_dynamics``      the dynamics ODE in MLX operations: the reference time,
                   and the integration that the kernel replaces
``_kernel``        the integration of the dynamics in Metal kernels: the
                   states, and their Jacobian for the derivatives
``_frames``        coorbital -> coprecessing -> inertial frame in MLX
                   operations
``_wigner``        the rotation to the inertial frame in Metal kernels,
                   with its VJP
``_waveform``      the modes and polarizations of a batch of binaries
``_driver``        LAL's grids for one binary or a batch, and the waveform
                   on them
``_conditioning``  the tapers and the Fourier transform of ``SimInspiralTD``
                   and ``SimInspiralFD``
=================  ==========================================================

The pure functions take a :class:`~waveformjax.Binary` of numbers or arrays
(converted to float32 MLX arrays), either one binary or a batch along a
leading axis of its fields (``mass_1`` of shape ``(B,)``, ``spin_1`` of shape
``(B, 3)``; fields without the axis are shared), and return MLX arrays
with the same leading axis. They work under ``mx.grad`` (and ``mx.vjp``)
with respect to every parameter of the binary (as MLX arrays) and under
``mx.compile``, the gradient included; MLX cannot differentiate the kernel
in forward mode, so ``mx.jvp`` fails, and ``mx.vmap`` does not apply to it
(the functions take the batch themselves). Every output is the strain
multiplied by ``1e21`` unless the option ``scaled_strain`` is false. Where
LAL would fail to find the reference time the result is NaN.

The functions on LAL's grid take one binary of numbers (or, for
``batch_td_polarizations``, a table of binaries as
:func:`waveformjax.batch_td_polarizations`), make LAL's checks and return
the result types of the JAX functions holding MLX arrays. The pure
functions of the conditioned waveforms have the lengths, tapers and epoch
of the binary they were made from and take one binary or a batch.
"""

from __future__ import annotations

from typing import Optional

from ...._options import WaveformOptions
from .._waveform import strain_scale
from . import _conditioning, _driver, _waveform
from ._data import NRSurData, default_data, load

__all__ = [
    "NRSurData",
    "load",
    "default_data",
    "polarizations_at",
    "modes_at",
    "td_polarizations",
    "td_modes",
    "batch_td_polarizations",
    "simulate_td",
    "simulate_fd",
    "simulate_td_function",
    "simulate_fd_function",
]


def _options(options: Optional[WaveformOptions]) -> WaveformOptions:
    return WaveformOptions() if options is None else options


def polarizations_at(binary, times, options: Optional[WaveformOptions] = None, data: Optional[NRSurData] = None):
    """``(h_+, h_x)`` at ``times`` (s, relative to the peak; zero outside
    the surrogate): ``(m,)`` each for one binary, ``(B, m)`` for a batch,
    whose ``times`` may be shared (``(m,)``) or its own (``(B, m)``).
    ``binary.f_ref = 0`` starts the dynamics at the start of the
    surrogate."""
    options = _options(options)
    data = default_data() if data is None else data
    binary, has_batch = _waveform.batched(binary)
    hp, hc = _waveform.polarizations_at(data, binary, times, options.mode_array, strain_scale(options))
    return (hp, hc) if has_batch else (hp[0], hc[0])


def modes_at(binary, times, options: Optional[WaveformOptions] = None, data: Optional[NRSurData] = None):
    """The inertial modes ``{(l, m): h_lm}`` (complex64) at ``times``, as
    :func:`polarizations_at`."""
    options = _options(options)
    data = default_data() if data is None else data
    binary, has_batch = _waveform.batched(binary)
    modes = _waveform.modes_at(data, binary, times, options.mode_array, strain_scale(options))
    return modes if has_batch else {mode: h[0] for mode, h in modes.items()}


def td_polarizations(binary, delta_t: float, f_min: float, options: Optional[WaveformOptions] = None,
                     data: Optional[NRSurData] = None):
    """``SimInspiralChooseTDWaveform``: a
    :class:`~waveformjax.TimeSeriesPolarizations` of float32 MLX arrays,
    with LAL's length and epoch. ``f_ref = 0`` is read as ``f_min``. Raises
    :class:`~waveformjax.InvalidParameterError` or
    :class:`~waveformjax.ModelError` where LAL fails."""
    return _driver.td_polarizations(binary, delta_t, f_min, _options(options), data)


def td_modes(binary, delta_t: float, f_min: float, options: Optional[WaveformOptions] = None,
             data: Optional[NRSurData] = None):
    """``SimInspiralChooseTDModes``: a :class:`~waveformjax.TimeSeriesModes`
    of the inertial modes (complex64 MLX arrays), as
    :func:`td_polarizations`."""
    return _driver.td_modes(binary, delta_t, f_min, _options(options), data)


def batch_td_polarizations(parameters, delta_t: float, f_min: float, options: Optional[WaveformOptions] = None, *,
                           strict: bool = False, chunk_size: Optional[int] = None, data: Optional[NRSurData] = None):
    """The polarizations of many binaries, each of LAL's length and epoch,
    as :func:`waveformjax.batch_td_polarizations` (a
    :class:`~waveformjax.BatchTimeSeries`): ``parameters`` is a mapping of
    arrays, a structured array or an ``(n, 12)`` table (see
    :func:`waveformjax.parameter_table`). Their grids are computed together
    and the binaries evaluated together on the GPU, at most ``chunk_size``
    at a time."""
    return _driver.batch_td_polarizations(parameters, delta_t, f_min, _options(options), strict=strict,
                                          chunk_size=chunk_size, data=data)


def simulate_td(binary, delta_t: float, f_min: float, options: Optional[WaveformOptions] = None,
                data: Optional[NRSurData] = None):
    """``SimInspiralTD``: the waveform of :func:`td_polarizations` tapered
    at the start (a :class:`~waveformjax.TimeSeriesPolarizations`)."""
    return _conditioning.simulate_td(binary, delta_t, f_min, _options(options), data)


def simulate_fd(binary, delta_f: float, f_min: float, f_max: float, options: Optional[WaveformOptions] = None,
                data: Optional[NRSurData] = None):
    """``SimInspiralFD``: the waveform of :func:`simulate_td` padded to a
    power of two and Fourier transformed (a
    :class:`~waveformjax.FrequencySeriesPolarizations` of complex64 MLX
    arrays on ``k delta_f``; ``delta_f = 0`` chooses it)."""
    return _conditioning.simulate_fd(binary, delta_f, f_min, f_max, _options(options), data)


def simulate_td_function(binary, delta_t: float, f_min: float, options: Optional[WaveformOptions] = None,
                         data: Optional[NRSurData] = None):
    """:func:`simulate_td` as a pure function ``f(binary) -> (hplus,
    hcross)`` with the length, tapers and epoch of ``binary`` (a
    :class:`~waveformjax.conversions.SimulationFunction`, whose
    ``reference`` is :func:`simulate_td` of ``binary``). It takes one binary
    or a batch, as :func:`polarizations_at`, and works under ``mx.grad`` and
    ``mx.compile``."""
    return _conditioning.simulate_td_function(binary, delta_t, f_min, _options(options), data)


def simulate_fd_function(binary, delta_f: float, f_min: float, f_max: float,
                         options: Optional[WaveformOptions] = None, data: Optional[NRSurData] = None):
    """:func:`simulate_fd` as a pure function of the binary (see
    :func:`simulate_td_function`)."""
    return _conditioning.simulate_fd_function(binary, delta_f, f_min, f_max, _options(options), data)
