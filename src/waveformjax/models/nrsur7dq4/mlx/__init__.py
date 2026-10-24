# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 in MLX, for Apple GPUs: the pure functions of
:mod:`waveformjax.models.nrsur7dq4` (``polarizations_at`` and ``modes_at``)
in float32, which MLX evaluates on the GPU by default.

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
``_kernel``        the integration of the dynamics in one Metal kernel,
                   with its Jacobian for the derivatives
``_frames``        coorbital -> coprecessing -> inertial frame
``_waveform``      the modes and polarizations of a batch of binaries
=================  ==========================================================

The functions take a :class:`~waveformjax.Binary` of numbers or arrays
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
"""

from __future__ import annotations

from typing import Optional

from ...._options import WaveformOptions
from .._waveform import strain_scale
from . import _waveform
from ._data import NRSurData, default_data, load

__all__ = ["NRSurData", "load", "default_data", "polarizations_at", "modes_at"]


def polarizations_at(binary, times, options: Optional[WaveformOptions] = None, data: Optional[NRSurData] = None):
    """``(h_+, h_x)`` at ``times`` (s, relative to the peak; zero outside
    the surrogate): ``(m,)`` each for one binary, ``(B, m)`` for a batch,
    whose ``times`` may be shared (``(m,)``) or its own (``(B, m)``).
    ``binary.f_ref = 0`` starts the dynamics at the start of the
    surrogate."""
    options = WaveformOptions() if options is None else options
    data = default_data() if data is None else data
    binary, has_batch = _waveform.batched(binary)
    hp, hc = _waveform.polarizations_at(data, binary, times, options.mode_array, strain_scale(options))
    return (hp, hc) if has_batch else (hp[0], hc[0])


def modes_at(binary, times, options: Optional[WaveformOptions] = None, data: Optional[NRSurData] = None):
    """The inertial modes ``{(l, m): h_lm}`` (complex64) at ``times``, as
    :func:`polarizations_at`."""
    options = WaveformOptions() if options is None else options
    data = default_data() if data is None else data
    binary, has_batch = _waveform.batched(binary)
    modes = _waveform.modes_at(data, binary, times, options.mode_array, strain_scale(options))
    return modes if has_batch else {mode: h[0] for mode, h in modes.items()}
