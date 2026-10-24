# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXAS in MLX, for Apple GPUs, in float32.

* :func:`polarizations`: ``(h_+, h_x)`` at given frequencies, for one
  binary or a batch, differentiable with ``mx.grad`` and ``mx.vjp``;
* :func:`polarizations_and_jacobian`: with their Jacobian with respect to
  the twelve parameters of the binary.

The waveform is that of :mod:`..core`: the set-up of each binary is
generated from the JAX function (a Metal kernel of one thread per binary),
the waveform at the frequencies is a kernel written by hand (one thread per
binary and frequency), and the derivatives contract the forward-mode
derivatives of both with the cotangents in one more kernel
(:mod:`._kernels`).

It needs the optional dependency ``mlx`` (``pip install waveformx[mlx]``)
and JAX (to trace the set-up, once per process).

The functions take a :class:`~waveformx.Binary` of numbers or arrays
(converted to float32 MLX arrays): one binary, or a batch along a leading
axis of its fields. The output is the strain multiplied by ``1e21``
unless ``scaled_strain`` is false; NaN where LAL fails.
"""

from __future__ import annotations

__all__ = ["polarizations", "polarizations_and_jacobian"]


def polarizations(binary, frequencies, scaled_strain=True):
    """``(h_+, h_x)`` of IMRPhenomXAS (see :func:`._api.polarizations`)."""
    from ._api import polarizations as function

    return function(binary, frequencies, scaled_strain)


def polarizations_and_jacobian(binary, frequencies, scaled_strain=True):
    """``(h_+, h_x)`` and their Jacobians (see :func:`._api.polarizations_and_jacobian`)."""
    from ._api import polarizations_and_jacobian as function

    return function(binary, frequencies, scaled_strain)
