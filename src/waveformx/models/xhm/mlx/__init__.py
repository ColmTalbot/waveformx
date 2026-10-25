# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXHM in MLX, for Apple GPUs, in float32.

* :func:`polarizations`: ``(h_+, h_x)`` at given frequencies, for one
  binary or a batch, differentiable with ``mx.grad`` and ``mx.vjp``;
* :func:`modes`: the modes ``h_{l,-m}``, differentiable likewise;
* :func:`polarizations_and_jacobian`: with their Jacobian with respect to
  the twelve parameters of the binary.

The waveform is that of :mod:`..core` (:mod:`._kernels`: the set-up
generated from the JAX functions, the waveform at the frequencies a kernel
calling functions generated from them). It needs the optional dependency
``mlx`` and JAX (to trace the set-up, once per process). The output is the
strain multiplied by ``1e21`` unless ``scaled_strain`` is false; NaN where
LAL fails.
"""

from __future__ import annotations

from ...._options import REFERENCE_MF

__all__ = ["polarizations", "modes", "polarizations_and_jacobian", "to_mass_scaled_reference", "to_physical_reference"]


def polarizations(binary, frequencies, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` of IMRPhenomXHM (see :func:`._api.polarizations`)."""
    from ._api import polarizations as function

    return function(binary, frequencies, scaled_strain, reference_mf)


def modes(binary, frequencies, scaled_strain=True, reference_mf=None):
    """The modes of IMRPhenomXHM (see :func:`._api.modes`)."""
    from ._api import modes as function

    return function(binary, frequencies, scaled_strain, reference_mf)


def polarizations_and_jacobian(binary, frequencies, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` and their Jacobians (see :func:`._api.polarizations_and_jacobian`)."""
    from ._api import polarizations_and_jacobian as function

    return function(binary, frequencies, scaled_strain, reference_mf)


def to_mass_scaled_reference(binary, reference_mf=REFERENCE_MF, f_min=None):
    """``binary`` at the mass-scaled reference (see :func:`...xas.mlx._api.to_mass_scaled_reference`)."""
    from ...xas.mlx._api import to_mass_scaled_reference as function

    return function(binary, reference_mf, f_min)


def to_physical_reference(binary, f_ref, reference_mf=REFERENCE_MF):
    """``binary`` at ``f_ref`` in Hz (see :func:`...xas.mlx._api.to_physical_reference`)."""
    from ...xas.mlx._api import to_physical_reference as function

    return function(binary, f_ref, reference_mf)
