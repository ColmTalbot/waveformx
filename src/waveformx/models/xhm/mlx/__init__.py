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
    """``(h_+, h_x)`` (complex64 MLX arrays) of IMRPhenomXHM at
    ``frequencies`` (Hz, positive and increasing; ``(N,)`` or, for a batch,
    ``(B, N)``): ``(N,)`` each for one binary, ``(B, N)`` for a batch (a
    :class:`~waveformx.Binary` whose fields have a leading axis). Only the
    aligned components of the spins enter. The strain is multiplied by
    ``1e21`` unless ``scaled_strain`` is false; NaN where LAL fails.
    Differentiable with ``mx.grad`` with respect to the fields of the
    binary. With ``reference_mf`` the binary is given at ``M f_ref =
    reference_mf`` (``WaveformOptions.reference_mf``; its ``f_ref`` zero,
    NaN otherwise)."""
    from ._api import polarizations as function

    return function(binary, frequencies, scaled_strain, reference_mf)


def modes(binary, frequencies, scaled_strain=True, reference_mf=None):
    """The modes ``{(l, m): h_{l,-m}(f)}`` (``m > 0``: the modes of
    negative ``m`` at the positive frequencies; ``h_{l,m}(-f) = (-1)^l
    conj(h_{l,-m}(f))``), those whose sum with the spherical harmonics is
    :func:`polarizations` (the (3,2) recycling the (2,2) in its ringdown;
    the higher modes at every frequency). Differentiable with ``mx.grad``."""
    from ._api import modes as function

    return function(binary, frequencies, scaled_strain, reference_mf)


def polarizations_and_jacobian(binary, frequencies, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` and their Jacobians with respect to the twelve
    parameters of the binary: each ``(..., N, 12)`` complex64 (zero for the
    in-plane spins)."""
    from ._api import polarizations_and_jacobian as function

    return function(binary, frequencies, scaled_strain, reference_mf)


def to_mass_scaled_reference(binary, reference_mf=REFERENCE_MF, f_min=None):
    """:func:`waveformx.to_mass_scaled_reference` in MLX (float32, the
    fields of the binary MLX arrays or numbers, one binary or a batch):
    differentiable with ``mx.grad``. For IMRPhenomXHM too."""
    from ...xas.mlx._api import to_mass_scaled_reference as function

    return function(binary, reference_mf, f_min)


def to_physical_reference(binary, f_ref, reference_mf=REFERENCE_MF):
    """:func:`waveformx.to_physical_reference` in MLX (see
    :func:`to_mass_scaled_reference`)."""
    from ...xas.mlx._api import to_physical_reference as function

    return function(binary, f_ref, reference_mf)
