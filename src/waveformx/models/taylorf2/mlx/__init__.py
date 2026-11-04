# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""TaylorF2 in MLX, for Apple GPUs, in float32.

* :func:`polarizations`: ``(h_+, h_x)`` at given frequencies, for one
  binary or a batch, differentiable with ``mx.grad`` and ``mx.vjp``;
* :func:`polarizations_and_jacobian`: with their Jacobian with respect to
  the twelve parameters of the binary.

The waveform is that of :mod:`..core`: the set-up of each binary is
generated from the JAX function (a Metal kernel of one thread per binary),
the waveform at the frequencies is a kernel written by hand (one thread per
binary and frequency), and the derivatives contract the forward-mode
derivatives of both with the cotangents in one more kernel
(:mod:`._tables`, shared with IMRPhenomD).

It needs the optional dependency ``mlx`` (``pip install waveformx[mlx]``)
and JAX (to trace the set-up, once per process).

The functions take a :class:`~waveformx.Binary` of numbers or arrays
(converted to float32 MLX arrays): one binary, or a batch along a leading
axis of its fields. The output is the strain multiplied by ``1e21``
unless ``scaled_strain`` is false.
"""

from __future__ import annotations

from ...._options import REFERENCE_MF

__all__ = ["polarizations", "polarizations_and_jacobian", "to_mass_scaled_reference", "to_physical_reference"]


def polarizations(binary, frequencies, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` (complex64 MLX arrays) of TaylorF2 at ``frequencies``
    (Hz, positive; ``(N,)`` or, for a batch, ``(B, N)``): ``(N,)`` each for
    one binary, ``(B, N)`` for a batch (a :class:`~waveformx.Binary` whose
    fields have a leading axis). Only the aligned components of the spins
    enter. The strain is multiplied by ``1e21`` unless ``scaled_strain`` is
    false. Differentiable with ``mx.grad`` with respect to the fields of
    the binary."""
    from ._api import polarizations as function

    return function(binary, frequencies, scaled_strain, reference_mf)


def polarizations_and_jacobian(binary, frequencies, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` and their Jacobians with respect to the twelve
    parameters of the binary (``mass_1``, ``mass_2``, ``spin_1``,
    ``spin_2``, ``luminosity_distance``, ``inclination``, ``phi_ref``,
    ``f_ref``): each ``(..., N, 12)`` complex64 (zero for the in-plane
    spins). ``reference_mf`` as for :func:`polarizations`."""
    from ._api import polarizations_and_jacobian as function

    return function(binary, frequencies, scaled_strain, reference_mf)


def to_mass_scaled_reference(binary, reference_mf=REFERENCE_MF):
    """:func:`waveformx.to_mass_scaled_reference` in MLX (float32, the
    fields of the binary MLX arrays or numbers, one binary or a batch):
    differentiable with ``mx.grad``."""
    from ._api import to_mass_scaled_reference as function

    return function(binary, reference_mf)


def to_physical_reference(binary, f_ref, reference_mf=REFERENCE_MF):
    """:func:`waveformx.to_physical_reference` in MLX (see
    :func:`to_mass_scaled_reference`)."""
    from ._api import to_physical_reference as function

    return function(binary, f_ref, reference_mf)
