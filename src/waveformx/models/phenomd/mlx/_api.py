# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The pure function of IMRPhenomD in MLX: binaries in, the polarizations
out, with derivatives for ``mx.grad`` and ``mx.vjp`` (see
:mod:`...taylorf2.mlx._tables`)."""

from __future__ import annotations

from ...taylorf2.mlx import _tables
from ._kernels import kernels

__all__ = ["polarizations", "polarizations_and_jacobian"]


def polarizations(binary, frequencies, scaled_strain=True):
    """``(h_+, h_x)`` (complex64 MLX arrays) of IMRPhenomD at
    ``frequencies`` (Hz, positive and increasing; ``(N,)`` or, for a batch,
    ``(B, N)``): ``(N,)`` each for one binary, ``(B, N)`` for a batch (a
    :class:`~waveformx.Binary` whose fields have a leading axis). Only the
    aligned components of the spins enter. The strain is multiplied by
    ``1e21`` unless ``scaled_strain`` is false; NaN where LAL fails.
    Differentiable with ``mx.grad`` with respect to the fields of the
    binary."""
    return _tables.polarizations(kernels(_tables.scale(scaled_strain)), binary, frequencies)


def polarizations_and_jacobian(binary, frequencies, scaled_strain=True):
    """``(h_+, h_x)`` and their Jacobians with respect to the twelve
    parameters of the binary (``mass_1``, ``mass_2``, ``spin_1``,
    ``spin_2``, ``luminosity_distance``, ``inclination``, ``phi_ref``,
    ``f_ref``): each ``(..., N, 12)`` complex64 (zero for the in-plane
    spins)."""
    return _tables.polarizations_and_jacobian(kernels(_tables.scale(scaled_strain)), binary, frequencies)
