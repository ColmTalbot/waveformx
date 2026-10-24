# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The pure function of TaylorF2 in MLX: binaries in, the polarizations
out, with derivatives for ``mx.grad`` and ``mx.vjp`` (see
:mod:`._tables`)."""

from __future__ import annotations

from . import _tables
from ...._options import REFERENCE_MF
from .. import reference
from ._kernels import kernels

__all__ = ["polarizations", "polarizations_and_jacobian", "to_mass_scaled_reference", "to_physical_reference"]


def polarizations(binary, frequencies, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` (complex64 MLX arrays) of TaylorF2 at ``frequencies``
    (Hz, positive; ``(N,)`` or, for a batch, ``(B, N)``): ``(N,)`` each for
    one binary, ``(B, N)`` for a batch (a :class:`~waveformx.Binary` whose
    fields have a leading axis). Only the aligned components of the spins
    enter. The strain is multiplied by ``1e21`` unless ``scaled_strain`` is
    false. Differentiable with ``mx.grad`` with respect to the fields of
    the binary."""
    return _tables.polarizations(kernels(_tables.scale(scaled_strain), _tables.reference(reference_mf)), binary, frequencies)


def polarizations_and_jacobian(binary, frequencies, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` and their Jacobians with respect to the twelve
    parameters of the binary (``mass_1``, ``mass_2``, ``spin_1``,
    ``spin_2``, ``luminosity_distance``, ``inclination``, ``phi_ref``,
    ``f_ref``): each ``(..., N, 12)`` complex64 (zero for the in-plane
    spins). ``reference_mf`` as for :func:`polarizations`."""
    return _tables.polarizations_and_jacobian(kernels(_tables.scale(scaled_strain), _tables.reference(reference_mf)), binary, frequencies)


_TO_MASS_SCALED, _TO_PHYSICAL = _tables.converters(reference.to_mass_scaled_reference, reference.to_physical_reference)


def to_mass_scaled_reference(binary, reference_mf=REFERENCE_MF):
    """:func:`..reference.to_mass_scaled_reference` in MLX (float32, the
    fields of the binary MLX arrays or numbers, one binary or a batch):
    differentiable with ``mx.grad``."""
    return _TO_MASS_SCALED(binary, _tables.reference(reference_mf))


def to_physical_reference(binary, f_ref, reference_mf=REFERENCE_MF):
    """:func:`..reference.to_physical_reference` in MLX (see
    :func:`to_mass_scaled_reference`)."""
    return _TO_PHYSICAL(binary, f_ref, _tables.reference(reference_mf))
