# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The converters between LAL's reference convention (``Binary.f_ref`` in
Hz) and the mass-scaled one (``WaveformOptions.reference_mf``), for every
model: :func:`to_mass_scaled_reference` and :func:`to_physical_reference`
dispatch to the functions of the model's ``reference`` module
(:data:`REFERENCE_MODULES`), which can also be called directly.

What changes with the reference depends on the model:

* TaylorF2, IMRPhenomD, IMRPhenomXAS and IMRPhenomXHM (aligned spins):
  ``phi_ref`` alone, by half the change of the (2,2) phase between the two
  references; the waveforms are equal to rounding. For IMRPhenomD,
  IMRPhenomXAS and IMRPhenomXHM, ``f_ref = 0`` is the first frequency,
  which the converters take as ``f_min``.
* IMRPhenomXPNR: the spins, the inclination and ``phi_ref``, carried
  with the SpinTaylor equations (``steps``, ``return_angle``).
* NRSur7dq4: the same, carried with the surrogate's own dynamics
  (``f_min``, ``return_angle``, ``data``).

The converters are pure: ``jax.jit``, ``jax.vmap`` and ``jax.grad`` work
(``approximant`` is static). The MLX models have the same functions in
their ``mlx`` packages.
"""

from __future__ import annotations

import importlib

from ._options import REFERENCE_MF
from .exceptions import UnknownApproximantError

__all__ = ["REFERENCE_MODULES", "reference_module", "to_mass_scaled_reference", "to_physical_reference"]

#: approximant -> the module of its converters
REFERENCE_MODULES = {
    "TaylorF2": "waveformx.models.taylorf2.reference",
    "IMRPhenomD": "waveformx.models.phenomd.reference",
    "IMRPhenomXAS": "waveformx.models.xas.reference",
    "IMRPhenomXHM": "waveformx.models.xas.reference",
    "IMRPhenomXPNR": "waveformx.models.xpnr.reference",
    "NRSur7dq4": "waveformx.models.nrsur7dq4.reference",
}


def reference_module(approximant: str):
    """The module of the converters of ``approximant`` (LAL's name)."""
    try:
        return importlib.import_module(REFERENCE_MODULES[approximant])
    except (KeyError, TypeError):
        raise UnknownApproximantError(
            f"unknown approximant {approximant!r}: waveformx implements {', '.join(REFERENCE_MODULES)}"
        ) from None


def to_mass_scaled_reference(binary, reference_mf=REFERENCE_MF, *args, approximant="IMRPhenomXPNR", **keywords):
    """``binary`` given at its reference frequency ``binary.f_ref`` (Hz,
    LAL's convention) as a binary given at the mass-scaled reference
    ``reference_mf / M`` (``f_ref`` zero), for the model ``approximant``
    with ``WaveformOptions(reference_mf=reference_mf)``. The other
    arguments are those of the model's function (see the module): e.g.
    ``f_min`` where ``f_ref = 0`` means the first frequency, ``steps`` and
    ``return_angle`` for IMRPhenomXPNR. ``approximant`` defaults to
    IMRPhenomXPNR, whose converters these functions were first."""
    return reference_module(approximant).to_mass_scaled_reference(binary, reference_mf, *args, **keywords)


def to_physical_reference(binary, f_ref, reference_mf=REFERENCE_MF, *args, approximant="IMRPhenomXPNR", **keywords):
    """The inverse of :func:`to_mass_scaled_reference`: ``binary`` given at
    the mass-scaled reference ``reference_mf / M`` (its ``f_ref`` ignored)
    as a binary given at ``f_ref`` (Hz, LAL's convention), for the model
    ``approximant``."""
    return reference_module(approximant).to_physical_reference(binary, f_ref, reference_mf, *args, **keywords)
