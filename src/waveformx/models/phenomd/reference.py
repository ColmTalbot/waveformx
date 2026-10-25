# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Converters between the reference conventions of IMRPhenomD: only
``phi_ref`` changes, by half the change of the model's phase between the
two references (``-Psi + t0 M f``, :func:`.core.coefficients`), as for
TaylorF2 (:mod:`..taylorf2.reference`). In LAL's convention ``f_ref = 0``
is the first frequency: :func:`to_mass_scaled_reference` then takes it as
``f_min`` (NaN without)."""

from __future__ import annotations

from ..taylorf2.reference import converters
from . import core

__all__ = ["to_mass_scaled_reference", "to_physical_reference", "phase"]


def phase(m1, m2, chi1, chi2, mf):
    """IMRPhenomD's phase ``-Psi + t0 M f`` at ``M f = mf`` (but for its
    constant)."""
    total = m1 + m2
    return core.coefficients(m1 / total, m2 / total, chi1, chi2)["phase"](mf)


to_mass_scaled_reference, to_physical_reference = converters(phase, "f_min")
