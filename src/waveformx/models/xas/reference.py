# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Converters between the reference conventions of IMRPhenomXAS and
IMRPhenomXHM: only ``phi_ref`` changes, by half the change of the phase of
the (2,2) mode between the two references (LAL's ``phase_22 / eta + linb M
f``, whose value at the reference ``phifRef`` subtracts), as for TaylorF2
(:mod:`..taylorf2.reference`). IMRPhenomXHM aligns its higher modes with
the (2,2) at the reference, so the mode ``(l, m)`` turns by ``m / 2`` times
that change: ``phi_ref`` is kept modulo ``2 pi`` (not ``pi``, which would
flip the sign of the modes of odd ``m``), and the same binary then gives the
same waveform in both models' conventions, and ``phi_ref`` agrees with
IMRPhenomXPNR's converters for aligned spins. In LAL's convention ``f_ref
= 0`` is the first frequency: :func:`to_mass_scaled_reference` then takes
it as ``f_min`` (NaN without)."""

from __future__ import annotations

import numpy as np

from ..taylorf2.reference import converters
from ..xpnr._math import powers
from ..xpnr.phase22 import phase_22, phase_coefficients, time_shift_22
from . import core

__all__ = ["to_mass_scaled_reference", "to_physical_reference", "phase"]


def phase(m1, m2, chi1, chi2, mf):
    """The phase of IMRPhenomXAS's (2,2) mode at ``M f = mf`` (but for its
    constant), with the waveform variables of :func:`.core.table`."""
    wf = core.waveform_variables((m1, m2, chi1, chi2, 1.0, 0.0, 0.0, 1.0), 1.0, 1.0)
    coefficients = phase_coefficients(wf)
    linb = time_shift_22(coefficients, wf)
    return phase_22(mf, powers(mf), coefficients) / wf.eta + linb * mf


to_mass_scaled_reference, to_physical_reference = converters(phase, "f_min", 4.0 * np.pi)
