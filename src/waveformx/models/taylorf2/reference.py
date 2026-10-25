# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Converters between the reference conventions of TaylorF2 (and, through
:func:`converters`, of IMRPhenomD).

In LAL's convention a binary's ``phi_ref`` is given at the reference
frequency ``Binary.f_ref`` (Hz); in the mass-scaled one
(``WaveformOptions.reference_mf``) at ``M f_ref = reference_mf``. For these
aligned-spin models of the (2,2) mode the reference only fixes the
constant of the phase: ``phi(f) = Phi(f) - Phi(f_ref) + 2 phi_ref`` (plus
``pi / 4`` for TaylorF2), with ``Phi`` the model's phase (LAL's ``-Psi``
and, for IMRPhenomD, its time shift ``t0 M f``). A binary is converted
from ``f_1`` to ``f_2`` by

    phi_ref' = phi_ref + (Phi(f_2) - Phi(f_1)) / 2,

reduced modulo ``pi`` (which leaves the (2,2) mode unchanged), so that both
descriptions give the same waveform to rounding. ``Phi`` is evaluated in
float64 whatever the precision of the binary: it is ``~10^4`` radians for
binary neutron stars.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..._options import REFERENCE_MF
from ..xas.core import reduce
from ..xpnr._math import MTSUN_SI, precision, precision_of
from . import core, pn

__all__ = ["to_mass_scaled_reference", "to_physical_reference", "converters", "phase"]


def phase(m1, m2, chi1, chi2, mf):
    """TaylorF2's ``Phi`` (``-Psi``) at ``M f = mf``."""
    total = m1 + m2
    v, logs = pn.phasing(m1 / total, m2 / total, chi1, chi2)
    return core.phase_mf(v, logs, mf)


def converters(phase_of, zero_is):
    """``(to_mass_scaled_reference, to_physical_reference)`` of a model
    whose phase is ``phase_of(m1, m2, chi1, chi2, mf)`` and for which
    ``f_ref = 0`` means ``zero_is``: ``"coalescence"`` (no reference phase)
    or ``"f_min"`` (the first frequency, which the converters then take as
    an argument)."""

    def phase_at(binary, f, f_min):
        m1, m2 = binary.mass_1, binary.mass_2
        chi1, chi2 = jnp.asarray(binary.spin_1)[..., 2], jnp.asarray(binary.spin_2)[..., 2]
        f = jnp.asarray(f, jnp.float64)
        if zero_is == "coalescence":
            given = f > 0.0
            safe = jnp.where(given, f, 1.0)
        else:
            f = jnp.where(f == 0.0, jnp.nan if f_min is None else jnp.asarray(f_min, jnp.float64), f)
            given, safe = True, jnp.where(f > 0.0, f, jnp.nan)
        value = phase_of(m1, m2, chi1, chi2, safe * (m1 + m2) * MTSUN_SI)
        return jnp.where(given, value, 0.0)

    def scaled(binary, reference_mf):
        m1, m2 = binary.mass_1, binary.mass_2
        chi1, chi2 = jnp.asarray(binary.spin_1)[..., 2], jnp.asarray(binary.spin_2)[..., 2]
        return phase_of(m1, m2, chi1, chi2, reference_mf)

    def shifted(binary, phi_from, phi_to, f_ref):
        dtype = precision_of(binary)
        phi = jnp.asarray(binary.phi_ref, jnp.float64) + 0.5 * reduce(phi_to - phi_from)
        return binary._replace(phi_ref=phi.astype(dtype), f_ref=jnp.asarray(f_ref, dtype))

    def as_float64(binary):
        return type(binary)(*(jnp.asarray(value, jnp.float64) for value in binary))

    def to_mass_scaled_reference(binary, reference_mf=REFERENCE_MF, f_min=None):
        """``binary`` given at its reference frequency ``binary.f_ref`` (Hz,
        LAL's convention) as a binary given at the mass-scaled reference
        ``reference_mf / M`` (``f_ref`` zero), for
        ``WaveformOptions(reference_mf=reference_mf)``: only ``phi_ref``
        changes. Pure: ``jax.jit``, ``jax.vmap`` and ``jax.grad`` work; in
        the precision of the binary, the phase computed in float64."""
        with precision(jnp.float64):
            b = as_float64(binary)
            out = shifted(binary, phase_at(b, b.f_ref, f_min), scaled(b, reference_mf), 0.0)
        return out

    def to_physical_reference(binary, f_ref, reference_mf=REFERENCE_MF):
        """The inverse of :func:`to_mass_scaled_reference`: ``binary`` given
        at the mass-scaled reference ``reference_mf / M`` (its ``f_ref``
        ignored) as a binary given at ``f_ref`` (Hz, LAL's convention)."""
        with precision(jnp.float64):
            b = as_float64(binary)
            out = shifted(binary, scaled(b, reference_mf), phase_at(b, f_ref, None), f_ref)
        return out

    return to_mass_scaled_reference, to_physical_reference


to_mass_scaled_reference, to_physical_reference = converters(phase, "coalescence")
to_mass_scaled_reference.__doc__ += """

    For TaylorF2 ``f_ref = 0`` is the reference at coalescence (no
    reference phase); ``f_min`` is not used."""
to_physical_reference.__doc__ += """ ``f_ref = 0`` is the reference at
    coalescence."""
