# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The polarizations of IMRPhenomXAS in JAX
(``SimInspiralChooseFDWaveformSequence``), from the tables of :mod:`.core`.

The function is plain JAX (no loops), so ``jax.jit``, ``jax.vmap`` and
every derivative (``jax.grad``, ``jax.jvp``, ``jax.jacfwd``) apply
directly. Under ``jax.vmap`` over binaries with shared frequencies, the
powers of the frequencies are computed once for the batch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..._options import strain_scale
from ..xpnr._math import precision, precision_of, real
from . import core

__all__ = ["polarizations", "aligned_parameters"]


def aligned_parameters(binary):
    """The parameters of :func:`.core.table` of a binary: the masses, the
    aligned spins, the distance, the inclination, ``phi_ref`` and
    ``f_ref`` (the in-plane spins do not enter)."""
    b = binary
    values = (b.mass_1, b.mass_2, jnp.asarray(b.spin_1)[..., 2], jnp.asarray(b.spin_2)[..., 2],
              b.luminosity_distance, b.inclination, b.phi_ref, b.f_ref)
    return tuple(jnp.asarray(v, real()) for v in values)


def polarizations(binary, frequencies, options):
    """``(h_+, h_x)`` of IMRPhenomXAS at ``frequencies`` (Hz, positive and
    increasing), as ``SimInspiralChooseFDWaveformSequence``: zero above
    ``0.3 / M`` (``0.33 / M`` for ``chi_eff > 0.99``), NaN where LAL fails.
    In float32 when the binary's parameters are float32; the strain times
    ``1e21`` unless ``options.scaled_strain`` is false."""
    with precision(precision_of(binary)):
        f = jnp.asarray(frequencies, real())
        table = core.table(aligned_parameters(binary), jax.lax.stop_gradient(f[0]), strain_scale(options))
        return core.evaluate(core.unpack(table), core.frequency_powers(f))
