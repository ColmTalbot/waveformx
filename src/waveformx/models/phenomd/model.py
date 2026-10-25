# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The polarizations of IMRPhenomD in JAX
(``SimInspiralChooseFDWaveformSequence``), from the tables of :mod:`.core`.

The table of each binary is computed in float64 whatever the precision:
in float32 the connection of the inspiral phase to the intermediate one
(its derivative at ``Mf = 0.018``, a sum of terms of ``~10^4`` that cancel
to ``~10^2``) would shift the merger in time by up to tens of
microseconds for heavy binaries. The waveform at the frequencies is in
the binary's precision.

Plain JAX (no loops), so that ``jax.jit``, ``jax.vmap`` and every
derivative apply directly; under ``jax.vmap`` over binaries with shared
frequencies the powers of the frequencies are computed once for the batch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..._options import strain_scale
from ..xas.model import aligned_parameters
from ..xpnr._math import precision, precision_of, real
from . import core

__all__ = ["polarizations", "aligned_parameters"]


def polarizations(binary, frequencies, options):
    """``(h_+, h_x)`` of IMRPhenomD at ``frequencies`` (Hz, positive and
    increasing), as ``SimInspiralChooseFDWaveformSequence``: every
    frequency is evaluated (LAL's sequence has no cutoff), NaN where LAL
    fails. In float32 when the binary's parameters are float32, but for the
    table of each binary, which is computed in float64 (see
    :func:`.core.narrow`); the strain times ``1e21`` unless
    ``options.scaled_strain`` is false. With ``options.reference_mf`` the
    binary is given at ``M f_ref = reference_mf`` (its ``f_ref`` zero; NaN
    otherwise)."""
    dtype = precision_of(binary)
    with precision(np.float64):
        params = tuple(jnp.asarray(p, jnp.float64) for p in aligned_parameters(binary))
        f_min = jax.lax.stop_gradient(jnp.asarray(frequencies, jnp.float64)[0])
        table = core.table(params, f_min, strain_scale(options), options.reference_mf)
    with precision(dtype):
        f = jnp.asarray(frequencies, real())
        return core.evaluate(core.unpack(core.narrow(table, dtype)), core.frequency_powers(f))
