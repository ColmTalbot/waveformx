# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The polarizations of TaylorF2 in JAX
(``SimInspiralChooseFDWaveformSequence``), from the tables of :mod:`.core`.

Plain JAX (no loops), so that ``jax.jit``, ``jax.vmap`` and every
derivative apply directly; under ``jax.vmap`` over binaries with shared
frequencies the powers of the frequencies are computed once for the batch.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..._options import strain_scale
from ..xas.model import aligned_parameters
from ..xpnr._math import precision, precision_of, real
from . import core

__all__ = ["polarizations", "aligned_parameters"]


def polarizations(binary, frequencies, options):
    """``(h_+, h_x)`` of TaylorF2 at ``frequencies`` (Hz, positive), as
    ``SimInspiralChooseFDWaveformSequence``: every frequency is evaluated
    (no cut at the innermost stable circular orbit, as LAL's sequence). In
    float32 when the binary's parameters are float32; the strain times
    ``1e21`` unless ``options.scaled_strain`` is false."""
    with precision(precision_of(binary)):
        f = jnp.asarray(frequencies, real())
        table = core.table(aligned_parameters(binary), strain_scale(options))
        return core.evaluate(core.unpack(table), core.frequency_powers(f))
