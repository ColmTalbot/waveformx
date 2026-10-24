# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The modes and polarizations of IMRPhenomXHM in JAX, from the tables of
:mod:`.core`.

The functions are plain JAX (no loops), so ``jax.jit``, ``jax.vmap`` and
every derivative apply directly. Under ``jax.vmap`` over binaries with
shared frequencies, the powers of the frequencies are computed once for the
batch. In float32 when the binary's parameters are float32; the strain
times ``1e21`` unless ``options.scaled_strain`` is false.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..._options import strain_scale
from ..xas import core as xas_core
from ..xas.model import aligned_parameters
from ..xpnr._math import precision, precision_of, real
from . import core

__all__ = ["modes", "polarizations", "modes_and_polarizations"]


def modes_and_polarizations(binary, frequencies, options):
    """The modes ``{(l, m): h_{l,-m}(f)}`` (``m > 0``: the modes of
    negative ``m`` at the positive frequencies, :data:`.core.MODES`) and
    ``(h_+, h_x)`` at ``frequencies`` (Hz, positive and increasing), as
    ``SimInspiralChooseFDWaveformSequence`` sums them: the (2,2) zero above
    ``0.3 / M`` (``0.33 / M`` for ``chi_eff > 0.99``), the higher modes at
    every frequency, the ringdown of the (3,2) with the (2,2) recycled; NaN
    where LAL fails."""
    with precision(precision_of(binary)):
        f = jnp.asarray(frequencies, real())
        table = core.table(aligned_parameters(binary), jax.lax.stop_gradient(f[0]), strain_scale(options))
        return core.modes_and_polarizations(core.unpack(table), xas_core.frequency_powers(f))


def polarizations(binary, frequencies, options):
    """``(h_+, h_x)`` (see :func:`modes_and_polarizations`)."""
    return modes_and_polarizations(binary, frequencies, options)[1]


def modes(binary, frequencies, options):
    """The modes ``{(l, m): h_{l,-m}(f)}`` (see
    :func:`modes_and_polarizations`); ``h_{l,m}(-f) = (-1)^l
    conj(h_{l,-m}(f))``."""
    return modes_and_polarizations(binary, frequencies, options)[0]
