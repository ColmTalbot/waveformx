# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``MODEL`` of TaylorF2: the contract of :mod:`waveformx.models`.

* :meth:`TaylorF2Model.polarizations` is pure (``jax.jit`` with
  ``options`` static, ``jax.vmap``, ``jax.grad``, ``jax.jvp``);
* :meth:`TaylorF2Model.validate` applies LALSimulation's checks to
  concrete values: no in-plane spins (a non-precessing approximant).

The modes (``SimInspiralChooseFDModes``) are not implemented.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import InvalidParameterError, UnsupportedError
from .. import check_reference_mf
from . import model

__all__ = ["MODEL", "TaylorF2Model", "check_aligned"]


def check_aligned(binary, approximant):
    """LAL's error for in-plane spins of a non-precessing approximant."""
    for name in ("spin_1", "spin_2"):
        spin = np.asarray(getattr(binary, name), dtype=float)
        if np.any(spin[:2] != 0.0):
            raise InvalidParameterError(
                f"non-zero in-plane spins were given, but {approximant} is a non-precessing approximant")


class TaylorF2Model:
    """TaylorF2 (``SimInspiralChooseFDWaveformSequence``)."""

    domain = "frequency"
    precessing = False

    def validate(self, binary, frequencies, options):
        """LALSimulation's checks of TaylorF2 for concrete values, and that
        ``f_ref`` is zero with ``options.reference_mf``."""
        check_aligned(binary, "TaylorF2")
        check_reference_mf(binary, options, "TaylorF2")

    def polarizations(self, binary, frequencies, options):
        """``SimInspiralChooseFDWaveformSequence`` (see :func:`.model.polarizations`)."""
        return model.polarizations(binary, frequencies, options)

    def modes(self, binary, frequencies, options):
        raise UnsupportedError("the modes of TaylorF2 are not implemented in waveformx")

    def choose_fd_modes(self, binary, delta_f, f_min, f_max, options):
        raise UnsupportedError("the modes of TaylorF2 are not implemented in waveformx")


MODEL = TaylorF2Model()
