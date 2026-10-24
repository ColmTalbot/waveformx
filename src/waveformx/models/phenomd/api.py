# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``MODEL`` of IMRPhenomD: the contract of :mod:`waveformx.models`.

* :meth:`PhenomDModel.polarizations` is pure (``jax.jit`` with ``options``
  static, ``jax.vmap``, ``jax.grad``, ``jax.jvp``);
* :meth:`PhenomDModel.validate` applies LALSimulation's checks to concrete
  values: no in-plane spins (a non-precessing approximant) and a final
  spin of at most one.

The modes (``SimInspiralChooseFDModes``) are not implemented.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import InvalidParameterError, UnsupportedError
from ..taylorf2.api import check_aligned
from . import fits, model

__all__ = ["MODEL", "PhenomDModel"]


class PhenomDModel:
    """IMRPhenomD (``SimInspiralChooseFDWaveformSequence``)."""

    domain = "frequency"
    precessing = False

    def validate(self, binary, frequencies, options):
        """LALSimulation's checks of IMRPhenomD for concrete values."""
        check_aligned(binary, "IMRPhenomD")
        m1, m2 = float(binary.mass_1), float(binary.mass_2)
        chi1, chi2 = float(np.asarray(binary.spin_1)[2]), float(np.asarray(binary.spin_2)[2])
        x1, x2 = m1 / (m1 + m2), m2 / (m1 + m2)
        if float(fits.final_spin(x1 * x2, x1 - x2, chi1, chi2)) > 1.0:
            raise InvalidParameterError("the final spin of IMRPhenomD exceeds one")

    def polarizations(self, binary, frequencies, options):
        """``SimInspiralChooseFDWaveformSequence`` (see :func:`.model.polarizations`)."""
        return model.polarizations(binary, frequencies, options)

    def modes(self, binary, frequencies, options):
        raise UnsupportedError("the modes of IMRPhenomD are not implemented in waveformx")

    def choose_fd_modes(self, binary, delta_f, f_min, f_max, options):
        raise UnsupportedError("the modes of IMRPhenomD are not implemented in waveformx")


MODEL = PhenomDModel()
