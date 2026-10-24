# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``MODEL`` of IMRPhenomXAS: the contract of :mod:`waveformx.models`.

* :meth:`XASModel.polarizations` is pure (``jax.jit`` with ``options``
  static, ``jax.vmap``, ``jax.grad``, ``jax.jvp``);
* :meth:`XASModel.validate` applies LALSimulation's checks to concrete
  values: no in-plane spins (a non-precessing approximant), and the checks
  of ``IMRPhenomXSetWaveformVariables``.

The modes (``SimInspiralChooseFDModes``) are not implemented.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ...exceptions import InvalidParameterError, UnsupportedError
from ..xpnr._math import lean
from ..xpnr.internals import waveform_struct
from . import model

__all__ = ["MODEL", "XASModel", "check"]


@jax.jit
def _flags(binary, f_min):
    """LAL's errors of the waveform variables."""
    with lean(True):
        params = model.aligned_parameters(binary)
        wf = waveform_struct(*params[:4], 0.0, params[7], params[6], f_min, 0.0, params[4], params[5])
        return dict(mass_ratio=wf.error_mass_ratio, final_mass=wf.error_final_mass,
                    final_spin=wf.error_final_spin, meco=wf.error_meco)


def check(binary, frequencies, approximant):
    """LALSimulation's checks of the aligned-spin IMRPhenomX models for
    concrete values (``approximant`` names the model in the errors)."""
    frequencies = np.asarray(frequencies, dtype=float)
    for name in ("spin_1", "spin_2"):
        spin = np.asarray(getattr(binary, name), dtype=float)
        if np.any(spin[:2] != 0.0):
            raise InvalidParameterError(
                f"non-zero in-plane spins were given, but {approximant} is a non-precessing approximant")
    if frequencies.size == 0:
        return
    binary = type(binary)(*(jnp.asarray(value, jnp.float64) for value in binary))
    flags = {k: bool(v) for k, v in _flags(binary, jnp.float64(frequencies[0])).items()}
    if flags["mass_ratio"]:
        raise InvalidParameterError(f"{approximant} is not valid at mass ratios beyond 1000")
    if flags["final_mass"]:
        raise InvalidParameterError("the fitted final mass is not physical")
    if flags["final_spin"]:
        raise InvalidParameterError("the fitted final spin exceeds one")
    if flags["meco"]:
        raise InvalidParameterError("f_MECO cannot be greater than f_ISCO")


class XASModel:
    """IMRPhenomXAS (``SimInspiralChooseFDWaveformSequence``)."""

    domain = "frequency"
    precessing = False

    def validate(self, binary, frequencies, options):
        """LALSimulation's checks of IMRPhenomXAS for concrete values."""
        check(binary, frequencies, "IMRPhenomXAS")

    def polarizations(self, binary, frequencies, options):
        """``SimInspiralChooseFDWaveformSequence`` (see :func:`.model.polarizations`)."""
        return model.polarizations(binary, frequencies, options)

    def modes(self, binary, frequencies, options):
        raise UnsupportedError("the modes of IMRPhenomXAS are not implemented in waveformx")

    def choose_fd_modes(self, binary, delta_f, f_min, f_max, options):
        raise UnsupportedError("the modes of IMRPhenomXAS are not implemented in waveformx")


MODEL = XASModel()
