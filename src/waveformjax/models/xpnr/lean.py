# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The lean IMRPhenomXPNR (``WaveformOptions(lean=True)``).

The extended model (``extended_spintaylor``) with the direct SpinTaylor
angles and the parity fix of lalsuite !2593, without the layers that only
reproduce LAL's numerics:

* the SpinTaylor solution is one DOP853 integration in both directions
  from the reference frequency (:mod:`.spintaylor.lean`), rather than LAL's
  adaptive Runge-Kutta-Fehlberg steps repeated one by one, LAL's sampling
  and its finer re-run of the end of the inspiral; the end of the inspiral
  is the extended model's (continuous) one, and LAL's second integration
  from the start of the grid of the PNR interpolants is not needed;
* masses are in solar masses (no round trips through kg), the mixing
  coefficients of the higher modes are not rounded to single precision,
  and nothing keeps XLA from fusing multiply-adds (:func:`._math.lean`);
* no prescriptions other than the SpinTaylor angles are compiled, and the
  higher modes take exact derivatives where LAL takes finite differences
  (as the extended model).

LAL's physics is kept: the fits, the ansatzes, the PNR construction of the
angles and the calibrated discrete choices of the higher-mode amplitudes.
It agrees with the extended model to the integration error of the
precession equations (mismatches ~1e-11 and below).

The rest of the model is shared with :mod:`.model`, where the static flag
``lean`` selects these paths (the default model's compiled graph does not
change). :func:`polarizations` is the pure function; the public functions
take ``lean=True``.
"""

from __future__ import annotations

from ..._options import WaveformOptions
from . import model

__all__ = ["polarizations", "options"]


def options(mode_array=None, scaled_strain=True, reference_mf=None):
    """The :class:`~waveformjax.WaveformOptions` of the lean model."""
    return WaveformOptions(lean=True, mode_array=mode_array, scaled_strain=scaled_strain, reference_mf=reference_mf)


def polarizations(binary, frequencies, mode_array=None, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` of the lean IMRPhenomXPNR at ``frequencies`` (Hz,
    positive and increasing): pure, for ``jax.jit``, ``jax.vmap``,
    ``jax.grad``, ``jax.jvp`` and ``jax.jacfwd`` with respect to the binary
    (forward-mode Jacobians, see :func:`.model.polarizations`); float32 for
    a binary of float32 parameters. NaN where the model fails. With
    ``reference_mf`` (``WaveformOptions.reference_mf``), the binary is given
    at the mass-scaled reference frequency ``reference_mf / M`` (and
    ``binary.f_ref`` must be zero)."""
    return model.polarizations(binary, frequencies, options(mode_array, scaled_strain, reference_mf))
