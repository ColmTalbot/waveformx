# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The models, and the contract between them and the public functions.

Each model package defines ``MODEL``, an object with the methods below. The
public functions (``waveformjax._api``) parse and check the arguments, call
``validate`` on concrete values (outside any JAX transformation), and then
call the pure functions, which they compile with ``jax.jit`` (``options``
is static: a hashable :class:`~waveformjax.WaveformOptions`).

Frequency-domain models (``MODEL.domain == "frequency"``):

``validate(binary, frequencies, options) -> None``
    LALSimulation's checks for concrete values (numpy floats in a
    :class:`~waveformjax.Binary`, numpy frequencies). Raises the errors of
    :mod:`waveformjax.exceptions`, e.g. ``PrecessionVersionUnavailableError``
    where LAL would change the precession prescription and
    ``options.version_fallback`` is ``False``.

``polarizations(binary, frequencies, options) -> (hplus, hcross)``
    ``SimInspiralChooseFDWaveformSequence`` at ``frequencies`` (Hz, a 1-d
    float64 array). Pure: no Python control flow on traced values, so that
    ``jax.jit``, ``jax.vmap`` over the leaves of ``binary`` and ``jax.grad``
    with respect to them work. Failures the model can only detect while
    computing give non-finite values (NaN), which the batch functions report
    as ``ModelError``.

``modes(binary, frequencies, options) -> FrequencyModes``
    The modes in the conventions of ``SimInspiralChooseFDModes`` on the
    two-sided frequencies ``-f[::-1], f`` (see waveformrs's
    ``FrequencyDomainModel::modes``).

``choose_fd_modes(binary, delta_f, f_min, f_max, options) -> FrequencyModes``
    ``SimInspiralChooseFDModes`` on its uniform two-sided grid.

Time-domain models (``MODEL.domain == "time"``):

``validate(binary, delta_t, f_min, options) -> None``

``td_polarizations(binary, delta_t, f_min, options) -> TimeSeriesPolarizations``
``td_modes(binary, delta_t, f_min, options) -> TimeSeriesModes``
    ``SimInspiralChooseTDWaveform`` / ``SimInspiralChooseTDModes``. The
    length of the output depends on the binary, so these are not jitted as a
    whole; they call ``polarizations_at``.

``polarizations_at(binary, times, options) -> (hplus, hcross)``
    The polarizations at ``times`` (s, relative to the peak as in LAL's
    output), pure and differentiable as above.
"""

from __future__ import annotations

import importlib

from ..exceptions import UnknownApproximantError

__all__ = ["APPROXIMANTS", "get"]

#: approximant -> (domain, precessing, module)
APPROXIMANTS = {
    "IMRPhenomXPNR": ("frequency", True, "waveformjax.models.xpnr"),
    "NRSur7dq4": ("time", True, "waveformjax.models.nrsur7dq4"),
}


def get(approximant: str):
    """The ``MODEL`` of the approximant named ``approximant`` (LAL's name)."""
    try:
        module = APPROXIMANTS[approximant][2]
    except (KeyError, TypeError):
        raise UnknownApproximantError(
            f"unknown approximant {approximant!r}: waveformjax implements "
            f"{', '.join(APPROXIMANTS)}"
        ) from None
    return importlib.import_module(module).MODEL
