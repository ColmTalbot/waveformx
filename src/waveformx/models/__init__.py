# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The models, and the contract between them and the public functions.

Each model package defines ``MODEL``, an object with the methods below. The
public functions (``waveformx._api``) parse and check the arguments, call
``validate`` on concrete values (outside any JAX transformation), and then
call the pure functions, which they compile with ``jax.jit`` (``options``
is static: a hashable :class:`~waveformx.WaveformOptions`).

Frequency-domain models (``MODEL.domain == "frequency"``):

``validate(binary, frequencies, options) -> None``
    The model's checks for concrete values (numpy floats in a
    :class:`~waveformx.Binary`, numpy frequencies). Raises the errors of
    :mod:`waveformx.exceptions`, e.g. ``ModelError`` where the SpinTaylor
    angles of IMRPhenomXPNR cannot be computed.

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
    output), pure and differentiable as above. NRSur7dq4 evaluates them in
    the precision of the binary's parameters (float32 or float64).
"""

from __future__ import annotations

import importlib

from ..exceptions import UnknownApproximantError

__all__ = ["APPROXIMANTS", "get", "check_reference_mf"]

#: approximant -> (domain, precessing, module)
APPROXIMANTS = {
    "TaylorF2": ("frequency", False, "waveformx.models.taylorf2"),
    "IMRPhenomD": ("frequency", False, "waveformx.models.phenomd"),
    "IMRPhenomXAS": ("frequency", False, "waveformx.models.xas"),
    "IMRPhenomXHM": ("frequency", False, "waveformx.models.xhm"),
    "IMRPhenomXPNR": ("frequency", True, "waveformx.models.xpnr"),
    "NRSur7dq4": ("time", True, "waveformx.models.nrsur7dq4"),
}


def get(approximant: str):
    """The ``MODEL`` of the approximant named ``approximant`` (LAL's name)."""
    try:
        module = APPROXIMANTS[approximant][2]
    except (KeyError, TypeError):
        raise UnknownApproximantError(
            f"unknown approximant {approximant!r}: waveformx implements "
            f"{', '.join(APPROXIMANTS)}"
        ) from None
    return importlib.import_module(module).MODEL


def check_reference_mf(binary, options, approximant, supported=True):
    """The checks of the mass-scaled reference (``options.reference_mf``)
    for concrete values: refused where not ``supported``, and
    ``binary.f_ref`` must be zero with it."""
    if options.reference_mf is None:
        return
    from ..exceptions import InvalidParameterError

    if not supported:
        raise InvalidParameterError(f"{approximant} does not take reference_mf")
    if float(binary.f_ref) != 0.0:
        raise InvalidParameterError(
            "with reference_mf, the reference frequency is reference_mf / M: f_ref must be zero "
            "(to_mass_scaled_reference converts a binary given at f_ref)")
