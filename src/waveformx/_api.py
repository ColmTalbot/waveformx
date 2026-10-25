# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The public functions, with the signatures of waveformrs's Python
bindings."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from . import conversions, models
from ._binary import (
    BINARY_NAMES,
    Binary,
    binaries_from_table,
    binary_from_keywords,
    check_binary,
    parameter_table,
    row,
)
from ._options import WaveformOptions, parse_options
from ._results import (
    BatchPolarizations,
    BatchTimeSeries,
    FrequencyGrid,
    FrequencyModes,
    FrequencySeriesPolarizations,
    TimeSeriesModes,
    TimeSeriesPolarizations,
    check_frequencies,
)
from .exceptions import InvalidParameterError, ModelError, UnsupportedError, WaveformError

__all__ = [
    "approximants",
    "is_precessing",
    "is_frequency_domain",
    "nrsur7dq4_data_file",
    "polarizations",
    "polarizations_on_grid",
    "modes",
    "choose_fd_modes",
    "td_polarizations",
    "td_modes",
    "simulate_td",
    "simulate_fd",
    "simulate_td_function",
    "simulate_fd_function",
    "batch_polarizations",
    "batch_td_polarizations",
    "polarizations_function",
    "td_polarizations_function",
]

NRSUR7DQ4_FILE = "NRSur7dq4_v1.0.h5"


def approximants(domain: Optional[str] = None) -> list:
    """LAL's names of the approximants (of one domain, ``"frequency"`` or
    ``"time"``)."""
    if domain not in (None, "frequency", "time"):
        raise ValueError('domain must be None, "frequency" or "time"')
    return [name for name, (d, _, _) in models.APPROXIMANTS.items() if domain in (None, d)]


def _entry(approximant: str):
    if not isinstance(approximant, str) or approximant not in models.APPROXIMANTS:
        models.get(approximant)  # raises UnknownApproximantError
    return models.APPROXIMANTS[approximant]


def is_precessing(approximant: str) -> bool:
    return _entry(approximant)[1]


def is_frequency_domain(approximant: str) -> bool:
    return _entry(approximant)[0] == "frequency"


def nrsur7dq4_data_file() -> Optional[str]:
    """The NRSur7dq4 data file from ``NRSUR7DQ4_DATA`` (the file) or
    ``LAL_DATA_PATH`` (its directory), or ``None``."""
    path = os.environ.get("NRSUR7DQ4_DATA")
    if path and Path(path).is_file():
        return str(path)
    for directory in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep):
        if directory and (Path(directory) / NRSUR7DQ4_FILE).is_file():
            return str(Path(directory) / NRSUR7DQ4_FILE)
    return None


def _split(keywords: dict):
    """The binary keywords and the option keywords."""
    binary = {k: v for k, v in keywords.items() if k in BINARY_NAMES}
    options = {k: v for k, v in keywords.items() if k not in BINARY_NAMES}
    return binary, options


def _request(options: Optional[WaveformOptions], keywords: dict):
    binary_keywords, option_keywords = _split(keywords)
    binary = binary_from_keywords(binary_keywords)
    options = parse_options(options, option_keywords)
    check_binary(binary)
    return binary, options


def _model(approximant: str, domain: str):
    model = models.get(approximant)
    if model.domain != domain:
        what = "frequency-domain" if domain == "frequency" else "time-domain"
        raise UnsupportedError(f"{approximant} has no {what} waveforms")
    return model


@functools.lru_cache(maxsize=None)
def _compiled(approximant: str, options: WaveformOptions, method: str, batched: bool):
    """``jax.jit`` of a model method with the options fixed (and
    ``jax.vmap`` over the binaries if ``batched``), cached per options."""
    function = functools.partial(getattr(models.get(approximant), method), options=options)
    if batched:
        function = jax.vmap(function, in_axes=(0, None))
    return jax.jit(function)


def polarizations_function(approximant: str, options: Optional[WaveformOptions] = None, **option_keywords) -> Callable:
    """The pure function ``f(binary, frequencies) -> (hplus, hcross)`` of a
    frequency-domain approximant, for use with ``jax.jit``, ``jax.vmap`` and
    ``jax.grad``. ``binary`` is a :class:`~waveformx.Binary` of float64
    arrays; no parameter checks are made.

    >>> f = polarizations_function("IMRPhenomXPNR")
    >>> loss = lambda m1: jnp.sum(jnp.abs(f(binary._replace(mass_1=m1), freqs)[0]) ** 2)
    >>> jax.grad(loss)(36.0)
    """
    model = _model(approximant, "frequency")
    return functools.partial(model.polarizations, options=parse_options(options, option_keywords))


def td_polarizations_function(approximant: str, options: Optional[WaveformOptions] = None, **option_keywords) -> Callable:
    """The pure function ``f(binary, times) -> (hplus, hcross)`` of a
    time-domain approximant at ``times`` (s, relative to the peak).
    NRSur7dq4 evaluates it in the precision of ``binary`` (float32 if its
    parameters are float32, else float64) and gives the strain times 1e21
    unless ``scaled_strain=False``."""
    model = _model(approximant, "time")
    return functools.partial(model.polarizations_at, options=parse_options(options, option_keywords))


def polarizations(approximant: str, frequencies: Any, *, options: Optional[WaveformOptions] = None, **keywords):
    """``(h_+, h_x)`` at ``frequencies`` (Hz), as
    ``SimInspiralChooseFDWaveformSequence``."""
    model = _model(approximant, "frequency")
    binary, options = _request(options, keywords)
    frequencies = check_frequencies(frequencies)
    validate_result = getattr(model, "validate_result", None)
    if validate_result is None:
        model.validate(binary, np.asarray(frequencies), options)
        return _compiled(approximant, options, "polarizations", False)(binary, frequencies)
    # (the model's costly checks once the waveform is known to need them)
    model.validate(binary, np.asarray(frequencies), options, integrate=False)
    result = _compiled(approximant, options, "polarizations", False)(binary, frequencies)
    validate_result(binary, np.asarray(frequencies), options, result)
    return result


def polarizations_on_grid(approximant: str, grid: Any, *, options: Optional[WaveformOptions] = None, **keywords):
    """``(h_+, h_x)`` on a :class:`FrequencyGrid` (or frequencies)."""
    if not isinstance(grid, FrequencyGrid):
        grid = FrequencyGrid(grid)
    return polarizations(approximant, grid.frequencies, options=options, **keywords)


def modes(approximant: str, frequencies: Any, *, options: Optional[WaveformOptions] = None, **keywords) -> FrequencyModes:
    """The modes ``h_lm`` at ``+-frequencies`` (``SimInspiralChooseFDModes``
    conventions)."""
    model = _model(approximant, "frequency")
    binary, options = _request(options, keywords)
    frequencies = check_frequencies(frequencies)
    model.validate(binary, np.asarray(frequencies), options)
    return model.modes(binary, frequencies, options=options)


def choose_fd_modes(
    approximant: str,
    delta_f: float,
    f_min: float,
    f_max: float = 0.0,
    *,
    options: Optional[WaveformOptions] = None,
    **keywords,
) -> FrequencyModes:
    """``SimInspiralChooseFDModes``."""
    model = _model(approximant, "frequency")
    binary, options = _request(options, keywords)
    if not delta_f > 0 or not f_min > 0 or f_max < 0:
        raise InvalidParameterError("delta_f and f_min must be positive and f_max non-negative")
    return model.choose_fd_modes(binary, float(delta_f), float(f_min), float(f_max), options=options)


def _check_td(delta_t: float, f_min: float) -> None:
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise InvalidParameterError("delta_t must be positive")
    if not np.isfinite(f_min) or f_min < 0:
        raise InvalidParameterError("f_min must be non-negative")


def td_polarizations(
    approximant: str, delta_t: float, f_min: float, *, options: Optional[WaveformOptions] = None, **keywords
) -> TimeSeriesPolarizations:
    """``SimInspiralChooseTDWaveform``."""
    model = _model(approximant, "time")
    binary, options = _request(options, keywords)
    _check_td(delta_t, f_min)
    model.validate(binary, float(delta_t), float(f_min), options)
    return model.td_polarizations(binary, float(delta_t), float(f_min), options=options)


def td_modes(
    approximant: str, delta_t: float, f_min: float, *, options: Optional[WaveformOptions] = None, **keywords
) -> TimeSeriesModes:
    """``SimInspiralChooseTDModes``."""
    model = _model(approximant, "time")
    binary, options = _request(options, keywords)
    _check_td(delta_t, f_min)
    model.validate(binary, float(delta_t), float(f_min), options)
    return model.td_modes(binary, float(delta_t), float(f_min), options=options)


def simulate_td(
    approximant: str, delta_t: float, f_min: float, *, options: Optional[WaveformOptions] = None, **keywords
) -> TimeSeriesPolarizations:
    """``SimInspiralTD``: the conditioned time-domain waveform (see
    :mod:`waveformx.conversions`)."""
    _entry(approximant)
    binary, options = _request(options, keywords)
    return conversions.simulate_td(approximant, binary, options, delta_t, f_min)


def simulate_fd(
    approximant: str,
    delta_f: float,
    f_min: float,
    f_max: float,
    *,
    options: Optional[WaveformOptions] = None,
    **keywords,
) -> FrequencySeriesPolarizations:
    """``SimInspiralFD``: the conditioned frequency-domain waveform on ``k
    delta_f`` (``delta_f = 0`` chooses the spacing; see
    :mod:`waveformx.conversions`)."""
    _entry(approximant)
    binary, options = _request(options, keywords)
    return conversions.simulate_fd(approximant, binary, options, delta_f, f_min, f_max)


def simulate_td_function(
    approximant: str, delta_t: float, f_min: float, *, options: Optional[WaveformOptions] = None, **keywords
) -> Callable:
    """:func:`simulate_td` as a pure function ``f(binary) -> (hplus,
    hcross)`` for ``jax.jit``, ``jax.vmap`` and ``jax.grad``, with the
    length, epoch and tapers of the binary given by ``keywords`` (a
    :class:`~waveformx.conversions.SimulationFunction`; its ``reference``
    is :func:`simulate_td` of that binary)."""
    _entry(approximant)
    binary, options = _request(options, keywords)
    return conversions.simulate_td_function(approximant, binary, options, delta_t, f_min)


def simulate_fd_function(
    approximant: str,
    delta_f: float,
    f_min: float,
    f_max: float,
    *,
    options: Optional[WaveformOptions] = None,
    **keywords,
) -> Callable:
    """:func:`simulate_fd` as a pure function of the binary (see
    :func:`simulate_td_function`)."""
    _entry(approximant)
    binary, options = _request(options, keywords)
    return conversions.simulate_fd_function(approximant, binary, options, delta_f, f_min, f_max)


def _validate_rows(table: np.ndarray, check: Callable[[Binary], None]):
    """``ok`` and ``{index: error}`` from checking each row."""
    ok = np.ones(len(table), dtype=bool)
    errors = {}
    for index in range(len(table)):
        try:
            binary = row(table, index)
            check_binary(binary)
            check(binary)
        except WaveformError as error:
            ok[index] = False
            errors[index] = error
    return ok, errors


def batch_polarizations(
    approximant: str,
    grid: Any,
    parameters: Any,
    *,
    threads: int = 1,
    strict: bool = False,
    exact: bool = False,
    chunk_size: Optional[int] = None,
    options: Optional[WaveformOptions] = None,
    **option_keywords,
) -> BatchPolarizations:
    """Polarizations of many binaries on one grid, with ``jax.vmap``.

    ``parameters`` is a mapping of arrays, a structured array or an ``(n,
    12)`` table (see :func:`waveformx.parameter_table`). Binaries that
    fail the checks, or whose waveform is not finite, are reported in
    ``ok`` and ``errors`` and left as zeros, unless ``strict`` (then the
    first error is raised). ``chunk_size`` bounds the binaries evaluated at
    once. ``threads`` and ``exact`` are accepted for compatibility with
    waveformrs and ignored: XLA parallelises the computation itself.
    """
    model = _model(approximant, "frequency")
    options = parse_options(options, option_keywords)
    if not isinstance(grid, FrequencyGrid):
        grid = FrequencyGrid(grid)
    frequencies = grid.frequencies
    table = parameter_table(parameters)
    ok, errors = _validate_rows(table, lambda b: model.validate(b, np.asarray(frequencies), options))
    if strict and errors:
        raise errors[min(errors)]
    n = len(table)
    hplus = np.zeros((n, len(grid)), dtype=np.complex128)
    hcross = np.zeros((n, len(grid)), dtype=np.complex128)
    if ok.any():
        # failed rows are replaced by a valid one, so that no NaN reaches
        # the compiled function, and zeroed afterwards
        safe = table.copy()
        safe[~ok] = table[np.argmax(ok)]
        function = _compiled(approximant, options, "polarizations", True)
        step = n if chunk_size is None else max(1, int(chunk_size))
        for start in range(0, n, step):
            stop = min(n, start + step)
            hp, hc = function(binaries_from_table(safe[start:stop]), frequencies)
            hplus[start:stop], hcross[start:stop] = np.asarray(hp), np.asarray(hc)
        finite = np.all(np.isfinite(hplus), axis=1) & np.all(np.isfinite(hcross), axis=1)
        for index in np.flatnonzero(ok & ~finite):
            ok[index] = False
            errors[int(index)] = ModelError(f"{approximant} could not be evaluated for binary {index}")
        if strict and errors:
            raise errors[min(errors)]
    hplus[~ok] = 0
    hcross[~ok] = 0
    return BatchPolarizations(jnp.asarray(hplus), jnp.asarray(hcross), ok, errors)


def batch_td_polarizations(
    approximant: str,
    delta_t: float,
    f_min: float,
    parameters: Any,
    *,
    threads: int = 1,
    strict: bool = False,
    options: Optional[WaveformOptions] = None,
    **option_keywords,
) -> BatchTimeSeries:
    """Time-domain polarizations of many binaries (one series each, of the
    binary's own length)."""
    model = _model(approximant, "time")
    options = parse_options(options, option_keywords)
    _check_td(delta_t, f_min)
    table = parameter_table(parameters)
    ok, errors = _validate_rows(table, lambda b: model.validate(b, float(delta_t), float(f_min), options))
    if strict and errors:
        raise errors[min(errors)]
    batch = getattr(model, "batch_td_polarizations", None)
    if batch is not None:
        waveforms = batch(table, ok, float(delta_t), float(f_min), options=options)
    else:
        waveforms = [
            model.td_polarizations(row(table, i), float(delta_t), float(f_min), options=options) if ok[i] else None
            for i in range(len(table))
        ]
    for index, waveform in enumerate(waveforms):
        if waveform is not None and not (
            np.all(np.isfinite(np.asarray(waveform.hplus))) and np.all(np.isfinite(np.asarray(waveform.hcross)))
        ):
            ok[index] = False
            errors[index] = ModelError(f"{approximant} could not be evaluated for binary {index}")
            waveforms[index] = None
    if strict and errors:
        raise errors[min(errors)]
    return BatchTimeSeries(list(waveforms), ok, errors)
