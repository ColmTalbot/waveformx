# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Frequency grids and the results of the waveform functions, with the
attributes of their waveformrs counterparts."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .exceptions import InvalidParameterError

__all__ = [
    "FrequencyGrid",
    "FrequencyModes",
    "TimeSeriesPolarizations",
    "FrequencySeriesPolarizations",
    "TimeSeriesModes",
    "BatchPolarizations",
    "BatchTimeSeries",
    "check_frequencies",
]

Mode = Tuple[int, int]


def check_frequencies(frequencies: Any) -> jnp.ndarray:
    """``frequencies`` as a 1-d float64 array, checked to be positive,
    finite and strictly increasing (concrete values only)."""
    values = np.asarray(frequencies, dtype=float)
    if values.ndim != 1:
        raise InvalidParameterError("frequencies must be a 1-d array")
    if values.size and (not np.all(np.isfinite(values)) or values[0] <= 0):
        raise InvalidParameterError("frequencies must be positive and finite")
    if np.any(np.diff(values) <= 0):
        raise InvalidParameterError("frequencies must be strictly increasing")
    return jnp.asarray(values, dtype=jnp.float64)


class FrequencyGrid:
    """Frequencies (Hz) for evaluating many waveforms on the same
    frequencies. In JAX the compiled function is cached by the grid's shape,
    so reusing a grid avoids recompilation."""

    def __init__(self, frequencies: Any):
        self._frequencies = check_frequencies(frequencies)

    @staticmethod
    def uniform(f_min: float, f_max: float, delta_f: float) -> "FrequencyGrid":
        """``f_min, f_min + delta_f, ...`` up to and including ``f_max`` (to
        within half a step)."""
        n = int(np.floor((f_max - f_min) / delta_f + 0.5)) + 1
        return FrequencyGrid(f_min + np.arange(n) * delta_f)

    @property
    def frequencies(self) -> jnp.ndarray:
        return self._frequencies

    def __len__(self) -> int:
        return int(self._frequencies.shape[0])

    def __repr__(self) -> str:
        f = self._frequencies
        return f"FrequencyGrid({len(self)} frequencies, {float(f[0]) if len(self) else 0:g}-{float(f[-1]) if len(self) else 0:g} Hz)"


@dataclasses.dataclass
class FrequencyModes:
    """Modes ``h_lm`` in the conventions of ``SimInspiralChooseFDModes``:
    ``frequencies`` (two-sided for :func:`choose_fd_modes`) and a dict
    ``{(l, m): array}``, with the angles ``theta`` and ``phi`` of the line
    of sight in the frame of the modes and the ``polarization_angle`` that
    rotates their polarizations into LAL's."""

    frequencies: Any
    modes: Dict[Mode, Any]
    theta: float = 0.0
    phi: float = 0.0
    polarization_angle: float = 0.0

    def __getitem__(self, mode: Mode):
        return self.modes[tuple(mode)]

    def __iter__(self):
        return iter((self.frequencies, self.modes))

    def polarizations(self) -> Tuple[Any, Any]:
        """``(h_+, h_x)`` from the modes, summed with the spin-weighted
        spherical harmonics at ``(theta, phi)`` and rotated by the
        polarization angle."""
        from ._harmonics import polarizations_from_modes

        return polarizations_from_modes(self)


@dataclasses.dataclass
class TimeSeriesPolarizations:
    """``h_+`` and ``h_x`` sampled every ``delta_t`` from ``epoch`` (s,
    relative to the peak)."""

    hplus: Any
    hcross: Any
    epoch: float
    delta_t: float

    @property
    def times(self):
        return self.epoch + self.delta_t * jnp.arange(self.hplus.shape[-1])

    def __iter__(self):
        return iter((self.hplus, self.hcross))


@dataclasses.dataclass
class FrequencySeriesPolarizations:
    """``h_+(f)`` and ``h_x(f)`` on the frequencies ``k delta_f``, ``k = 0,
    1, ...``, with the ``epoch`` (s) of the time series they are the Fourier
    transform of."""

    hplus: Any
    hcross: Any
    epoch: float
    delta_f: float

    @property
    def frequencies(self):
        return self.delta_f * jnp.arange(self.hplus.shape[-1])

    def __iter__(self):
        return iter((self.hplus, self.hcross))


@dataclasses.dataclass
class TimeSeriesModes:
    """Modes ``{(l, m): h_lm}`` sampled every ``delta_t`` from ``epoch``."""

    modes: Dict[Mode, Any]
    epoch: float
    delta_t: float

    @property
    def times(self):
        first = next(iter(self.modes.values()))
        return self.epoch + self.delta_t * jnp.arange(first.shape[-1])

    def __getitem__(self, mode: Mode):
        return self.modes[tuple(mode)]

    def __iter__(self):
        return iter((self.times, self.modes))


@dataclasses.dataclass
class BatchPolarizations:
    """``(n_binaries, n_frequencies)`` arrays ``hplus`` and ``hcross``, with
    ``ok`` (whether each binary succeeded) and ``errors`` (``{index:
    exception}``); failed rows are zero."""

    hplus: Any
    hcross: Any
    ok: np.ndarray
    errors: Dict[int, Exception]

    def __iter__(self):
        return iter((self.hplus, self.hcross))


@dataclasses.dataclass
class BatchTimeSeries:
    """One :class:`TimeSeriesPolarizations` (or ``None`` for a failure) per
    binary, with ``ok`` and ``errors``."""

    waveforms: List[Optional[TimeSeriesPolarizations]]
    ok: np.ndarray
    errors: Dict[int, Exception]

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, index: int):
        return self.waveforms[index]

    def __iter__(self):
        return iter(self.waveforms)
