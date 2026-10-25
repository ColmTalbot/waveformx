# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The settings of LALSimulation's ``LALDict`` that the models take."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping, Optional, Tuple

from .exceptions import InvalidParameterError

__all__ = ["WaveformOptions", "OPTION_NAMES", "REFERENCE_MF", "parse_options"]

PRECESSION_VERSIONS = {"nnlo": 102, "msa": 223, "spintaylor": 320, "spin_taylor": 320}


def _precession_version(value: Any) -> int:
    if isinstance(value, str):
        try:
            return PRECESSION_VERSIONS[value.lower()]
        except KeyError:
            raise InvalidParameterError(
                f"unknown precession version {value!r}: use 102 (\"NNLO\"), 223 (\"MSA\") "
                "or 320 (\"SpinTaylor\")"
            ) from None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("precession_version must be an int (LAL's PhenomXPrecVersion) or a str")
    if value not in (102, 223, 320):
        raise InvalidParameterError(
            f"precession version {value} is not implemented: use 102 (NNLO), 223 (MSA) or "
            "320 (SpinTaylor)"
        )
    return value


def _mode_array(value: Any) -> Optional[Tuple[Tuple[int, int], ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError("mode_array must be a sequence of (l, m) pairs or None")
    modes = []
    for item in value:
        try:
            ell, emm = (int(x) for x in item)
        except (TypeError, ValueError):
            raise TypeError("mode_array must be a sequence of (l, m) pairs or None") from None
        modes.append((ell, emm))
    return tuple(modes)


def _reference_mf(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("reference_mf must be a float or None")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError("reference_mf must be a float or None") from None
    if not 0.0 < value < 0.015:
        raise InvalidParameterError(
            f"reference_mf = {value} must lie in (0, 0.015): the inspiral of some binaries ends at M f = 0.015"
        )
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


@dataclasses.dataclass(frozen=True)
class WaveformOptions:
    """The settings of LALSimulation's ``LALDict`` that the models take.

    Instances are immutable and hashable, so they can be static arguments of
    ``jax.jit``.

    * ``precession_version``: LAL's ``PhenomXPrecVersion`` (102 ``"NNLO"``,
      223 ``"MSA"`` or 320 ``"SpinTaylor"``). Kept for compatibility with
      waveformrs; IMRPhenomXPNR always uses the SpinTaylor angles.
    * ``mode_array``: LAL's ``ModeArray`` (``(l, m)`` pairs) or ``None`` for
      all modes. For IMRPhenomXPNR, the co-precessing modes.
    * ``extended_phase_grid``: interpolate the PNR angles of
      IMRPhenomXPNR on a grid whose spacing grows like ``f^2`` (constant
      steps of the precession phase) from LAL's spacing, up to ``1e-4`` in
      ``M f`` (default ``False``), rather than on LAL's uniform grid: far
      fewer points for light binaries, at the cost of a larger difference
      from LAL.
    * ``reference_mf``: for every model, the
      reference frequency of the binary in units of the total mass,
      ``M f_ref`` (default ``None``:
      LAL's convention, ``Binary.f_ref`` in Hz). The spins, the
      inclination and ``phi_ref`` are then given at ``f_ref =
      reference_mf / M``, late in the inspiral for every binary
      (:data:`REFERENCE_MF` is the recommended value), and the waveform
      works for every mode array and every binary, whether ``f_ref`` lies
      inside the frequencies or not. ``Binary.f_ref`` must then be zero (the
      waveform is NaN otherwise, and its derivative with respect to it
      zero). For the aligned-spin models only the reference phase moves
      with it; for NRSur7dq4 the reference is the orbital frequency ``pi
      reference_mf`` of the surrogate, which must lie inside it (for
      ``REFERENCE_MF`` it does for every binary of its range). Whatever ``reference_mf`` is, IMRPhenomXPNR first carries the
      binary to :data:`REFERENCE_MF` and takes everything that depends on
      the reference there (the aligned spins of the co-precessing modes,
      J, the PNR fits, the offsets of the angles, the phase alignment and
      the basis of the polarizations), so a binary converted between two
      references gives the same waveform; other values cost an
      integration between the two references (and a compilation each, as
      the option is static). :func:`waveformx.to_mass_scaled_reference` and
      :func:`waveformx.to_physical_reference` (with ``approximant``)
      convert binaries between the two conventions.
    * ``scaled_strain``: whether the models return the strain multiplied by
      ``1e21`` (default ``True``), which keeps the strain, its square and
      its derivatives within the range of float32; ``False`` gives the
      strain itself, as LAL.
    """

    precession_version: int = 320
    mode_array: Optional[Tuple[Tuple[int, int], ...]] = None
    extended_phase_grid: bool = False
    scaled_strain: bool = True
    reference_mf: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "precession_version", _precession_version(self.precession_version))
        object.__setattr__(self, "mode_array", _mode_array(self.mode_array))
        _bool(self.extended_phase_grid, "extended_phase_grid")
        _bool(self.scaled_strain, "scaled_strain")
        object.__setattr__(self, "reference_mf", _reference_mf(self.reference_mf))

    @staticmethod
    def lalsimulation_defaults() -> "WaveformOptions":
        """LALSimulation 6.2.0's defaults for an empty ``LALDict``: the MSA
        angles (223), all modes and the strain itself (not scaled)."""
        return WaveformOptions(precession_version=223, scaled_strain=False)

    def replace(self, **options: Any) -> "WaveformOptions":
        """A copy with the given options changed."""
        _check_names(options)
        return dataclasses.replace(self, **options)

    def as_dict(self) -> dict:
        """The options as keyword arguments."""
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}


OPTION_NAMES = tuple(field.name for field in dataclasses.fields(WaveformOptions))

#: The recommended ``reference_mf``: late in the inspiral, which ends at
#: ``M f >= 0.015`` for mass ratios up to 30 and spins up to 0.99.
REFERENCE_MF = 0.01

#: The factor of the strain unless ``scaled_strain`` is false.
STRAIN_SCALE = 1e21


def strain_scale(options: WaveformOptions) -> float:
    """:data:`STRAIN_SCALE` if ``options.scaled_strain``, else 1."""
    return STRAIN_SCALE if options.scaled_strain else 1.0


def _check_names(options: Iterable[str]) -> None:
    for name in options:
        if name not in OPTION_NAMES:
            raise TypeError(
                f"unexpected keyword argument {name!r}: the options are {', '.join(OPTION_NAMES)}"
            )


def parse_options(options: Optional[WaveformOptions], keywords: Mapping[str, Any]) -> WaveformOptions:
    """``options`` (default ``WaveformOptions()``) changed by ``keywords``."""
    if options is None:
        options = WaveformOptions()
    elif not isinstance(options, WaveformOptions):
        raise TypeError("options must be a WaveformOptions or None")
    if not keywords:
        return options
    return options.replace(**keywords)
