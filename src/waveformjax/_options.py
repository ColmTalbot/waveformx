# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The settings of LALSimulation's ``LALDict`` that the models take."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping, Optional, Tuple

from .exceptions import InvalidParameterError

__all__ = ["WaveformOptions", "OPTION_NAMES", "parse_options"]

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
      waveformrs; IMRPhenomXPNR always uses its own prescription (330).
    * ``mode_array``: LAL's ``ModeArray`` (``(l, m)`` pairs) or ``None`` for
      all modes. For IMRPhenomXPNR, the co-precessing modes.
    * ``version_fallback``: whether IMRPhenomXPNR may change its precession
      prescription as LALSimulation does (default ``True``). Without, only
      the SpinTaylor angles are computed, and the waveform is NaN where LAL
      would leave them.
    * ``pnr_parity_fix``: whether IMRPhenomXPNR applies the fix of lalsuite
      merge request !2593, which restores the mirror symmetry of the
      waveform (default ``True``, as LALSimulation master); ``False``
      reproduces LALSimulation 6.2.0 and 6.2.1.
    * ``extended_spintaylor``: IMRPhenomXPNR with the SpinTaylor angles for
      every binary (default ``False``): also where LALSimulation cannot use
      them (a reference frequency beyond the PN energy test, small in-plane
      spins), with LAL's discrete switches replaced by continuous versions
      (the end of the inspiral, the two-spin construction for every
      binary, the connection of the PNR ``beta``, the edge of the PNR
      transition region at ``q = 12``) and exact derivatives in the higher
      modes where LAL takes finite differences, so that the waveform and
      its derivatives are continuous. This is not LAL's waveform: where LAL
      uses the SpinTaylor angles the two agree closely except near LAL's
      own switches (``report/extended``). ``version_fallback`` is then
      ignored.
    * ``scaled_strain``: whether NRSur7dq4 returns the strain multiplied by
      ``1e21`` (default ``True``), which keeps the strain, its square and
      its derivatives within the range of float32; ``False`` gives the
      strain itself, as LAL. IMRPhenomXPNR always gives the strain itself.
    """

    precession_version: int = 320
    mode_array: Optional[Tuple[Tuple[int, int], ...]] = None
    version_fallback: bool = True
    pnr_parity_fix: bool = True
    extended_spintaylor: bool = False
    scaled_strain: bool = True

    def __post_init__(self):
        object.__setattr__(self, "precession_version", _precession_version(self.precession_version))
        object.__setattr__(self, "mode_array", _mode_array(self.mode_array))
        _bool(self.version_fallback, "version_fallback")
        _bool(self.pnr_parity_fix, "pnr_parity_fix")
        _bool(self.extended_spintaylor, "extended_spintaylor")
        _bool(self.scaled_strain, "scaled_strain")

    @staticmethod
    def lalsimulation_defaults() -> "WaveformOptions":
        """LALSimulation 6.2.0's defaults for an empty ``LALDict``: the MSA
        angles (223), all modes, IMRPhenomXPNR without the parity fix and
        the strain itself (not scaled)."""
        return WaveformOptions(precession_version=223, pnr_parity_fix=False, scaled_strain=False)

    def replace(self, **options: Any) -> "WaveformOptions":
        """A copy with the given options changed."""
        _check_names(options)
        return dataclasses.replace(self, **options)

    def as_dict(self) -> dict:
        """The options as keyword arguments."""
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}


OPTION_NAMES = tuple(field.name for field in dataclasses.fields(WaveformOptions))


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
