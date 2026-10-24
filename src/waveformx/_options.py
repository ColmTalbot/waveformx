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
      waveformrs; IMRPhenomXPNR always uses its own prescription (330).
    * ``mode_array``: LAL's ``ModeArray`` (``(l, m)`` pairs) or ``None`` for
      all modes. For IMRPhenomXPNR, the co-precessing modes.
    * ``version_fallback``: whether IMRPhenomXPNR may change its precession
      prescription as LALSimulation does (default ``False``). Without, only
      the SpinTaylor angles are computed (which compiles much faster): the
      public functions raise an error, and the pure functions give NaN,
      where LAL would leave them (in-plane spins below ``1e-6``, a failed
      SpinTaylor integration).
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
    * ``extended_direct_angles``: with ``extended_spintaylor``, evaluate the
      SpinTaylor angles directly from the solution of the precession
      equations (default ``True``) rather than from LAL's splines through
      its samples of it. Much faster for light binaries, whose samples
      number millions; it differs from LAL by the interpolation error of
      LAL's splines, largest for heavy binaries, whose samples are sparse.
    * ``extended_phase_grid``: with ``extended_spintaylor``, interpolate the
      PNR angles on a grid whose spacing grows like ``f^2`` (constant steps
      of the precession phase) from LAL's spacing, up to ``1e-4`` in ``M f``
      (default ``False``), rather than on LAL's uniform grid: far fewer
      points for light binaries, at the cost of a larger difference from
      LAL.
    * ``lean``: the lean IMRPhenomXPNR (default ``False``): the extended
      model with the direct angles and the parity fix, without the layers
      that reproduce LAL's numerics: the precession equations integrated
      once with a high-order method rather than step by step as LAL does
      (and without LAL's finer re-run of the end of the inspiral), masses
      in solar masses, no rounding to single precision of stored values.
      LAL's physics (fits, ansatzes, the discrete choices of the higher
      modes) is kept. It agrees with the extended model to its
      integration error; the other precession options are ignored.
    * ``reference_mf``: for the models that take it (the lean
      IMRPhenomXPNR, TaylorF2 and IMRPhenomD; the others refuse it in
      their checks), the reference frequency of the binary in units of the
      total mass, ``M f_ref`` (default ``None``:
      LAL's convention, ``Binary.f_ref`` in Hz). The spins, the
      inclination and ``phi_ref`` are then given at ``f_ref =
      reference_mf / M``, late in the inspiral for every binary
      (:data:`REFERENCE_MF` is the recommended value), and the waveform
      works for every mode array and every binary, whether ``f_ref`` lies
      inside the frequencies or not. ``Binary.f_ref`` must then be zero (the
      waveform is NaN otherwise, and its derivative with respect to it
      zero). Whatever ``reference_mf`` is, the model first carries the
      binary to :data:`REFERENCE_MF` and takes everything that depends on
      the reference there (the aligned spins of the co-precessing modes,
      J, the PNR fits, the offsets of the angles, the phase alignment and
      the basis of the polarizations), so a binary converted between two
      references gives the same waveform; other values cost an
      integration between the two references (and a compilation each, as
      the option is static). :func:`waveformx.to_mass_scaled_reference` and
      :func:`waveformx.to_physical_reference` convert binaries between
      the two conventions.
    * ``scaled_strain``: whether the models return the strain multiplied by
      ``1e21`` (default ``True``), which keeps the strain, its square and
      its derivatives within the range of float32; ``False`` gives the
      strain itself, as LAL.
    """

    precession_version: int = 320
    mode_array: Optional[Tuple[Tuple[int, int], ...]] = None
    version_fallback: bool = False
    pnr_parity_fix: bool = True
    extended_spintaylor: bool = False
    extended_direct_angles: bool = True
    extended_phase_grid: bool = False
    lean: bool = False
    scaled_strain: bool = True
    reference_mf: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "precession_version", _precession_version(self.precession_version))
        object.__setattr__(self, "mode_array", _mode_array(self.mode_array))
        _bool(self.version_fallback, "version_fallback")
        _bool(self.pnr_parity_fix, "pnr_parity_fix")
        _bool(self.extended_spintaylor, "extended_spintaylor")
        _bool(self.extended_direct_angles, "extended_direct_angles")
        _bool(self.extended_phase_grid, "extended_phase_grid")
        _bool(self.lean, "lean")
        _bool(self.scaled_strain, "scaled_strain")
        object.__setattr__(self, "reference_mf", _reference_mf(self.reference_mf))

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
