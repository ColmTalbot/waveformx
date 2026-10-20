# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The parameters of a binary, as a JAX pytree."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Tuple

import jax.numpy as jnp
import numpy as np

from .exceptions import InvalidParameterError

__all__ = ["Binary", "BINARY_NAMES", "COLUMNS", "binary_from_keywords", "check_binary", "parameter_table", "binaries_from_table"]


class Binary(NamedTuple):
    """The parameters of ``SimInspiralChooseFDWaveform``, with the masses in
    solar masses and the distance in Mpc.

    A ``NamedTuple``, so a JAX pytree: every field may be a traced array,
    and ``jax.vmap`` maps over a leading batch axis of all fields (``spin_1``
    and ``spin_2`` then have shape ``(n, 3)``).

    * ``mass_1``, ``mass_2``: component masses (solar masses);
    * ``spin_1``, ``spin_2``: dimensionless spin vectors ``(x, y, z)`` at the
      reference frequency, with the orbital angular momentum along ``z``;
    * ``luminosity_distance``: Mpc;
    * ``inclination``: angle between the line of sight and the orbital
      angular momentum at ``f_ref`` (rad);
    * ``phi_ref``: orbital reference phase (rad);
    * ``f_ref``: reference frequency (Hz); zero selects LAL's default of the
      model (the starting frequency).
    """

    mass_1: Any
    mass_2: Any
    spin_1: Any
    spin_2: Any
    luminosity_distance: Any = 1.0
    inclination: Any = 0.0
    phi_ref: Any = 0.0
    f_ref: Any = 0.0


BINARY_NAMES = Binary._fields

#: The columns of a table of binaries (``parameter_table``), as in waveformrs.
COLUMNS = (
    "mass_1",
    "mass_2",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "luminosity_distance",
    "inclination",
    "phi_ref",
    "f_ref",
)


def _real(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}") from None


def _spin(value: Any, name: str) -> Tuple[float, float, float]:
    try:
        spin = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a sequence of three real numbers (x, y, z)") from None
    if spin.shape != (3,):
        raise TypeError(f"{name} must be a sequence of three real numbers (x, y, z)")
    return tuple(float(x) for x in spin)


def binary_from_keywords(keywords: Mapping[str, Any]) -> Binary:
    """The binary from keyword arguments (``mass_1`` and ``mass_2``
    required), as float64 JAX arrays. Raises ``TypeError`` for unknown or
    missing keywords."""
    unknown = [name for name in keywords if name not in BINARY_NAMES]
    if unknown:
        raise TypeError(
            f"unexpected keyword argument {unknown[0]!r}: the binary parameters are "
            f"{', '.join(BINARY_NAMES)}"
        )
    for name in ("mass_1", "mass_2"):
        if name not in keywords:
            raise TypeError(f"missing required keyword argument {name!r}")
    values = {
        "mass_1": _real(keywords["mass_1"], "mass_1"),
        "mass_2": _real(keywords["mass_2"], "mass_2"),
        "spin_1": _spin(keywords.get("spin_1", (0.0, 0.0, 0.0)), "spin_1"),
        "spin_2": _spin(keywords.get("spin_2", (0.0, 0.0, 0.0)), "spin_2"),
    }
    for name in ("luminosity_distance", "inclination", "phi_ref", "f_ref"):
        default = Binary._field_defaults[name]
        values[name] = _real(keywords.get(name, default), name)
    return Binary(**{name: jnp.asarray(value, dtype=jnp.float64) for name, value in values.items()})


def check_binary(binary: Binary) -> None:
    """The checks of LALSimulation common to the models, on concrete
    values; raises ``InvalidParameterError``."""
    m1, m2 = float(binary.mass_1), float(binary.mass_2)
    if not (np.isfinite(m1) and np.isfinite(m2)) or m1 <= 0 or m2 <= 0:
        raise InvalidParameterError("masses must be positive")
    for name in ("spin_1", "spin_2"):
        spin = np.asarray(getattr(binary, name), dtype=float)
        if not np.all(np.isfinite(spin)) or np.linalg.norm(spin) > 1.0:
            raise InvalidParameterError(f"{name} must have magnitude <= 1")
    distance = float(binary.luminosity_distance)
    if not np.isfinite(distance) or distance <= 0:
        raise InvalidParameterError("luminosity_distance must be positive")
    f_ref = float(binary.f_ref)
    if not np.isfinite(f_ref) or f_ref < 0:
        raise InvalidParameterError("f_ref must be positive, or zero for the model's default")
    for name in ("inclination", "phi_ref"):
        if not np.isfinite(float(getattr(binary, name))):
            raise InvalidParameterError(f"{name} must be finite")


def parameter_table(parameters: Any) -> np.ndarray:
    """An ``(n, 12)`` float64 table with the columns of :data:`COLUMNS` from

    * a mapping of arrays with the keyword names (``spin_1`` and ``spin_2``
      of shape ``(n, 3)``, or the columns ``spin_1x`` ... ``spin_2z``;
      missing optional parameters take their defaults),
    * a structured array or anything with those fields (a pandas
      ``DataFrame``), or
    * an ``(n, 12)`` array already in this layout.
    """
    if isinstance(parameters, Mapping) or getattr(getattr(parameters, "dtype", None), "names", None) or hasattr(parameters, "columns"):
        fields = parameters
        names = set(parameters.keys() if hasattr(parameters, "keys") else parameters.dtype.names)
        n = None
        for name in names:
            n = len(np.atleast_1d(np.asarray(fields[name])))
            break
        if n is None:
            raise TypeError("parameters is empty")
        columns = []
        for column in COLUMNS:
            if column in names:
                values = np.asarray(fields[column], dtype=float)
            elif column[:-1] in ("spin_1", "spin_2") and column[:-1] in names:
                spins = np.asarray(fields[column[:-1]], dtype=float).reshape(n, 3)
                values = spins[:, "xyz".index(column[-1])]
            elif column in ("mass_1", "mass_2"):
                raise TypeError(f"parameters must contain {column!r}")
            else:
                values = np.full(n, Binary._field_defaults.get(column, 0.0), dtype=float)
            columns.append(np.broadcast_to(values, (n,)))
        unknown = names - set(COLUMNS) - set(BINARY_NAMES)
        if unknown:
            raise TypeError(f"unknown parameters {sorted(unknown)}: the columns are {', '.join(COLUMNS)}")
        return np.stack(columns, axis=1)
    table = np.asarray(parameters, dtype=float)
    if table.ndim != 2 or table.shape[1] != len(COLUMNS):
        raise ValueError(f"parameters must have {len(COLUMNS)} columns ({', '.join(COLUMNS)})")
    return table


def binaries_from_table(table: np.ndarray) -> Binary:
    """A batched :class:`Binary` (leading axis ``n``) from an ``(n, 12)``
    table."""
    table = jnp.asarray(table, dtype=jnp.float64)
    return Binary(
        mass_1=table[:, 0],
        mass_2=table[:, 1],
        spin_1=table[:, 2:5],
        spin_2=table[:, 5:8],
        luminosity_distance=table[:, 8],
        inclination=table[:, 9],
        phi_ref=table[:, 10],
        f_ref=table[:, 11],
    )


def row(table: np.ndarray, index: int) -> Binary:
    """Row ``index`` of a table as an unbatched :class:`Binary` of floats."""
    r = table[index]
    return Binary(r[0], r[1], r[2:5], r[5:8], r[8], r[9], r[10], r[11])
