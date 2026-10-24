# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The NRSur7dq4 surrogate data, read from ``NRSur7dq4_v1.0.h5``.

The file holds, for every dynamics node and every empirical-interpolation
node of the coorbital waveform, a polynomial fit in the seven parameters
``(log q, chi1x, chi1y, chiHat, chi2x, chi2y, chi_a)``: a list of
coefficients and, for each, the orders of the monomial it multiplies
(``PrecessingNRSur_LoadFitData`` in LALSimulation).

Here every fit of the dynamics is turned into a dense row over the union of
the monomials used by the dynamics fits (a few hundred), so that the nine
fits of a node are one matrix-vector product. The fits at the empirical
nodes of the waveform are stored by the order of the mass-ratio parameter
and the spin monomial, so that the mass ratio, which is the same at every
node, is folded into the coefficients once. The data are
returned as :class:`NRSurData`, a pytree of float64 (and integer) JAX
arrays, read once per file and cached; a float32 copy (cast from the
float64 arrays, including the factorised spline systems) is cached too
when it is first asked for.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ...exceptions import DataFileError, DataFileNotFoundError
from ._spline import Knots, factorize

__all__ = [
    "NRSurData",
    "precision",
    "LMAX",
    "MODES",
    "N_FITS",
    "PIECES",
    "load",
    "default_data",
    "data_file",
]

#: Highest ``ell`` of the surrogate (``NRSUR_LMAX``).
LMAX = 4
#: The ``(ell, m)`` modes in LAL's ``MultiModalWaveform`` order.
MODES = tuple((ell, m) for ell in range(2, LMAX + 1) for m in range(-ell, ell + 1))
#: The fits of a dynamics node, in the order of ``DynamicsNodeFitData``.
DYNAMICS_FITS = (
    "omega",
    "omega_orb_0",
    "omega_orb_1",
    "chiA_0",
    "chiA_1",
    "chiA_2",
    "chiB_0",
    "chiB_1",
    "chiB_2",
)
N_FITS = len(DYNAMICS_FITS)

#: The data pieces of the coorbital waveform, ``(ell, m, kind)``, with
#: ``kind`` one of ``real``, ``imag`` (``m = 0``) and ``Re+``, ``Re-``,
#: ``Im+``, ``Im-`` (``m > 0``), in the order they are stored here.
PIECES = tuple(
    piece
    for ell in range(2, LMAX + 1)
    for piece in (
        [(ell, 0, "real"), (ell, 0, "imag")]
        + [(ell, m, kind) for m in range(1, ell + 1) for kind in ("Re+", "Re-", "Im+", "Im-")]
    )
)
#: The file stores ``Y_pm = (h^{l,-m} pm conj(h^{l,m})) / 2``; with ``X_pm =
#: pm conj(Y_pm)`` the bases of ``Re[X_-]`` and ``Im[X_+]`` change sign
#: (``PrecessingNRSur_LoadCoorbitalEllModes``).
_INVERTED = ("Re-", "Im+")

#: Most factors of non-zero order in a term of the fits.
MAX_FACTORS = 5

#: Name of the data file (``NRSUR7DQ4_DATAFILE``).
DATA_FILE = "NRSur7dq4_v1.0.h5"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NRSurData:
    """The surrogate data as JAX arrays (a pytree whose leaves are the
    arrays; ``ei_bounds`` is static).

    * ``t_ds``: the dynamics nodes (units of the total mass, peak at 0),
      without the half nodes;
    * ``t_ds_half``: the half nodes ``t_1/2, t_3/2, t_5/2``;
    * ``monomials``: ``(n_monomials, MAX_FACTORS)`` the monomials of the
      fit parameters (the union over the dynamics fits) as indices into the table of
      powers ``x_j^k`` at ``4 j + k`` (see :mod:`._fits`): the factors with
      ``k > 0`` in the order of ``j``, padded with the index 0 of ``x_0^0 =
      1``;
    * ``dynamics``: ``(n_ds + 3, 9, n_monomials)`` coefficients of the nine
      fits of every dynamics node (:data:`DYNAMICS_FITS`), the three half
      nodes last;
    * ``t_coorb``: the times of the coorbital waveform;
    * ``ei_time``: the index into ``t_coorb`` of each empirical node;
    * ``ei_coefs``: ``(n_ei, 4, n_spin)`` coefficients of the fit at each
      empirical node, by the order of the mass-ratio parameter and the spin
      monomial (``ei_spins``);
    * ``ei_basis``: ``(n_ei, n_coorb)`` empirical-interpolation basis, with
      LAL's sign changes applied;
    * ``ds_knots``, ``coorb_knots``: ``t_ds`` and ``t_coorb`` with their
      factorised spline systems (:class:`._spline.Knots`);
    * ``coorb_index``, ``coorb_delx``: the interval of ``t_ds`` of each time
      of ``t_coorb`` and the offset from its left knot;
    * ``ei_bounds``: the empirical nodes of data piece ``p`` (:data:`PIECES`)
      are ``ei_bounds[p]:ei_bounds[p + 1]`` (static);
    * ``ei_spins``: the orders of the six spin parameters in each spin
      monomial of the empirical-node fits (static).
    """

    t_ds: jnp.ndarray
    t_ds_half: jnp.ndarray
    monomials: jnp.ndarray
    dynamics: jnp.ndarray
    t_coorb: jnp.ndarray
    ei_time: jnp.ndarray
    ei_coefs: jnp.ndarray
    ei_basis: jnp.ndarray
    ds_knots: Knots
    coorb_knots: Knots
    coorb_index: jnp.ndarray
    coorb_delx: jnp.ndarray
    ei_bounds: Tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
    ei_spins: Tuple[Tuple[int, ...], ...] = dataclasses.field(metadata=dict(static=True))

    @property
    def n_ds(self) -> int:
        """Number of dynamics nodes (without the half nodes)."""
        return int(self.t_ds.shape[0])

    @property
    def n_coorb(self) -> int:
        """Number of coorbital times."""
        return int(self.t_coorb.shape[0])

    @property
    def dtype(self):
        """The floating-point type of the arrays (float64 or float32)."""
        return self.dynamics.dtype

    def astype(self, dtype) -> "NRSurData":
        """The data with the floating-point arrays converted to ``dtype``."""
        return jax.tree_util.tree_map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, self
        )


def data_file() -> Optional[str]:
    """The data file from ``NRSUR7DQ4_DATA`` (the file) or ``LAL_DATA_PATH``
    (its directory), or ``None`` (:func:`waveformx.nrsur7dq4_data_file`)."""
    from ..._api import nrsur7dq4_data_file

    return nrsur7dq4_data_file()


def _check_fit(coefs, orders, where: str):
    """A fit's coefficients and ``(n, 7)`` basis-function orders, checked."""
    coefs = np.asarray(coefs, dtype=float)
    orders = np.asarray(orders, dtype=np.int64)
    if coefs.ndim != 1 or orders.shape != (coefs.size, 7):
        raise DataFileError(f"{where}: {coefs.size} coefficients with basis orders of shape {orders.shape}")
    if np.any(orders < 0) or np.any(orders[:, 0] > 3) or np.any(orders[:, 1:] > 2):
        # LAL's table of powers has 4 powers of q and 3 of each spin
        raise DataFileError(f"{where}: basis-function order out of range")
    return coefs, orders


class _Union:
    """The union of the monomials (or of their spin parts) used by a set of
    fits, each keyed by its tuple of orders."""

    def __init__(self):
        self.index = {}

    def columns(self, keys) -> np.ndarray:
        columns = [self.index.setdefault(tuple(int(k) for k in key), len(self.index)) for key in keys]
        return np.array(columns, dtype=np.int64)

    def keys(self) -> np.ndarray:
        keys = [None] * len(self.index)
        for key, column in self.index.items():
            keys[column] = key
        return np.array(keys, dtype=np.int32).reshape(len(keys), -1)


def _factors(orders: np.ndarray) -> np.ndarray:
    """Monomials (``(n, 7)`` orders) as indices ``4 j + k`` into the table
    of powers: their factors of non-zero order, padded with index 0
    (``x_0^0 = 1``). Multiplying by the order-zero factors (exactly one) is
    exact, so the product equals LAL's product over all seven factors."""
    factors = np.zeros((len(orders), MAX_FACTORS), dtype=np.int32)
    for row, monomial in zip(factors, orders):
        nonzero = [4 * j + k for j, k in enumerate(monomial) if k > 0]
        if len(nonzero) > MAX_FACTORS:
            raise DataFileError("a fit of the NRSur7dq4 data has too many factors per term")
        row[: len(nonzero)] = nonzero
    return factors


def _read(group, name: str) -> np.ndarray:
    try:
        return group[name][()]
    except KeyError:
        raise DataFileError(f"{group.name}/{name} is missing from the NRSur7dq4 data file") from None


def _load_numpy(path: str) -> dict:
    """The data file as numpy arrays (``PrecessingNRSur_Init``)."""
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise DataFileError("reading the NRSur7dq4 data needs h5py") from error
    try:
        handle = h5py.File(path, "r")
    except OSError as error:
        raise DataFileError(f"cannot read the NRSur7dq4 data file {path}: {error}") from error
    with handle as f:
        t_with_halves = np.asarray(_read(f, "t_ds"), dtype=float)
        if t_with_halves.ndim != 1 or t_with_halves.size < 10:
            raise DataFileError("too few dynamics nodes in the NRSur7dq4 data file")
        n_ds = t_with_halves.size - 3
        # nodes are stored as ds_node_0 (t_0), ds_node_1 (t_1/2), ds_node_2
        # (t_1), ds_node_3 (t_3/2), ds_node_4 (t_2), ds_node_5 (t_5/2),
        # ds_node_6 (t_3), and ds_node_{n+3} (t_n) for n >= 3
        t_ds = np.concatenate([t_with_halves[0:6:2], t_with_halves[6:]])
        t_half = t_with_halves[1:6:2]
        if np.any(np.diff(t_ds) <= 0):
            raise DataFileError("the dynamics nodes of the NRSur7dq4 data are not increasing")
        monomials = _Union()
        dynamics = []
        groups = [2 * i if i < 3 else i + 3 for i in range(n_ds)] + [1, 3, 5]
        for j in groups:
            group = f.get(f"ds_node_{j}")
            if group is None:
                raise DataFileError(f"ds_node_{j} is missing from the NRSur7dq4 data file")
            for name in DYNAMICS_FITS:
                coefs, orders = _check_fit(
                    _read(group, f"{name}_coefs"), _read(group, f"{name}_bfOrders"), f"ds_node_{j}/{name}"
                )
                dynamics.append((coefs, monomials.columns(orders)))

        t_coorb = np.asarray(_read(f, "t_coorb"), dtype=float)
        n_coorb = t_coorb.size
        if n_coorb < 4 or np.any(np.diff(t_coorb) <= 0):
            raise DataFileError("the coorbital times of the NRSur7dq4 data are not increasing")
        if t_coorb[0] < t_ds[0] or t_coorb[-1] > t_ds[-1]:
            raise DataFileError("the coorbital times extend beyond the dynamics nodes")

        spins = _Union()
        ei_fits, ei_time, ei_basis, ei_bounds = [], [], [], [0]
        for ell, m, kind in PIECES:
            name = f"hCoorb_{ell}_{m}_{kind}"
            group = f.get(name)
            if group is None:
                raise DataFileError(f"{name} is missing from the NRSur7dq4 data file")
            nodes = np.asarray(_read(group, "nodeIndices"), dtype=np.int64).reshape(-1)
            basis = np.asarray(_read(group, "EIBasis"), dtype=float)
            if basis.shape != (nodes.size, n_coorb) or np.any(nodes < 0) or np.any(nodes >= n_coorb):
                raise DataFileError(f"{name}: inconsistent empirical-interpolation data")
            if kind in _INVERTED:
                basis = -basis
            modelers = group.get("nodeModelers")
            if modelers is None:
                raise DataFileError(f"{name}/nodeModelers is missing from the NRSur7dq4 data file")
            for i in range(nodes.size):
                coefs, orders = _check_fit(
                    _read(modelers, f"coefs_{i}"), _read(modelers, f"bfOrders_{i}"), f"{name}/{i}"
                )
                ei_fits.append((coefs, orders[:, 0], spins.columns(orders[:, 1:])))
            ei_time.append(nodes)
            ei_basis.append(basis)
            ei_bounds.append(ei_bounds[-1] + int(nodes.size))

    dense = np.zeros((len(dynamics), len(monomials.index)))
    for row, (coefs, columns) in zip(dense, dynamics):
        np.add.at(row, columns, coefs)
    ei_coefs = np.zeros((len(ei_fits), 4, len(spins.index)))
    for row, (coefs, q_orders, columns) in zip(ei_coefs, ei_fits):
        np.add.at(row, (q_orders, columns), coefs)
    return dict(
        t_ds=t_ds,
        t_ds_half=t_half,
        monomials=_factors(monomials.keys()),
        dynamics=dense.reshape(n_ds + 3, len(DYNAMICS_FITS), -1),
        t_coorb=t_coorb,
        ei_time=np.concatenate(ei_time).astype(np.int32),
        ei_coefs=ei_coefs,
        ei_basis=np.concatenate(ei_basis, axis=0),
        ei_bounds=tuple(ei_bounds),
        ei_spins=tuple(tuple(int(k) for k in key) for key in spins.keys()),
    )


@functools.lru_cache(maxsize=4)
def _load_cached(path: str) -> NRSurData:
    arrays = _load_numpy(path)
    static = {name: arrays.pop(name) for name in ("ei_bounds", "ei_spins")}
    t_ds, t_coorb = arrays["t_ds"], arrays["t_coorb"]
    index = np.clip(np.searchsorted(t_ds, t_coorb, side="right") - 1, 0, t_ds.size - 2)
    # concrete arrays even when the first call is inside a JAX trace (the
    # pure functions under jax.jit): they are cached beyond the trace
    with jax.ensure_compile_time_eval():
        return NRSurData(
            **{name: jnp.asarray(value) for name, value in arrays.items()},
            ds_knots=factorize(t_ds),
            coorb_knots=factorize(t_coorb),
            coorb_index=jnp.asarray(index, dtype=jnp.int32),
            coorb_delx=jnp.asarray(t_coorb - t_ds[index]),
            **static,
        )


def precision(dtype=None) -> np.dtype:
    """The floating-point type of an evaluation for ``dtype``: float32 if it
    is float32, else (also for ``None``) float64, which is float32 where
    JAX's 64-bit floats are disabled."""
    dtype = np.dtype(dtype) if dtype is not None else None
    return jax.dtypes.canonicalize_dtype(np.float32 if dtype == np.float32 else np.float64)


@functools.lru_cache(maxsize=4)
def _load_as(path: str, dtype: str) -> NRSurData:
    data = _load_cached(path)
    if data.dtype == dtype:
        return data
    with jax.ensure_compile_time_eval():
        return data.astype(dtype)


def load(path=None, dtype=None) -> NRSurData:
    """The surrogate data of the file ``path`` (default: :func:`data_file`),
    read once per file, with floating-point arrays of ``dtype`` (float64 by
    default, or float32).

    Raises :class:`~waveformx.DataFileNotFoundError` if there is no file
    and :class:`~waveformx.DataFileError` if it cannot be read.
    """
    if dtype is not None and np.dtype(dtype) not in (np.float32, np.float64):
        raise TypeError(f"the NRSur7dq4 data are float32 or float64, not {np.dtype(dtype)}")
    if path is None:
        path = data_file()
        if path is None:
            raise DataFileNotFoundError(
                f"unable to find {DATA_FILE}: set NRSUR7DQ4_DATA to the file or LAL_DATA_PATH to its "
                "directory (the data are available at https://zenodo.org/records/14999310)"
            )
    path = Path(path)
    if not path.is_file():
        raise DataFileNotFoundError(f"the NRSur7dq4 data file {path} does not exist")
    return _load_as(str(path.resolve()), str(precision(dtype)))


def default_data(dtype=None) -> NRSurData:
    """The data of the file found by :func:`data_file` (float64, or
    ``dtype``)."""
    return load(None, dtype)
