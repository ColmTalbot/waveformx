# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The NRSur7dq4 data as float32 MLX arrays.

The data are read by the JAX implementation (:func:`.._data.load`, in
float64) and converted once. What the JAX implementation computes with
``lax.scan``, which MLX does not have, is prepared here in float64 as
fixed linear operators:

* the natural cubic spline of the dynamics states (on ``t_ds``) evaluated
  at the coorbital times is a fixed linear map of the states,
  ``ds_to_coorb`` (``(n_coorb, n_ds)``), which replaces the two
  substitution sweeps and the evaluation of :mod:`.._spline`;
* the spline through values on ``t_coorb``, evaluated at times that depend
  on the binary, needs its second derivatives: the inverse of the
  tridiagonal system (``coorb_system_inverse``) maps the right-hand side to
  them, again in place of the two sweeps.

The Adams-Bashforth weights of the dynamics (:func:`.._dynamics.ab4_tables`)
are computed in float64 too.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import mlx.core as mx
import numpy as np

from .. import _data as _jax_data
from .._dynamics import ab4_tables

__all__ = ["NRSurData", "load", "default_data", "spline_system", "spline_matrix"]


@dataclasses.dataclass(frozen=True)
class NRSurData:
    """The surrogate data as float32 (and int32) MLX arrays.

    * ``t_ds``: the dynamics nodes, without the half nodes;
    * ``monomials``: ``(n_monomials, 5)`` the monomials of the dynamics fits
      as indices into the table of powers (see :mod:`.._fits`);
    * ``dynamics``: ``(n_ds + 3, 9, n_monomials)`` the fits of every
      dynamics node, the three half nodes last;
    * ``omega_fits``: ``(n_monomials, n_ds)`` the orbital-frequency fits of
      the nodes (``dynamics[:n_ds, 0, :]`` transposed);
    * ``ab4_forward``, ``ab4_backward``: ``(n_ds, 4)`` the Adams-Bashforth
      weights of the steps to every node;
    * ``ds_to_coorb``: ``(n_coorb, n_ds)`` the natural cubic spline on
      ``t_ds`` evaluated at ``t_coorb``, as a matrix;
    * ``ei_time``, ``ei_coefs``, ``ei_basis``: the fits at the empirical
      nodes and the empirical-interpolation basis (as in :mod:`.._data`);
    * ``coorb_x``, ``coorb_h``, ``coorb_h_inv``: ``t_coorb``, its steps and
      their inverses;
    * ``coorb_system_inverse``: ``(n_coorb - 2, n_coorb - 2)`` the inverse of
      the tridiagonal system of the natural spline on ``t_coorb``;
    * ``ei_bounds``, ``ei_spins``: static, as in :mod:`.._data`.
    """

    t_ds: mx.array
    monomials: mx.array
    dynamics: mx.array
    omega_fits: mx.array
    ab4_forward: mx.array
    ab4_backward: mx.array
    ds_to_coorb: mx.array
    ei_time: mx.array
    ei_coefs: mx.array
    ei_basis: mx.array
    coorb_x: mx.array
    coorb_h: mx.array
    coorb_h_inv: mx.array
    coorb_system_inverse: mx.array
    ei_bounds: Tuple[int, ...]
    ei_spins: Tuple[Tuple[int, ...], ...]

    @property
    def n_ds(self) -> int:
        """Number of dynamics nodes (without the half nodes)."""
        return int(self.t_ds.shape[0])

    @property
    def n_coorb(self) -> int:
        """Number of coorbital times."""
        return int(self.coorb_x.shape[0])


def spline_system(x) -> Tuple[np.ndarray, np.ndarray]:
    """The system ``T c = D y`` of GSL's natural cubic spline on the knots
    ``x`` (float64): ``T`` ``(n - 2, n - 2)`` tridiagonal, with diagonal
    ``2 (h_i + h_{i+1})`` and off-diagonal ``h_{i+1}``, and ``D`` ``(n - 2,
    n)`` the right-hand side ``3 (dy_{i+1} / h_{i+1} - dy_i / h_i)``; ``c``
    are the inner second-derivative coefficients (``c_0 = c_{n-1} = 0``)."""
    x = np.asarray(x, dtype=float)
    h = np.diff(x)
    n = x.size
    system = np.diag(2.0 * (h[:-1] + h[1:])) + np.diag(h[1:-1], 1) + np.diag(h[1:-1], -1)
    rows = np.arange(n - 2)
    rhs = np.zeros((n - 2, n))
    rhs[rows, rows] = 3.0 / h[:-1]
    rhs[rows, rows + 1] = -3.0 / h[:-1] - 3.0 / h[1:]
    rhs[rows, rows + 2] = 3.0 / h[1:]
    return system, rhs


def spline_matrix(x, index, delx) -> np.ndarray:
    """The natural cubic spline on the knots ``x`` evaluated in the
    intervals ``index`` at the offsets ``delx`` from their left knots, as
    an ``(m, n)`` matrix applied to the values at the knots (float64; the
    formulas of ``cspline_eval``)."""
    x = np.asarray(x, dtype=float)
    index = np.asarray(index)
    delx = np.asarray(delx, dtype=float)[:, None]
    n = x.size
    system, rhs = spline_system(x)
    c = np.zeros((n, n))
    c[1:-1] = np.linalg.solve(system, rhs)
    y = np.eye(n)
    dx = np.diff(x)[index][:, None]
    y_i, y_ip1 = y[index], y[index + 1]
    c_i, c_ip1 = c[index], c[index + 1]
    b = (y_ip1 - y_i) / dx - dx * (c_ip1 + 2.0 * c_i) / 3.0
    d = (c_ip1 - c_i) / (3.0 * dx)
    return y_i + delx * (b + delx * (c_i + delx * d))


def _f32(value) -> mx.array:
    return mx.array(np.asarray(value, dtype=np.float32))


def _i32(value) -> mx.array:
    return mx.array(np.asarray(value, dtype=np.int32))


def _convert(data: _jax_data.NRSurData) -> NRSurData:
    """The float32 MLX data from the float64 JAX data."""
    t_ds = np.asarray(data.t_ds)
    t_coorb = np.asarray(data.t_coorb)
    dynamics = np.asarray(data.dynamics)
    n_ds = t_ds.size
    forward, backward = (np.asarray(table) for table in ab4_tables(t_ds))
    system, _ = spline_system(t_coorb)
    h = np.diff(t_coorb)
    return NRSurData(
        t_ds=_f32(t_ds),
        monomials=_i32(data.monomials),
        dynamics=_f32(dynamics),
        omega_fits=_f32(dynamics[:n_ds, 0, :].T),
        ab4_forward=_f32(forward),
        ab4_backward=_f32(backward),
        ds_to_coorb=_f32(spline_matrix(t_ds, np.asarray(data.coorb_index), np.asarray(data.coorb_delx))),
        ei_time=_i32(data.ei_time),
        ei_coefs=_f32(data.ei_coefs),
        ei_basis=_f32(data.ei_basis),
        coorb_x=_f32(t_coorb),
        coorb_h=_f32(h),
        coorb_h_inv=_f32(1.0 / h),
        coorb_system_inverse=_f32(np.linalg.inv(system)),
        ei_bounds=data.ei_bounds,
        ei_spins=data.ei_spins,
    )


#: The converted data by the identity of the (cached) JAX data, which are
#: kept alive with them.
_CONVERTED = {}


def load(path=None) -> NRSurData:
    """The float32 MLX data of the file ``path`` (default: the file found by
    :func:`waveformjax.nrsur7dq4_data_file`), converted once per file.
    Raises as :func:`waveformjax.models.nrsur7dq4.load`."""
    # the JAX loader resolves the path, reads the file once and caches it
    data = _jax_data.load(path, np.float64)
    if id(data) not in _CONVERTED:
        _CONVERTED[id(data)] = (data, _convert(data))
    return _CONVERTED[id(data)][1]


def default_data() -> NRSurData:
    """The data of the default file."""
    return load(None)
