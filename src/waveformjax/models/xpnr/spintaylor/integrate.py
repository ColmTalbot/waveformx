# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""A differentiable loop over a traced number of steps.

``jax.lax.scan`` needs a static length and runs all its iterations;
``jax.lax.while_loop`` stops early but cannot be differentiated in reverse
mode. :func:`fixed_steps` applies ``y_{i+1}, out_i = step(aux, y_i)`` for
``i < n`` with ``n`` traced, in a ``while_loop`` (so the cost is that of the
``n`` steps actually taken), and supplies the reverse-mode derivative with
``jax.custom_vjp``: the discrete adjoint, a second ``while_loop`` running
backwards through the stored states and applying ``jax.vjp`` of one step.
The derivatives are those of the discrete map, exactly (to rounding) what
``jax.grad`` of an unrolled loop would give.

Forward-mode differentiation (``jax.jvp``, ``jax.jacfwd``) of
:func:`fixed_steps` is not available (a limitation of ``custom_vjp``).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

__all__ = ["fixed_steps"]


def _layout(step, aux, y0):
    """Ravel/unravel functions for one stored row ``(y_i, out_i)``."""
    _, out = jax.eval_shape(step, aux, y0)
    out = jax.tree_util.tree_map(lambda o: jnp.zeros(o.shape, o.dtype), out)
    row, unravel = ravel_pytree((y0, out))
    return row.shape[0], row.dtype, unravel


def _forward(step, aux, y0, n, capacity):
    size, dtype, unravel = _layout(step, aux, y0)
    rows = jnp.zeros((capacity, size), dtype)
    n = jnp.minimum(n, capacity)
    # the parameters as one flat array: XLA carries every leaf of a
    # while_loop separately, which costs per iteration
    flat_aux, unravel_aux = ravel_pytree(aux)

    def cond(carry):
        return carry[0] < n

    def body(carry):
        i, y, rows = carry
        y_next, out = step(unravel_aux(flat_aux), y)
        row, _ = ravel_pytree((y, out))
        rows = jax.lax.dynamic_update_slice(rows, row[None].astype(dtype), (i, jnp.zeros_like(i)))
        return i + 1, y_next, rows

    _, y_last, rows = jax.lax.while_loop(cond, body, (jnp.zeros((), n.dtype), y0, rows))
    return rows, y_last, unravel


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 5))
def _fixed_steps_rows(step, vjp_step, aux, y0, n, capacity):
    rows, y_last, _ = _forward(step, aux, y0, n, capacity)
    return rows, y_last


def _fwd(step, vjp_step, aux, y0, n, capacity):
    rows, y_last, _ = _forward(step, aux, y0, n, capacity)
    return (rows, y_last), (aux, rows, n, y0)


def _bwd(step, vjp_step, capacity, residuals, cotangents):
    aux, rows, n, y0 = residuals
    ct_rows, ct_last = cotangents
    _, _, unravel = _layout(step, aux, y0)
    n = jnp.minimum(n, capacity)
    flat_aux, unravel_aux = ravel_pytree(aux)

    def flat_step(flat_aux, y):
        return vjp_step(unravel_aux(flat_aux), y)

    def cond(carry):
        return carry[0] >= 0

    def body(carry):
        i, lam, abar = carry
        y, _ = unravel(jax.lax.dynamic_index_in_dim(rows, i, keepdims=False))
        ct_y, ct_out = unravel(jax.lax.dynamic_index_in_dim(ct_rows, i, keepdims=False))
        _, vjp = jax.vjp(flat_step, flat_aux, y)
        da, dy = vjp((lam, ct_out))
        lam = jax.tree_util.tree_map(jnp.add, dy, ct_y)
        return i - 1, lam, abar + da

    _, lam, abar = jax.lax.while_loop(cond, body, (n - 1, ct_last, jnp.zeros_like(flat_aux)))
    abar = unravel_aux(abar)
    n_bar = np.zeros(jnp.shape(n), dtype=jax.dtypes.float0)
    return abar, lam, n_bar


_fixed_steps_rows.defvjp(_fwd, _bwd)


def fixed_steps(step, aux, y0, n, capacity, vjp_step=None):
    """``y_{i+1}, out_i = step(aux, y_i)`` for ``i = 0 .. n - 1``.

    ``step`` must be a pure function of the pytrees ``aux`` (parameters)
    and ``y`` (state) that does not close over traced values; ``n`` is a
    traced integer, ``capacity`` a Python int bounding it. Returns ``(ys,
    outs, y_n)``: the states before each step and the outputs, stacked along
    a leading axis of length ``capacity`` (zeros beyond ``n``), and the
    final state. ``vjp_step`` (default ``step``), the same map computed
    differently (e.g. with fewer copies of the equations), is the one
    differentiated.
    """
    n = jnp.asarray(n)
    rows, y_last = _fixed_steps_rows(step, step if vjp_step is None else vjp_step, aux, y0, n, capacity)
    _, _, unravel = _layout(step, aux, y0)
    ys, outs = jax.vmap(unravel)(rows)
    return ys, outs, y_last
