# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""A loop over a traced number of steps.

``jax.lax.scan`` needs a static length and runs all its iterations;
``jax.lax.while_loop`` stops early. :func:`fixed_steps` applies ``y_{i+1},
out_i = step(aux, y_i)`` for ``i < n`` with ``n`` traced, in a
``while_loop`` (so the cost is that of the ``n`` steps actually taken),
recording the states and outputs. It is differentiable in forward mode
(``jax.jvp``, ``jax.jacfwd``): the derivatives are those of the discrete
map. The model's reverse-mode derivatives come from its forward-mode
Jacobian (see :func:`..model.polarizations`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .._math import isolated
from .._transforms import while_loop

__all__ = ["fixed_steps"]


def _layout(step, aux, y0):
    """Ravel/unravel functions for one stored row ``(y_i, out_i)``."""
    _, out = jax.eval_shape(step, aux, y0)
    out = jax.tree_util.tree_map(lambda o: jnp.zeros(o.shape, o.dtype), out)
    row, unravel = ravel_pytree((y0, out))
    return row.shape[0], row.dtype, unravel


def fixed_steps(step, aux, y0, n, capacity):
    """``y_{i+1}, out_i = step(aux, y_i)`` for ``i = 0 .. n - 1``.

    ``step`` is a pure function of the pytrees ``aux`` (parameters) and
    ``y`` (state); ``n`` is a traced integer, ``capacity`` a Python int
    bounding it. Returns ``(ys, outs, y_n)``: the states before each step
    and the outputs, stacked along a leading axis of length ``capacity``
    (zeros beyond ``n``), and the final state.
    """
    n = jnp.minimum(jnp.asarray(n), capacity)
    size, dtype, unravel = _layout(step, aux, y0)
    rows = jnp.zeros((capacity, size), dtype)

    def advance(aux, y):
        y_next, out = step(aux, y)
        row, _ = ravel_pytree((y, out))
        return y_next, row.astype(dtype)

    def test(consts, small):
        return small[0] < consts[1]

    def body(consts, small, rows, active):
        aux, _ = consts
        i, y = small
        # the step in a loop of its own, so that its many small operations
        # run on the calling thread (see _math.isolated)
        y_next, row = isolated(advance, aux, y)
        # (inactive under jax.vmap: the row written back unchanged)
        row = jnp.where(active, row, rows[i])
        rows = jax.lax.dynamic_update_slice(rows, row[None], (i, jnp.zeros_like(i)))
        return (i + 1, y_next), rows

    (_, y_last), rows = while_loop(test, body, (aux, n), (jnp.zeros((), n.dtype), y0), rows)
    ys, outs = jax.vmap(unravel)(rows)
    return ys, outs, y_last
