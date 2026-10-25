# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""How the IMRPhenomXPNR functions of a binary behave under ``jax.vmap``
and automatic differentiation.

The model is made of loops whose numbers of iterations depend on the binary
(``jax.lax.while_loop``) and of choices among prescriptions. Loops with a
traced number of iterations can be differentiated in forward mode but not
in reverse mode, and reverse mode through the many scalar set-up
computations of the model compiles slowly. So the derivatives are taken in
forward mode only:

* :func:`differentiable` makes ``f(binary, frequencies)`` a
  ``jax.custom_jvp`` function whose rule computes the Jacobian with respect
  to the parameters of the binary with ``jax.jvp`` along each of them (only
  those with non-zero tangents) and contracts it with the tangents. That
  contraction is linear, so ``jax.grad`` (which transposes it) works as well
  as ``jax.jvp`` and ``jax.jacfwd``. Tangents of the frequencies are
  propagated in forward mode only.
* the Jacobian is a ``jax.custom_batching.custom_vmap`` function, so that
  under ``jax.vmap`` of a derivative the binaries are batched inside the
  forward-mode differentiation (the batched model is then differentiated,
  rather than the derivative of a single binary batched).

Under ``jax.vmap`` the model is ``jax.vmap`` of the single binary (or,
with ``block``, a ``jax.lax.map`` over blocks of that many binaries, each
block vectorised: the memory is that of a block rather than of the whole
batch, a Jacobian's tangents included), with its choices made by
:func:`switch` and :func:`cond`, which compute a branch for
the whole batch only if some binary of the batch takes it (``jax.lax.switch``
and ``jax.lax.cond`` with a batched predicate compute every branch). Its loops
(:func:`while_loop`) run to the longest of the batch, keeping their large
buffers out of the per-iteration selection of ``jax.lax.while_loop``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from ._math import precision_of

__all__ = ["cond", "differentiable", "switch", "while_loop"]


def _leaves(binary, dtype):
    leaves, treedef = jax.tree_util.tree_flatten(binary)
    return [jnp.asarray(leaf, dtype) for leaf in leaves], treedef


def _vmappable(single, batched):
    """``single(binary, frequencies)`` with ``batched(binaries, frequencies,
    frequencies_batched)`` (the binaries with a leading axis) as its rule
    under ``jax.vmap``."""
    function = jax.custom_batching.custom_vmap(single)

    @function.def_vmap
    def rule(axis_size, in_batched, binary, frequencies):
        binary_batched, frequencies_batched = in_batched
        flags = jax.tree_util.tree_leaves(binary_batched)
        if not any(flags) and not frequencies_batched:
            out = single(binary, frequencies)
            return out, jax.tree_util.tree_map(lambda _: False, out)
        leaves, treedef = jax.tree_util.tree_flatten(binary)
        leaves = [x if b else jnp.broadcast_to(jnp.asarray(x), (axis_size,) + jnp.shape(x)) for x, b in zip(leaves, flags)]
        out = batched(jax.tree_util.tree_unflatten(treedef, leaves), frequencies, frequencies_batched)
        return out, jax.tree_util.tree_map(lambda _: True, out)

    return function


def _parameters(leaves, active, batched=False):
    """The leaves flagged in ``active`` raveled into one vector of
    parameters (per binary, with ``batched``), and what puts a vector back
    in their place."""
    chosen = [leaf for leaf, a in zip(leaves, active) if a]
    if batched:
        params = jnp.concatenate([leaf.reshape(leaf.shape[0], -1) for leaf in chosen], axis=-1)
    else:
        params = jnp.concatenate([leaf.reshape(-1) for leaf in chosen])

    def rebuild(params):
        out, start = [], 0
        for leaf, a in zip(leaves, active):
            if a:
                size = leaf[0].size if batched else leaf.size
                out.append(params[..., start:start + size].reshape(leaf.shape))
                start += size
            else:
                out.append(leaf)
        return out

    return params, rebuild


def _jacobian_function(single, batched, active):
    """``(f, J)``: ``f(binary, frequencies)`` and its Jacobian with respect to
    the leaves of the binary flagged in ``active`` (raveled into a trailing
    axis of the outputs), from ``jax.jvp`` along each parameter."""

    def single_jacobian(binary, frequencies):
        leaves, treedef = _leaves(binary, precision_of(binary))
        params, rebuild = _parameters(leaves, active)

        def along(direction):
            return jax.jvp(lambda p: single(jax.tree_util.tree_unflatten(treedef, rebuild(p)), frequencies),
                           (params,), (direction,))

        return jax.vmap(along, out_axes=(None, -1))(jnp.eye(params.shape[-1], dtype=params.dtype))

    def batched_jacobian(binaries, frequencies, frequencies_batched):
        leaves, treedef = _leaves(binaries, precision_of(binaries))
        params, rebuild = _parameters(leaves, active, batched=True)

        def along(direction):
            # the same direction for every binary: each output depends on
            # its own binary only
            tangent = jnp.broadcast_to(direction, params.shape)
            return jax.jvp(
                lambda p: batched(jax.tree_util.tree_unflatten(treedef, rebuild(p)), frequencies, frequencies_batched),
                (params,), (tangent,))

        return jax.vmap(along, out_axes=(None, -1))(jnp.eye(params.shape[-1], dtype=params.dtype))

    return _vmappable(single_jacobian, batched_jacobian)


def _vmapped(function, in_batched):
    """``function`` mapped over the leading axis of the batched leaves of
    its arguments, with every output batched."""
    if not any(jax.tree_util.tree_leaves(in_batched)):
        return function
    in_axes = jax.tree_util.tree_map(lambda b: 0 if b else None, in_batched)
    return jax.vmap(function, in_axes=tuple(in_axes), out_axes=0)


def _split_static(tree):
    """The leaves of ``tree`` that are JAX arrays, and what rebuilds the
    tree from them with the other leaves (Python values) unchanged."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    dynamic = [isinstance(leaf, jax.Array) for leaf in leaves]
    static = [None if d else leaf for leaf, d in zip(leaves, dynamic)]

    def rebuild(values):
        values = iter(values)
        return jax.tree_util.tree_unflatten(treedef, [next(values) if d else s for s, d in zip(static, dynamic)])

    return [leaf for leaf, d in zip(leaves, dynamic) if d], rebuild


def _traced(function, *args):
    """``function`` (of the pytrees ``args``) under ``jax.jit``, traced once
    (``jax.jit`` caches the trace), and the shapes of its outputs: the
    batching rules below map the traced function rather than tracing the
    Python code again (which nested rules would multiply)."""
    jitted = jax.jit(function)
    return jitted, jax.eval_shape(jitted, *args)


def _broadcast(tree, batched, axis_size):
    return [x if b else jnp.broadcast_to(x, (axis_size,) + jnp.shape(x)) for x, b in zip(tree, batched)]


def switch(index, branches, operands):
    """``jax.lax.switch(index, branches, operands)`` which, under
    ``jax.vmap`` with a batched ``index``, computes each branch for the
    whole batch only if some element of the batch takes it (where several
    are taken, their results are selected element by element). The
    branches must not close over batched values: pass them in
    ``operands`` (whose leaves that are not JAX arrays stay Python values)."""
    operands, rebuild = _split_static(operands)
    traced = [_traced(lambda values, branch=branch: branch(rebuild(values)), operands) for branch in branches]
    branches = [replay for replay, _ in traced]
    shapes = traced[0][1]

    def plain(index, operands):
        return jax.lax.switch(index, branches, operands)

    function = jax.custom_batching.custom_vmap(plain)

    @function.def_vmap
    def rule(axis_size, in_batched, index, operands):
        index_batched, operands_batched = in_batched
        if not index_batched:
            if not any(operands_batched):
                out = plain(index, operands)
                return out, jax.tree_util.tree_map(lambda _: False, out)
            out = jax.lax.switch(index, [_vmapped(b, (operands_batched,)) for b in branches], operands)
            return out, jax.tree_util.tree_map(lambda _: True, out)
        if not any(operands_batched):
            operands = _broadcast(operands, operands_batched, axis_size)
            operands_batched = [True] * len(operands)
        # each branch taken by some binary merges its values into the
        # output (one set of zeros to start from, and no zeros of their
        # own for the branches not taken: the output passes through)
        out = jax.tree_util.tree_map(lambda s: jnp.zeros((axis_size,) + s.shape, s.dtype), shapes)
        for i, branch in enumerate(branches):
            def merge(args, branch=branch):
                out, operands, taken = args
                value = _vmapped(branch, (operands_batched,))(operands)
                return jax.tree_util.tree_map(
                    lambda new, old: jnp.where(taken.reshape((-1,) + (1,) * (new.ndim - 1)), new, old), value, out)

            taken = index == i
            out = jax.lax.cond(jnp.any(taken), merge, lambda args: args[0], (out, operands, taken))
        return out, jax.tree_util.tree_map(lambda _: True, out)

    return function(jnp.asarray(index, jnp.int32), operands)


def cond(pred, true_fn, false_fn, operands):
    """``jax.lax.cond(pred, true_fn, false_fn, operands)`` with the batching
    of :func:`switch`."""
    return switch(jnp.asarray(pred).astype(jnp.int32), [false_fn, true_fn], operands)


def while_loop(cond_fun, body_fun, consts, small, large=()):
    """``jax.lax.while_loop`` over the carry ``(small, large)``:
    ``cond_fun(consts, small)`` and ``body_fun(consts, small, large,
    active) -> (small, large)``, with ``active`` true.

    Under ``jax.vmap`` with a batched predicate, ``jax.lax.while_loop``
    runs while any element of the batch is active and keeps the carry of
    the others by selecting all of it at every iteration, which costs the
    size of the carry. Here the loop also runs while any element is active,
    but only the ``small`` carry is selected: ``body_fun`` gets the active
    elements (``active``) and must leave the ``large`` carry of the others
    unchanged itself (e.g. writing back what a dynamic update would
    overwrite). The functions must not close over batched values: pass them
    in ``consts``."""
    consts, rebuild_consts = _split_static(consts)
    small, rebuild_small = _split_static(small)
    large, rebuild_large = _split_static(large)
    n_small = len(small)

    def flat_cond(consts, carry):
        return cond_fun(rebuild_consts(consts), rebuild_small(carry[:n_small]))

    def flat_body(consts, carry, active):
        s, l = body_fun(rebuild_consts(consts), rebuild_small(carry[:n_small]), rebuild_large(carry[n_small:]), active)
        return _split_static(s)[0] + _split_static(l)[0]

    carry = small + large
    test, _ = _traced(flat_cond, consts, carry)
    step, _ = _traced(flat_body, consts, carry, jnp.asarray(True))

    def plain(consts, carry):
        return jax.lax.while_loop(lambda c: test(consts, c), lambda c: step(consts, c, True), carry)

    function = jax.custom_batching.custom_vmap(plain)

    @function.def_vmap
    def rule(axis_size, in_batched, consts, carry):
        consts_batched, carry_batched = in_batched
        if not any(consts_batched) and not any(carry_batched):
            out = plain(consts, carry)
            return out, [False] * len(out)
        carry = _broadcast(carry, carry_batched, axis_size)
        consts_axes = [0 if b else None for b in consts_batched]
        tests = jax.vmap(test, in_axes=(consts_axes, 0))
        steps = jax.vmap(step, in_axes=(consts_axes, 0, 0))

        def body(carry):
            active = tests(consts, carry)
            new = steps(consts, carry, active)
            return [jnp.where(active.reshape((-1,) + (1,) * (x.ndim - 1)), x, old) if k < n_small else x
                    for k, (x, old) in enumerate(zip(new, carry))]

        out = jax.lax.while_loop(lambda c: jnp.any(tests(consts, c)), body, carry)
        return out, [True] * len(out)

    out = function(consts, carry)
    return rebuild_small(out[:n_small]), rebuild_large(out[n_small:])


def _blocked(single, block):
    """``single`` under ``jax.vmap`` as ``batched(binaries, frequencies,
    frequencies_batched)``: ``jax.vmap`` over the binaries, or with
    ``block`` (static) ``jax.lax.map`` over blocks of ``block`` of them."""
    if block is None:
        def batched(binaries, frequencies, frequencies_batched=False):
            return jax.vmap(single, in_axes=(0, 0 if frequencies_batched else None))(binaries, frequencies)
    else:
        def batched(binaries, frequencies, frequencies_batched=False):
            if frequencies_batched:
                return jax.lax.map(lambda args: single(*args), (binaries, frequencies), batch_size=block)
            return jax.lax.map(lambda binary: single(binary, frequencies), binaries, batch_size=block)
    return batched


def differentiable(single, batched=None, block=None, jacobian_block=None):
    """``f(binary, frequencies)`` computing ``single(binary, frequencies)``
    (``batched`` under ``jax.vmap``, default ``jax.vmap(single)``), with
    derivatives from forward-mode Jacobians (see the module docstring).
    With ``block`` (static), the binaries of a ``jax.vmap`` are computed
    ``block`` at a time, and those of the Jacobian ``jacobian_block`` (by
    default ``block``) at a time."""
    primal = single
    if batched is None:
        batched = _blocked(single, block)
        if block is not None:
            primal = _vmappable(single, batched)
            batched = _blocked(single, block if jacobian_block is None else jacobian_block)
    jacobians = {}

    @jax.custom_jvp
    def function(binary, frequencies):
        return primal(binary, frequencies)

    def rule(primals, tangents):
        binary, frequencies = primals
        t_binary, t_frequencies = tangents
        t_leaves = jax.tree_util.tree_leaves(t_binary, is_leaf=lambda t: isinstance(t, SymbolicZero))
        active = tuple(not isinstance(t, SymbolicZero) for t in t_leaves)
        if any(active):
            if active not in jacobians:
                jacobians[active] = _jacobian_function(single, batched, active)
            out, jac = jacobians[active](binary, frequencies)
            direction = jnp.concatenate([jnp.asarray(t, precision_of(binary)).reshape(-1)
                                         for t, a in zip(t_leaves, active) if a])
            t_out = jax.tree_util.tree_map(lambda j: j @ direction.astype(j.dtype), jac)
        else:
            out = single(binary, frequencies)
            t_out = jax.tree_util.tree_map(jnp.zeros_like, out)
        if not isinstance(t_frequencies, SymbolicZero):
            _, t_freq = jax.jvp(lambda f: single(binary, f), (frequencies,), (t_frequencies,))
            t_out = jax.tree_util.tree_map(jnp.add, t_out, t_freq)
        return out, t_out

    function.defjvp(rule, symbolic_zeros=True)

    def call(binary, frequencies):
        # (float32 for a binary of float32 parameters, see _math.precision)
        dtype = precision_of(binary)
        leaves, treedef = _leaves(binary, dtype)
        return function(jax.tree_util.tree_unflatten(treedef, leaves), jnp.asarray(frequencies, dtype))

    return call
