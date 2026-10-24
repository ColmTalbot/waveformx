# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Stages of the lean model (:mod:`._stages`) as generated Metal kernels.

A :class:`StageKernel` traces a JAX function of pytrees ``f(*rows,
*points)`` (``rows``: the binary's values; ``points``: the values at one
point) in float32, lowers it and its forward-mode derivative (``jax.jvp``)
to straight-line programs (:mod:`._program`) and runs them as Metal
kernels (:mod:`._metal`) over a batch of binaries and, for a function of a
point, over the points of each binary:

* :meth:`StageKernel.values`: the outputs, one thread per binary and point;
* :meth:`StageKernel.tangents`: the outputs and their derivatives along
  ``P`` directions given the derivatives of the inputs, one thread per
  binary, point and direction (each thread computes the values as well:
  recomputing is cheaper on the GPU than keeping them).

The leaves of the pytrees go in and out of the kernels packed into one
float array per kind (values of the binary, values at the points, and
their tangents), the booleans and integers as floats. Leaves are MLX
arrays with leading axes ``(B,)`` (the binary's values) or ``(B, N)`` (the
points' values, or ``(1, N)`` when the batch shares them), followed by the
leaf's shape; tangents have a trailing axis of the ``P`` directions.
"""

from __future__ import annotations

import itertools

import jax
import mlx.core as mx
import numpy as np

from .._math import lean as lean_mode
from .._math import precision
from . import _metal
from ._program import BOOL, FLOAT, INT, Complex, kind_of, lower
from ._stages import combine, partition

__all__ = ["StageKernel", "pack", "unpack"]

_COUNTER = itertools.count()
_MX_KIND = {FLOAT: mx.float32, INT: mx.int32, BOOL: mx.bool_}


def _size(shape):
    return int(np.prod(shape)) if shape else 1


def pack(leaves, shapes, prefix):
    """The leaves (MLX arrays of shape ``prefix + shape``) as one float32
    array ``prefix + (sum of sizes,)``."""
    if not leaves:
        return mx.zeros(prefix + (0,), dtype=mx.float32)
    parts = []
    for leaf, shape in zip(leaves, shapes):
        leaf = mx.array(leaf) if not isinstance(leaf, mx.array) else leaf
        leaf = mx.broadcast_to(leaf, prefix + tuple(shape))
        parts.append(leaf.reshape(prefix + (_size(shape),)).astype(mx.float32))
    return mx.concatenate(parts, axis=-1)


def pack_tangents(tangents, shapes, prefix, n_directions):
    """Tangents (``prefix + shape + (P,)``, or ``None`` for zero) as one
    array ``prefix + (sum of sizes, P)``."""
    if not tangents:
        return mx.zeros(prefix + (0, n_directions), dtype=mx.float32)
    parts = []
    for tangent, shape in zip(tangents, shapes):
        if tangent is None:
            parts.append(mx.zeros(prefix + (_size(shape), n_directions), dtype=mx.float32))
        else:
            tangent = mx.broadcast_to(tangent, prefix + tuple(shape) + (n_directions,))
            parts.append(tangent.reshape(prefix + (_size(shape), n_directions)))
    return mx.concatenate(parts, axis=-2)


def unpack(packed, shapes, kinds, prefix):
    """The inverse of :func:`pack` (with the leaves' kinds)."""
    out, start = [], 0
    for shape, kind in zip(shapes, kinds):
        n = _size(shape)
        leaf = packed[..., start:start + n].reshape(prefix + tuple(shape))
        if kind != FLOAT:
            leaf = leaf.astype(_MX_KIND[kind])
        out.append(leaf)
        start += n
    return out


def unpack_tangents(packed, shapes, prefix, n_directions):
    out, start = [], 0
    for shape in shapes:
        n = _size(shape)
        out.append(packed[..., start:start + n, :].reshape(prefix + tuple(shape) + (n_directions,)))
        start += n
    return out


class StageKernel:
    """``function(*rows, *points)`` (a JAX function of pytrees, traced at
    ``examples``, the first ``n_rows`` of which are the binary's values)
    as Metal kernels. ``name`` names the kernels."""

    def __init__(self, function, examples, n_rows, name, tangents=True):
        self.name = f"{name}_{next(_COUNTER)}"
        parts = [partition(example) for example in examples]
        self.metas = [meta for _, meta in parts]
        flat_examples = [v for values, _ in parts for v in values]
        self.n_leaves = [len(values) for values, _ in parts]
        self.n_row_leaves = sum(self.n_leaves[:n_rows])
        holder = {}

        def rebuild(flat):
            it = iter(flat)
            return [combine([next(it) for _ in range(n)], meta) for n, meta in zip(self.n_leaves, self.metas)]

        def flat_function(*flat):
            with precision(np.float32), lean_mode(True):
                out, holder["meta"] = partition(function(*rebuild(flat)))
            return out

        avals = [jax.ShapeDtypeStruct(tuple(v.shape), np.dtype(v.dtype), weak_type=getattr(v, "weak_type", False))
                 for v in flat_examples]
        closed = jax.make_jaxpr(flat_function)(*avals)
        self.out_meta = holder["meta"]
        self.out_avals = [jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype, weak_type=v.aval.weak_type)
                          for v in closed.jaxpr.outvars]
        self.in_shapes = [tuple(a.shape) for a in avals]
        self.in_kinds = [kind_of(a.dtype) for a in avals]
        self.out_shapes = [tuple(v.aval.shape) for v in closed.jaxpr.outvars]
        self.out_kinds = [kind_of(v.aval.dtype) for v in closed.jaxpr.outvars]
        self.program = lower(closed)
        self.float_in = [i for i, k in enumerate(self.in_kinds) if k == FLOAT]
        self.float_out = [i for i, k in enumerate(self.out_kinds) if k == FLOAT]
        self.jvp_program = None
        if tangents:
            self.jvp_program = self._lower_jvp(flat_function, avals)
        self._kernels = {}

    def out_example(self):
        """The outputs as a pytree of shapes (an example for the next
        stage)."""
        return combine(self.out_avals, self.out_meta)

    def _lower_jvp(self, flat_function, avals):
        float_in = self.float_in

        def jvp_function(*args):
            primals = list(args[:len(avals)])
            tangent_floats = list(args[len(avals):])
            tangents = []
            it = iter(tangent_floats)
            for i, a in enumerate(avals):
                if i in float_in:
                    tangents.append(next(it))
                else:
                    tangents.append(np.zeros(a.shape, dtype=jax.dtypes.float0))
            out, out_tangents = jax.jvp(flat_function, primals, tangents)
            return list(out) + [out_tangents[i] for i in self.float_out]

        tangent_avals = [avals[i] for i in float_in]
        closed = jax.make_jaxpr(jvp_function)(*avals, *tangent_avals)
        return lower(closed)

    # --- kernel sources --------------------------------------------------------

    def _layout(self):
        rows = list(range(self.n_row_leaves))
        points = list(range(self.n_row_leaves, len(self.in_shapes)))
        sizes = [_size(s) for s in self.in_shapes]
        offset = {}
        start = 0
        for i in rows:
            offset[i] = start
            start += sizes[i]
        n_row_in = start
        start = 0
        for i in points:
            offset[i] = start
            start += sizes[i]
        n_point_in = start
        # tangents: the float leaves, rows then points
        t_offset = {}
        start = 0
        for i in rows:
            if i in self.float_in:
                t_offset[i] = start
                start += sizes[i]
        n_row_t = start
        start = 0
        for i in points:
            if i in self.float_in:
                t_offset[i] = start
                start += sizes[i]
        n_point_t = start
        out_sizes = [_size(s) for s in self.out_shapes]
        out_offset = list(np.concatenate([[0], np.cumsum(out_sizes)]).astype(int))
        t_out_offset = {}
        start = 0
        for i in self.float_out:
            t_out_offset[i] = start
            start += out_sizes[i]
        return dict(offset=offset, n_row_in=n_row_in, n_point_in=n_point_in, t_offset=t_offset, n_row_t=n_row_t,
                    n_point_t=n_point_t, out_offset=out_offset, n_out=out_offset[-1], t_out_offset=t_out_offset,
                    n_out_t=start)

    def _source(self, tangent):
        lay = self._layout()
        n_row = self.n_row_leaves
        program = self.jvp_program if tangent else self.program
        n_primal = len(self.in_shapes)
        float_in = self.float_in

        def input_name(index, k, kind):
            if index < n_primal:
                if index < n_row:
                    value = f"xr[row_in + {lay['offset'][index] + k}]"
                else:
                    value = f"xp[point_in + {lay['offset'][index] + k}]"
            else:
                leaf = float_in[index - n_primal]
                if leaf < n_row:
                    value = f"tr[(row_t + {lay['t_offset'][leaf] + k}) * n_dir + dir]"
                else:
                    value = f"tp[(point_t + {lay['t_offset'][leaf] + k}) * n_dir + dir]"
            return value if kind == FLOAT else f"static_cast<{_metal._TYPES[kind]}>({value})"

        lines, name_of = _metal.body_lines(program, input_name)
        n_out_leaves = len(self.out_shapes)
        writes = []
        for i in range(n_out_leaves):
            for k, x in enumerate(program.outputs[i].reshape(-1)):
                if isinstance(x, Complex):
                    raise NotImplementedError("complex outputs")
                target = f"y[out + {lay['out_offset'][i] + k}]"
                writes.append(f"{target} = static_cast<float>({name_of(x)});")
        t_writes = []
        if tangent:
            for j, i in enumerate(self.float_out):
                for k, x in enumerate(program.outputs[n_out_leaves + j].reshape(-1)):
                    target = f"dy[(out_t + {lay['t_out_offset'][i] + k}) * n_dir + dir]"
                    t_writes.append(f"{target} = static_cast<float>({name_of(x)});")
        head = [
            "uint tid = thread_position_in_grid.x;",
            "uint n_points = dims[0];",
            "uint n_rows = dims[1];",
            "uint n_dir = dims[2];",
            "uint shared = dims[3];",
            "uint total = n_rows * n_points * (n_dir > 0 ? n_dir : 1);",
            "if (tid >= total) return;",
            "uint dir = n_dir > 0 ? tid % n_dir : 0;",
            "uint pr = n_dir > 0 ? tid / n_dir : tid;",
            "uint b = pr / n_points;",
            "uint point = pr % n_points;",
            f"size_t row_in = (size_t)b * {lay['n_row_in']};",
            f"size_t point_in = ((size_t)(shared ? 0 : b) * n_points + point) * {lay['n_point_in']};",
            f"size_t out = (size_t)pr * {lay['n_out']};",
        ]
        if tangent:
            head += [
                f"size_t row_t = (size_t)b * {lay['n_row_t']};",
                f"size_t point_t = ((size_t)(shared ? 0 : b) * n_points + point) * {lay['n_point_t']};",
                f"size_t out_t = (size_t)pr * {lay['n_out_t']};",
            ]
        body = head + lines
        if tangent:
            body += ["if (dir == 0) {"] + writes + ["}"] + t_writes
        else:
            body += writes
        return "\n".join(body), lay

    def _kernel(self, tangent):
        if tangent not in self._kernels:
            source, lay = self._source(tangent)
            inputs = ["xr", "xp", "dims"] + (["tr", "tp"] if tangent else [])
            outputs = ["y"] + (["dy"] if tangent else [])
            kernel = mx.fast.metal_kernel(name=f"{self.name}_{'jvp' if tangent else 'f'}", input_names=inputs,
                                          output_names=outputs, source=source, header=_metal.HEADER)
            self._kernels[tangent] = (kernel, lay)
        return self._kernels[tangent]

    # --- evaluation ------------------------------------------------------------

    def _pack_inputs(self, leaves, batch, n_points, shared):
        n_row = self.n_row_leaves
        xr = pack(leaves[:n_row], self.in_shapes[:n_row], (batch,))
        point_prefix = (1 if shared else batch, n_points)
        xp = pack(leaves[n_row:], self.in_shapes[n_row:], point_prefix)
        return xr, xp

    def _call(self, leaves, tangents, batch, n_points, shared, n_directions):
        tangent = tangents is not None
        kernel, lay = self._kernel(tangent)
        xr, xp = self._pack_inputs(leaves, batch, n_points, shared)
        dims = mx.array([n_points, batch, n_directions if tangent else 0, 1 if shared else 0], dtype=mx.uint32)
        inputs = [xr, xp, dims]
        output_shapes = [(batch * n_points * lay["n_out"],)]
        output_dtypes = [mx.float32]
        if tangent:
            n_row = self.n_row_leaves
            row_t = [tangents[i] for i in range(n_row) if i in self.float_in]
            row_shapes = [self.in_shapes[i] for i in range(n_row) if i in self.float_in]
            point_t = [tangents[i] for i in range(n_row, len(self.in_shapes)) if i in self.float_in]
            point_shapes = [self.in_shapes[i] for i in range(n_row, len(self.in_shapes)) if i in self.float_in]
            tr = pack_tangents(row_t, row_shapes, (batch,), n_directions)
            tp = pack_tangents(point_t, point_shapes, (1 if shared else batch, n_points), n_directions)
            inputs += [tr, tp]
            output_shapes.append((batch * n_points * lay["n_out_t"] * n_directions,))
            output_dtypes.append(mx.float32)
        threads = batch * n_points * (n_directions if tangent else 1)
        group = 64
        grid = max(-(-threads // group) * group, group)
        out = kernel(inputs=inputs, grid=(grid, 1, 1), threadgroup=(group, 1, 1), output_shapes=output_shapes,
                     output_dtypes=output_dtypes)
        return out, lay

    def _split(self, leaves):
        n_row = self.n_row_leaves
        return leaves[:n_row], leaves[n_row:]

    def values(self, *trees, batch, n_points=None, shared=False):
        """The outputs (a pytree like the function's) for the inputs
        ``trees`` (pytrees of MLX arrays); ``n_points`` for a function of a
        point (``None``: a function of the binary only)."""
        leaves = self._leaves(trees)
        per_point = n_points is not None
        out, lay = self._call(leaves, None, batch, n_points or 1, shared, 0)
        prefix = (batch, n_points) if per_point else (batch,)
        y = out[0].reshape(prefix + (lay["n_out"],))
        outs = unpack(y, self.out_shapes, self.out_kinds, prefix)
        return combine(outs, self.out_meta)

    def tangents(self, trees, tangent_trees, batch, n_directions, n_points=None, shared=False):
        """``(outputs, tangents)``: the outputs and their tangents (pytrees,
        ``None`` for the tangents of booleans and integers) given the
        tangents of the inputs (pytrees like ``trees`` with a trailing axis
        of ``n_directions``, ``None`` for zero)."""
        leaves = self._leaves(trees)
        t_leaves = self._tangent_leaves(tangent_trees, trees)
        per_point = n_points is not None
        out, lay = self._call(leaves, t_leaves, batch, n_points or 1, shared, n_directions)
        prefix = (batch, n_points) if per_point else (batch,)
        y = out[0].reshape(prefix + (lay["n_out"],))
        outs = unpack(y, self.out_shapes, self.out_kinds, prefix)
        dy = out[1].reshape(prefix + (lay["n_out_t"], n_directions))
        float_shapes = [self.out_shapes[i] for i in self.float_out]
        douts = unpack_tangents(dy, float_shapes, prefix, n_directions)
        t_all = [None] * len(self.out_shapes)
        for i, t in zip(self.float_out, douts):
            t_all[i] = t
        return combine(outs, self.out_meta), combine(t_all, self.out_meta)

    def _leaves(self, trees):
        leaves = []
        for tree, n, meta in zip(trees, self.n_leaves, self.metas):
            values, m = partition(tree)
            if len(values) != n:
                raise ValueError(f"{self.name}: an input has {len(values)} leaves, {n} expected")
            leaves += values
        return leaves

    def _tangent_leaves(self, tangent_trees, trees):
        leaves = []
        for tree, t_tree, n in zip(trees, tangent_trees, self.n_leaves):
            if t_tree is None:
                leaves += [None] * n
            else:
                leaves += _flatten_like(t_tree, tree)
        return leaves


def _flatten_like(t_tree, tree):
    """The leaves of ``t_tree`` at the array leaves of ``tree`` (``None``
    for a zero tangent)."""
    from ._stages import _is_dynamic

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    t_leaves = treedef.flatten_up_to(t_tree)
    return [t for leaf, t in zip(leaves, t_leaves) if _is_dynamic(leaf)]
