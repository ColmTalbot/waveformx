# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
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


class Packed:
    """A pytree whose array leaves are the columns of one float array
    (``prefix + (n,)``, booleans and integers as floats) and their tangents
    (``prefix + (n_float, P)``, the float leaves only, or ``None``): the
    outputs of a stage as they come out of its kernel, passed to the next
    stage as they are (one block of its inputs rather than one per leaf)
    and unpacked only where a field is read."""

    def __init__(self, values, tangents, shapes, kinds, meta, prefix):
        self.values = values
        self.tangents = tangents
        self.shapes = shapes
        self.kinds = kinds
        self.meta = meta
        self.prefix = prefix
        self._tree = None
        self._tangent_tree = None

    def tree(self):
        """The pytree of MLX arrays."""
        if self._tree is None:
            self._tree = combine(unpack(self.values, self.shapes, self.kinds, self.prefix), self.meta)
        return self._tree

    def tangent_tree(self):
        """The pytree of the tangents (``None`` for the leaves without)."""
        if self.tangents is None:
            return None
        if self._tangent_tree is None:
            floats = [i for i, k in enumerate(self.kinds) if k == FLOAT]
            t = unpack_tangents(self.tangents, [self.shapes[i] for i in floats], self.prefix,
                                self.tangents.shape[-1])
            leaves = [None] * len(self.shapes)
            for i, x in zip(floats, t):
                leaves[i] = x
            self._tangent_tree = combine(leaves, self.meta)
        return self._tangent_tree

    def select(self, index):
        """The binaries ``index`` (the first axis)."""
        prefix = (int(index.shape[0]),) + tuple(self.prefix[1:])
        tangents = None if self.tangents is None else self.tangents[index]
        return Packed(self.values[index], tangents, self.shapes, self.kinds, self.meta, prefix)


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
        self.n_row_args = n_rows
        holder = {}

        def rebuild(flat):
            it = iter(flat)
            return [combine([next(it) for _ in range(n)], meta) for n, meta in zip(self.n_leaves, self.metas)]

        def flat_function(*flat):
            with precision(np.float32), lean_mode(True):
                result = function(*rebuild(flat))
                out, holder["meta"] = partition(result)
                if isinstance(result, tuple):
                    holder["elements"] = [partition(e) for e in result]
            return out

        avals = [jax.ShapeDtypeStruct(tuple(v.shape), np.dtype(v.dtype), weak_type=getattr(v, "weak_type", False))
                 for v in flat_examples]
        closed = jax.make_jaxpr(flat_function)(*avals)
        self.out_meta = holder["meta"]
        self.out_elements = None
        if "elements" in holder:
            self.out_elements = [(len(values), meta) for values, meta in holder["elements"]]
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

    def _arg_of_leaf(self):
        out = []
        for a, n in enumerate(self.n_leaves):
            out += [a] * n
        return out

    def _layout(self, gspec=None):
        gathered = dict(gspec[0]) if gspec else {}
        arg_of = self._arg_of_leaf()
        rows = list(range(self.n_row_leaves))
        points = [i for i in range(self.n_row_leaves, len(self.in_shapes)) if arg_of[i] not in gathered]
        g_offset, g_t_offset, g_where = {}, {}, {}
        for a, spec in gathered.items():
            column, t_column = spec[0], spec[1]
            row_offset = spec[2] if len(spec) > 2 else 0
            slot = spec[3] if len(spec) > 3 else 0
            start = sum(self.n_leaves[:a])
            col, tcol = column, t_column
            for i in range(start, start + self.n_leaves[a]):
                g_offset[i] = col
                g_where[i] = (row_offset, slot)
                col += _size(self.in_shapes[i])
                if i in self.float_in:
                    if tcol is None:
                        g_t_offset[i] = None
                    else:
                        g_t_offset[i] = tcol
                        tcol += _size(self.in_shapes[i])
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
                    n_out_t=start, g_offset=g_offset, g_t_offset=g_t_offset, g_where=g_where,
                    g_width=gspec[1] if gspec else 0, g_t_width=gspec[2] if gspec else 0,
                    g_slots=gspec[3] if gspec else 0)

    def _source(self, tangent, gspec=None, counted=False):
        lay = self._layout(gspec)
        n_row = self.n_row_leaves
        program = self.jvp_program if tangent else self.program
        n_primal = len(self.in_shapes)
        float_in = self.float_in

        def input_name(index, k, kind):
            if index < n_primal:
                if index in lay["g_offset"]:
                    value = f"xg[g_in_{_where_name(lay['g_where'][index])} + {lay['g_offset'][index] + k}]"
                elif index < n_row:
                    value = f"xr[row_in + {lay['offset'][index] + k}]"
                else:
                    value = f"xp[point_in + {lay['offset'][index] + k}]"
            else:
                leaf = float_in[index - n_primal]
                if leaf in lay["g_offset"]:
                    column = lay["g_t_offset"].get(leaf)
                    where = _where_name(lay["g_where"][leaf])
                    value = "0.0f" if column is None else f"tg[(g_t_{where} + {column + k}) * n_dir + dir]"
                elif leaf < n_row:
                    value = f"tr[(row_t + {lay['t_offset'][leaf] + k}) * n_dir + dir]"
                else:
                    value = f"tp[(point_t + {lay['t_offset'][leaf] + k}) * n_dir + dir]"
            return value if kind == FLOAT else f"static_cast<{_metal._TYPES[kind]}>({value})"

        lines, name_of = _metal.body_lines(program, input_name, n_primal=n_primal if tangent else None)
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
        if counted:
            # (the points of a binary beyond its count are left at zero)
            head.append("if (point >= (uint)cn[b]) return;")
        if gspec:
            head.append("uint n_table = dims[4];")
            for row_offset, slot in sorted(set(lay["g_where"].values())):
                name = _where_name((row_offset, slot))
                head += [
                    f"uint g_row_{name} = min((uint)gi[((size_t)b * n_points + point) * {lay['g_slots']} + {slot}]"
                    f" + {row_offset}u, n_table - 1);",
                    f"size_t g_in_{name} = ((size_t)b * n_table + g_row_{name}) * {lay['g_width']};",
                ]
                if tangent:
                    head.append(f"size_t g_t_{name} = ((size_t)b * n_table + g_row_{name}) * {lay['g_t_width']};")
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

    def _kernel(self, tangent, gspec=None, counted=False):
        key = (tangent, gspec, counted)
        if key not in self._kernels:
            source, lay = self._source(tangent, gspec, counted)
            inputs = ["xr", "xp", "dims"] + (["tr", "tp"] if tangent else [])
            if gspec:
                inputs += ["xg", "gi"] + (["tg"] if tangent else [])
            if counted:
                inputs.append("cn")
            outputs = ["y"] + (["dy"] if tangent else [])
            suffix = ("jvp" if tangent else "f") + (f"_g{len(self._kernels)}" if gspec or counted else "")
            kernel = mx.fast.metal_kernel(name=f"{self.name}_{suffix}", input_names=inputs, output_names=outputs,
                                          source=source, header=_metal.HEADER)
            self._kernels[key] = (kernel, lay)
        return self._kernels[key]

    # --- evaluation ------------------------------------------------------------

    def _parts(self, args, targs, n_directions, skip=()):
        """Per argument: ``("packed", Packed)`` or ``("leaves", values,
        tangents)`` (``None`` for the arguments in ``skip``)."""
        parts = []
        for k, (arg, n, meta) in enumerate(zip(args, self.n_leaves, self.metas)):
            if k in skip:
                parts.append(None)
                continue
            if isinstance(arg, Packed):
                if len(arg.shapes) != n:
                    raise ValueError(f"{self.name}: argument {k} has {len(arg.shapes)} leaves, {n} expected")
                parts.append(("packed", arg))
                continue
            values, _ = partition(arg)
            if len(values) != n:
                raise ValueError(f"{self.name}: argument {k} has {len(values)} leaves, {n} expected")
            t = None
            if n_directions and targs is not None and targs[k] is not None:
                t = _flatten_like(targs[k], arg)
            parts.append(("leaves", values, t))
        return parts

    def _pack(self, parts, which, prefix, n_directions, tangents, skip=()):
        """The packed values (or tangents) of the arguments ``which`` (but
        those in ``skip``)."""
        blocks, start = [], sum(self.n_leaves[:which.start])
        for a, (part, n) in enumerate(zip(parts[which], self.n_leaves[which]), start=which.start):
            if a in skip:
                start += n
                continue
            shapes = self.in_shapes[start:start + n]
            kinds = self.in_kinds[start:start + n]
            float_shapes = [sh for sh, kd in zip(shapes, kinds) if kd == FLOAT]
            if part[0] == "packed":
                p = part[1]
                if tangents:
                    if p.tangents is None:
                        blocks.append(mx.zeros(prefix + (sum(_size(sh) for sh in float_shapes), n_directions)))
                    else:
                        blocks.append(mx.broadcast_to(p.tangents, prefix + p.tangents.shape[len(p.prefix):]))
                else:
                    blocks.append(mx.broadcast_to(p.values, prefix + p.values.shape[len(p.prefix):]))
            else:
                _, values, t = part
                if tangents:
                    t = t if t is not None else [None] * len(values)
                    blocks.append(pack_tangents([x for x, kd in zip(t, kinds) if kd == FLOAT], float_shapes, prefix,
                                                n_directions))
                else:
                    blocks.append(pack(values, shapes, prefix))
            start += n
        if not blocks:
            return mx.zeros(prefix + ((0, n_directions) if tangents else (0,)), dtype=mx.float32)
        return mx.concatenate(blocks, axis=-2 if tangents else -1)

    def run(self, args, targs=None, *, batch, n_directions=0, n_points=None, shared=False, gather=None, counts=None):
        """The outputs for the arguments ``args`` (pytrees of MLX arrays or
        :class:`Packed`) and, with ``n_directions``, their tangents
        (``targs``: pytrees of tangents, or ``None``, for the arguments
        that are not :class:`Packed`): a :class:`Packed` for each element
        of the function's tuple of results (or one).

        ``gather`` reads arguments at the points from the rows of a table
        rather than from arrays of the points: a dict with ``args`` (``{arg
        index: (column, tangent column or None[, row offset[, slot]])}``,
        the leaves of the argument in consecutive columns of the row
        ``index[b, n, slot] + row offset``, clipped to the table), ``table``
        (``(B, M, width)``), ``table_t`` (``(B, M, width_t, P)``, with
        tangents) and ``index`` (``(B, N)``, or ``(B, N, slots)``); the
        gathered arguments are then ignored in ``args``.

        ``counts`` (``(B,)``, int) leaves the points of each binary from its
        count on at zero, without evaluating them (the padding of grids of
        different sizes)."""
        tangent = bool(n_directions)
        gspec = None
        if gather is not None:
            width_t = gather["table_t"].shape[2] if tangent and gather.get("table_t") is not None else 0
            index = gather["index"]
            slots = int(index.shape[2]) if index.ndim == 3 else 1
            gspec = (tuple(sorted((a, tuple(v)) for a, v in gather["args"].items())), int(gather["table"].shape[2]),
                     int(width_t), slots)
        kernel, lay = self._kernel(tangent, gspec, counts is not None)
        n_args = len(self.n_leaves)
        n_row_args = self.n_row_args
        skip = tuple(gather["args"]) if gather is not None else ()
        args = [None if a in skip else x for a, x in enumerate(args)]
        parts = self._parts(args, targs, n_directions, skip)
        points = n_points or 1
        point_prefix = (1 if shared else batch, points)
        rows = slice(0, n_row_args)
        pts = slice(n_row_args, n_args)
        xr = self._pack(parts, rows, (batch,), 0, False)
        xp = self._pack(parts, pts, point_prefix, 0, False, skip)
        n_table = int(gather["table"].shape[1]) if gather is not None else 0
        dims = mx.array([points, batch, n_directions if tangent else 0, 1 if shared else 0, n_table], dtype=mx.uint32)
        inputs = [xr, xp, dims]
        output_shapes = [(batch * points * lay["n_out"],)]
        if tangent:
            inputs += [self._pack(parts, rows, (batch,), n_directions, True),
                       self._pack(parts, pts, point_prefix, n_directions, True, skip)]
            output_shapes.append((batch * points * lay["n_out_t"] * n_directions,))
        if gather is not None:
            inputs += [gather["table"].astype(mx.float32), mx.stop_gradient(gather["index"]).astype(mx.int32)]
            if tangent:
                table_t = gather.get("table_t")
                inputs.append(table_t if table_t is not None else mx.zeros((1,)))
        extra = {}
        if counts is not None:
            inputs.append(mx.stop_gradient(counts).astype(mx.int32))
            extra = dict(init_value=0.0)
        threads = batch * points * (n_directions if tangent else 1)
        group = 64
        grid = max(-(-threads // group) * group, group)
        out = kernel(inputs=inputs, grid=(grid, 1, 1), threadgroup=(group, 1, 1), output_shapes=output_shapes,
                     output_dtypes=[mx.float32] * len(output_shapes), **extra)
        prefix = (batch, n_points) if n_points is not None else (batch,)
        y = out[0].reshape(prefix + (lay["n_out"],))
        dy = out[1].reshape(prefix + (lay["n_out_t"], n_directions)) if tangent else None
        return self._split_outputs(y, dy, prefix)

    def _split_outputs(self, y, dy, prefix):
        if self.out_elements is None:
            return Packed(y, dy, self.out_shapes, self.out_kinds, self.out_meta, prefix)
        out, leaf, column, t_column = [], 0, 0, 0
        for n, meta in self.out_elements:
            shapes = self.out_shapes[leaf:leaf + n]
            kinds = self.out_kinds[leaf:leaf + n]
            width = sum(_size(sh) for sh in shapes)
            t_width = sum(_size(sh) for sh, kd in zip(shapes, kinds) if kd == FLOAT)
            values = y[..., column:column + width]
            tangents = None if dy is None else dy[..., t_column:t_column + t_width, :]
            out.append(Packed(values, tangents, shapes, kinds, meta, prefix))
            leaf += n
            column += width
            t_column += t_width
        return tuple(out)

    def values(self, *trees, batch, n_points=None, shared=False):
        """The outputs as a pytree like the function's."""
        out = self.run(trees, batch=batch, n_points=n_points, shared=shared)
        return tuple(p.tree() for p in out) if isinstance(out, tuple) else out.tree()

    def tangents(self, trees, tangent_trees, batch, n_directions, n_points=None, shared=False):
        """``(outputs, tangents)`` as pytrees (``None`` for the tangents of
        booleans and integers), given the tangents of the inputs (pytrees
        like ``trees`` with a trailing axis of ``n_directions``, ``None``
        for zero)."""
        out = self.run(trees, tangent_trees, batch=batch, n_directions=n_directions, n_points=n_points,
                       shared=shared)
        if isinstance(out, tuple):
            return tuple(p.tree() for p in out), tuple(p.tangent_tree() for p in out)
        return out.tree(), out.tangent_tree()


def _where_name(where):
    """The name of the row of a gathered input: ``r<offset>s<slot>``."""
    return f"r{where[0]}s{where[1]}"


def _flatten_like(t_tree, tree):
    """The leaves of ``t_tree`` at the array leaves of ``tree`` (``None``
    for a zero tangent)."""
    from ._stages import _is_dynamic

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    t_leaves = treedef.flatten_up_to(t_tree)
    return [t for leaf, t in zip(leaves, t_leaves) if _is_dynamic(leaf)]
