#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXHM (``waveformx.models.xhm``): its accuracy against
LALSimulation, of float32 and of MLX against float64, of the derivatives,
and its cost on the CPU (JAX) and the Apple GPU (MLX). Writes
``report/data/xhm.json``.

The binaries and frequencies are those of ``report/xas.py``:

* Against LAL: the largest relative error and the weighted mismatch of the
  polarizations of JAX float64 (``SimInspiralChooseFDWaveformSequence``),
  and of its modes against ``SimIMRPhenomXHMFrequencySequenceOneMode`` (the
  (3,2) apart: LAL's one-mode function does not recycle the (2,2), and LAL
  fixes its intermediate phase with finite differences).
* float32: the weighted mismatch of the polarizations and of the worst mode
  of JAX float32 and of MLX against JAX float64, for binaries from 20 Hz and
  binary neutron stars.
* Derivatives: the gradient of a noise-free log-likelihood, JAX float32 and
  MLX against JAX float64, as ratios to the bound of
  ``tests/test_xpnr_float32.py``.
* Timings: JAX (float64, float32), LALSimulation and MLX, as for
  IMRPhenomXAS.

Usage::

    python report/xhm.py [--n 200] [--n-bns 60] [--seed 2]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import xas as common_xas  # noqa: E402
from common import environment, lal_sequence, median_time, mismatch, psd, write  # noqa: E402
from xas import (  # noqa: E402
    BATCHES, BENCH, LARGE, SMALL, _mlx_time, _psd_interpolated, _vector, binary, draws, geometric, jax_timings,
    keywords, percentiles,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import lal  # noqa: E402
import lalsimulation as lalsim  # noqa: E402
import numpy as np  # noqa: E402

import waveformx as wf  # noqa: E402
from waveformx.models.xhm import core, model  # noqa: E402

try:
    import mlx.core as mx

    from waveformx.models.xhm import mlx as xhm_mlx
except ImportError:
    mx = None

OPTIONS = wf.WaveformOptions()
POLARIZATIONS = wf.polarizations_function("IMRPhenomXHM")
MODES = jax.jit(lambda b, f: model.modes_and_polarizations(b, f, OPTIONS))


def lal_one_mode(d, f, ell, emm):
    vector = lal.CreateREAL8Vector(len(f))
    vector.data = np.asarray(f, float)
    return lalsim.SimIMRPhenomXHMFrequencySequenceOneMode(
        vector, d["m1"] * lal.MSUN_SI, d["m2"] * lal.MSUN_SI, d["chi1"], d["chi2"], ell, emm, 400.0 * 1e6 * lal.PC_SI,
        d["phi_ref"], d["fref"], None).data.data.copy()


def against_lal(n, seed):
    records = []
    for d in draws(n, seed):
        f = geometric(d)
        # (below the cutoff, where LAL's one-mode functions end)
        f = f[f <= 0.3 / (d["total"] * common_xas.MTSUN_SI)]
        reference = lal_sequence(keywords(d), f, approximant="IMRPhenomXHM")
        modes, pols = MODES(binary(d), jnp.asarray(f))
        ours = [np.asarray(h) * 1e-21 for h in pols]
        scale = max(np.max(np.abs(r)) for r in reference)
        w = 1.0 / psd(f)
        record = dict(total=d["total"], q=d["m2"] / d["m1"],
                      relative_error=float(max(np.max(np.abs(o - r)) for o, r in zip(ours, reference)) / scale),
                      mismatch=float(max(mismatch(o, r, w) for o, r in zip(ours, reference))))
        errors = []
        for (ell, emm), h in modes.items():
            theirs = lal_one_mode(d, f, ell, -emm)
            h = np.asarray(h) * 1e-21
            if (ell, emm) == (3, 2):
                record["mismatch_32"] = float(mismatch(h, theirs, w)) if np.any(theirs) else 0.0
            elif np.any(theirs):
                errors.append(np.max(np.abs(h - theirs)) / np.max(np.abs(theirs)))
        record["modes_relative_error"] = float(max(errors))
        records.append(record)
    return dict(records=records, relative_error=percentiles([r["relative_error"] for r in records]),
                mismatch=percentiles([r["mismatch"] for r in records]),
                modes_relative_error=percentiles([r["modes_relative_error"] for r in records]),
                mismatch_32=percentiles([r["mismatch_32"] for r in records]))


def float32(n, seed, n_bns):
    def compare(sample, grid):
        out = []
        for d in sample:
            f = grid(d)
            w = 1.0 / psd(f) if len(f) <= 4096 else _psd_interpolated(f)
            m64, p64 = MODES(binary(d), jnp.asarray(f))
            m32, p32 = MODES(binary(d, np.float32), jnp.asarray(f, jnp.float32))
            nonzero = [mode for mode in m64 if np.any(np.asarray(m64[mode]))]
            record = dict(total=d["total"],
                          jax32=float(max(mismatch(np.asarray(a, complex), np.asarray(b), w) for a, b in zip(p32, p64))),
                          jax32_modes=float(max(mismatch(np.asarray(m32[k], complex), np.asarray(m64[k]), w)
                                                for k in nonzero)))
            if mx is not None:
                b = binary(d, np.float32)
                hp, hc = xhm_mlx.polarizations(b, f)
                mm = xhm_mlx.modes(b, f)
                record["mlx"] = float(max(mismatch(np.asarray(a, complex), np.asarray(r), w)
                                          for a, r in zip((hp, hc), p64)))
                record["mlx_modes"] = float(max(mismatch(np.asarray(mm[k], complex), np.asarray(m64[k]), w)
                                                for k in nonzero))
            out.append(record)
        return out

    def summary(records):
        return {name: percentiles([r.get(name, np.nan) for r in records])
                for name in ("jax32", "mlx", "jax32_modes", "mlx_modes")}

    population = compare(draws(n, seed, f_min=20.0), geometric)
    bns = compare(draws(n_bns, seed + 1, f_min=20.0, bns=True), geometric)
    uniform = compare(draws(max(n_bns // 10, 3), seed + 2, f_min=20.0, bns=True),
                      lambda d: np.float32(20.0 + np.arange(int((2048.0 - 20.0) * 256)) / 256.0).astype(np.float64))
    return dict(population=population, bns=bns, bns_uniform=uniform, summary=dict(
        population=summary(population), bns=summary(bns), bns_uniform=summary(uniform)))


def gradients(n, seed):
    def log_likelihood(b, f, data):
        hp, hc = POLARIZATIONS(b, f)
        return -0.5 * jnp.sum(jnp.abs(hp - data[0].astype(hp.dtype)) ** 2 + jnp.abs(hc - data[1].astype(hc.dtype)) ** 2)

    grad = jax.jit(jax.grad(log_likelihood))
    function = jax.jit(POLARIZATIONS)
    out = []
    for d in draws(n, seed, f_min=20.0):
        f = geometric(d, 800)
        b64 = binary(d)
        data = [np.asarray(h) for h in function(b64._replace(mass_1=b64.mass_1 * 1.01), jnp.asarray(f))]
        g64 = _vector(grad(b64, jnp.asarray(f), [jnp.asarray(x) for x in data]))
        g32 = _vector(grad(binary(d, np.float32), jnp.asarray(f, jnp.float32),
                           [jnp.asarray(x.astype(np.complex64)) for x in data]))
        bound = 1e-2 * np.abs(g64) + 1e-3 * np.max(np.abs(g64))
        record = dict(total=d["total"], ratio_jax32=float(np.max(np.abs(g32 - g64) / bound)))
        if mx is not None:
            dm = [mx.array(x.astype(np.complex64)) for x in data]

            def loss(*fields):
                hp, hc = xhm_mlx.polarizations(wf.Binary(*fields), f)
                return -0.5 * mx.sum(mx.abs(hp - dm[0]) ** 2 + mx.abs(hc - dm[1]) ** 2)

            fields = [mx.array(np.asarray(v, np.float32)) for v in binary(d, np.float32)]
            gm = mx.grad(loss, argnums=tuple(range(8)))(*fields)
            gm = np.concatenate([np.atleast_1d(np.asarray(x, np.float64)) for x in gm])
            record["ratio_mlx"] = float(np.max(np.abs(gm - g64) / bound))
        out.append(record)
    return dict(records=out, n=len(out),
                ratio_jax32=percentiles([r["ratio_jax32"] for r in out]),
                ratio_mlx=percentiles([r.get("ratio_mlx", np.nan) for r in out]),
                within_jax32=sum(r["ratio_jax32"] <= 1 for r in out),
                within_mlx=sum(r.get("ratio_mlx", np.inf) <= 1 for r in out))


def lal_timings():
    def run(f):
        return lambda: lal_sequence(BENCH, f, approximant="IMRPhenomXHM")

    return dict(single=median_time(run(SMALL), repeats=21, number=20), large=median_time(run(LARGE), repeats=5))


def mlx_timings():
    out = dict(value={}, grad={}, jacobian={}, modes={}, large_value={}, large_grad={})
    small, large = mx.array(SMALL.astype(np.float32)), mx.array(LARGE.astype(np.float32))

    def batch(size):
        if size == 1:
            return wf.Binary(*(np.float32(np.asarray(BENCH[name])) for name in wf.Binary._fields))
        return wf.Binary(*(mx.array(np.asarray(v, np.float32)) for v in common_xas.bench_binary(np.float64, size)))

    for grid, sizes, prefix in ((small, BATCHES, ""), (large, (1, 16), "large_")):
        for size in sizes:
            b = batch(size)
            out[prefix + "value"][str(size)] = _mlx_time(lambda: xhm_mlx.polarizations(b, grid))
            m1 = mx.array(np.asarray(b.mass_1, np.float32))

            def loss(m):
                hp, _ = xhm_mlx.polarizations(b._replace(mass_1=m), grid)
                return mx.sum(mx.abs(hp) ** 2)

            gradient = mx.grad(loss)
            out[prefix + "grad"][str(size)] = _mlx_time(lambda: gradient(m1))
            if not prefix:
                out["jacobian"][str(size)] = _mlx_time(lambda: xhm_mlx.polarizations_and_jacobian(b, grid)[1])
                out["modes"][str(size)] = _mlx_time(lambda: list(xhm_mlx.modes(b, grid).values()))
    script = (
        "import time; t0 = time.perf_counter(); import numpy as np, mlx.core as mx, waveformx as wf; "
        "from waveformx.models.xhm import mlx as m; t1 = time.perf_counter(); "
        "b = wf.Binary(30.0, 25.0, (0, 0, .3), (0, 0, -.2), 400.0, .4, .1, 20.0); "
        "f = np.geomspace(20, 1024, 1000); mx.eval(*m.polarizations(b, f)); t2 = time.perf_counter(); "
        "g = mx.grad(lambda x: mx.sum(mx.abs(m.polarizations(b._replace(mass_1=x), f)[0]) ** 2)); "
        "mx.eval(g(mx.array(30.0))); t3 = time.perf_counter(); print(t1 - t0, t2 - t1, t3 - t2)")
    first = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.split()
    out["first_calls"] = dict(imports=float(first[0]), value=float(first[1]), grad=float(first[2]))
    out["device"] = mx.device_info().get("device_name")
    out["mlx"] = mx.__version__
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--n-bns", type=int, default=60)
    parser.add_argument("--n-gradients", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    payload = dict(environment=environment(), n=args.n, n_bns=args.n_bns, seed=args.seed, table_size=core.SIZE)
    payload["lal"] = against_lal(args.n, args.seed)
    print("against LAL:", {k: payload["lal"][k] for k in ("relative_error", "mismatch", "modes_relative_error",
                                                          "mismatch_32")}, flush=True)
    payload["float32"] = float32(args.n, args.seed, args.n_bns)
    print("float32:", payload["float32"]["summary"], flush=True)
    payload["gradients"] = gradients(args.n_gradients, args.seed)
    print("gradients:", {k: payload["gradients"][k] for k in ("ratio_jax32", "ratio_mlx")}, flush=True)
    timings = dict(frequencies=len(SMALL), large=len(LARGE))
    pure = lambda b, f: POLARIZATIONS(b, f)  # noqa: E731
    for name, dtype in (("float64", jnp.float64), ("float32", jnp.float32)):
        start = time.perf_counter()
        timings[name] = jax_timings(pure, dtype)
        print(name, timings[name], f"({time.perf_counter() - start:.0f} s)", flush=True)
    timings["lal"] = lal_timings()
    print("lal", timings["lal"], flush=True)
    if mx is not None:
        timings["mlx"] = mlx_timings()
        print("mlx", timings["mlx"], flush=True)
    payload["timings"] = timings
    write("xhm.json", payload)


if __name__ == "__main__":
    main()
