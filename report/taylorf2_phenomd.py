#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""TaylorF2 and IMRPhenomD (``waveformx.models.taylorf2``,
``waveformx.models.phenomd``): their accuracy against LALSimulation, of
float32 and of MLX against float64, of the derivatives, and their cost on
the CPU (JAX) and the Apple GPU (MLX), as ``report/xas.py`` measures
IMRPhenomXAS. Writes ``report/data/taylorf2.json`` or
``report/data/phenomd.json``.

* Against LAL: random aligned-spin binaries (total masses log-uniform in
  2-500 solar masses, mass ratios uniform in 1/20-1, spins uniform in
  [-0.99, 0.99], ``f_min`` among 10, 15 and 20 Hz, ``f_ref = f_min`` (30%)
  or uniform up to ``min(5 f_min, f_max)``), 1000 geometric frequencies
  from ``f_min`` to ``f_max``: the Schwarzschild ISCO for TaylorF2 (LAL's
  end of its uniform grid), ``Mf = 0.2`` for IMRPhenomD (likewise), and at
  most 2048 Hz. The largest relative error and the weighted mismatch (aLIGO
  design, the larger of the polarizations) of JAX float64.
* float32, derivatives and timings as ``report/xas.py``.
* IMRPhenomD: the errors of the closed-form QNM fits
  (``report/data/phenomd_qnm.json``, from ``scripts/fit_phenomd_qnm.py``).

MLX is optional: without it the MLX parts are left out.

Usage::

    python report/taylorf2_phenomd.py --model IMRPhenomD [--n 300] [--n-bns 100] [--seed 2]
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import DATA, environment, lal_sequence, median_time, psd, write  # noqa: E402
from xas import (BATCHES, BENCH, LARGE, SMALL, _mlx_time, _psd_interpolated, _vector, bench_binary,  # noqa: E402
                 geometric, jax_timings, percentiles, weighted)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import waveformx as wf  # noqa: E402

try:
    import mlx.core as mx
except ImportError:
    mx = None

MTSUN_SI = 4.925490947641267e-06
#: approximant -> (module, output file)
MODELS = {"TaylorF2": ("taylorf2", "taylorf2.json"), "IMRPhenomD": ("phenomd", "phenomd.json")}


def f_max(approximant, total):
    """The end of the frequencies of a binary of total mass ``total``."""
    if approximant == "TaylorF2":
        return min(1.0 / (6.0 ** 1.5 * np.pi * total * MTSUN_SI), 2048.0)
    return min(0.2 / (total * MTSUN_SI), 2048.0)


def keywords(d):
    return dict(mass_1=d["m1"], mass_2=d["m2"], spin_1=(0.0, 0.0, d["chi1"]), spin_2=(0.0, 0.0, d["chi2"]),
                luminosity_distance=400.0, inclination=d["inclination"], phi_ref=d["phi_ref"], f_ref=d["fref"])


def binary(d, dtype=np.float64):
    k = keywords(d)
    return wf.Binary(*(np.asarray(k[name], dtype) for name in wf.Binary._fields))


def draws(approximant, n, seed, f_min=None, bns=False):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        if bns:
            m1, m2 = rng.uniform(1.0, 2.5, 2)
            chi1, chi2 = rng.uniform(-0.6, 0.6, 2)
        else:
            total = np.exp(rng.uniform(np.log(2.0), np.log(500.0)))
            q = rng.uniform(1 / 20, 1.0)
            m1, m2 = total / (1 + q), total * q / (1 + q)
            chi1, chi2 = rng.uniform(-0.99, 0.99, 2)
        total = m1 + m2
        fmin = f_min if f_min is not None else float(rng.choice([10.0, 15.0, 20.0]))
        fmax = f_max(approximant, total)
        if fmax <= 1.2 * fmin:
            continue
        fref = fmin if rng.uniform() < 0.3 else float(rng.uniform(fmin, min(5 * fmin, fmax)))
        r = lambda x: float(np.float32(x))  # noqa: E731
        out.append(dict(m1=r(max(m1, m2)), m2=r(min(m1, m2)), chi1=r(chi1), chi2=r(chi2),
                        inclination=r(rng.uniform(0, np.pi)), phi_ref=r(rng.uniform(0, 2 * np.pi)),
                        fref=r(fref), fmin=fmin, fmax=fmax, total=float(total)))
    return out


class Measurement:
    def __init__(self, approximant):
        self.approximant = approximant
        self.module = MODELS[approximant][0]
        self.function = wf.polarizations_function(approximant)
        self.mlx = importlib.import_module(f"waveformx.models.{self.module}.mlx") if mx is not None else None

    def against_lal(self, n, seed):
        function = jax.jit(self.function)
        records = []
        for d in draws(self.approximant, n, seed):
            f = geometric(d)
            reference = lal_sequence(keywords(d), f, approximant=self.approximant)
            ours = [np.asarray(h) * 1e-21 for h in function(binary(d), jnp.asarray(f))]
            scale = max(np.max(np.abs(r)) for r in reference)
            error = max(np.max(np.abs(o - r)) for o, r in zip(ours, reference)) / scale
            records.append(dict(total=d["total"], q=d["m2"] / d["m1"], relative_error=float(error),
                                mismatch=float(weighted(ours, reference, f))))
        return dict(records=records, relative_error=percentiles([r["relative_error"] for r in records]),
                    mismatch=percentiles([r["mismatch"] for r in records]))

    def float32(self, n, seed, n_bns):
        function = jax.jit(self.function)

        def compare(sample, grid):
            out = []
            for d in sample:
                f = grid(d)
                weights = 1.0 / psd(f) if len(f) <= 4096 else _psd_interpolated(f)
                reference = function(binary(d), jnp.asarray(f))
                record = dict(total=d["total"])
                single = function(binary(d, np.float32), jnp.asarray(f, jnp.float32))
                record["jax32"] = float(weighted(single, reference, f, weights))
                if self.mlx is not None:
                    record["mlx"] = float(weighted(self.mlx.polarizations(binary(d, np.float32), f), reference, f,
                                                   weights))
                out.append(record)
            return out

        def summary(records):
            return {name: percentiles([r.get(name, np.nan) for r in records]) for name in ("jax32", "mlx")}

        population = compare(draws(self.approximant, n, seed, f_min=20.0), geometric)
        bns = compare(draws(self.approximant, n_bns, seed + 1, f_min=20.0, bns=True), geometric)
        uniform = compare(draws(self.approximant, max(n_bns // 10, 3), seed + 2, f_min=20.0, bns=True),
                          lambda d: np.float32(20.0 + np.arange(int((d["fmax"] - 20.0) * 256)) / 256.0)
                          .astype(np.float64))
        return dict(population=population, bns=bns, bns_uniform=uniform, summary=dict(
            population=summary(population), bns=summary(bns), bns_uniform=summary(uniform),
            light=summary([r for r in population if r["total"] < 5.0])))

    def gradients(self, n, seed):
        function = jax.jit(self.function)

        def log_likelihood(b, f, data):
            hp, hc = self.function(b, f)
            return -0.5 * jnp.sum(jnp.abs(hp - data[0].astype(hp.dtype)) ** 2
                                  + jnp.abs(hc - data[1].astype(hc.dtype)) ** 2)

        grad = jax.jit(jax.grad(log_likelihood))
        out = []
        for d in draws(self.approximant, n, seed, f_min=20.0):
            f = geometric(d, 800)
            b64 = binary(d)
            data = [np.asarray(h) for h in function(b64._replace(mass_1=b64.mass_1 * 1.01), jnp.asarray(f))]
            g64 = _vector(grad(b64, jnp.asarray(f), [jnp.asarray(x) for x in data]))
            g32 = _vector(grad(binary(d, np.float32), jnp.asarray(f, jnp.float32),
                               [jnp.asarray(x.astype(np.complex64)) for x in data]))
            bound = 1e-2 * np.abs(g64) + 1e-3 * np.max(np.abs(g64))
            record = dict(total=d["total"], ratio_jax32=float(np.max(np.abs(g32 - g64) / bound)))
            if self.mlx is not None:
                dm = [mx.array(x.astype(np.complex64)) for x in data]

                def loss(*fields):
                    hp, hc = self.mlx.polarizations(wf.Binary(*fields), f)
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

    def lal_timings(self):
        def run(f):
            return lambda: lal_sequence(BENCH, f, approximant=self.approximant)

        return dict(single=median_time(run(SMALL), repeats=21, number=20), large=median_time(run(LARGE), repeats=5))

    def mlx_timings(self):
        model = self.mlx
        out = dict(value={}, grad={}, jacobian={}, large_value={}, large_grad={})
        small, large = mx.array(SMALL.astype(np.float32)), mx.array(LARGE.astype(np.float32))

        def batch(size):
            if size == 1:
                return wf.Binary(*(np.float32(np.asarray(BENCH[name])) for name in wf.Binary._fields))
            return wf.Binary(*(mx.array(np.asarray(v, np.float32)) for v in bench_binary(np.float64, size)))

        for grid, sizes, prefix in ((small, BATCHES, ""), (large, (1, 16), "large_")):
            for size in sizes:
                b = batch(size)
                out[prefix + "value"][str(size)] = _mlx_time(lambda: model.polarizations(b, grid))
                m1 = mx.array(np.asarray(b.mass_1, np.float32))

                def loss(m):
                    hp, _ = model.polarizations(b._replace(mass_1=m), grid)
                    return mx.sum(mx.abs(hp) ** 2)

                gradient = mx.grad(loss)
                out[prefix + "grad"][str(size)] = _mlx_time(lambda: gradient(m1))
                if not prefix:
                    out["jacobian"][str(size)] = _mlx_time(lambda: model.polarizations_and_jacobian(b, grid)[1])
        script = (
            "import time; t0 = time.perf_counter(); import numpy as np, mlx.core as mx, waveformx as wf; "
            f"from waveformx.models.{self.module} import mlx as m; t1 = time.perf_counter(); "
            "b = wf.Binary(30.0, 25.0, (0, 0, .3), (0, 0, -.2), 400.0, .4, .1, 20.0); "
            "f = np.geomspace(20, 1024, 1000); mx.eval(*m.polarizations(b, f)); t2 = time.perf_counter(); "
            "g = mx.grad(lambda x: mx.sum(mx.abs(m.polarizations(b._replace(mass_1=x), f)[0]) ** 2)); "
            "mx.eval(g(mx.array(30.0))); t3 = time.perf_counter(); print(t1 - t0, t2 - t1, t3 - t2)")
        first = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                               check=True).stdout.split()
        out["first_calls"] = dict(imports=float(first[0]), value=float(first[1]), grad=float(first[2]))
        out["device"] = mx.device_info().get("device_name")
        out["mlx"] = mx.__version__
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--n-bns", type=int, default=100)
    parser.add_argument("--n-gradients", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    m = Measurement(args.model)
    core = importlib.import_module(f"waveformx.models.{m.module}.core")
    payload = dict(environment=environment(), approximant=args.model, n=args.n, n_bns=args.n_bns, seed=args.seed,
                   table_size=core.SIZE)
    # (the load of the machine before and after: the timings want it idle)
    load = [os.getloadavg()[0]]
    payload["lal"] = m.against_lal(args.n, args.seed)
    print("against LAL:", payload["lal"]["relative_error"], payload["lal"]["mismatch"], flush=True)
    payload["float32"] = m.float32(args.n, args.seed, args.n_bns)
    print("float32:", payload["float32"]["summary"], flush=True)
    payload["gradients"] = m.gradients(args.n_gradients, args.seed)
    print("gradients:", {k: payload["gradients"][k] for k in ("ratio_jax32", "ratio_mlx")}, flush=True)
    timings = dict(frequencies=len(SMALL), large=len(LARGE))
    for name, dtype in (("float64", jnp.float64), ("float32", jnp.float32)):
        # (a new function each time, so that its compilation is timed)
        timings[name] = jax_timings(lambda b, f: m.function(b, f), dtype)
        print(name, timings[name], flush=True)
    timings["lal"] = m.lal_timings()
    print("lal", timings["lal"], flush=True)
    if mx is not None:
        timings["mlx"] = m.mlx_timings()
        print("mlx", timings["mlx"], flush=True)
    payload["timings"] = timings
    payload["load_average"] = load + [os.getloadavg()[0]]
    if args.model == "IMRPhenomD":
        payload["qnm_fit"] = json.loads((DATA / "phenomd_qnm.json").read_text())
    write(MODELS[args.model][1], payload)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"{time.perf_counter() - start:.0f} s")
