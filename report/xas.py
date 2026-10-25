#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXAS (``waveformx.models.xas``): its accuracy against
LALSimulation, of float32 and of MLX against float64, of the derivatives,
and its cost on the CPU (JAX) and the Apple GPU (MLX). Writes
``report/data/xas.json``.

* Against LAL: random aligned-spin binaries (total masses log-uniform in
  2-500 solar masses, mass ratios uniform in 1/20-1, spins uniform in
  [-0.99, 0.99], ``f_min`` among 10, 15 and 20 Hz, ``f_ref = f_min`` (30%)
  or uniform up to ``min(5 f_min, f_max)``), 1000 geometric frequencies
  from ``f_min`` to ``min(1.05 x 0.33 / M, 2048 Hz)``: the largest relative
  error and the weighted mismatch (aLIGO design, the larger of the
  polarizations) of JAX float64.
* float32: the same binaries from 20 Hz, and binary neutron stars
  (1-2.5 solar masses, spins up to 0.6) from 20 Hz on 1000 geometric
  frequencies and on a uniform grid of 1/256 Hz to 2048 Hz: the weighted
  mismatch of JAX float32 and of MLX against JAX float64 at the
  float32-rounded inputs.
* Derivatives: the gradient of a noise-free log-likelihood (the data those
  of ``mass_1`` 1% larger), JAX float32 and MLX (``mx.grad``) against JAX
  float64, as ratios to the bound of ``tests/test_xpnr_float32.py``
  (``1e-2 |g| + 1e-3 max |g|``).
* Timings: JAX on the CPU (float64 and float32; the first call, one binary
  on 1000 geometric frequencies and on 2^20 uniform ones, batches under
  ``jax.vmap``, the gradient), the (2,2) mode of IMRPhenomXPNR evaluated as
  LAL does (``models/xpnr/xas.py``), LALSimulation, and MLX on the GPU (the
  values, the gradient and the Jacobian against the batch size).

MLX is optional: without it the MLX parts are left out.

Usage::

    python report/xas.py [--n 300] [--n-bns 100] [--seed 2]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import DATA, environment, lal_sequence, median_time, mismatch, psd, write  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import waveformx as wf  # noqa: E402
from waveformx.models.xas import core  # noqa: E402
from waveformx.models.xpnr import xas as lal_xas  # noqa: E402
from waveformx.models.xpnr._math import precision, precision_of, real  # noqa: E402
from waveformx.models.xpnr.internals import waveform_struct  # noqa: E402

try:
    import mlx.core as mx

    from waveformx.models.xas import mlx as xas_mlx
except ImportError:
    mx = None

MTSUN_SI = 4.925490947641267e-06
OPTIONS = wf.WaveformOptions()
POLARIZATIONS = wf.polarizations_function("IMRPhenomXAS")


def keywords(d):
    return dict(mass_1=d["m1"], mass_2=d["m2"], spin_1=(0.0, 0.0, d["chi1"]), spin_2=(0.0, 0.0, d["chi2"]),
                luminosity_distance=400.0, inclination=d["inclination"], phi_ref=d["phi_ref"], f_ref=d["fref"])


def binary(d, dtype=np.float64):
    """The binary of a draw."""
    k = keywords(d)
    return wf.Binary(*(np.asarray(k[name], dtype) for name in wf.Binary._fields))


def draws(n, seed, f_min=None, bns=False):
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
        fmax = min(0.33 / (total * MTSUN_SI) * 1.05, 2048.0)
        if fmax <= 1.2 * fmin:
            continue
        fref = fmin if rng.uniform() < 0.3 else float(rng.uniform(fmin, min(5 * fmin, fmax)))
        # (the parameters rounded to float32: the inputs of LAL and of both
        # precisions)
        r = lambda x: float(np.float32(x))  # noqa: E731
        out.append(dict(m1=r(max(m1, m2)), m2=r(min(m1, m2)), chi1=r(chi1), chi2=r(chi2),
                        inclination=r(rng.uniform(0, np.pi)), phi_ref=r(rng.uniform(0, 2 * np.pi)),
                        fref=r(fref), fmin=fmin, fmax=fmax, total=float(total)))
    return out


def geometric(d, n=1000):
    return np.float32(np.geomspace(d["fmin"], d["fmax"], n)).astype(np.float64)


def percentiles(values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return None
    return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)), max=float(values.max()),
                n=int(values.size))


def weighted(a, b, f, weights=None):
    w = 1.0 / psd(f) if weights is None else weights
    return max(mismatch(np.asarray(x, np.complex128), np.asarray(y), w) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def against_lal(n, seed):
    function = jax.jit(POLARIZATIONS)
    records = []
    for d in draws(n, seed):
        f = geometric(d)
        reference = lal_sequence(keywords(d), f, approximant="IMRPhenomXAS")
        ours = [np.asarray(h) * 1e-21 for h in function(binary(d), jnp.asarray(f))]
        scale = max(np.max(np.abs(r)) for r in reference)
        error = max(np.max(np.abs(o - r)) for o, r in zip(ours, reference)) / scale
        records.append(dict(total=d["total"], q=d["m2"] / d["m1"], relative_error=float(error),
                            mismatch=float(weighted(ours, reference, f))))
    return dict(records=records, relative_error=percentiles([r["relative_error"] for r in records]),
                mismatch=percentiles([r["mismatch"] for r in records]))


def float32(n, seed, n_bns):
    function = jax.jit(POLARIZATIONS)

    def compare(sample, grid):
        out = []
        for d in sample:
            f = grid(d)
            weights = 1.0 / psd(f) if len(f) <= 4096 else _psd_interpolated(f)
            reference = function(binary(d), jnp.asarray(f))
            record = dict(total=d["total"])
            single = function(binary(d, np.float32), jnp.asarray(f, jnp.float32))
            record["jax32"] = float(weighted(single, reference, f, weights))
            if mx is not None:
                b = binary(d, np.float32)
                record["mlx"] = float(weighted(xas_mlx.polarizations(b, f), reference, f, weights))
            out.append(record)
        return out

    def summary(records):
        return {name: percentiles([r.get(name, np.nan) for r in records]) for name in ("jax32", "mlx")}

    population = compare(draws(n, seed, f_min=20.0), geometric)
    bns = compare(draws(n_bns, seed + 1, f_min=20.0, bns=True), geometric)
    uniform = compare(draws(max(n_bns // 10, 3), seed + 2, f_min=20.0, bns=True),
                      lambda d: np.float32(20.0 + np.arange(int((2048.0 - 20.0) * 256)) / 256.0).astype(np.float64))
    return dict(population=population, bns=bns, bns_uniform=uniform, summary=dict(
        population=summary(population), bns=summary(bns), bns_uniform=summary(uniform),
        light=summary([r for r in population if r["total"] < 5.0])))


def _psd_interpolated(f):
    knots = np.geomspace(f[0], f[-1], 2000)
    return 1.0 / np.interp(f, knots, psd(knots))


def gradients(n, seed):
    draws_ = draws(n, seed, f_min=20.0)

    def log_likelihood(b, f, data):
        hp, hc = POLARIZATIONS(b, f)
        return -0.5 * jnp.sum(jnp.abs(hp - data[0].astype(hp.dtype)) ** 2 + jnp.abs(hc - data[1].astype(hc.dtype)) ** 2)

    grad = jax.jit(jax.grad(log_likelihood))
    function = jax.jit(POLARIZATIONS)
    out = []
    for d in draws_:
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
                hp, hc = xas_mlx.polarizations(wf.Binary(*fields), f)
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


def _vector(b):
    return np.concatenate([np.atleast_1d(np.asarray(getattr(b, name), np.float64)) for name in wf.Binary._fields])


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------

BENCH = dict(mass_1=30.0, mass_2=25.0, spin_1=(0.0, 0.0, 0.3), spin_2=(0.0, 0.0, -0.2), luminosity_distance=400.0,
             inclination=0.4, phi_ref=0.1, f_ref=20.0)
SMALL = np.geomspace(20.0, 1024.0, 1000)
LARGE = 20.0 + np.arange(1 << 20) / 512.0
BATCHES = (1, 16, 256, 4096)


def lal_faithful(b, f, options=OPTIONS):
    """The (2,2) mode of IMRPhenomXPNR as LAL evaluates it
    (``models/xpnr/xas.py``) with the polarizations of IMRPhenomXAS: the
    implementation before this one."""
    with precision(precision_of(b)):
        f = jnp.asarray(f, real())
        w = waveform_struct(b.mass_1, b.mass_2, b.spin_1[2], b.spin_2[2], 0.0, b.f_ref,
                            b.phi_ref, f[0], f[-1], b.luminosity_distance, b.inclination, 1e21)
        s = lal_xas.xas_setup(w)
        h = lal_xas.h22(w.m_sec * f, w, s, lal_xas.phif_ref(w, s))
        c = jnp.cos(b.inclination)
        return -core.YLM * 0.5 * (1 + c * c) * h, 1j * core.YLM * c * h


def bench_binary(dtype, size=None, seed=0):
    rng = np.random.default_rng(seed)
    if size is None:
        return wf.Binary(*(jnp.asarray(np.asarray(BENCH[name], float), dtype) for name in wf.Binary._fields))
    m1 = rng.uniform(5.0, 100.0, size)
    z = np.zeros(size)
    values = [m1, m1 * rng.uniform(0.2, 1.0, size), np.stack([z, z, rng.uniform(-0.9, 0.9, size)], 1),
              np.stack([z, z, rng.uniform(-0.9, 0.9, size)], 1), np.full(size, 400.0), np.full(size, 0.4),
              np.full(size, 0.1), np.full(size, 20.0)]
    return wf.Binary(*(jnp.asarray(v, dtype) for v in values))


def jax_timings(function, dtype, batches=BATCHES):
    out = {}
    single = jax.jit(function)
    b = bench_binary(dtype)
    small, large = jnp.asarray(SMALL, dtype), jnp.asarray(LARGE, dtype)
    start = time.perf_counter()
    jax.block_until_ready(single(b, small))
    out["compile"] = time.perf_counter() - start
    out["single"] = median_time(lambda: single(b, small), repeats=21, number=20)
    out["large"] = median_time(lambda: single(b, large), repeats=5)
    batched = jax.jit(jax.vmap(function, in_axes=(0, None)))
    out["batch"] = {}
    for size in batches:
        bb = bench_binary(dtype, size)
        if size == batches[0]:
            start = time.perf_counter()
            jax.block_until_ready(batched(bb, small))
            out["compile_vmap"] = time.perf_counter() - start
        out["batch"][str(size)] = median_time(lambda: batched(bb, small), repeats=7)
    loss = jax.jit(jax.grad(lambda m: jnp.sum(jnp.abs(function(b._replace(mass_1=m), small)[0]) ** 2)))
    start = time.perf_counter()
    jax.block_until_ready(loss(b.mass_1))
    out["compile_grad"] = time.perf_counter() - start
    out["grad"] = median_time(lambda: loss(b.mass_1), repeats=21, number=10)
    return out


def lal_timings():
    def run(f):
        return lambda: lal_sequence(BENCH, f, approximant="IMRPhenomXAS")

    return dict(single=median_time(run(SMALL), repeats=21, number=20), large=median_time(run(LARGE), repeats=5))


def _mlx_time(function, repeats=21, number=5):
    mx.eval(function())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            mx.eval(function())
        times.append((time.perf_counter() - start) / number)
    return float(np.median(times))


def mlx_timings():
    out = dict(value={}, grad={}, jacobian={}, large_value={}, large_grad={})
    small, large = mx.array(SMALL.astype(np.float32)), mx.array(LARGE.astype(np.float32))

    def batch(size):
        if size == 1:
            return wf.Binary(*(np.float32(np.asarray(BENCH[name])) for name in wf.Binary._fields))
        return wf.Binary(*(mx.array(np.asarray(v, np.float32)) for v in bench_binary(np.float64, size)))

    for grid, sizes, prefix in ((small, BATCHES, ""), (large, (1, 16), "large_")):
        for size in sizes:
            b = batch(size)
            out[prefix + "value"][str(size)] = _mlx_time(lambda: xas_mlx.polarizations(b, grid))
            m1 = mx.array(np.asarray(b.mass_1, np.float32))

            def loss(m):
                hp, _ = xas_mlx.polarizations(b._replace(mass_1=m), grid)
                return mx.sum(mx.abs(hp) ** 2)

            gradient = mx.grad(loss)
            out[prefix + "grad"][str(size)] = _mlx_time(lambda: gradient(m1))
            if not prefix:
                out["jacobian"][str(size)] = _mlx_time(lambda: xas_mlx.polarizations_and_jacobian(b, grid)[1])
    script = (
        "import time; t0 = time.perf_counter(); import numpy as np, mlx.core as mx, waveformx as wf; "
        "from waveformx.models.xas import mlx as m; t1 = time.perf_counter(); "
        "b = wf.Binary(30.0, 25.0, (0, 0, .3), (0, 0, -.2), 400.0, .4, .1, 20.0); "
        "f = np.geomspace(20, 1024, 1000); mx.eval(*m.polarizations(b, f)); t2 = time.perf_counter(); "
        "g = mx.grad(lambda x: mx.sum(mx.abs(m.polarizations(b._replace(mass_1=x), f)[0]) ** 2)); "
        "mx.eval(g(mx.array(30.0))); t3 = time.perf_counter(); print(t1 - t0, t2 - t1, t3 - t2)")
    first = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.split()
    out["first_calls"] = dict(imports=float(first[0]), value=float(first[1]), grad=float(first[2]))
    info = mx.device_info()
    out["device"] = info.get("device_name")
    out["mlx"] = mx.__version__
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--n-bns", type=int, default=100)
    parser.add_argument("--n-gradients", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    payload = dict(environment=environment(), n=args.n, n_bns=args.n_bns, seed=args.seed, table_size=core.SIZE)
    payload["lal"] = against_lal(args.n, args.seed)
    print("against LAL:", payload["lal"]["relative_error"], payload["lal"]["mismatch"], flush=True)
    payload["float32"] = float32(args.n, args.seed, args.n_bns)
    print("float32:", payload["float32"]["summary"], flush=True)
    payload["gradients"] = gradients(args.n_gradients, args.seed)
    print("gradients:", {k: payload["gradients"][k] for k in ("ratio_jax32", "ratio_mlx")}, flush=True)
    timings = dict(frequencies=len(SMALL), large=len(LARGE))
    pure = lambda b, f: POLARIZATIONS(b, f)  # noqa: E731
    for name, function, dtype in (("float64", pure, jnp.float64), ("float32", pure, jnp.float32),
                                  ("lal_faithful_float64", lal_faithful, jnp.float64)):
        timings[name] = jax_timings(function, dtype)
        print(name, timings[name], flush=True)
    timings["lal"] = lal_timings()
    print("lal", timings["lal"], flush=True)
    if mx is not None:
        timings["mlx"] = mlx_timings()
        print("mlx", timings["mlx"], flush=True)
    payload["timings"] = timings
    write("xas.json", payload)


if __name__ == "__main__":
    main()
