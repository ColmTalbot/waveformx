#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The lean IMRPhenomXPNR in MLX on the Apple GPU against the JAX lean
model on the CPU: the accuracy of the float32 MLX waveform and of its
derivatives, and its cost. Writes ``report/data/mlx_xpnr.json``.

* Accuracy on the random binaries of ``report/extended`` (their own
  frequencies, 1024 linear from ``f_min``): the weighted mismatch (aLIGO
  design, the larger of the polarizations) of the MLX waveform and of
  JAX's float32 lean waveform against JAX's float64 lean waveform; where
  each is finite.
* Derivatives on the first of those binaries: the gradient of the
  noise-free log-likelihood of ``tests/test_xpnr_float32.py`` (the data
  those of ``mass_1`` 1% larger), MLX (``mx.grad``) and JAX float32
  against JAX float64, as ratios to that test's bound (``1e-2 |g| + 1e-3
  max |g|``).
* Timings on the benchmark grid (8033 frequencies, 20-1024 Hz): the pure
  function and the gradient of the power against the number of binaries
  per call (random binaries of 20-120 solar masses), MLX on the GPU and
  the JAX lean model on the CPU (``jax.jit``, ``jax.vmap``), and the first
  calls of MLX in a new process (tracing the model and compiling its
  kernels).

It needs MLX (``pip install waveformx[mlx]``).

Usage::

    python report/mlx_xpnr.py [--n 200] [--n-gradients 30] [--seed 4]
    python report/mlx_xpnr.py --only-timings   # re-time, keep the rest
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "extended"))

from common import BENCHMARK_FREQUENCIES, BINARY, DATA, environment, median_time, write  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import waveformx as wf  # noqa: E402
import xcommon  # noqa: E402
from waveformx.models.xpnr import lean  # noqa: E402
from waveformx.models.xpnr import mlx as xpnr_mlx  # noqa: E402


def mlx_environment():
    info = mx.device_info()
    return dict(mlx=mx.__version__, device=info.get("device_name"), architecture=info.get("architecture"),
                memory_gb=round(info.get("memory_size", 0) / 2 ** 30), macos=platform.mac_ver()[0])


def percentiles(values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return None
    return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)), max=float(values.max()),
                n=int(values.size))


def as_binary(d, dtype=np.float64):
    return wf.Binary(*(np.asarray(v, dtype) for v in (d["m1"], d["m2"], d["s1"], d["s2"], 500.0,
                                                       d.get("inclination", 0.5), d.get("phi_ref", 0.3), d["fref"])))


def batch_of(binaries, dtype=np.float32):
    return wf.Binary(*(np.stack([np.asarray(getattr(b, name), dtype) for b in binaries]) for name in wf.Binary._fields))


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def accuracy(n, seed, chunk=32):
    draws = xcommon.draws(n, seed)
    lean64 = jax.jit(lambda b, f: lean.polarizations(b, f))
    records = []
    start = time.perf_counter()
    for first in range(0, n, chunk):
        part = draws[first:first + chunk]
        freqs = np.stack([np.asarray(xcommon.frequencies(d, 1024)) for d in part])
        hp, hc = xpnr_mlx.polarizations(batch_of([as_binary(d) for d in part]), freqs.astype(np.float32))
        mx.eval(hp, hc)
        for k, d in enumerate(part):
            f = freqs[k]
            ref = [np.asarray(x) for x in lean64(as_binary(d), jnp.asarray(f))]
            single = [np.asarray(x).astype(np.complex128) for x in lean64(
                wf.Binary(*(jnp.asarray(v, jnp.float32) for v in as_binary(d))), jnp.asarray(f, jnp.float32))]
            got = [np.asarray(hp[k]).astype(np.complex128), np.asarray(hc[k]).astype(np.complex128)]
            finite = [bool(np.all(np.isfinite(x[0]))) for x in (ref, single, got)]
            record = dict(index=first + k, total=d["M"], q=d["q"], f_min=d["fmin"], finite_double=finite[0],
                          finite_jax32=finite[1], finite_mlx=finite[2])
            if finite[0]:
                for name, value in (("jax32", single), ("mlx", got)):
                    if np.all(np.isfinite(value[0])):
                        record[f"mismatch_{name}"] = max(xcommon.weighted_mismatch(value[0], ref[0], f),
                                                         xcommon.weighted_mismatch(value[1], ref[1], f))
            records.append(record)
        print(f"{first + len(part)}/{n} ({time.perf_counter() - start:.0f} s)", flush=True)
    finite = dict(
        both=sum(r["finite_double"] and r["finite_mlx"] for r in records),
        double_only=sum(r["finite_double"] and not r["finite_mlx"] for r in records),
        mlx_only=sum(r["finite_mlx"] and not r["finite_double"] for r in records),
        jax32_not_finite=sum(r["finite_double"] and not r["finite_jax32"] for r in records))
    summary = dict(mlx=percentiles([r.get("mismatch_mlx", np.nan) for r in records]),
                   jax32=percentiles([r.get("mismatch_jax32", np.nan) for r in records]), finite=finite)
    return records, summary


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------


def gradients(n, seed):
    draws = xcommon.draws(n, seed)

    def log_likelihood(b, f, data):
        hp, hc = lean.polarizations(b, f)
        overlap = jnp.vdot(data[0].astype(hp.dtype), hp) + jnp.vdot(data[1].astype(hc.dtype), hc)
        return jnp.real(overlap) - 0.5 * jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2)

    grad = jax.jit(jax.grad(log_likelihood))
    lean64 = jax.jit(lambda b, f: lean.polarizations(b, f))
    out = []
    start = time.perf_counter()
    for k, d in enumerate(draws):
        f = np.asarray(xcommon.frequencies(d, 1024))
        b = as_binary(d)
        data = lean64(b._replace(mass_1=b.mass_1 * 1.01), jnp.asarray(f))
        if not np.all(np.isfinite(np.asarray(data[0]))):
            continue
        double = np.concatenate([np.ravel(np.asarray(x)) for x in grad(b, jnp.asarray(f), data)])
        single = np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in grad(
            wf.Binary(*(jnp.asarray(v, jnp.float32) for v in b)), jnp.asarray(f, jnp.float32), data)])
        d32 = [mx.array(np.asarray(x).astype(np.complex64)) for x in data]

        def mlx_log_likelihood(*fields):
            hp, hc = xpnr_mlx.polarizations(wf.Binary(*fields), f)
            overlap = mx.sum(mx.conj(d32[0]) * hp) + mx.sum(mx.conj(d32[1]) * hc)
            return mx.real(overlap) - 0.5 * mx.sum(mx.abs(hp) ** 2 + mx.abs(hc) ** 2)

        g = mx.grad(mlx_log_likelihood, argnums=list(range(8)))(*[mx.array(np.asarray(v, np.float32)) for v in b])
        got = np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in g])
        bound = 1e-2 * np.abs(double) + 1e-3 * np.max(np.abs(double))
        out.append(dict(index=k, double=double.tolist(), mlx=got.tolist(), jax32=single.tolist(),
                        ratio_mlx=float(np.max(np.abs(got - double) / bound)),
                        ratio_jax32=float(np.max(np.abs(single - double) / bound)),
                        finite_mlx=bool(np.all(np.isfinite(got))), finite_jax32=bool(np.all(np.isfinite(single)))))
        print(f"gradient {k}: mlx {out[-1]['ratio_mlx']:.2e}, jax32 {out[-1]['ratio_jax32']:.2e} of the bound "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
    summary = dict(n=len(out), ratio_mlx=percentiles([g["ratio_mlx"] for g in out]),
                   ratio_jax32=percentiles([g["ratio_jax32"] for g in out]),
                   within_mlx=sum(g["ratio_mlx"] <= 1 for g in out), within_jax32=sum(g["ratio_jax32"] <= 1 for g in out),
                   finite_mlx=sum(g["finite_mlx"] for g in out), finite_jax32=sum(g["finite_jax32"] for g in out))
    return out, summary


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------


def mlx_time(function, repeats=7, number=1):
    mx.eval(function())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            mx.eval(function())
        times.append((time.perf_counter() - start) / number)
    return float(np.median(times))


FIRST_CALLS = """
import json, time
start = time.perf_counter()
import mlx.core as mx
import numpy as np
import waveformx as wf
from waveformx.models.xpnr import mlx as xpnr_mlx
imports = time.perf_counter() - start
f = np.arange(160, 8193) / 8.0
binary = wf.Binary(36.0, 29.0, (0.4, 0.2, 0.3), (-0.2, 0.3, -0.1), 500.0, 0.7, 1.2, 20.0)
start = time.perf_counter()
mx.eval(xpnr_mlx.polarizations(binary, f))
value = time.perf_counter() - start
args = [mx.array(np.asarray(x, dtype=np.float32)) for x in binary]
start = time.perf_counter()
mx.eval(mx.grad(lambda *a: mx.sum(mx.abs(xpnr_mlx.polarizations(wf.Binary(*a), f)[0]) ** 2), argnums=list(range(8)))(*args))
grad = time.perf_counter() - start
print(json.dumps(dict(imports=imports, value=value, grad=grad)))
"""

MLX_SIZES = (1, 16, 64, 256, 1024)
MLX_GRADIENT_SIZES = (1, 16, 64, 256)
JAX_SIZES = (1, 16, 64)
JAX_GRADIENT_SIZES = (1, 16)


def draws_for_timing(n, seed=7):
    """Random binaries of 20-120 solar masses, mass ratios 1/8-1, spins up
    to 0.8, ``f_ref = 20`` Hz (the benchmark grid starts at 20 Hz)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        total = rng.uniform(20.0, 120.0)
        q = rng.uniform(1 / 8, 1.0)
        spins = []
        for _ in range(2):
            a, c, p = rng.uniform(0, 0.8), rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)
            s = np.sqrt(1 - c * c)
            spins.append([a * s * np.cos(p), a * s * np.sin(p), a * c])
        out.append(dict(m1=total / (1 + q), m2=total - total / (1 + q), s1=spins[0], s2=spins[1], fref=20.0))
    return out


def timings():
    f = BENCHMARK_FREQUENCIES
    others = [as_binary(d) for d in draws_for_timing(max(MLX_SIZES))]
    single = wf.Binary(**BINARY)
    out = dict(frequencies=len(f))
    first = subprocess.run([sys.executable, "-c", FIRST_CALLS], capture_output=True, text=True, check=True)
    entry = dict(first_calls=json.loads(first.stdout.strip().splitlines()[-1]), value={}, grad={})
    print("mlx first calls", entry["first_calls"], flush=True)

    def power(*fields):
        hp, hc = xpnr_mlx.polarizations(wf.Binary(*fields), f)
        return mx.sum(mx.abs(hp) ** 2 + mx.abs(hc) ** 2)

    grad = mx.grad(power, argnums=list(range(8)))
    for size in MLX_SIZES:
        b = single if size == 1 else batch_of(others[:size])
        fields = [mx.array(np.asarray(x, np.float32)) for x in b]
        entry["value"][str(size)] = mlx_time(lambda: xpnr_mlx.polarizations(b, f), repeats=7 if size > 16 else 11)
        if size in MLX_GRADIENT_SIZES:
            entry["grad"][str(size)] = mlx_time(lambda: grad(*fields), repeats=3 if size > 16 else 5)
        print("mlx", size, f"{entry['value'][str(size)] * 1e3:.1f} ms",
              f"grad {entry['grad'].get(str(size), float('nan')) * 1e3:.1f} ms", flush=True)
    out["mlx"] = entry
    for name, dtype in (("jax_float64", jnp.float64), ("jax_float32", jnp.float32)):
        fj = jnp.asarray(f, dtype)

        def waveform(b):
            return lean.polarizations(b, fj)

        def total_power(b):
            hp, hc = waveform(b)
            return jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2)

        value_jit, batch_jit = jax.jit(waveform), jax.jit(jax.vmap(waveform))
        grad_jit, batch_grad_jit = jax.jit(jax.grad(total_power)), jax.jit(jax.vmap(jax.grad(total_power)))
        entry = dict(value={}, grad={})
        for size in JAX_SIZES:
            if size == 1:
                b = wf.Binary(*(jnp.asarray(v, dtype) for v in single))
                value_f, grad_f = value_jit, grad_jit
            else:
                b = wf.Binary(*(jnp.asarray(v, dtype) for v in batch_of(others[:size], np.float64)))
                value_f, grad_f = batch_jit, batch_grad_jit
            entry["value"][str(size)] = median_time(lambda: value_f(b), repeats=7 if size > 1 else 11)
            if size in JAX_GRADIENT_SIZES:
                entry["grad"][str(size)] = median_time(lambda: grad_f(b), repeats=3 if size > 1 else 5)
            print(name, size, f"{entry['value'][str(size)] * 1e3:.1f} ms",
                  f"grad {entry['grad'].get(str(size), float('nan')) * 1e3:.1f} ms", flush=True)
        out[name] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--n-gradients", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output", default="mlx_xpnr.json")
    parser.add_argument("--only-timings", action="store_true")
    args = parser.parse_args()
    if args.only_timings:
        payload = json.loads((DATA / args.output).read_text())
        payload.update(timings=timings(), environment=environment(), mlx_environment=mlx_environment())
        write(args.output, payload)
        return
    records, summary = accuracy(args.n, args.seed)
    print("mismatch: mlx", summary["mlx"], "jax float32", summary["jax32"], summary["finite"], flush=True)
    grads, grad_summary = gradients(args.n_gradients, args.seed)
    print("gradients:", grad_summary, flush=True)
    payload = dict(environment=environment(), mlx_environment=mlx_environment(), n=args.n, seed=args.seed,
                   summary=summary, gradient_summary=grad_summary, timings=timings(), records=records,
                   gradients=grads)
    write(args.output, payload)


if __name__ == "__main__":
    main()
