#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 in MLX on the Apple GPU against the JAX implementation on the
CPU: the accuracy of the float32 MLX waveform and of its derivatives, LAL's
grids of its batch function, its conditioned waveforms, and its cost.

Accuracy is measured on the pairs of ``float32_nrsur7dq4.py`` (same seed
and number), in the same way: both binaries of a pair evaluated on one
time grid, projected on a detector, tapered, and compared through inner
products weighted by the aLIGO design sensitivity from 20 Hz. The MLX
waveform (``polarizations_at`` at the float32 times) and JAX's float32
waveform are both compared with JAX's float64 one. The derivatives of a
noise-free log-likelihood are compared for some nearby pairs.

The grids of ``batch_td_polarizations`` (computed for the whole batch) are
compared with the JAX driver's, binary by binary, on the binaries of the
pairs, from 20 Hz and from the start of the surrogate. The conditioned
waveforms (``simulate_td``, ``simulate_fd``) are compared with the JAX
conversions on a few binaries.

Timings (on an idle machine): the pure function on 8192 samples and its
gradient for batches of 1 to 4096 binaries (MLX, compiled) or 1 to 256
(JAX, ``jax.vmap``, in float32 and float64), the public functions on LAL's
grid (one binary, and a batch of 1024 random binaries at 4096 Hz from the
start of the surrogate), and ``simulate_fd``.

It needs MLX (``pip install waveformx[mlx]``), and LALSuite as the other
measurements do.

Usage::

    python report/mlx_nrsur7dq4.py [--n-pairs 150] [--n-gradients 40]
    python report/mlx_nrsur7dq4.py --only-timings   # re-time, keep the rest
"""

import argparse
import json
import platform
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np

import float32_nrsur7dq4 as f32
from common import DATA, NAMES, environment, max_relative_error, median_time, wf, write
from waveformx._binary import row
from waveformx.models.nrsur7dq4 import _data, _driver, _waveform
from waveformx.models.nrsur7dq4 import mlx as nrsur
from waveformx.models.nrsur7dq4._dynamics import START_TIME
from waveformx.models.nrsur7dq4.mlx import _driver as mlx_driver

OPTIONS = wf.WaveformOptions()


def mlx_environment():
    info = mx.device_info()
    return dict(mlx=mx.__version__, device=info.get("device_name"), architecture=info.get("architecture"),
                memory_gb=round(info.get("memory_size", 0) / 2 ** 30), macos=platform.mac_ver()[0])


def mlx_binary(keywords):
    return wf.Binary(*(mx.array(np.asarray(keywords[name], dtype=np.float32)) for name in wf.Binary._fields))


def mlx_batch(binaries):
    return wf.Binary(*(mx.array(np.stack([np.asarray(b[name], dtype=np.float32) for b in binaries]))
                       for name in wf.Binary._fields))


def mlx_detector(b, times, psi, hp, hc):
    """:func:`float32_nrsur7dq4.detector` in MLX (batched along the first
    axis of ``b``)."""
    m = ((b.mass_1 + b.mass_2) * _waveform.MTSUN_SI)[..., None]
    x = mx.clip((times - START_TIME * m) / (400.0 * m), 0.0, 1.0)
    return (hp * np.cos(2 * psi) + hc * np.sin(2 * psi)) * (0.5 - 0.5 * mx.cos(np.pi * x))


def mlx_strain(b, times, psi):
    return mlx_detector(b, times, psi, *nrsur.polarizations_at(b, times, OPTIONS))


def mlx_log_likelihood(b1, w2, times, psi, inv_asd, dt, df):
    """:func:`float32_nrsur7dq4.log_likelihood` in MLX."""
    h = mlx_strain(b1, times, psi)
    w1 = mx.fft.rfft(h, n=2 * h.shape[-1]) * dt * inv_asd

    def inner(a, b):
        return 4 * df * mx.sum(mx.real(a) * mx.real(b) + mx.imag(a) * mx.imag(b))

    return inner(w1, w2) - 0.5 * inner(w1, w1)


def flat(gradient):
    return np.concatenate([np.atleast_1d(np.asarray(x, np.float64)) for x in gradient])


# ---------------------------------------------------------------------------
# accuracy on the pairs
# ---------------------------------------------------------------------------

def measure(case, gradients):
    """The MLX and JAX float32 waveforms of a pair against JAX float64."""
    n, dt = case["n"], case["dt"]
    ia = f32.inv_asd(n, dt)
    times64 = jnp.asarray(case["t_lo"]) + jnp.asarray(dt) * jnp.arange(n)
    times32 = jnp.asarray(case["t_lo"], jnp.float32) + jnp.asarray(dt, jnp.float32) * jnp.arange(n, dtype=jnp.float32)
    psi = case["psi"]
    double = [np.asarray(f32.strain_jit(f32.binary_of(b, f32.DOUBLE), times64, jnp.asarray(psi)))
              for b in case["binaries"]]
    single = [np.asarray(f32.strain_jit(f32.binary_of(b, f32.SINGLE), times32, jnp.asarray(psi, jnp.float32)),
                         np.float64) for b in case["binaries"]]
    times_mx = mx.array(np.asarray(times32))
    pair = mlx_strain(mlx_batch(case["binaries"]), times_mx, psi)
    mx.eval(pair)
    gpu = [np.array(pair[k]).astype(np.float64) for k in range(2)]

    def white(h):
        return np.fft.rfft(h, 2 * n) * dt * ia

    def dot(a, b):
        return 4 / (2 * n * dt) * np.sum(a.real * b.real + a.imag * b.imag)

    def normalised(a, b):
        return dot(a, b) / np.sqrt(dot(a, a) * dot(b, b))

    ref = [white(h) for h in double]
    norm = np.sqrt(dot(ref[0], ref[0]) * dot(ref[1], ref[1]))
    o64 = normalised(*ref)
    record = dict(kind=case["kind"], totals=[b["mass_1"] + b["mass_2"] for b in case["binaries"]],
                  samples=n, overlap=float(o64))
    for name, hs in (("mlx", gpu), ("jax32", single)):
        w = [white(h) for h in hs]
        record[f"d_overlap_{name}"] = float(normalised(*w) - o64)
        record[f"epsilon_{name}"] = float((dot(*w) - dot(*ref)) / norm)
        record[f"noise_weighted_{name}"] = [float(np.sqrt(dot(a - b, a - b) / dot(b, b))) for a, b in zip(w, ref)]
        record[f"max_relative_{name}"] = [max_relative_error(a, b) for a, b in zip(hs, double)]
        record[f"finite_{name}"] = bool(all(np.all(np.isfinite(h)) for h in hs))
    if gradients:
        args64 = (jnp.asarray(ia), jnp.asarray(dt), jnp.asarray(1 / (2 * n * dt)))
        b64 = [f32.binary_of(b, f32.DOUBLE) for b in case["binaries"]]
        w2 = f32.whiten_jit(f32.strain_jit(b64[1], times64, jnp.asarray(psi)), *args64[:2])
        reference = flat(f32.grad_log_likelihood(b64[0], w2, times64, jnp.asarray(psi), *args64))
        args32 = (mx.array(ia.astype(np.float32)), dt, 1 / (2 * n * dt))
        w2 = mx.fft.rfft(pair[1], n=2 * n) * dt * args32[0]
        gradient = mx.grad(mlx_log_likelihood)(mlx_binary(case["binaries"][0]), w2, times_mx, psi, *args32)
        record["grad_log_likelihood"] = dict(double=reference.tolist(), mlx=flat(gradient).tolist())
    return record


def summarise(records):
    out = {}
    for kind in ("random", "nearby", "all"):
        chosen = [r for r in records if kind == "all" or r["kind"] == kind]
        entry = dict(n=len(chosen))
        for name in ("mlx", "jax32"):
            for key in ("d_overlap", "epsilon"):
                entry[f"{key}_{name}"] = f32.percentiles([r[f"{key}_{name}"] for r in chosen])
            for key in ("noise_weighted", "max_relative"):
                entry[f"{key}_{name}"] = f32.percentiles([x for r in chosen for x in r[f"{key}_{name}"]])
            entry[f"finite_{name}"] = all(r[f"finite_{name}"] for r in chosen)
        out[kind] = entry
    chosen = [r["grad_log_likelihood"] for r in records if "grad_log_likelihood" in r]
    if chosen:
        double = np.array([g["double"] for g in chosen])
        single = np.array([g["mlx"] for g in chosen])
        columns = [i for i, name in enumerate(NAMES) if name != "f_ref"]
        error = np.abs(single - double)[:, columns]
        out["gradients"] = dict(
            n=len(chosen), relative=f32.percentiles((error / np.abs(double[:, columns])).ravel()),
            over_typical=f32.percentiles((error / np.sqrt(np.mean(double[:, columns] ** 2, axis=0))).ravel()),
            finite=bool(np.all(np.isfinite(single))))
    return out


# ---------------------------------------------------------------------------
# LAL's grids and the conditioned waveforms
# ---------------------------------------------------------------------------

def check_grids(binaries):
    """The grids of the batch against the JAX driver's, binary by binary."""
    table = wf.parameter_table({name: np.array([b[name] for b in binaries], dtype=float) for name in wf.Binary._fields})
    out = {}
    for f_min, delta_t in ((20.0, 1 / 4096), (0.0, 1 / 4096)):
        grids, errors = mlx_driver.grids(table, delta_t, f_min, OPTIONS)
        same = same_errors = valid = 0
        t0 = []
        for i in range(len(table)):
            try:
                grid = _driver.setup(row(table, i), delta_t, f_min, OPTIONS)
            except wf.exceptions.WaveformError as error:
                same_errors += i in errors and str(errors[i]) == str(error)
                continue
            valid += 1
            same += i not in errors and (grids.n[i], grids.epoch[i]) == (grid.n, grid.epoch)
            t0.append(abs(grids.t0[i] - grid.t0))
        out[f"{f_min:g}"] = dict(n=len(table), valid=valid, same_length_and_epoch=same,
                                 lal_errors=len(table) - valid, same_errors=same_errors,
                                 max_t0_difference=float(max(t0)) if t0 else 0.0)
        print(f"grids from {f_min:g} Hz: {same}/{valid} the same, {same_errors}/{len(table) - valid} errors",
              flush=True)
    return out


CONDITIONED = [
    ("40 + 20, precessing, from 20 Hz", dict(mass_1=40.0, mass_2=20.0, spin_1=[0.4, -0.3, 0.2],
                                             spin_2=[0.1, 0.5, -0.3], luminosity_distance=400.0, inclination=0.7,
                                             phi_ref=0.1, f_ref=20.0), 20.0),
    ("25 + 60, f_ref = 0, from the start", dict(mass_1=25.0, mass_2=60.0, spin_1=[0.7, 0.0, 0.7],
                                                spin_2=[0.0, -0.9, 0.0], luminosity_distance=1500.0, inclination=2.1,
                                                phi_ref=-0.4, f_ref=0.0), 0.0),
    ("240 + 40, from the start", dict(mass_1=240.0, mass_2=40.0, spin_1=[0.0, 0.5, -0.5], spin_2=[0.3, 0.3, 0.3],
                                      luminosity_distance=3000.0, inclination=1.3, phi_ref=2.0, f_ref=0.0), 0.0),
]


def check_conditioned():
    from waveformx import conversions

    records = []
    for label, keywords, f_min in CONDITIONED:
        binary = wf.Binary(**keywords)
        for function, arguments in (("simulate_td", (1 / 4096, f_min)), ("simulate_fd", (0.0, f_min, 2048.0)),
                                    ("simulate_fd", (1 / 8, f_min, 2048.0))):
            expected = getattr(conversions, function)("NRSur7dq4", binary, OPTIONS, *arguments)
            result = getattr(nrsur, function)(binary, *arguments, OPTIONS)
            step = "delta_t" if function == "simulate_td" else "delta_f"
            same = (result.hplus.shape == expected.hplus.shape and result.epoch == expected.epoch
                    and getattr(result, step) == getattr(expected, step))
            errors = [max_relative_error(np.array(a), np.asarray(b)) for a, b in zip(result, expected)]
            records.append(dict(label=label, function=function, arguments=list(arguments), same_grid=bool(same),
                                samples=int(result.hplus.shape[-1]), relative_error=max(errors)))
            print(f"{function} {label} {arguments}: same grid {same}, error {max(errors):.1e}", flush=True)
    return records


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------

def mlx_time(function, repeats=7, number=1):
    """:func:`common.median_time` for MLX (evaluating the result)."""
    mx.eval(function())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            mx.eval(function())
        times.append((time.perf_counter() - start) / number)
    return float(np.median(times))


def first_call(function):
    start = time.perf_counter()
    mx.eval(function())
    return time.perf_counter() - start


#: The first calls in a new process: reading and converting the data, then
#: the waveform and its gradient, each compiling its Metal kernels.
FIRST_CALLS = """
import json, time
start = time.perf_counter()
import mlx.core as mx
import numpy as np
import waveformx as wf
from waveformx.models.nrsur7dq4 import mlx as nrsur
nrsur.default_data()
data = time.perf_counter() - start
times = mx.array(np.float32(-1.9) + np.float32(1 / 4096) * np.arange(8192, dtype=np.float32))
binary = wf.Binary(*(mx.array(np.asarray(x, dtype=np.float32)) for x in
                     (40.0, 20.0, (0.4, 0.2, 0.3), (-0.2, 0.3, -0.1), 500.0, 0.7, 1.2, 20.0)))
start = time.perf_counter()
mx.eval(nrsur.polarizations_at(binary, times))
value = time.perf_counter() - start
start = time.perf_counter()
mx.eval(mx.grad(lambda b: mx.sum(nrsur.polarizations_at(b, times)[0] ** 2))(binary))
grad = time.perf_counter() - start
print(json.dumps(dict(data=data, value=value, grad=grad)))
"""


def first_calls():
    """:data:`FIRST_CALLS` in a new process (s)."""
    out = subprocess.run([sys.executable, "-c", FIRST_CALLS], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


MLX_SIZES = (1, 16, 64, 256, 1024, 4096)
MLX_GRADIENT_SIZES = (1, 16, 64, 256, 1024)
JAX_SIZES = (1, 16, 64, 256)


def pure_timings():
    """Median times (s) of the pure function on 8192 samples at 4096 Hz
    (``h_+``) and of the gradient of its power, per call, against the
    batch size: MLX (compiled), JAX in float32 and float64."""
    rng = np.random.default_rng(1)
    base = dict(mass_1=40.0, mass_2=20.0, spin_1=[0.4, 0.2, 0.3], spin_2=[-0.2, 0.3, -0.1],
                luminosity_distance=500.0, inclination=0.7, phi_ref=1.2, f_ref=20.0)
    others = [dict(f32.draw(rng), f_ref=0.0) for _ in range(max(MLX_SIZES))]
    out = dict(samples=8192)
    times = mx.array(np.float32(-1.9) + np.float32(1 / 4096) * np.arange(8192, dtype=np.float32))

    def value(b):
        return nrsur.polarizations_at(b, times, OPTIONS)[0]

    def power(b):
        return mx.sum(value(b) ** 2)

    compiled_value, compiled_grad = mx.compile(value), mx.compile(mx.grad(power))
    single = mlx_binary(base)
    entry = dict(first_calls=first_calls(), first_call_compiled=first_call(lambda: compiled_value(single)),
                 value={}, grad={})
    print("mlx first calls", entry["first_calls"], flush=True)
    for size in MLX_SIZES:
        b = single if size == 1 else mlx_batch(others[:size])
        entry["value"][str(size)] = mlx_time(lambda: compiled_value(b), repeats=7 if size > 64 else 21,
                                             number=1 if size > 64 else 5)
        if size in MLX_GRADIENT_SIZES:
            entry["grad"][str(size)] = mlx_time(lambda: compiled_grad(b), repeats=5 if size > 64 else 11)
        print("mlx", size, f"{entry['value'][str(size)] * 1e3:.2f} ms",
              f"grad {entry['grad'].get(str(size), float('nan')) * 1e3:.1f} ms", flush=True)
    out["mlx"] = entry
    for name, dtype in (("jax_float32", jnp.float32), ("jax_float64", jnp.float64)):
        times_jax = jnp.asarray(-1.9, dtype) + jnp.asarray(1 / 4096, dtype) * jnp.arange(8192, dtype=dtype)

        def waveform(b):
            return f32.POLARIZATIONS(b, times_jax)[0]

        value_jit = jax.jit(waveform)
        batch_jit = jax.jit(jax.vmap(waveform))
        grad_jit = jax.jit(jax.grad(lambda b: jnp.sum(waveform(b) ** 2)))
        batch_grad_jit = jax.jit(jax.vmap(jax.grad(lambda b: jnp.sum(waveform(b) ** 2))))
        entry = dict(value={}, grad={})
        for size in JAX_SIZES:
            if size == 1:
                b = f32.binary_of(base, dtype)
                value_f, grad_f = value_jit, grad_jit
            else:
                b = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[f32.binary_of(o, dtype) for o in others[:size]])
                value_f, grad_f = batch_jit, batch_grad_jit
            entry["value"][str(size)] = median_time(lambda: value_f(b), repeats=7 if size > 1 else 21,
                                                    number=1 if size > 1 else 5)
            if size <= 64:
                entry["grad"][str(size)] = median_time(lambda: grad_f(b), repeats=5 if size > 1 else 11)
            print(name, size, f"{entry['value'][str(size)] * 1e3:.2f} ms",
                  f"grad {entry['grad'].get(str(size), float('nan')) * 1e3:.1f} ms", flush=True)
        out[name] = entry
    return out


def public_timings():
    """The public functions on LAL's grid: one binary at 4096 Hz from 20 Hz
    (``td_polarizations``, ``simulate_fd`` up to 2048 Hz in steps of 1/8
    Hz), and 1024 random binaries at 4096 Hz from the start of the
    surrogate (``batch_td_polarizations``)."""
    from waveformx import conversions

    binary = wf.Binary(40.0, 20.0, (0.4, 0.2, 0.3), (-0.2, 0.3, -0.1), 500.0, 0.7, 1.2, 20.0)
    out = {}
    out["td_polarizations"] = dict(
        mlx=mlx_time(lambda: tuple(nrsur.td_polarizations(binary, 1 / 4096, 20.0, OPTIONS)), repeats=21, number=5),
        jax=median_time(lambda: tuple(wf.td_polarizations("NRSur7dq4", 1 / 4096, 20.0, **binary._asdict())),
                        repeats=21, number=5))
    out["simulate_fd"] = dict(
        mlx=mlx_time(lambda: tuple(nrsur.simulate_fd(binary, 1 / 8, 20.0, 2048.0, OPTIONS)), repeats=11),
        jax=median_time(lambda: tuple(conversions.simulate_fd("NRSur7dq4", binary, OPTIONS, 1 / 8, 20.0, 2048.0)),
                        repeats=11))
    rng = np.random.default_rng(2)
    draws = [dict(f32.draw(rng), f_ref=0.0) for _ in range(1024)]
    table = wf.parameter_table({name: np.array([d[name] for d in draws], dtype=float) for name in wf.Binary._fields})

    def mlx_batch_call():
        batch = nrsur.batch_td_polarizations(table, 1 / 4096, 0.0, OPTIONS)
        return [w.hplus for w in batch]

    def jax_batch_call():
        batch = wf.batch_td_polarizations("NRSur7dq4", 1 / 4096, 0.0, table)
        return [w.hplus for w in batch]

    start = time.perf_counter()
    mlx_driver.grids(table, 1 / 4096, 0.0, OPTIONS)
    grids = time.perf_counter() - start
    out["batch"] = dict(n=len(table), samples=int(sum(mlx_driver.grids(table, 1 / 4096, 0.0, OPTIONS)[0].n)),
                        mlx_first_call=first_call(mlx_batch_call), mlx=mlx_time(mlx_batch_call, repeats=5),
                        mlx_grids=median_time(lambda: mlx_driver.grids(table, 1 / 4096, 0.0, OPTIONS)[0].n,
                                              repeats=5),
                        mlx_grids_first_call=grids,
                        jax=median_time(jax_batch_call, repeats=3))
    print(out, flush=True)
    return out


def timings():
    return dict(pure=pure_timings(), public=public_timings())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-pairs", type=int, default=150, help="pairs of each kind (as float32_nrsur7dq4.py)")
    parser.add_argument("--n-gradients", type=int, default=40, help="nearby pairs whose gradients are compared")
    parser.add_argument("--seed", type=int, default=20261023, help="the seed of float32_nrsur7dq4.py")
    parser.add_argument("--output", default="mlx_nrsur7dq4.json")
    parser.add_argument("--only-timings", action="store_true",
                        help="measure the timings again and keep the rest of the existing output")
    args = parser.parse_args()
    if wf.nrsur7dq4_data_file() is None:
        raise SystemExit("NRSur7dq4_v1.0.h5 not found: set NRSUR7DQ4_DATA")
    if args.only_timings:
        payload = json.loads((DATA / args.output).read_text())
        payload.update(timings=timings(), environment=environment(), mlx_environment=mlx_environment())
        write(args.output, payload)
        return
    data = _data.default_data()
    rng = np.random.default_rng(args.seed)
    cases = f32.draw_pairs(rng, args.n_pairs, data)
    records, compared = [], 0
    start = time.perf_counter()
    for index, case in enumerate(cases):
        gradients = case["kind"] == "nearby" and compared < args.n_gradients
        compared += gradients
        records.append(dict(measure(case, gradients), index=index))
        if index % 25 == 0:
            r = records[-1]
            print(f"{index:4d} {r['kind']:6s} overlap {r['overlap']: .3f} change: mlx {r['d_overlap_mlx']: .1e}, "
                  f"jax float32 {r['d_overlap_jax32']: .1e} ({time.perf_counter() - start:.0f} s)", flush=True)
    binaries = [b for case in cases for b in case["binaries"]]
    payload = dict(environment=environment(), mlx_environment=mlx_environment(), seed=args.seed,
                   n_pairs=args.n_pairs, f_low=f32.F_LOW, summary=summarise(records), grids=check_grids(binaries),
                   conditioned=check_conditioned(), timings=timings(), records=records)
    s = payload["summary"]["all"]
    print("change of the overlap: mlx", s["d_overlap_mlx"], "jax float32", s["d_overlap_jax32"])
    print("gradients:", payload["summary"].get("gradients"))
    write(args.output, payload)


if __name__ == "__main__":
    main()
