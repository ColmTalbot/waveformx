#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Compilation and evaluation times of waveformjax, against waveformrs and
LALSimulation, on the binaries and grids of waveformrs's Python benchmark
(``python/benchmarks/benchmark.py``).

* single waveforms: IMRPhenomXPNR on 8033 frequencies (20-1024 Hz in steps
  of 1/8 Hz), NRSur7dq4 in the time domain (4096 Hz from 20 Hz), and the
  conversions (``simulate_td`` of IMRPhenomXPNR at 4096 Hz from 20 Hz,
  ``simulate_fd`` of NRSur7dq4 up to 2048 Hz from 20 Hz). For waveformjax:
  the first call (LAL's checks, tracing and compilation), later calls of the
  public function (checks on the host, the compiled model, the copy of the
  result), and the compiled pure function alone;
* batches: ``jax.vmap`` of the pure functions over batches of 1 to 256
  binaries around the benchmark binary (compilation and throughput), the
  public batch functions for 64 binaries, against waveformrs's batches (one
  and all threads) and a loop over LALSimulation;
* the share of the CPU that XLA keeps busy (process CPU time over wall
  time).

Single waveforms are timed over a cycle of 16 binaries around the benchmark
binary, so that no code reuses the result of a previous call (waveformrs
keeps the last NRSur7dq4 evaluation of a workspace). LALSimulation's
conversions of IMRPhenomXPNR are timed with multibanding off (the work
waveformjax and waveformrs do) and with its default, multibanding on. Run
with ``XLA_FLAGS="--xla_cpu_multi_thread_eigen=false
intra_op_parallelism_threads=1" ... --parts single --output
timings_one_core.json`` for JAX on one core.

LALSimulation runs with ``OMP_NUM_THREADS=1``. The times are medians of
several repeats on an otherwise idle machine.

Usage::

    python report/benchmark.py [--quick]
"""

import argparse
import time

import jax
import jax.numpy as jnp
import lal
import lalsimulation as lalsim
import numpy as np

from common import (BENCHMARK_FREQUENCIES, BINARY, NRSUR, binary_of, cpu_share, environment, lal_sequence,
                    lal_td_arguments, median_time, rs, wf, write)

XPNR_OPTIONS = dict(pnr_parity_fix=False)


def draws(binary, n, seed=7):
    """``n`` binaries around ``binary`` (waveformrs's benchmark draws)."""
    rng = np.random.default_rng(seed)
    spins = rng.uniform(-0.5, 0.5, (2, n, 3))
    return dict(
        mass_1=binary["mass_1"] * rng.uniform(0.9, 1.1, n), mass_2=binary["mass_2"] * rng.uniform(0.9, 1.1, n),
        spin_1=spins[0], spin_2=spins[1], luminosity_distance=np.full(n, binary["luminosity_distance"]),
        inclination=rng.uniform(0.0, np.pi, n), phi_ref=rng.uniform(0.0, 2 * np.pi, n),
        f_ref=np.full(n, binary["f_ref"]),
    )


def row(parameters, i):
    return {k: (v[i] if np.ndim(v) > 0 else v) for k, v in parameters.items()}


def batch_binary(parameters):
    return wf.Binary(
        mass_1=jnp.asarray(parameters["mass_1"]), mass_2=jnp.asarray(parameters["mass_2"]),
        spin_1=jnp.asarray(parameters["spin_1"]), spin_2=jnp.asarray(parameters["spin_2"]),
        luminosity_distance=jnp.asarray(parameters["luminosity_distance"]),
        inclination=jnp.asarray(parameters["inclination"]), phi_ref=jnp.asarray(parameters["phi_ref"]),
        f_ref=jnp.asarray(parameters["f_ref"]),
    )


def first_call(function):
    start = time.perf_counter()
    jax.block_until_ready(function())
    return time.perf_counter() - start


def lal_nrsur(binary, delta_t, f_min):
    return lambda: lalsim.SimInspiralChooseTDWaveform(*lal_td_arguments(binary, delta_t, f_min), None,
                                                      lalsim.NRSur7dq4)


CYCLE = 16


def rows_near(binary, n=CYCLE):
    """``n`` binaries around ``binary`` as plain keyword dicts."""
    parameters = draws(binary, n, seed=11)
    out = []
    for i in range(n):
        r = row(parameters, i)
        out.append({k: tuple(float(x) for x in v) if np.ndim(v) else float(v) for k, v in r.items()})
    return out


def cycle_time(function, items, repeats):
    """Median over ``repeats`` of the mean time per item of a pass over
    ``items`` (after one pass, which compiles what needs compiling): every
    call sees a new binary, so that no code can reuse a previous result."""
    for item in items:
        jax.block_until_ready(function(item))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for item in items:
            jax.block_until_ready(function(item))
        times.append((time.perf_counter() - start) / len(items))
    return float(np.median(times))


def lal_params(multibanding):
    params = lal.CreateDict()
    if not multibanding:
        lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(params, 0)
        lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(params, 0)
    return params


def single(repeats):
    """One waveform per call, for binaries around the benchmark binary."""
    from waveformjax._api import _request
    from waveformjax.models import get

    out = {}
    frequencies = BENCHMARK_FREQUENCIES
    jf = jnp.asarray(frequencies)
    rows = rows_near(BINARY)
    entry = dict(first_call=first_call(lambda: wf.polarizations("IMRPhenomXPNR", frequencies, **XPNR_OPTIONS,
                                                                **BINARY)))
    entry["public"] = cycle_time(lambda r: wf.polarizations("IMRPhenomXPNR", frequencies, **XPNR_OPTIONS, **r), rows,
                                 repeats)
    model = get("IMRPhenomXPNR")
    requests = [_request(None, dict(r, **XPNR_OPTIONS)) for r in rows]
    entry["checks"] = cycle_time(lambda q: model.validate(q[0], frequencies, q[1]), requests, repeats)
    pure = jax.jit(wf.polarizations_function("IMRPhenomXPNR", **XPNR_OPTIONS))
    binaries = [binary_of(r) for r in rows]
    entry["pure"] = cycle_time(lambda b: pure(b, jf), binaries, repeats)
    entry["cpu_share"] = cpu_share(lambda: pure(binaries[0], jf))
    entry["lal"] = cycle_time(lambda r: lal_sequence(r, frequencies), rows, repeats)
    if rs is not None:
        entry["waveformrs"] = cycle_time(lambda r: rs.polarizations("IMRPhenomXPNR", frequencies, **XPNR_OPTIONS, **r),
                                         rows, repeats)
        grid = rs.FrequencyGrid(frequencies)
        entry["waveformrs_grid"] = cycle_time(
            lambda r: rs.polarizations_on_grid("IMRPhenomXPNR", grid, **XPNR_OPTIONS, **r), rows, repeats)
    out["IMRPhenomXPNR"] = entry
    print("IMRPhenomXPNR", {k: round(v, 5) for k, v in entry.items()}, flush=True)

    if wf.nrsur7dq4_data_file() is not None:
        delta_t, f_min = 1 / 4096, 20.0
        rows = rows_near(NRSUR)
        entry = dict(first_call=first_call(lambda: wf.td_polarizations("NRSur7dq4", delta_t, f_min, **NRSUR).hplus))
        entry["public"] = cycle_time(lambda r: wf.td_polarizations("NRSur7dq4", delta_t, f_min, **r).hplus, rows,
                                     repeats)
        model = get("NRSur7dq4")
        requests = [_request(None, dict(r)) for r in rows]
        entry["checks"] = cycle_time(lambda q: model.validate(q[0], delta_t, f_min, q[1]), requests, repeats)
        reference = wf.td_polarizations("NRSur7dq4", delta_t, f_min, **NRSUR)
        times = jnp.asarray(np.asarray(reference.times))
        entry["samples"] = int(times.shape[0])
        pure = jax.jit(wf.td_polarizations_function("NRSur7dq4"))
        binaries = [binary_of(r) for r in rows]
        entry["pure"] = cycle_time(lambda b: pure(b, times), binaries, repeats)
        entry["cpu_share"] = cpu_share(lambda: pure(binaries[0], times))
        entry["lal"] = cycle_time(lambda r: lal_nrsur(r, delta_t, f_min)(), rows, repeats)
        if rs is not None:
            entry["waveformrs"] = cycle_time(lambda r: rs.td_polarizations("NRSur7dq4", delta_t, f_min, **r), rows,
                                             repeats)
        out["NRSur7dq4"] = entry
        print("NRSur7dq4", {k: round(v, 5) for k, v in entry.items()}, flush=True)

    conversions = [
        ("simulate_td IMRPhenomXPNR", "IMRPhenomXPNR", "simulate_td", (1 / 4096, 20.0), BINARY, XPNR_OPTIONS),
        ("simulate_fd NRSur7dq4", "NRSur7dq4", "simulate_fd", (0.0, 20.0, 2048.0), NRSUR, {}),
    ]
    for label, approximant, name, arguments, binary, options in conversions:
        if approximant == "NRSur7dq4" and wf.nrsur7dq4_data_file() is None:
            continue
        rows = rows_near(binary)
        ours = getattr(wf, name)
        entry = dict(first_call=first_call(lambda: ours(approximant, *arguments, **options, **binary).hplus))
        entry["public"] = cycle_time(lambda r: ours(approximant, *arguments, **options, **r).hplus, rows, repeats)
        code = lalsim.GetApproximantFromString(approximant)

        def lal_call(r, multibanding=False):
            params = lal_params(multibanding) if approximant == "IMRPhenomXPNR" else None
            if name == "simulate_td":
                return lalsim.SimInspiralTD(*lal_td_arguments(r, *arguments), params, code)
            return lalsim.SimInspiralFD(*lal_td_arguments(r, 0.0, 0.0)[:14], *arguments, r["f_ref"], params, code)

        entry["lal"] = cycle_time(lal_call, rows, repeats)
        if approximant == "IMRPhenomXPNR":
            entry["lal_multibanding"] = cycle_time(lambda r: lal_call(r, True), rows, repeats)
        if rs is not None:
            theirs = getattr(rs, name)
            entry["waveformrs"] = cycle_time(lambda r: theirs(approximant, *arguments, **options, **r), rows, repeats)
        out[label] = entry
        print(label, {k: round(v, 5) for k, v in entry.items()}, flush=True)
    return out


def batches(sizes, repeats):
    out = {"IMRPhenomXPNR": {}, "NRSur7dq4": {}}
    frequencies = jnp.asarray(BENCHMARK_FREQUENCIES)
    model = wf.polarizations_function("IMRPhenomXPNR", **XPNR_OPTIONS)
    for n in sizes:
        parameters = draws(BINARY, n)
        binaries = batch_binary(parameters)
        function = jax.jit(jax.vmap(model, in_axes=(0, None)))
        compile_time = first_call(lambda: function(binaries, frequencies))
        seconds = median_time(lambda: function(binaries, frequencies), max(3, repeats // 2))
        share = cpu_share(lambda: function(binaries, frequencies), repeats=2)
        out["IMRPhenomXPNR"][str(n)] = dict(compile=compile_time, seconds=seconds, per_second=n / seconds,
                                            cpu_share=share)
        print(f"IMRPhenomXPNR vmap {n}: compile {compile_time:.1f} s, {n / seconds:.1f} waveforms/s, "
              f"cpu share {share:.1f}", flush=True)

    if wf.nrsur7dq4_data_file() is not None:
        # LAL's output grid of the benchmark binary
        times = jnp.asarray(np.asarray(wf.td_polarizations("NRSur7dq4", 1 / 4096, 20.0, **NRSUR).times))
        model = wf.td_polarizations_function("NRSur7dq4")
        for n in sizes:
            parameters = draws(NRSUR, n)
            binaries = batch_binary(parameters)
            function = jax.jit(jax.vmap(model, in_axes=(0, None)))
            compile_time = first_call(lambda: function(binaries, times))
            seconds = median_time(lambda: function(binaries, times), max(3, repeats // 2))
            share = cpu_share(lambda: function(binaries, times), repeats=2)
            out["NRSur7dq4"][str(n)] = dict(compile=compile_time, seconds=seconds, per_second=n / seconds,
                                            cpu_share=share, samples=int(times.shape[0]))
            print(f"NRSur7dq4 vmap {n}: compile {compile_time:.1f} s, {n / seconds:.1f} waveforms/s, "
                  f"cpu share {share:.1f}", flush=True)
    return out


def public_batches(n, repeats):
    out = {}
    parameters = draws(BINARY, n)
    frequencies = BENCHMARK_FREQUENCIES
    entry = dict(n=n)
    entry["first_call"] = first_call(lambda: wf.batch_polarizations("IMRPhenomXPNR", frequencies, parameters,
                                                                    **XPNR_OPTIONS).hplus)
    entry["waveformjax"] = n / median_time(
        lambda: wf.batch_polarizations("IMRPhenomXPNR", frequencies, parameters, **XPNR_OPTIONS).hplus, repeats)
    lal_times = [median_time(lambda: lal_sequence(row(parameters, i), frequencies), 3) for i in range(min(n, 8))]
    entry["lal"] = 1 / float(np.mean(lal_times))
    if rs is not None:
        for threads in (1, 0):
            entry[f"waveformrs_{threads}"] = n / median_time(
                lambda: rs.batch_polarizations("IMRPhenomXPNR", frequencies, parameters, threads=threads,
                                               **XPNR_OPTIONS), repeats)
    out["IMRPhenomXPNR"] = entry
    print("public batch IMRPhenomXPNR", entry, flush=True)
    if wf.nrsur7dq4_data_file() is not None:
        parameters = draws(NRSUR, n)
        delta_t, f_min = 1 / 4096, 20.0
        entry = dict(n=n)
        entry["first_call"] = first_call(lambda: wf.batch_td_polarizations("NRSur7dq4", delta_t, f_min,
                                                                           parameters).waveforms[0].hplus)
        entry["waveformjax"] = n / median_time(
            lambda: wf.batch_td_polarizations("NRSur7dq4", delta_t, f_min, parameters).waveforms[-1].hplus,
            repeats)
        lal_times = [median_time(lal_nrsur(row(parameters, i), delta_t, f_min), 3) for i in range(min(n, 8))]
        entry["lal"] = 1 / float(np.mean(lal_times))
        if rs is not None:
            for threads in (1, 0):
                entry[f"waveformrs_{threads}"] = n / median_time(
                    lambda: rs.batch_td_polarizations("NRSur7dq4", delta_t, f_min, parameters, threads=threads),
                    repeats)
        out["NRSur7dq4"] = entry
        print("public batch NRSur7dq4", entry, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--parts", nargs="+", default=["single", "batches", "public_batches"])
    parser.add_argument("--output", default="timings.json")
    args = parser.parse_args()
    repeats = 3 if args.quick else 7
    out = dict(environment=environment(), lal_omp_threads=1, n_frequencies=len(BENCHMARK_FREQUENCIES))
    if "single" in args.parts:
        out["single"] = single(repeats)
    if "batches" in args.parts:
        out["batches"] = batches(args.sizes, repeats)
    if "public_batches" in args.parts:
        out["public_batches"] = public_batches(64, max(3, repeats // 2))
    write(args.output, out)


if __name__ == "__main__":
    main()
