#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The time of one long IMRPhenomXPNR waveform: writes
``report/data/xpnr_long.json``.

A precessing neutron star–black hole binary (10 + 1.4 solar masses, spins
(0.5, 0.2, 0.4) and (0.02, -0.01, 0.03), inclination 0.7, phi_ref 1.2) on
uniform grids up to 2048 Hz: from ``f_min = f_ref = 10`` Hz with ``df =
1/64`` Hz (130,433 frequencies) and ``1/128`` Hz (260,865), and from 20 Hz
with ``df = 1/32`` Hz (64,897). Median of 7 calls after a warm-up of:

* the jitted pure function (JAX, float64) and the public
  ``waveformx.polarizations`` (its checks on the host, then the compiled
  model);
* the MLX port (float32, on the GPU), with ``--mlx-python``;
* waveformrs (``polarizations`` and ``polarizations_on_grid``) and
  LALSimulation's ``SimInspiralChooseFDWaveformSequence`` (no multibanding)
  and ``SimInspiralChooseFDWaveform`` (its multibanding), single-threaded.

Usage::

    python report/xpnr_long.py [--mlx-python PYTHON]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

NSBH = dict(mass_1=10.0, mass_2=1.4, spin_1=(0.5, 0.2, 0.4), spin_2=(0.02, -0.01, 0.03), luminosity_distance=200.0,
            inclination=0.7, phi_ref=1.2)
#: ``(f_min, 1 / df)`` of the grids.
GRIDS = ((10.0, 64), (10.0, 128), (20.0, 32))
REPEATS = 7


def grid(f_min, duration):
    df = 1.0 / duration
    return np.arange(round(f_min / df), round(2048.0 / df) + 1) * df


def median_time(function, block=lambda x: x):
    block(function())
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        block(function())
        times.append(time.perf_counter() - t)
    return float(np.median(times))


MLX = r'''
import json, sys, time
import numpy as np
import mlx.core as mx
import waveformx as wf
from waveformx.models.xpnr import mlx as xpnr_mlx
nsbh, grids, repeats = json.loads(sys.argv[1]), json.loads(sys.argv[2]), int(sys.argv[3])
out = {}
for f_min, duration in grids:
    df = 1.0 / duration
    f = (np.arange(round(f_min / df), round(2048.0 / df) + 1) * df).astype(np.float32)
    b = wf.Binary(nsbh["mass_1"], nsbh["mass_2"], nsbh["spin_1"], nsbh["spin_2"], nsbh["luminosity_distance"],
                  nsbh["inclination"], nsbh["phi_ref"], f_min)
    call = lambda: xpnr_mlx.polarizations(b, f)
    mx.eval(call())
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        mx.eval(call())
        times.append(time.perf_counter() - t)
    out[f"{f_min:g}_{duration}"] = float(np.median(times))
print(json.dumps(out))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mlx-python", default=None)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(HERE))
    import jax
    import jax.numpy as jnp
    import lal
    import lalsimulation as lalsim
    from common import binary_of, environment, lal_sequence, max_relative_error, rs, wf, write

    pure = jax.jit(wf.polarizations_function("IMRPhenomXPNR"))
    results = {}
    for f_min, duration in GRIDS:
        key = f"{f_min:g}_{duration}"
        f = grid(f_min, duration)
        keywords = dict(NSBH, f_ref=f_min)
        b, jf = binary_of(keywords), jnp.asarray(f)
        row = dict(n_frequencies=len(f))
        row["jax"] = median_time(lambda: pure(b, jf), jax.block_until_ready)
        row["jax_public"] = median_time(lambda: wf.polarizations("IMRPhenomXPNR", f, **keywords),
                                        jax.block_until_ready)
        if rs is not None:
            row["waveformrs"] = median_time(lambda: rs.polarizations("IMRPhenomXPNR", f, **keywords))
            on_grid = rs.FrequencyGrid(f)
            row["waveformrs_grid"] = median_time(lambda: rs.polarizations_on_grid("IMRPhenomXPNR", on_grid, **keywords))
        row["lal"] = median_time(lambda: lal_sequence(keywords, f))

        def lal_multibanding():
            k = keywords
            return lalsim.SimInspiralChooseFDWaveform(
                k["mass_1"] * lal.MSUN_SI, k["mass_2"] * lal.MSUN_SI, *k["spin_1"], *k["spin_2"],
                k["luminosity_distance"] * 1e6 * lal.PC_SI, k["inclination"], k["phi_ref"], 0.0, 0.0, 0.0,
                1.0 / duration, f_min, 2048.0, k["f_ref"], None, lalsim.IMRPhenomXPNR)

        row["lal_multibanding"] = median_time(lal_multibanding)
        hp, hc = pure(b, jf)
        lp, lc = lal_sequence(keywords, f)
        row["max_relative_error_lal"] = max(max_relative_error(np.asarray(hp) * 1e-21, lp),
                                            max_relative_error(np.asarray(hc) * 1e-21, lc))
        results[key] = row
        print(key, {k: (round(v * 1e3, 1) if k not in ("n_frequencies", "max_relative_error_lal") else v)
                    for k, v in row.items()}, flush=True)
    if args.mlx_python:
        out = subprocess.run([args.mlx_python, "-c", MLX, json.dumps(NSBH), json.dumps(GRIDS), str(REPEATS)],
                             capture_output=True, text=True, check=True)
        for key, seconds in json.loads(out.stdout.strip().splitlines()[-1]).items():
            results[key]["mlx"] = seconds
            print(key, "mlx", round(seconds * 1e3, 1), flush=True)
    write("xpnr_long.json", dict(environment=environment(), binary=NSBH, results=results))


if __name__ == "__main__":
    main()
