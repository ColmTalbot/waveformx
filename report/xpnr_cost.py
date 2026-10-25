#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The cost of compiling and running IMRPhenomXPNR: writes
``report/data/xpnr_cost.json``.

* tracing + lowering + compiling (one process per case, so that nothing is
  cached) of the public function on 64 frequencies, all modes: the
  waveform, ``jax.vmap`` over 4 binaries and ``jax.grad`` of a sum of
  squares, with the default interpolants of the PNR angles and with those
  of ``extended_phase_grid``;
* the run time of the jitted pure function (median of 21 calls), with
  either grid, at the benchmark binary and grid and for a light binary
  (8 + 5 solar masses from 10 Hz, ``df = 1/16`` Hz), whose grid of the PNR
  interpolants is large.

Usage::

    python report/xpnr_cost.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CASE = r'''
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import MODEL
grid, kind = sys.argv[1], sys.argv[2]
options = WaveformOptions(extended_phase_grid=grid == "phase")
F = jnp.asarray(np.linspace(20.0, 512.0, 64))
X0 = jnp.asarray([36.0, 29.0, 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 25.0])
def of(x):
    return Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                  inclination=x[9], phi_ref=x[10], f_ref=x[11])
f = lambda x: MODEL.polarizations(of(x), F, options)
if kind == "eval":
    g, x = f, X0
elif kind == "vmap":
    g, x = jax.vmap(f), jnp.stack([X0 * (1 + 0.01 * i) for i in range(4)])
else:
    g, x = jax.grad(lambda x: sum(jnp.sum(jnp.abs(h) ** 2) for h in f(x))), X0
t0 = time.perf_counter()
lowered = jax.jit(g).trace(x).lower()
compiled = lowered.compile()
t1 = time.perf_counter()
print(json.dumps(dict(seconds=t1 - t0)))
'''

RUN = r'''
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import model
CASES = {
    "benchmark": (Binary(mass_1=jnp.float64(36.0), mass_2=jnp.float64(29.0), spin_1=jnp.asarray([0.4, 0.2, 0.3]),
                         spin_2=jnp.asarray([-0.2, 0.3, -0.1]), luminosity_distance=jnp.float64(500.0),
                         inclination=jnp.float64(0.7), phi_ref=jnp.float64(1.2), f_ref=jnp.float64(20.0)),
                  jnp.asarray(np.arange(160, 8193) / 8.0)),
    "light": (Binary(mass_1=jnp.float64(8.0), mass_2=jnp.float64(5.0), spin_1=jnp.asarray([0.3, 0.3, 0.3]),
                     spin_2=jnp.asarray([-0.3, 0.2, 0.0]), luminosity_distance=jnp.float64(500.0),
                     inclination=jnp.float64(0.7), phi_ref=jnp.float64(1.2), f_ref=jnp.float64(10.0)),
              jnp.asarray(np.arange(160, 16385) / 16.0)),
}
out = {}
for case, (b, F) in CASES.items():
    for grid in ("default", "phase"):
        options = WaveformOptions(extended_phase_grid=grid == "phase")
        f = jax.jit(lambda b, F=F, o=options: model.polarizations(b, F, o))
        jax.block_until_ready(f(b))
        times = []
        for _ in range(21):
            t = time.perf_counter()
            jax.block_until_ready(f(b))
            times.append(time.perf_counter() - t)
        out[f"{case}_{grid}"] = float(np.median(times))
print(json.dumps(out))
'''


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    sys.path.insert(0, str(HERE))
    from common import environment, write

    compile_seconds = {}
    for grid in ("default", "phase"):
        for kind in ("eval", "vmap", "grad"):
            out = subprocess.run([sys.executable, "-c", CASE, grid, kind], capture_output=True, text=True, check=True)
            compile_seconds[f"{grid}_{kind}"] = json.loads(out.stdout.strip().splitlines()[-1])["seconds"]
            print(grid, kind, f"{compile_seconds[f'{grid}_{kind}']:.1f} s", flush=True)
    out = subprocess.run([sys.executable, "-c", RUN], capture_output=True, text=True, check=True)
    runtime = json.loads(out.stdout.strip().splitlines()[-1])
    print("runtime", runtime, flush=True)
    write("xpnr_cost.json", dict(environment=environment(), compile_seconds=compile_seconds, runtime=runtime))


if __name__ == "__main__":
    main()
