#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The lean IMRPhenomXPNR (``lean=True``) against the extended model, and
the cost of compiling and running it against the default model: writes
``report/data/lean_xpnr.json``.

* the weighted mismatch (aLIGO design, the larger of the polarizations) of
  the lean and extended models over random binaries of
  ``report/extended``;
* tracing + lowering + compiling (one process per case, so that nothing is
  cached) of the public function on 64 frequencies, all modes: the
  waveform, ``jax.vmap`` over 4 binaries and ``jax.grad`` of a sum of
  squares;
* the run time of the jitted pure function at the benchmark binary and
  grid (median of 21 calls).

Usage::

    python report/lean_xpnr.py [--n 1400] [--seed 4]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

CASE = r'''
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import MODEL
name, kind = sys.argv[1], sys.argv[2]
options = {"lean": WaveformOptions(lean=True), "default": WaveformOptions()}[name]
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
F = jnp.asarray(np.arange(160, 8193) / 8.0)
B = Binary(mass_1=jnp.float64(36.0), mass_2=jnp.float64(29.0), spin_1=jnp.asarray([0.4, 0.2, 0.3]),
           spin_2=jnp.asarray([-0.2, 0.3, -0.1]), luminosity_distance=jnp.float64(500.0),
           inclination=jnp.float64(0.7), phi_ref=jnp.float64(1.2), f_ref=jnp.float64(20.0))
out = {}
for name, options in (("default", WaveformOptions()), ("extended", WaveformOptions(extended_spintaylor=True)),
                      ("lean", WaveformOptions(lean=True))):
    f = jax.jit(lambda b, o=options: model.polarizations(b, F, o))
    jax.block_until_ready(f(B))
    times = []
    for _ in range(21):
        t = time.perf_counter()
        jax.block_until_ready(f(B))
        times.append(time.perf_counter() - t)
    out[name] = float(np.median(times))
print(json.dumps(out))
'''


def mismatches(n, seed):
    sys.path.insert(0, str(HERE / "extended"))
    import numpy as np
    import xcommon as x
    from xcommon import jax

    lean = jax.jit(x.polarizations_fn("lean"))
    extended = jax.jit(x.polarizations_fn("extended"))
    values, finite_differs = [], 0
    for d in x.draws(n, seed):
        f = x.frequencies(d, 1024)
        b = x.binary(d)
        a = [np.asarray(v) for v in lean(b, f)]
        e = [np.asarray(v) for v in extended(b, f)]
        fa, fe = bool(np.isfinite(a[0]).all()), bool(np.isfinite(e[0]).all())
        if fa != fe:
            finite_differs += 1
            continue
        if fa:
            values.append(max(x.weighted_mismatch(a[0], e[0], f), x.weighted_mismatch(a[1], e[1], f)))
    return values, finite_differs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    from common import environment, write

    compile_seconds = {}
    for name in ("lean", "default"):
        for kind in ("eval", "vmap", "grad"):
            out = subprocess.run([sys.executable, "-c", CASE, name, kind], capture_output=True, text=True, check=True)
            compile_seconds[f"{name}_{kind}"] = json.loads(out.stdout.strip().splitlines()[-1])["seconds"]
            print(name, kind, f"{compile_seconds[f'{name}_{kind}']:.1f} s", flush=True)
    out = subprocess.run([sys.executable, "-c", RUN], capture_output=True, text=True, check=True)
    runtime = json.loads(out.stdout.strip().splitlines()[-1])
    print("runtime", runtime, flush=True)
    t = time.time()
    values, finite_differs = mismatches(args.n, args.seed)
    print(f"lean vs extended: {len(values)} binaries, max {max(values):.2e}, {finite_differs} differ in finiteness "
          f"({time.time() - t:.0f} s)", flush=True)
    write("lean_xpnr.json", dict(environment=environment(), compile_seconds=compile_seconds, runtime=runtime,
                                 n=args.n, seed=args.seed, mismatches=values, finite_differs=finite_differs))


if __name__ == "__main__":
    main()
