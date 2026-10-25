#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The memory IMRPhenomXPNR uses: writes ``report/data/xpnr_memory.json``.

For a light binary (1.62 + 1.08 M☉ from 10 Hz, f_ref = 20 Hz, the (2,2)
mode) on uniform frequencies up to 2048 Hz, and for the waveform, ``jax.vmap``
over 64 binaries, ``jax.jacfwd`` and ``jax.vmap(jax.jacfwd)``: the
temporary buffers of the compiled function (``memory_analysis()``), its
output, and the peak resident memory of the process (compilation included),
one process per case. With ``--src`` and ``--label`` the cases are measured
for another checkout (its ``src`` directory) and stored under that label,
e.g. the version before a change::

    python report/xpnr_memory.py                      # this checkout
    python report/xpnr_memory.py --src ../old/src --label before --compile-only

``--compile-only`` skips running the compiled functions (the peak resident
memory is then that of compilation): for versions whose batched Jacobian
does not fit in memory. With ``--mlx-python`` (a Python with MLX), the peak
GPU memory of the MLX port for 64 binaries (6 + 4 M☉) on 4096 frequencies,
values and Jacobian, is stored too.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CASE = r'''
import json, resource, sys
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import MODEL
kind, mass, f_min, n, batch, run = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), \
    int(sys.argv[5]), sys.argv[6] == "1"
options = WaveformOptions(mode_array=((2, 2), (2, -2)))
q = 1.5
F = jnp.asarray(np.linspace(f_min, min(0.3 / (mass * 4.925491e-6), 2048.0), n))
x0 = jnp.asarray([mass * q / (1 + q), mass / (1 + q), 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 20.0])
def of(x):
    return Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                  inclination=x[9], phi_ref=x[10], f_ref=x[11])
f = lambda x: MODEL.polarizations(of(x), F, options)
xs = jnp.stack([x0 * (1 + 0.002 * (i % 7)) for i in range(batch)])
g, x = {"eval": (f, x0), "vmap": (jax.vmap(f), xs), "jacfwd": (jax.jacfwd(f), x0),
        "vmap_jacfwd": (jax.vmap(jax.jacfwd(f)), xs)}[kind]
compiled = jax.jit(g).trace(x).lower().compile()
analysis = compiled.memory_analysis()
if run:
    jax.block_until_ready(compiled(x))
scale = 2.0 ** 20 if sys.platform == "darwin" else 2.0 ** 10
print(json.dumps(dict(temp_mb=analysis.temp_size_in_bytes / 2 ** 20, output_mb=analysis.output_size_in_bytes / 2 ** 20,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale)))
'''

MLX_CASE = r'''
import json, sys
import numpy as np
import mlx.core as mx
import waveformx as wf
from waveformx.models.xpnr import mlx as xpnr_mlx
batch, s = 64, 1 + 0.002 * (np.arange(64) % 7)
b = wf.Binary(mass_1=6.0 * s, mass_2=4.0 * s, spin_1=np.tile([0.4, 0.2, 0.3], (batch, 1)),
              spin_2=np.tile([-0.2, 0.3, -0.1], (batch, 1)), luminosity_distance=500.0 * s, inclination=0.7 * s,
              phi_ref=1.2 * s, f_ref=20.0 * s)
f = np.linspace(20.0, 1024.0, 4096)
out = {}
for name, function in (("eval_mb", xpnr_mlx.polarizations), ("jacobian_mb", xpnr_mlx.polarizations_and_jacobian)):
    mx.eval(function(b, f, mode_array=((2, 2), (2, -2))))
    mx.reset_peak_memory()
    mx.eval(function(b, f, mode_array=((2, 2), (2, -2))))
    out[name] = mx.get_peak_memory() / 2 ** 20
print(json.dumps(out))
'''

#: (kind, total mass, f_min, frequencies, binaries)
CASES = [
    ("eval", 2.7, 10.0, 1024, 1), ("eval", 2.7, 10.0, 16384, 1), ("eval", 2.7, 10.0, 262144, 1),
    ("jacfwd", 2.7, 10.0, 1024, 1), ("jacfwd", 2.7, 10.0, 16384, 1), ("jacfwd", 2.7, 10.0, 262144, 1),
    ("vmap", 2.7, 10.0, 1024, 64), ("vmap", 2.7, 10.0, 16384, 64),
    ("vmap_jacfwd", 2.7, 10.0, 1024, 64), ("vmap_jacfwd", 2.7, 10.0, 16384, 64),
    ("eval", 60.0, 20.0, 16384, 1), ("jacfwd", 60.0, 20.0, 16384, 1),
]


def key(case):
    return "{}_{:g}_{:g}_{}_{}".format(*case)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", default=str(ROOT / "src"), help="the src directory of the version measured")
    parser.add_argument("--label", default="after", help="the label of the version measured")
    parser.add_argument("--compile-only", action="store_true", help="do not run the compiled functions")
    parser.add_argument("--mlx-python", help="a Python with MLX: also measure the MLX port")
    parser.add_argument("--mlx-only", action="store_true", help="measure the MLX port only (with --mlx-python)")
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    from common import DATA, environment, write

    path = DATA / "xpnr_memory.json"
    data = json.loads(path.read_text()) if path.exists() else dict(cases=[list(c) for c in CASES], results={})
    env = dict(os.environ, PYTHONPATH=args.src, WAVEFORMX_TEST_CACHE="")
    commit = subprocess.run(["git", "-C", args.src, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    results = {}
    if args.mlx_only:
        results = data["results"].get(args.label, {}).get("values", {})
    for case in [] if args.mlx_only else CASES:
        out = subprocess.run([sys.executable, "-c", CASE, *map(str, case), "0" if args.compile_only else "1"],
                             capture_output=True, text=True, env=env)
        if out.returncode:
            print(key(case), "failed:", out.stderr.strip().splitlines()[-1:], flush=True)
            continue
        results[key(case)] = json.loads(out.stdout.strip().splitlines()[-1])
        print(key(case), results[key(case)], flush=True)
    if args.mlx_python:
        out = subprocess.run([args.mlx_python, "-c", MLX_CASE], capture_output=True, text=True, env=env, check=True)
        results["mlx"] = json.loads(out.stdout.strip().splitlines()[-1])
        print("mlx", results["mlx"], flush=True)
    compile_only = args.compile_only
    if args.mlx_only:
        compile_only = data["results"].get(args.label, {}).get("compile_only", compile_only)
    data["results"][args.label] = dict(commit=commit, compile_only=compile_only, values=results)
    data["environment"] = environment()
    write("xpnr_memory.json", data)


if __name__ == "__main__":
    main()
