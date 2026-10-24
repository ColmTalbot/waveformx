#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The mass-scaled reference frequency of the lean IMRPhenomXPNR
(``WaveformOptions(lean=True, reference_mf=0.01)``): writes
``report/data/reference_xpnr.json``.

* finite waveforms per mode array, over random binaries (mass ratios to
  30, total masses 3-400, spins to 0.99, the band from 10-30 Hz to
  1024 Hz), with LAL's convention (``f_ref = f_min``) and the mass-scaled
  one;
* against LAL with the parity fix (``PATCHED_LALSIMULATION``, optional):
  binaries drawn at ``f_ref = 20 Hz`` (q to 12, 20-90 solar masses, spins
  to 0.9) converted to the mass-scaled reference
  (``to_mass_scaled_reference``); the lean model with it against LAL given
  the binary converted to ``f_ref = 0.01 / M`` (inside the band), and LAL
  at that reference against LAL at 20 Hz (the model's own dependence on
  the reference). Mismatches with white noise, 20-1024 Hz in steps of
  1/8 Hz, maximised over time and phase, the polarizations in the same
  basis;
  LAL at 20 Hz is also compared with the consistent model directly
  (``lal_20_vs_lean``: the lean model given the binary at 20 Hz carries it
  to its canonical reference, so this is the distance from LAL's test
  point), and LAL at a test point near the canonical reference (the same
  spins and angles given at ``f_ref = M_near / M``, ``M_near`` off 0.01
  by a relative offset log-uniform in [0.002, 0.2]) with the consistent
  model (``lal_near_vs_lean``): how LAL's distance from the consistent
  model grows with the distance of its reference from the canonical one;
* compilation (one process per convention, no cache) and the run time of
  the jitted pure function per total mass (all modes, 20-1024 Hz in steps
  of 1/8 Hz, median of 9 calls): LAL's convention, the canonical
  ``reference_mf = 0.01`` and the references 0.006 and 0.002, which the
  model first carries to 0.01.

Usage::

    OMP_NUM_THREADS=1 python report/reference_xpnr.py [--n 256] [--n-lal 40]
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
REF = 0.01
MTSUN_SI = 4.925490947641267e-06
MODES = {"21": ((2, 1), (2, -1)), "32": ((3, 2), (3, -2)), "33": ((3, 3), (3, -3)), "44": ((4, 4), (4, -4)),
         "all": None}
MASSES = (10.0, 20.0, 60.0, 150.0, 400.0)

TIMING = r'''
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from waveformx import Binary
from waveformx.models.xpnr import lean
ref = None if sys.argv[1] == "lal" else float(sys.argv[1])
F = jnp.asarray(np.arange(20.0, 1024.0, 0.125))
f = jax.jit(lambda b, F: lean.polarizations(b, F, reference_mf=ref))
def binary(M, q=2.0):
    a = lambda x: jnp.asarray(x, jnp.float64)
    return Binary(a(M * q / (1 + q)), a(M / (1 + q)), a([0.5, -0.3, 0.2]), a([0.2, 0.3, -0.4]), a(500.), a(0.8),
                  a(0.4), a(0.0 if ref else 20.0))
t = time.perf_counter()
jax.block_until_ready(f(binary(30.0), F))
out = {"compile": time.perf_counter() - t, "runtime": {}}
for M in json.loads(sys.argv[2]):
    b = binary(M)
    jax.block_until_ready(f(b, F))
    ts = []
    for _ in range(9):
        t = time.perf_counter()
        jax.block_until_ready(f(b, F))
        ts.append(time.perf_counter() - t)
    out["runtime"][str(M)] = float(np.median(ts))
print(json.dumps(out))
'''


def finite_counts(n, seed):
    import jax
    import jax.numpy as jnp

    from waveformx import Binary
    from waveformx.models.xpnr import lean

    rng = np.random.default_rng(seed)
    q = np.exp(rng.uniform(0, np.log(30), n))
    mass = np.exp(rng.uniform(np.log(3), np.log(400), n))

    def spins():
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1)[:, None] * rng.uniform(0, 0.99, n)[:, None]

    f_min = rng.uniform(10, 30, n)
    batch = Binary(jnp.asarray(mass * q / (1 + q)), jnp.asarray(mass / (1 + q)), jnp.asarray(spins()),
                   jnp.asarray(spins()), jnp.full(n, 500.0), jnp.asarray(rng.uniform(0, np.pi, n)),
                   jnp.asarray(rng.uniform(0, 2 * np.pi, n)), jnp.asarray(f_min))
    frequencies = jnp.asarray(np.stack([np.geomspace(f, 1024.0, 1000) for f in f_min]))
    counts = {}
    for key, modes in MODES.items():
        for convention, ref in (("lal", None), ("scaled", REF)):
            b = batch if ref is None else batch._replace(f_ref=jnp.zeros(n))
            hp, hc = jax.jit(jax.vmap(lambda b, f: lean.polarizations(b, f, mode_array=modes, reference_mf=ref)))(
                b, frequencies)
            finite = np.all(np.isfinite(np.asarray(hp)), axis=1) & np.all(np.isfinite(np.asarray(hc)), axis=1)
            counts[f"{convention}_{key}"] = int(finite.sum())
            print(convention, key, counts[f"{convention}_{key}"], "/", n, flush=True)
    return counts


def mismatch(h1, h2):
    n = 1 << 18
    z = np.zeros(n, complex)
    z[:len(h1[0])] = np.conj(h1[0]) * h2[0] + np.conj(h1[1]) * h2[1]
    norm = np.sqrt(sum(np.real(np.vdot(x, x)) for x in h1) * sum(np.real(np.vdot(x, x)) for x in h2))
    return float(1.0 - np.max(np.abs(np.fft.ifft(z) * n)) / norm)


def rotated(h, psi):
    c, s = np.cos(2.0 * psi), np.sin(2.0 * psi)
    return [h[0] * c - h[1] * s, h[1] * c + h[0] * s]


def against_lal(n, seed, library):
    import jax
    import jax.numpy as jnp
    import lal
    import lalsimulation as lalsim

    import waveformx as wf
    from waveformx.models.xpnr import lean

    waveformrs = HERE.parents[1] / "waveformrs" / "scripts"
    sys.path.insert(0, str(waveformrs))
    from lalsimulation_ctypes import Library, Overlay

    patched = Overlay(lalsim, Library(library))
    F = np.arange(20.0, 1024.0, 0.125)

    def lal_h(b):
        sequence = lal.CreateREAL8Vector(len(F))
        sequence.data = F
        hp, hc = patched.SimInspiralChooseFDWaveformSequence(
            float(b.phi_ref), float(b.mass_1) * lal.MSUN_SI, float(b.mass_2) * lal.MSUN_SI, *np.asarray(b.spin_1),
            *np.asarray(b.spin_2), float(b.f_ref), 500e6 * lal.PC_SI, float(b.inclination), lal.CreateDict(),
            lalsim.IMRPhenomXPNR, sequence)
        return [hp.data.data * 1e21, hc.data.data * 1e21]

    to_scaled = jax.jit(lambda b: wf.to_mass_scaled_reference(b, return_angle=True))
    to_physical = jax.jit(lambda b, f: wf.to_physical_reference(b, f, return_angle=True))
    scaled = jax.jit(lambda b, f: lean.polarizations(b, f, reference_mf=REF))
    rng = np.random.default_rng(seed)
    records = []
    while len(records) < n:
        q, mass = rng.uniform(1, 12), rng.uniform(20, 90)
        spins = [rng.normal(size=3) for _ in range(2)]
        spins = [s / np.linalg.norm(s) * rng.uniform(0, 0.9) for s in spins]
        binary = wf.Binary(*(jnp.asarray(v, jnp.float64) for v in (
            mass * q / (1 + q), mass / (1 + q), spins[0], spins[1], 500.0, rng.uniform(0, np.pi),
            rng.uniform(0, 2 * np.pi), 20.0)))
        there, psi = to_scaled(binary)
        f_ref = REF / (mass * MTSUN_SI)
        physical, _ = to_physical(there, f_ref)
        try:
            at_20, at_ref = lal_h(binary), lal_h(physical)
        except RuntimeError:
            continue
        # (the lean model at the mass-scaled reference and LAL at f_ref =
        # 0.01 / M share L at the reference, and so the basis of the
        # polarizations; LAL at 20 Hz has that of L at 20 Hz)
        h = [np.asarray(x) for x in scaled(there, jnp.asarray(F))]
        # (LAL's test point against the consistent model: the binary given at
        # 20 Hz is the one the model carries to the canonical reference,
        # whatever reference_mf it is given)
        records.append(dict(q=q, mass=mass, f_ref=f_ref, lean_vs_lal=mismatch(at_ref, h),
                            lal_vs_lal=mismatch(at_20, rotated(at_ref, float(psi))),
                            lal_20_vs_lean=mismatch(at_20, rotated(h, float(psi)))))
        # LAL's test point near the canonical reference: the same spins and
        # angles given at f_ref = M_near / M, M_near off REF by a relative
        # offset log-uniform in [0.002, 0.2], either sign (the comparison of
        # the consistent model with LAL, by the distance from REF)
        offset = np.exp(rng.uniform(np.log(0.002), np.log(0.2))) * rng.choice([-1.0, 1.0])
        mf_near = REF * (1.0 + offset)
        near = binary._replace(f_ref=jnp.asarray(mf_near / (mass * MTSUN_SI), jnp.float64))
        near_there, near_psi = to_scaled(near)
        try:
            at_near = lal_h(near)
        except RuntimeError:
            records.pop()
            continue
        h_near = [np.asarray(x) for x in scaled(near_there, jnp.asarray(F))]
        records[-1].update(mf_near=mf_near, offset=offset, lal_near_vs_lean=mismatch(at_near, rotated(h_near, float(near_psi))))
        print(f"q {q:.1f} M {mass:.0f} lean vs LAL {records[-1]['lean_vs_lal']:.1e} "
              f"LAL 20 Hz vs LAL {f_ref:.0f} Hz {records[-1]['lal_vs_lal']:.1e} "
              f"LAL 20 Hz vs lean {records[-1]['lal_20_vs_lean']:.1e} "
              f"LAL at Mf {mf_near:.4f} vs lean {records[-1]['lal_near_vs_lean']:.1e}", flush=True)
    return records


def timings():
    env = dict(os.environ, JAX_COMPILATION_CACHE_DIR="")
    out = {}
    for convention, key in (("lal", "lal"), (str(REF), "scaled"), ("0.006", "scaled_0.006"),
                            ("0.002", "scaled_0.002")):
        result = subprocess.run([sys.executable, "-c", TIMING, convention, json.dumps(MASSES)], env=env,
                                capture_output=True, text=True, check=True)
        out[key] = json.loads(result.stdout.strip().splitlines()[-1])
        print(convention, out[key], flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--n-lal", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--lalsimulation-library", default=os.environ.get("PATCHED_LALSIMULATION"))
    parser.add_argument("--output", default=str(HERE / "data" / "reference_xpnr.json"))
    args = parser.parse_args()
    import jax

    jax.config.update("jax_enable_x64", True)
    data = dict(reference_mf=REF, n=args.n, finite=finite_counts(args.n, args.seed))
    if args.lalsimulation_library:
        data["lal"] = against_lal(args.n_lal, args.seed, args.lalsimulation_library)
    data["timings"] = timings()
    Path(args.output).write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
