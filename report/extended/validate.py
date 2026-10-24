#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The extended angles against LAL's where LAL uses the SpinTaylor angles:
the mismatch (aLIGO design weights, no maximisation, the larger of the two
polarizations) over random binaries (:func:`xcommon.draws`) for which LAL
keeps the SpinTaylor angles, for the widths of the blends of the PNR
``beta`` given (``BETA_SOFT_WIDTH`` in ``Mf``; the angle bands scale with
it), and the strict model (LAL's to rounding) as a check.

Usage::

    python report/extended/validate.py [--n 2000] [--widths 0.002,0.005,0.01] [--variant lean]
"""

import argparse
import time

import numpy as np

import xcommon as x
from xcommon import jax, jnp
from waveformx.models.xpnr import pnr_angles as pa

N_FREQUENCIES = 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--widths", default=str(pa.BETA_SOFT_WIDTH))
    parser.add_argument("--output", default="validation.json")
    parser.add_argument("--variant", default="extended", help="extended or lean")
    args = parser.parse_args()
    import waveformrs as rs
    draws = []
    for d in x.draws(4 * args.n, args.seed):
        f = np.linspace(d["fmin"], x.f_max(d["M"]), 64)
        try:
            rs.polarizations("IMRPhenomXPNR", f, version_fallback=False, mass_1=d["m1"], mass_2=d["m2"], spin_1=d["s1"],
                             spin_2=d["s2"], luminosity_distance=500.0, inclination=0.5, phi_ref=0.3, f_ref=d["fref"])
        except Exception:  # noqa: BLE001 (LAL leaves the SpinTaylor angles)
            continue
        draws.append(d)
        if len(draws) == args.n:
            break
    # one binary at a time: under jax.vmap the loops of a batch run as long
    # as those of its lightest binary
    strict = jax.jit(x.polarizations_fn("strict"))
    reference = []
    for d in draws:
        fr = x.frequencies(d, N_FREQUENCIES)
        hp, hc = strict(x.binary(d), fr)
        reference.append((np.asarray(hp), np.asarray(hc), np.asarray(fr)))
    # the strict model is LAL's (waveformrs's) to rounding
    checks = []
    for d, (hp, hc, f) in list(zip(draws, reference))[:64]:
        rp, rc = rs.polarizations("IMRPhenomXPNR", f, mass_1=d["m1"], mass_2=d["m2"], spin_1=d["s1"], spin_2=d["s2"],
                                  luminosity_distance=500.0, inclination=0.5, phi_ref=0.3, f_ref=d["fref"])
        checks.append(max(x.weighted_mismatch(hp, rp, f), x.weighted_mismatch(hc, rc, f)))
    print("strict against waveformrs: max mismatch %.1e" % max(checks))
    results = {}
    for width in (float(w) for w in args.widths.split(",")):
        pa.BETA_SOFT_WIDTH = width
        pa.BETA_SOFT_ANGLE = 10.0 * width
        t = time.time()
        extended = jax.jit(x.polarizations_fn(args.variant))
        mismatches = []
        for d, (rp, rc, f) in zip(draws, reference):
            a, b = (np.asarray(v) for v in extended(x.binary(d), jnp.asarray(f)))
            mismatches.append(None if not np.all(np.isfinite(a)) else
                              max(x.weighted_mismatch(a, rp, f), x.weighted_mismatch(b, rc, f)))
        results[str(width)] = mismatches
        mm = np.array([m for m in mismatches if m is not None])
        print(f"width {width}: {time.time() - t:.0f} s, median {np.median(mm):.1e}, 99% {np.percentile(mm, 99):.1e}, "
              f"max {mm.max():.1e}, fraction above 1e-3 {np.mean(mm > 1e-3):.4f}", flush=True)
    x.write(args.output, dict(n=len(draws), seed=args.seed, variant=args.variant, strict_vs_waveformrs=checks, draws=draws,
                                    mismatches=results))


if __name__ == "__main__":
    main()
