#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The jumps of the waveform along 1-D scans of five binaries in four
parameters, for LAL's angles and the extended ones.

Along each scan (101 points), the discrete quantities of the SpinTaylor
solution and of the PNR set-up (the numbers of samples of the backward,
forward and fine runs, the knots, the grid of the PNR interpolants, LAL's
switches) are recorded; where any changes between neighbours, the change
is located by bisection (to adjacent floats) and the jump of the
polarizations across it measured (``|h+ - h-| / |h|`` with aLIGO weights,
and the mismatch).

Usage::

    python report/extended/jumps.py
"""

import time

import numpy as np

import xcommon as x
from xcommon import jax, jnp
from waveformjax.models.xpnr import pnr_angles as pa

BINARIES = {
    "A (36+29, f_min 20 Hz)": dict(m1=36., m2=29., s1=[0.4, 0.2, 0.3], s2=[-0.2, 0.3, -0.1], fref=20., fmin=20.),
    "B (100+60)": dict(m1=100., m2=60., s1=[0.5, 0.3, 0.2], s2=[0.1, -0.4, 0.3], fref=20., fmin=20.),
    "C (8+3)": dict(m1=8., m2=3., s1=[0.6, 0.0, 0.2], s2=[0.0, 0.1, 0.0], fref=20., fmin=20.),
    "D (50+5)": dict(m1=50., m2=5., s1=[0.7, 0.0, 0.1], s2=[0.0, 0.0, 0.0], fref=20., fmin=20.),
    "E (180+120, f_ref beyond)": dict(m1=180., m2=120., s1=[0.4, 0.2, 0.3], s2=[-0.2, 0.3, -0.1], fref=60., fmin=15.),
}
#: the parameters scanned and the widths of the scans (relative for the
#: masses and f_ref, which is scanned upwards from its value)
PARAMETERS = {"mass_1": 0.01, "chi1_x": 0.02, "chi1_z": 0.02, "f_ref": 0.04}
DISCRETE = ["n_b", "n_fwd", "copy_length", "n_fine", "len_pn", "split", "grid_start", "grid_n", "again",
            "fmax_switch", "beta_flat", "alpha_off", "upper_none", "attach", "remap_small", "two_spin"]
K = 101
MAX_CHANGES = 12
ITERATIONS = 44


def internals_fn(extended):
    setup = x.setup_fn(extended)

    def f(b, frequencies):
        prec, w, grid, flags = setup(b, frequencies)
        st, sa, ap, bp = prec.st, prec.st_angles, prec.alpha_params, prec.beta_params
        out = dict(n_b=st.n_b, n_fwd=st.n_fwd, copy_length=st.copy_length, n_fine=st.n_fine, len_pn=st.len_pn,
                   split=st.split, grid_start=grid[0], grid_n=grid[1], again=flags.again,
                   fmax_switch=sa.mfmax_pn > w.f_ring - w.f_damp, beta_flat=sa.beta_mrd.flat,
                   alpha_off=ap.mf_alpha_lower == 100.0, upper_none=bp.mf_beta_upper == 100.0,
                   attach=pa.attach_mr_beta(bp), remap_small=x.pnr.chi_in_plane(prec.chi1, prec.chi2) < 1e-3,
                   two_spin=x.pnr.two_spin(prec, 330))
        return {k: jnp.asarray(v, jnp.float64) for k, v in out.items()}

    return jax.jit(jax.vmap(f, in_axes=(0, None)))


def set_parameter(d, name, value):
    if name == "mass_1":
        return dict(d, m1=value)
    if name == "f_ref":
        return dict(d, fref=value)
    which, k = name.split("_")
    key = "s1" if which == "chi1" else "s2"
    s = list(d[key])
    s["xyz".index(k)] = value
    return dict(d, **{key: s})


def get_parameter(d, name):
    if name == "mass_1":
        return d["m1"]
    if name == "f_ref":
        return d["fref"]
    which, k = name.split("_")
    return d["s1" if which == "chi1" else "s2"]["xyz".index(k)]


def survey(variant, internals, polarizations, rng):
    records = []
    for bname, d in BINARIES.items():
        f = x.frequencies(d, 2048)
        for pname, width in PARAMETERS.items():
            x0 = get_parameter(d, pname)
            span = width * (abs(x0) if pname in ("mass_1", "f_ref") else 1.0)
            xs = x0 + (np.linspace(0, span, K) if pname == "f_ref" else np.linspace(-span / 2, span / 2, K))

            def batch(values):
                return x.stack([x.binary(set_parameter(d, pname, v)) for v in values])

            key = np.stack([np.asarray(internals(batch(xs), f)[k]) for k in DISCRETE], axis=1)
            changes = np.nonzero(np.any(np.diff(key, axis=0) != 0, axis=1))[0]
            if len(changes) > MAX_CHANGES:
                changes = np.sort(rng.choice(changes, MAX_CHANGES, replace=False))
            if len(changes) == 0:
                continue
            n = len(changes)
            changes = np.concatenate([changes, np.full(MAX_CHANGES - n, changes[0])])
            lo, hi, k0 = xs[changes].copy(), xs[changes + 1].copy(), key[changes]
            for _ in range(ITERATIONS):
                mid = 0.5 * (lo + hi)
                km = np.stack([np.asarray(internals(batch(mid), f)[k]) for k in DISCRETE], axis=1)
                same = np.all(km == k0, axis=1)
                lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
            il = internals(batch(lo), f)
            ih = internals(batch(hi), f)
            hl = np.asarray(polarizations(batch(lo), f)[0])
            hh = np.asarray(polarizations(batch(hi), f)[0])
            for j in range(n):
                changed = [k for k in DISCRETE if float(il[k][j]) != float(ih[k][j])]
                records.append(dict(binary=bname, parameter=pname, at=float(lo[j]), changed=changed,
                                    jump=x.relative_change(hh[j], hl[j], f), mismatch=x.weighted_mismatch(hh[j], hl[j], f)))
            print(variant, bname, pname, n, "changes", flush=True)
    return records


def main():
    out = {}
    for variant in ("lal", "extended"):
        t = time.time()
        rng = np.random.default_rng(0)
        polarizations = jax.jit(jax.vmap(x.polarizations_fn(variant if variant == "extended" else "strict"),
                                         in_axes=(0, None)))
        out[variant] = survey(variant, internals_fn(variant == "extended"), polarizations, rng)
        print(variant, f"{time.time() - t:.0f} s")
    x.write("jumps.json", dict(binaries=BINARIES, parameters=PARAMETERS, points=K, jumps=out))


if __name__ == "__main__":
    main()
