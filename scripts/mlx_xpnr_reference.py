#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The reference values of the MLX port of the lean IMRPhenomXPNR: the
float64 intermediates of every stage (:func:`waveformjax.models.xpnr.mlx._stages.reference`)
and the float64 Jacobians of ``h_+`` and ``h_x`` with respect to the twelve
parameters of the binary, at random binaries of ``report/extended`` and at
edge cases, written to an ``.npz`` file (one entry per binary and value,
``<case>/<name>``).

Usage::

    python scripts/mlx_xpnr_reference.py OUT.npz [--n 20] [--seed 11] [--n-frequencies 1024] [--no-jacobians]
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import waveformjax as wf
from waveformjax.models.xpnr import lean
from waveformjax.models.xpnr.mlx import _stages

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "report" / "extended"))
import xcommon  # noqa: E402

#: Edge cases: (name, parameters of xcommon.binary, f_min).
EDGE_CASES = [
    ("generic", dict(m1=36.0, m2=29.0, s1=[0.4, 0.2, 0.3], s2=[-0.2, 0.3, -0.1], fref=20.0), 20.0),
    ("aligned", dict(m1=36.0, m2=29.0, s1=[0.0, 0.0, 0.3], s2=[0.0, 0.0, -0.1], fref=20.0), 20.0),
    ("non_spinning", dict(m1=36.0, m2=29.0, s1=[0.0, 0.0, 0.0], s2=[0.0, 0.0, 0.0], fref=20.0), 20.0),
    ("equal_q1", dict(m1=30.0, m2=30.0, s1=[0.3, 0.1, 0.2], s2=[0.3, 0.1, 0.2], fref=20.0), 20.0),
    ("equal_aligned", dict(m1=30.0, m2=30.0, s1=[0.0, 0.0, 0.2], s2=[0.0, 0.0, 0.2], fref=20.0), 20.0),
    ("single", dict(m1=36.0, m2=29.0, s1=[0.5, 0.2, 0.3], s2=[0.0, 0.0, 0.0], fref=20.0), 20.0),
    ("small_in_plane", dict(m1=36.0, m2=29.0, s1=[5e-7, 0.0, 0.3], s2=[0.0, 0.0, -0.1], fref=20.0), 20.0),
    ("q12", dict(m1=60.0, m2=5.0, s1=[0.5, 0.3, -0.2], s2=[0.1, 0.2, 0.3], fref=20.0), 20.0),
    ("q15", dict(m1=75.0, m2=5.0, s1=[0.6, 0.3, -0.2], s2=[0.1, 0.2, 0.3], fref=20.0), 20.0),
    ("q18", dict(m1=90.0, m2=5.0, s1=[0.6, 0.3, -0.2], s2=[0.1, 0.2, 0.3], fref=20.0), 20.0),
    ("heavy", dict(m1=200.0, m2=100.0, s1=[0.4, -0.3, 0.2], s2=[0.2, 0.1, -0.5], fref=10.0), 10.0),
    ("light", dict(m1=9.0, m2=6.0, s1=[0.3, 0.3, 0.3], s2=[-0.3, 0.2, 0.0], fref=20.0), 20.0),
    ("light_low", dict(m1=7.0, m2=4.0, s1=[0.6, 0.1, -0.3], s2=[0.2, -0.4, 0.1], fref=15.0), 10.0),
    ("fref_above", dict(m1=36.0, m2=29.0, s1=[0.4, 0.2, 0.3], s2=[-0.2, 0.3, -0.1], fref=45.0), 20.0),
    ("beyond", dict(m1=180.0, m2=120.0, s1=[0.4, 0.2, 0.3], s2=[-0.2, 0.3, -0.1], fref=60.0), 20.0),
    ("high_spin", dict(m1=40.0, m2=20.0, s1=[0.0, 0.8, 0.55], s2=[0.6, 0.0, -0.75], fref=20.0), 20.0),
]

PARAMETERS = ("m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z", "distance", "inclination", "phi_ref", "f_ref")


def of(x):
    return wf.Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                     inclination=x[9], phi_ref=x[10], f_ref=x[11])


def vector(d):
    return np.array([d["m1"], d["m2"], *d["s1"], *d["s2"], 500.0, d.get("inclination", 0.5), d.get("phi_ref", 0.3),
                     d["fref"]])


def flatten(prefix, tree, out):
    """The array leaves of ``tree`` as ``out[prefix/path]``."""
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if isinstance(leaf, (jax.Array, np.ndarray, np.generic)):
            out[prefix + jax.tree_util.keystr(path)] = np.asarray(leaf)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-frequencies", type=int, default=1024)
    parser.add_argument("--no-jacobians", action="store_true")
    args = parser.parse_args()
    cases = list(EDGE_CASES)
    for i, d in enumerate(xcommon.draws(args.n, args.seed)):
        cases.append((f"random{i:02d}", d, d["fmin"]))
    jacobian = jax.jit(lambda x, f: jax.jacfwd(lambda x: lean.polarizations(of(x), f))(x))
    out = {"parameters": np.array(PARAMETERS)}
    names = []
    for name, d, fmin in cases:
        t = time.time()
        x = vector(d)
        f = np.linspace(fmin, xcommon.f_max(d["m1"] + d["m2"]), args.n_frequencies)
        b = of(jnp.asarray(x))
        (hp, hc), values = _stages.reference(b, jnp.asarray(f), keep=True)
        out[f"{name}/x"] = x
        out[f"{name}/frequencies"] = f
        out[f"{name}/hp"] = np.asarray(hp)
        out[f"{name}/hc"] = np.asarray(hc)
        sol = values.pop("sol")
        n = int(sol.n)
        for field in ("y", "k0", "h", "dense", "omega_end", "t0"):
            out[f"{name}/sol/{field}"] = np.asarray(getattr(sol, field))[:n]
        for field in ("n", "n_back", "order", "t_end", "omega_first", "omega_last", "f_min_pn", "failed",
                      "incomplete"):
            value = np.asarray(getattr(sol, field))
            out[f"{name}/sol/{field}"] = value[:n] if value.ndim else value
        splines = values.pop("splines")
        n_grid = int(values["grid"][1])
        for field in _stages.SPLINES:
            out[f"{name}/splines/{field}"] = np.asarray(getattr(splines, field))[:n_grid]
            out[f"{name}/splines/m_{field}"] = np.asarray(getattr(splines, "m_" + field))[:n_grid]
        for key, value in values.items():
            flatten(f"{name}/{key}", value, out)
        if not args.no_jacobians:
            jp, jc = jacobian(jnp.asarray(x), jnp.asarray(f))
            out[f"{name}/jac_hp"] = np.asarray(jp)
            out[f"{name}/jac_hc"] = np.asarray(jc)
        names.append(name)
        print(f"{name}: n_steps {n}, grid {n_grid}, finite {np.isfinite(hp).all()} ({time.time() - t:.1f} s)",
              flush=True)
    out["cases"] = np.array(names)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
