#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The smoothness of the waveform in its parameters: the overlap between
the waveform at a random binary and at points along each parameter around
it, and where it is not smooth, what jumps.

Frequencies: 20-1024 Hz in steps of 1/8 Hz, aLIGO design weights; the
overlap is ``<h0, h> / sqrt(<h0, h0> <h, h>)`` with ``h_+`` and ``h_x``
together, no maximisation. Along each parameter the scan spans the range
over which ``1 - O`` reaches 10^-2 (from the curvature at the binary;
inside ``|chi| < 0.999`` for the spins, upwards from ``f_ref = f_min`` for
the reference frequency).

* ``scans``: the first random binary, LAL's model and the extended one,
  401 points per parameter.
* ``survey``: random binaries, 101 points per parameter. A point where the
  fifth differences of ``1 - O`` stand out of their noise (a jump gives ~10
  times its size, a smooth function ``h^5`` times its fifth derivative) is
  located by bisection to adjacent floats, and what changes across it is
  recorded: the discrete quantities of the SpinTaylor solution and of the
  PNR grid, the choices of the higher-mode amplitudes, the angles on the
  PNR grid and the co-precessing modes. LAL's model only counts them.

Usage::

    python report/extended/smoothness.py [--n 24] [--jobs 8]

(``JAX_COMPILATION_CACHE_DIR`` keeps the compiled model between the
workers and runs.)
"""

import argparse
import concurrent.futures
import multiprocessing
import time

import numpy as np

import xcommon as x
from xcommon import jax, jnp
from waveformx.models.xpnr import hm

F_MIN, F_MAX, DELTA_F = 20.0, 1024.0, 0.125
NAMES = ["mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z", "luminosity_distance",
         "inclination", "phi_ref", "f_ref"]
#: the parameters scanned (the distance only scales the waveform)
SCANNED = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
#: a non-smooth point: fifth differences beyond THRESHOLD robust standard
#: deviations; those beyond LOCATE are located (at most MAX_LOCATED per scan)
THRESHOLD = 8.0
LOCATE = 20.0
MAX_LOCATED = 3
ITERATIONS = 44
SCAN_SEED, SURVEY_SEED = 1, 11
HIGHER = [m for m in x.MODES if m != (2, 2)]


def frequencies():
    return np.arange(F_MIN, F_MAX + 0.5 * DELTA_F, DELTA_F)


def point(seed, index):
    """A random binary of :func:`xcommon.draws`, starting at 20 Hz (the
    reference frequency no lower)."""
    d = x.draws(index + 1, seed)[index]
    return dict(d, fmin=F_MIN, fref=max(d["fref"], F_MIN))


def vector(d):
    return np.array([d["m1"], d["m2"], *d["s1"], *d["s2"], 500.0, 0.5, 0.3, d["fref"]])


def binary(v):
    return x.wf.Binary(mass_1=v[0], mass_2=v[1], spin_1=v[2:5], spin_2=v[5:8], luminosity_distance=v[8],
                       inclination=v[9], phi_ref=v[10], f_ref=v[11])


class Model:
    """The polarizations of a variant (``h_+`` and ``h_x`` stacked) and the
    overlap with those at ``v0``."""

    def __init__(self, variant, v0):
        f = frequencies()
        self.grid = jnp.asarray(f)
        self.w = np.tile(1 / x.psd(f), 2)
        self.variant = variant
        self._pol = jax.jit(x.polarizations_fn(variant))
        self.h0 = self.h(v0)
        self.n0 = self.inner(self.h0, self.h0)

    def h(self, v):
        hp, hc = self._pol(binary(jnp.asarray(v)), self.grid)
        return np.concatenate([np.asarray(hp), np.asarray(hc)])

    def inner(self, a, b):
        return float(np.sum(self.w * (a * np.conj(b)).real))

    def overlap(self, v):
        h = self.h(v)
        return self.inner(self.h0, h) / np.sqrt(self.n0 * self.inner(h, h))


def scan_range(model, v0, i):
    """The offsets along parameter ``i`` over which ``1 - O`` reaches
    10^-2."""
    u = np.zeros(12)
    u[i] = 1.0
    name = NAMES[i]
    delta = 1e-4
    sides = (1,) if name == "f_ref" else (-1, 1)
    g = np.mean([1 - model.overlap(v0 + s * delta * u) for s in sides]) * 2 / delta ** 2
    half = np.sqrt(2e-2 / g) if np.isfinite(g) and g > 0 else 1e-3
    if 2 <= i < 8:
        s = v0[2:5] if i < 5 else v0[5:8]
        k = (i - 2) % 3
        top = np.sqrt(max(0.999 ** 2 - (np.sum(s ** 2) - s[k] ** 2), 0.0))
        half = min(half, top - abs(s[k]))
    if name in ("mass_1", "mass_2"):
        half = min(half, 0.2 * v0[i])
    return u, ((0.0, 2 * half) if name == "f_ref" else (-half, half))


def flags_of(eps, y):
    """The points where the fifth differences of ``y`` stand out:
    ``(a, b, size / sigma)``, largest first, and ``sigma``."""
    d5 = np.diff(y, 5)
    sigma = 1.4826 * np.median(np.abs(d5 - np.median(d5)))
    groups = []
    for k in np.nonzero(np.abs(d5) > THRESHOLD * sigma)[0]:
        if groups and k - groups[-1][-1] <= 5:
            groups[-1].append(k)
        else:
            groups.append([k])
    out = []
    for g in groups:
        k = g[int(np.argmax(np.abs(d5[g])))]
        out.append((float(eps[k]), float(eps[k + 5]), float(np.max(np.abs(d5[g])) / sigma)))
    out.sort(key=lambda r: -r[2])
    return out, float(sigma)


def locate(model, v0, u, a, b):
    """Bisection to the non-smooth point in ``[a, b]``: the half whose
    midpoint deviates most from the chord. Returns the bracket, the relative
    jump of the polarizations and the mismatch across it."""
    norm = lambda z: np.sqrt(model.inner(z, z))  # noqa: E731
    ha, hb = model.h(v0 + a * u), model.h(v0 + b * u)
    for _ in range(ITERATIONS):
        m = 0.5 * (a + b)
        hm = model.h(v0 + m * u)
        left = norm(model.h(v0 + 0.5 * (a + m) * u) - 0.5 * (ha + hm))
        right = norm(model.h(v0 + 0.5 * (m + b) * u) - 0.5 * (hm + hb))
        if left >= right:
            b, hb = m, hm
        else:
            a, ha = m, hm
    mismatch = 1 - model.inner(ha, hb) / np.sqrt(model.inner(ha, ha) * model.inner(hb, hb))
    return a, b, norm(hb - ha) / np.sqrt(model.n0), float(mismatch)


def internals_fn(grid):
    """What changes across a located point (extended model)."""
    setup = x.setup_fn(True)
    flags = x.VARIANTS["extended"]

    def f(v):
        b = binary(v)
        prec, w, g, fl = setup(b, grid)
        st, sa, ap, bp = prec.st, prec.st_angles, prec.alpha_params, prec.beta_params
        discrete = dict(n_b=st.n_b, n_fwd=st.n_fwd, copy_length=st.copy_length, n_fine=st.n_fine, len_pn=st.len_pn,
                        fwd_steps=st.fwd.accepted, fine_steps=st.fine.accepted, back_steps=st.back.n,
                        grid_start=g[0], grid_n=g[1], again=fl.again)
        x.pnr.clear_cache()
        data = x.model._decide(b, grid, 0.0, grid[0], grid[-1], x.MODES, True, **flags)
        wf = data.wf
        coefficients = hm.modes_coefficients(wf, x.model.phase_coefficients(wf), HIGHER)
        for (ell, emm), c in zip(HIGHER, coefficients):
            a = c.amp
            discrete[f"amp{ell}{emm}"] = (a.inspiral_amp_version * 1e6 + a.intermediate_amp_version * 10
                                          + jnp.where(a.amp_emr, 1, 0))
            discrete[f"order{ell}{emm}"] = a.inter_amp_pol_order
        discrete["s_tot_r>=0.8"] = wf.s_tot_r >= 0.8
        modes = jnp.stack(x.model._coprecessing_modes(wf, x.MODES, grid))
        angles = dict(alpha=jnp.stack(data.alpha), cos_beta=jnp.stack(data.cos_beta), epsilon=jnp.stack(data.epsilon))
        return ({k: jnp.asarray(val, jnp.float64) for k, val in discrete.items()}, modes, angles,
                dict(ftrans_mrd=sa.ftrans_mrd / w.m_sec, alpha_lower=ap.mf_alpha_lower / w.m_sec,
                     beta_lower=bp.mf_beta_lower / w.m_sec, beta_upper=bp.mf_beta_upper / w.m_sec))

    return jax.jit(f)


def classify(changed, modes, angles):
    """The cause of a jump, from what changes across it."""
    hm = [k for k in changed if k.startswith(("amp", "order", "s_tot"))]
    if hm:
        return "higher-mode choice"
    if angles["alpha"] > 1.0 or angles["epsilon"] > 1.0:
        return "alpha branch at the pole"
    if any(k in changed for k in ("grid_start", "grid_n")):
        return "PNR grid"
    if any(k in changed for k in ("n_b", "n_fwd", "copy_length", "n_fine", "len_pn", "again")):
        return "SpinTaylor knots"
    if any(k in changed for k in ("fwd_steps", "fine_steps", "back_steps")):
        return "integration steps"
    return "none (numerical noise)"


def describe(internals, v0, u, a, b):
    da, ma, aa, special = internals(jnp.asarray(v0 + a * u))
    db, mb, ab, _ = internals(jnp.asarray(v0 + b * u))
    changed = {k: [float(da[k]), float(db[k])] for k in da if float(da[k]) != float(db[k])}
    mode_jumps = [float(np.max(np.abs(np.asarray(mb[i]) - np.asarray(ma[i]))) / np.max(np.abs(np.asarray(ma[i]))))
                  for i in range(len(x.MODES))]
    angle_jumps = {k: float(np.max(np.abs(np.asarray(ab[k]) - np.asarray(aa[k])))) for k in aa}
    return dict(changed=changed, modes=mode_jumps, angles=angle_jumps,
                special={k: float(val) for k, val in special.items()},
                cause=classify(changed, mode_jumps, angle_jumps))


def scan_binary(d, variant, n, locate_events):
    v0 = vector(d)
    model = Model(variant, v0)
    internals = internals_fn(model.grid) if locate_events else None
    out = {}
    for i in SCANNED:
        u, (lo, hi) = scan_range(model, v0, i)
        eps = np.linspace(lo, hi, n)
        o = np.array([model.overlap(v0 + e * u) for e in eps])
        entry = dict(lo=lo, hi=hi, overlap=o.tolist())
        if not np.all(np.isfinite(o)):
            entry["nan"] = int(np.sum(~np.isfinite(o)))
            out[NAMES[i]] = entry
            continue
        flagged, sigma = flags_of(eps, 1 - o)
        entry.update(sigma=sigma, flags=[dict(a=a, b=b, significance=s) for a, b, s in flagged])
        events = []
        for a, b, significance in flagged:
            if not locate_events or significance < LOCATE or len(events) == MAX_LOCATED:
                continue
            a, b, jump, mismatch = locate(model, v0, u, a, b)
            events.append(dict(at=a, width=b - a, significance=significance, jump=float(jump), mismatch=mismatch,
                               **describe(internals, v0, u, a, b)))
        entry["events"] = events
        out[NAMES[i]] = entry
    return out


def survey_one(args):
    index, variant, n = args
    t = time.time()
    d = point(SURVEY_SEED, index)
    record = dict(index=index, variant=variant, binary={k: np.asarray(val).tolist() for k, val in d.items()})
    record["scans"] = scan_binary(d, variant, n, variant == "extended")
    for name, s in record["scans"].items():
        if not s.get("events"):
            s.pop("overlap", None)
    record["seconds"] = time.time() - t
    print(f"{variant} {index}: {record['seconds']:.0f}s", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    t = time.time()
    first = point(SCAN_SEED, 0)
    scans = {variant: scan_binary(first, variant, 401, False) for variant in ("lal", "extended")}
    print(f"scans: {time.time() - t:.0f}s", flush=True)
    tasks = [(i, v, 101) for i in range(args.n) for v in ("extended", "lal")]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=context) as pool:
        survey = list(pool.map(survey_one, tasks))
    x.write("smoothness.json", dict(grid=[F_MIN, F_MAX, DELTA_F], first=dict(first, s1=list(first["s1"]),
                                                                             s2=list(first["s2"])),
                                    names=NAMES, scans=scans, survey=survey, threshold=THRESHOLD, locate=LOCATE))


if __name__ == "__main__":
    main()
