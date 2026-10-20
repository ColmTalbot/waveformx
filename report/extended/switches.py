#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The waveform across LAL's switches, for LAL's model (with its fallbacks)
and the extended angles.

* ``boundaries``: across the edges of the PNR windows (``q = 8.5``, ``q =
  12``, ``chi = 0.85``), the two-spin construction (``|chi2| = 1e-3``), the
  remapped J frame (``chi_perp = 1e-3``), the NNLO angles and the PNR
  switch (``chi_perp = 1e-6``, ``1e-7``) and the reference frequency at
  which LAL's SpinTaylor angles stop: the relative change of the
  polarizations across ``1e-12`` (relative) of the threshold, and the
  mismatch.
* ``two_spin``: the jump at ``|chi2| = 1e-3`` for random binaries.
* ``beta``: along random lines in the spin magnitude of two-spin binaries
  (``q`` up to 12), the decisions of LAL's connection frequencies of the PNR
  ``beta`` (``IMRPhenomX_PNR_BetaConnectionFrequencies``,
  ``IMRPhenomX_PNR_AttachMRBeta``) are located by bisection; the jump of
  LAL's waveform there, and the change of the extended one across ``1e-4``
  of spin around it.
* ``q12``: both along the mass ratio from 11 to 15.
* ``steps``: at the steepest point of a smoothed switch, the mismatch
  between points ``h`` apart for decreasing ``h`` (``~h^2`` where the
  waveform is continuous).

Usage::

    python report/extended/switches.py
"""

import time

import numpy as np

import xcommon as x
from xcommon import jax, jnp
from waveformjax.models.xpnr import pnr_angles as pa
from waveformjax.models.xpnr.api import _prescription

LAL = jax.jit(x.polarizations_fn("lal"))
EXTENDED = jax.jit(x.polarizations_fn("extended"))


#: The relative distance on either side of a threshold at which it is
#: crossed (well above rounding, which the conversions to the masses and the
#: spin magnitudes add, and well below the smooth variation of the waveform).
EPSILON = 1e-12


def across(make, at, epsilon=EPSILON):
    """The relative change and the mismatch of LAL's and the extended
    polarizations between ``at (1 -+ epsilon)``."""
    lo, hi = at * (1.0 - epsilon), at * (1.0 + epsilon)
    out = {}
    for name, f in (("lal", LAL), ("extended", EXTENDED)):
        (bl, fr), (bh, _) = make(lo), make(hi)
        a, b = np.asarray(f(bl, fr)[0]), np.asarray(f(bh, fr)[0])
        out[name] = dict(jump=x.relative_change(b, a, fr), mismatch=x.weighted_mismatch(b, a, fr))
    return out


def single(d, n=1024):
    return x.binary(d), x.frequencies(d, n)


def boundaries():
    base = dict(s1=[0.5, 0.0, 0.3], s2=[0.0, 0.2, 0.1], fref=20., fmin=20.)

    def mass_ratio(total, spins):
        return lambda q: single(dict(spins, m1=total * q / (1 + q), m2=total / (1 + q)))

    cases = []
    for total, s1, s2 in ((60.0, [0.4, 0.1, 0.2], [0.1, 0.0, 0.1]), (120.0, [0.5, 0.0, 0.3], [0.0, 0.2, 0.1])):
        spins = dict(base, s1=s1, s2=s2)
        cases.append((f"q = 8.5 (M = {total:.0f})", mass_ratio(total, spins), 8.5))
        cases.append((f"q = 12 (M = {total:.0f})", mass_ratio(total, spins), 12.0))
    cases.append(("chi = 0.85", lambda a: single(dict(base, m1=30., m2=10., s1=list(a * np.array([0.6, 0.2, 0.5]) /
                                                                                     np.linalg.norm([0.6, 0.2, 0.5])),
                                                      s2=[0., 0., 0.])), 0.85))
    for q, label in ((1.5, "q = 1.5"), (8.3, "q = 8.3")):
        cases.append((f"|chi2| = 1e-3 ({label})",
                      lambda a, q=q: single(dict(base, m1=60. * q / (1 + q), m2=60. / (1 + q), s1=[0.5, 0.2, 0.3],
                                                 s2=list(a * np.array([0.6, 0.6, 0.5]) / np.linalg.norm([0.6, 0.6, 0.5])))),
                      1e-3))
    for at in (1e-3, 1e-6, 1e-7):
        cases.append((f"chi_perp = {at:.0e}", lambda a: single(dict(base, m1=30., m2=20., s1=[0.6 * a, 0.8 * a, 0.3],
                                                                    s2=[0., 0., 0.2])), at))
    out = []
    for label, make, at in cases:
        out.append(dict(label=label, at=at, **across(make, at)))
        print(out[-1], flush=True)

    # the reference frequency where LAL's SpinTaylor angles stop
    beyond = dict(m1=180., m2=120., s1=[0.4, 0.2, 0.3], s2=[-0.2, 0.3, -0.1], fmin=15.)
    fr = x.frequencies(dict(beyond, fref=15.))

    def make(f_ref):
        return x.binary(dict(beyond, fref=f_ref)), fr

    def failed(f_ref):
        return bool(_prescription(make(f_ref)[0], fr[0], fr[-1], x.MODES, True)["st_failed"])

    lo, hi = 15.0, 60.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if not failed(mid) else (lo, mid)
    # (LAL's decision is fuzzy within ~1e-12 of the boundary, where the
    # energy test is at the level of rounding: crossed across 1e-9)
    out.append(dict(label="f_ref at the PN energy test (M = 300)", at=lo, **across(make, lo, 1e-9)))
    print(out[-1], flush=True)
    return out


def two_spin(n=200):
    vl = jax.jit(jax.vmap(x.polarizations_fn("lal"), in_axes=(0, None)))
    ve = jax.jit(jax.vmap(x.polarizations_fn("extended"), in_axes=(0, None)))
    rng = np.random.default_rng(11)
    out = []
    for _ in range(n):
        q = rng.uniform(1.0, 20.0)
        total = np.exp(rng.uniform(np.log(10), np.log(200)))
        a1, c1, p1 = rng.uniform(0.05, 0.99), rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)
        c2, p2 = rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)
        s1 = a1 * np.array([np.sqrt(1 - c1 ** 2) * np.cos(p1), np.sqrt(1 - c1 ** 2) * np.sin(p1), c1])
        n2 = np.array([np.sqrt(1 - c2 ** 2) * np.cos(p2), np.sqrt(1 - c2 ** 2) * np.sin(p2), c2])
        d = dict(m1=total * q / (1 + q), m2=total / (1 + q), s1=list(s1), fref=20., fmin=20.)
        fr = x.frequencies(d)
        bs = x.stack([x.binary(d, s2=list(n2 * a)) for a in (np.nextafter(1e-3, 0.0), np.nextafter(1e-3, 1.0))])
        hl, he = np.asarray(vl(bs, fr)[0]), np.asarray(ve(bs, fr)[0])
        if not (np.all(np.isfinite(hl)) and np.all(np.isfinite(he))):
            continue
        out.append(dict(q=q, M=total, a1=a1, lal=x.weighted_mismatch(hl[1], hl[0], fr),
                        extended=x.weighted_mismatch(he[1], he[0], fr)))
    return out


def beta_lines(n_lines=60, points=64, iterations=48):
    """LAL's decisions on the PNR ``beta`` along random lines."""
    setup = x.setup_fn(False)

    def decisions(b, frequencies):
        prec, w, grid, flags = setup(b, frequencies)
        bp = prec.beta_params
        rec = pa._Branches()
        pa._beta_connection_frequencies(x.pnr.precompute_beta_coefficients(prec, w, 330), False, rec)
        return jnp.concatenate([jnp.stack([jnp.asarray(c, jnp.float64) for c in rec.recorded]),
                                jnp.stack([pa.attach_mr_beta(bp).astype(jnp.float64)])])

    vd = jax.jit(jax.vmap(decisions, in_axes=(0, None)))
    vl = jax.jit(jax.vmap(x.polarizations_fn("lal"), in_axes=(0, None)))
    ve = jax.jit(jax.vmap(x.polarizations_fn("extended"), in_axes=(0, None)))
    rng = np.random.default_rng(3)
    out = []
    for line in range(n_lines):
        q = rng.uniform(1.0, 12.0)
        total = rng.uniform(20.0, 150.0)
        c, p = rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)
        direction = np.array([np.sqrt(1 - c * c) * np.cos(p), np.sqrt(1 - c * c) * np.sin(p), c])
        d = dict(m1=total * q / (1 + q), m2=total / (1 + q), s2=[0.02, 0.01, 0.03], fref=20., fmin=20.)
        fr = x.frequencies(d)

        def batch(values):
            return x.stack([x.binary(d, s1=list(a * direction)) for a in values])

        xs = np.linspace(0.05, 0.95, points)
        k = np.asarray(vd(batch(xs), fr))
        idx = np.nonzero(np.any(np.diff(k, axis=0) != 0, axis=1))[0][:6]
        if len(idx) == 0:
            continue
        m = len(idx)
        idx = np.concatenate([idx, np.full(6 - m, idx[0])])
        lo, hi, k0 = xs[idx].copy(), xs[idx + 1].copy(), k[idx]
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            same = np.all(np.asarray(vd(batch(mid), fr)) == k0, axis=1)
            lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
        hl = [np.asarray(vl(batch(v), fr)[0]) for v in (lo, hi)]
        he = [np.asarray(ve(batch(v), fr)[0]) for v in (lo - 5e-5, hi + 5e-5)]
        for j in range(m):
            if not all(np.all(np.isfinite(h[j])) for h in hl + he):
                continue
            out.append(dict(q=q, M=total, a=float(lo[j]), lal=x.weighted_mismatch(hl[1][j], hl[0][j], fr),
                            extended_1e4=x.weighted_mismatch(he[1][j], he[0][j], fr)))
        print(line, flush=True)
    return out


def steps():
    """The change of the extended waveform at the steepest point of a
    smoothed switch of LAL's (here the attachment of the MR ``beta`` of a
    binary found by beta_lines), for decreasing steps."""
    d = dict(m1=124 * 9.6 / 10.6, m2=124 / 10.6, s2=[0.002, 0.003, 0.01], fref=20., fmin=20.)
    direction = np.array([np.sqrt(1 - 0.54 ** 2), 0.0, -0.54])
    fr = x.frequencies(d)

    def h(a):
        return np.asarray(EXTENDED(x.binary(d, s1=list(a * direction)), fr)[0])

    xs = np.linspace(0.75, 0.85, 201)
    hs = [h(a) for a in xs]
    st = [x.weighted_mismatch(hs[i + 1], hs[i], fr) for i in range(200)]
    i = int(np.argmax(st))
    centre = 0.5 * (xs[i] + xs[i + 1])
    out = []
    for step in (5e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
        out.append(dict(step=step, mismatch=x.weighted_mismatch(h(centre + step / 2), h(centre - step / 2), fr)))
    return dict(centre=centre, steps=out)


def q12_line(points=161):
    """LAL's and the extended waveform along the mass ratio from 11 to 15
    (``M = 120``): the mismatch between them, and between neighbours."""
    d = dict(s1=[0.5, 0.0, 0.3], s2=[0.0, 0.2, 0.1], fref=20., fmin=20.)
    fr = x.frequencies(dict(d, m1=110., m2=10.))
    qs = np.linspace(11.0, 15.0, points)
    lal, ext = [], []
    for q in qs:
        b = x.binary(dict(d, m1=120. * q / (1 + q), m2=120. / (1 + q)))
        lal.append(np.asarray(LAL(b, fr)[0]))
        ext.append(np.asarray(EXTENDED(b, fr)[0]))
    return dict(q=list(qs), difference=[x.weighted_mismatch(e, l, fr) for e, l in zip(ext, lal)],
                lal_steps=[x.weighted_mismatch(lal[i + 1], lal[i], fr) for i in range(points - 1)],
                extended_steps=[x.weighted_mismatch(ext[i + 1], ext[i], fr) for i in range(points - 1)])


def main():
    t = time.time()
    out = dict(boundaries=boundaries(), q12=q12_line())
    print(f"boundaries {time.time() - t:.0f} s")
    out["two_spin"] = two_spin()
    print(f"two-spin {time.time() - t:.0f} s")
    out["beta"] = beta_lines()
    print(f"beta {time.time() - t:.0f} s")
    out["steps"] = steps()
    x.write("switches.json", out)


if __name__ == "__main__":
    main()
