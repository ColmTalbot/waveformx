#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR against NRSur7dq4 on a fixed set of binaries: writes
``report/data/crossmodel.json``.

The signal is NRSur7dq4's ``h_+`` (LAL's ``SimInspiralFD``, or ours through
:func:`waveformx.simulate_fd`) of the whole surrogate (``f_min = 0``; LAL
tapers its start): conditioned from a frequency near the band, its first
cycles are tapered and filtered inside the band, which changes the mismatch
from 20 Hz by ``~1e-4`` from 10 Hz and ``~3e-3`` from 20 Hz. The template IMRPhenomXPNR (LAL's
``SimInspiralChooseFDWaveformSequence`` or our lean model, ``lean=True``) at
the same binary, with the same reference frequency (20 Hz) and LAL's frame
conventions, on 20-1024 Hz in steps of 1/8 Hz with the aLIGO design PSD. The
mismatch is maximised over time and the antenna patterns of the template
(:func:`waveformx.match.polarization_mismatch`) and optimised over the
parameters that the two models define differently
(:func:`waveformx.match.with_conventions`): the reference phase and a
common rotation of the in-plane spins about the orbital angular momentum,
the orientation of the frame in which the spins are given. The angle
between the in-plane spins is physical and is not optimised (freeing it
lets the optimisation reach different binaries). The best of a grid of 16
starts (4 reference phases times 4 rotations), then Nelder-Mead from the
best 3, the same procedure for LAL's model and ours. For our model, the
gradient optimisation of :func:`waveformx.match.optimize` from the same 3
starts is recorded as a check of it.

The benchmark binaries (``report/data/crossmodel_points.json``) are chosen
once with ``--select``: draws in NRSur7dq4's range (``q <= 4``, ``|chi| <=
0.8``, isotropic spins and orientation, total masses 90-200 Msun, kept if
the surrogate covers LAL's conditioning band from 10 Hz, so that it starts
well below the band),
LAL's optimised mismatch for each, and 20 binaries spread over the
parameters (farthest-point sampling in ``q``, ``M``, ``chi_eff``,
``chi_p`` and ``cos(inclination)``) among those where LAL's models agree
(mismatch below ``1e-3``) and 20 where they disagree (above ``1e-2``).
Keep the set fixed: it tracks the model over time.

Usage::

    python report/crossmodel.py --select [--n-draws 500] [--workers 16]
    python report/crossmodel.py [--points crossmodel_points.json]
"""

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA,
    environment,
    jnp,
    json,
    lal_sequence,
    lal_td_arguments,
    lalsim,
    np,
    psd,
    wf,
    write,
)

DELTA_F, F_LOW, F_MAX = 0.125, 20.0, 1024.0
F_REF = 20.0
FREQUENCIES = np.arange(round(F_LOW / DELTA_F), round(F_MAX / DELTA_F) + 1) * DELTA_F
FIRST = int(round(F_LOW / DELTA_F))
#: the surrogate must cover the conditioning from this frequency
COVERED = 10.0
POINTS = "crossmodel_points.json"
#: the grid of starts: reference phases times common rotations of the spins
STARTS = np.array([[p, a] for p in np.arange(4) * np.pi / 2 for a in np.arange(4) * np.pi / 2])
KEEP = 3
AGREE, DISAGREE, PER_REGION = 1e-3, 1e-2, 20

_PSD = None
_MISMATCH = None


def setup():
    """The PSD and the compiled mismatch (once per process)."""
    global _PSD, _MISMATCH
    if _MISMATCH is None:
        _PSD = jnp.asarray(psd(FREQUENCIES))
        _MISMATCH = __import__("jax").jit(
            lambda s, p, c: wf.match.polarization_mismatch(s, p, c, jnp.asarray(FREQUENCIES), _PSD))
    return _MISMATCH


def band(series):
    return np.asarray(series)[FIRST:FIRST + len(FREQUENCIES)]


def lal_nrsur(keywords):
    """LAL's NRSur7dq4 ``h_+`` from ``SimInspiralFD`` on the band."""
    hp, _ = lalsim.SimInspiralFD(*lal_td_arguments(keywords, 0.0, 0.0)[:14], DELTA_F, 0.0, F_MAX,
                                 keywords["f_ref"], None, lalsim.NRSur7dq4)
    return band(hp.data.data)


def our_nrsur(keywords, f_min=0.0):
    """Our NRSur7dq4 ``h_+`` from :func:`waveformx.simulate_fd` (raises
    where the surrogate does not cover the conditioning from ``f_min``, as
    LAL does, which may then crash; its strain is times ``1e21``)."""
    out = wf.simulate_fd("NRSur7dq4", DELTA_F, f_min, F_MAX, **keywords)
    assert out.delta_f == DELTA_F
    return band(out.hplus)


def conventions(x):
    """The shifts of :func:`waveformx.match.with_conventions` of ``x =
    (delta phi_ref, rotation)``: both spins rotated alike."""
    return jnp.stack([x[0], x[1], x[1]]) if isinstance(x, jnp.ndarray) else np.array([x[0], x[1], x[1]])


def shifted(keywords, x):
    """``keywords`` with the convention shifts ``x = (delta phi_ref,
    rotation)``."""
    x = conventions(np.asarray(x))
    def rotate(spin, angle):
        c, s = np.cos(angle), np.sin(angle)
        return (c * spin[0] - s * spin[1], s * spin[0] + c * spin[1], spin[2])

    return dict(keywords, phi_ref=keywords["phi_ref"] + x[0], spin_1=rotate(keywords["spin_1"], x[1]),
                spin_2=rotate(keywords["spin_2"], x[2]))


def lal_xpnr(keywords):
    return lal_sequence(keywords, FREQUENCIES)


_LEAN = None


def lean_function():
    global _LEAN
    if _LEAN is None:
        _LEAN = __import__("jax").jit(wf.polarizations_function("IMRPhenomXPNR", lean=True))
    return _LEAN


_DEFAULT = None


def default_function():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = __import__("jax").jit(wf.polarizations_function("IMRPhenomXPNR"))
    return _DEFAULT


def lean_binary(keywords, x=None):
    binary = wf.Binary(*(jnp.asarray(keywords[k], jnp.float64) for k in wf.Binary._fields))
    return binary if x is None else wf.match.with_conventions(binary, conventions(x))


def our_xpnr(keywords):
    return lean_function()(lean_binary(keywords), jnp.asarray(FREQUENCIES))


def optimised(signal, template, keywords):
    """The mismatch of ``signal`` with ``template(keywords)`` optimised over
    the convention shifts: ``dict(mismatch, shifts, unoptimised, starts,
    evaluations)``."""
    import scipy.optimize

    mismatch = setup()
    signal = jnp.asarray(signal)
    count = [0]

    def objective(x):
        count[0] += 1
        hp, hc = template(shifted(keywords, x))
        value = float(mismatch(signal, jnp.asarray(hp), jnp.asarray(hc)))
        return value if np.isfinite(value) else 1.0

    grid = np.array([objective(x) for x in STARTS])
    best = []
    for index in np.argsort(grid)[:KEEP]:
        x0 = STARTS[index]
        out = scipy.optimize.minimize(objective, x0, method="Nelder-Mead", options=dict(
            initial_simplex=x0 + np.vstack([np.zeros(2), 0.3 * np.eye(2)]), xatol=1e-4, fatol=1e-9, maxfev=400))
        best.append((out.fun, out.x))
    value, x = min(best, key=lambda b: b[0])
    return dict(mismatch=float(value), shifts=[float(v) for v in x], unoptimised=float(grid[0]),
                starts=[int(i) for i in np.argsort(grid)[:KEEP]], evaluations=count[0])


def _lean_objective(x, signal, binary, psd_):
    f = wf.polarizations_function("IMRPhenomXPNR", lean=True)
    frequencies = jnp.asarray(FREQUENCIES)
    hp, hc = f(wf.match.with_conventions(binary, conventions(x)), frequencies)
    return wf.match.polarization_mismatch(signal, hp, hc, frequencies, psd_)


def gradient_optimised(signal, keywords, starts):
    """Our lean model with :func:`waveformx.match.optimize` (L-BFGS-B with
    the gradient through the model) from ``starts``."""
    setup()
    result = wf.match.optimize(_lean_objective, starts, (jnp.asarray(signal), lean_binary(keywords), _PSD))
    return dict(mismatch=result.fun, shifts=[float(v) for v in result.x])


# ---------------------------------------------------------------------------
# the benchmark set
# ---------------------------------------------------------------------------


def draw(rng):
    """A binary in NRSur7dq4's range (see the module docstring)."""
    q = rng.uniform(1.0, 4.0)
    total = rng.uniform(90.0, 200.0)

    def spin():
        magnitude = rng.uniform(0.0, 0.8)
        cos_theta, phi = rng.uniform(-1.0, 1.0), rng.uniform(0.0, 2 * np.pi)
        sin_theta = np.sqrt(1 - cos_theta**2)
        return (magnitude * sin_theta * np.cos(phi), magnitude * sin_theta * np.sin(phi), magnitude * cos_theta)

    return dict(mass_1=total * q / (1 + q), mass_2=total / (1 + q), spin_1=spin(), spin_2=spin(),
                luminosity_distance=500.0, inclination=float(np.arccos(rng.uniform(-1.0, 1.0))),
                phi_ref=rng.uniform(0.0, 2 * np.pi), f_ref=F_REF)


def features(keywords):
    """``q, M, chi_eff, chi_p, cos(inclination)``."""
    m1, m2 = keywords["mass_1"], keywords["mass_2"]
    s1, s2 = np.asarray(keywords["spin_1"]), np.asarray(keywords["spin_2"])
    q = m1 / m2
    chi_eff = (m1 * s1[2] + m2 * s2[2]) / (m1 + m2)
    b1, b2 = 2 + 1.5 / q, 2 + 1.5 * q
    chi_p = max(b1 * m1**2 * np.hypot(*s1[:2]), b2 * m2**2 * np.hypot(*s2[:2])) / (b1 * m1**2)
    return dict(q=q, total_mass=m1 + m2, chi_eff=chi_eff, chi_p=chi_p, cos_inclination=np.cos(keywords["inclination"]))


def _select_one(keywords):
    """LAL's optimised mismatch at one draw, or ``None`` outside
    NRSur7dq4's range or where the surrogate does not cover the
    conditioning from :data:`COVERED` (checked with our NRSur7dq4: LAL can
    crash after such a failure)."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        our_nrsur(keywords, COVERED)
    except wf.WaveformError as error:
        return dict(keywords=keywords, error=str(error))
    start = time.perf_counter()
    signal = lal_nrsur(keywords)
    out = optimised(signal, lal_xpnr, keywords)
    out["seconds"] = time.perf_counter() - start
    return dict(keywords=keywords, lal=out)


def _init_worker():
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def spread(candidates, n):
    """``n`` of ``candidates`` spread over the features (farthest-point
    sampling from the first, with features scaled to the unit interval)."""
    if len(candidates) <= n:
        return list(range(len(candidates)))
    names = ["q", "total_mass", "chi_eff", "chi_p", "cos_inclination"]
    x = np.array([[c["features"][k] for k in names] for c in candidates])
    x = (x - x.min(0)) / np.where(np.ptp(x, 0) > 0, np.ptp(x, 0), 1)
    chosen = [0]
    distance = np.linalg.norm(x - x[0], axis=1)
    while len(chosen) < n:
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(x - x[chosen[-1]], axis=1))
    return chosen


def select(n_draws, seed, workers):
    rng = np.random.default_rng(seed)
    draws = [draw(rng) for _ in range(n_draws)]
    context = multiprocessing.get_context("spawn")
    start = time.perf_counter()
    with context.Pool(workers, initializer=_init_worker) as pool:
        results = []
        for i, result in enumerate(pool.imap(_select_one, draws)):
            results.append(result)
            if "lal" in result:
                print(i, f"{result['lal']['mismatch']:.2e}", flush=True)
    valid = [dict(r, features=features(r["keywords"])) for r in results if "lal" in r]
    # (in the order of the draws, so the selection is reproducible)
    agree = [r for r in valid if r["lal"]["mismatch"] < AGREE]
    disagree = [r for r in valid if r["lal"]["mismatch"] > DISAGREE]
    points = []
    for label, group in (("agree", agree), ("disagree", disagree)):
        for index in spread(group, PER_REGION):
            r = group[index]
            points.append(dict(label=label, keywords=r["keywords"], features=r["features"], lal=r["lal"]))
    payload = dict(
        description="Benchmark binaries for IMRPhenomXPNR against NRSur7dq4 (report/crossmodel.py --select)",
        settings=dict(n_draws=n_draws, seed=seed, delta_f=DELTA_F, f_low=F_LOW, f_max=F_MAX, f_ref=F_REF,
                      covered=COVERED, agree_below=AGREE, disagree_above=DISAGREE, per_region=PER_REGION),
        environment=environment(), seconds=time.perf_counter() - start,
        n_valid=len(valid), n_agree=len(agree), n_disagree=len(disagree),
        draws=[dict(features=r["features"], lal_mismatch=r["lal"]["mismatch"], keywords=r["keywords"]) for r in valid],
        points=points,
    )
    write(POINTS, payload)


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


def measure(points_file, gradient):
    points = json.loads((DATA / points_file).read_text())["points"]
    out = []
    start = time.perf_counter()
    for i, point in enumerate(points):
        keywords = dict(point["keywords"], spin_1=tuple(point["keywords"]["spin_1"]),
                        spin_2=tuple(point["keywords"]["spin_2"]))
        lal_signal = lal_nrsur(keywords)
        our_signal = our_nrsur(keywords)
        setup()
        record = dict(label=point["label"], features=point["features"], selection=point["lal"]["mismatch"])
        # our NRSur7dq4 against LAL's (no optimisation)
        record["nrsur_ours_vs_lal"] = float(1 - wf.match.match(jnp.asarray(lal_signal), jnp.asarray(our_signal),
                                                               jnp.asarray(FREQUENCIES), _PSD))
        # the floor from the start of the surrogate: the whole surrogate
        # against the surrogate conditioned from COVERED
        record["nrsur_conditioning"] = float(1 - wf.match.match(
            jnp.asarray(our_signal), jnp.asarray(our_nrsur(keywords, COVERED)), jnp.asarray(FREQUENCIES), _PSD))
        record["lal_xpnr_vs_lal_nrsur"] = optimised(lal_signal, lal_xpnr, keywords)
        record["our_xpnr_vs_lal_nrsur"] = optimised(lal_signal, our_xpnr, keywords)
        record["our_xpnr_vs_our_nrsur"] = optimised(our_signal, our_xpnr, keywords)
        # whether LAL used the SpinTaylor angles: our default model, which
        # follows LAL's, is NaN where LAL would switch to other angles
        record["lal_spintaylor"] = bool(np.all(np.isfinite(np.asarray(
            default_function()(lean_binary(keywords), jnp.asarray(FREQUENCIES))[0]))))
        lal_hp, _ = lal_xpnr(keywords)
        record["our_xpnr_vs_lal_xpnr"] = optimised(lal_hp, our_xpnr, keywords)
        if gradient:
            starts = STARTS[record["our_xpnr_vs_lal_nrsur"]["starts"]]
            record["our_xpnr_vs_lal_nrsur_gradient"] = gradient_optimised(lal_signal, keywords, starts)
        print(i, point["label"], {k: f"{v['mismatch']:.2e}" for k, v in record.items() if isinstance(v, dict)
                                  and "mismatch" in v}, f"NRSur {record['nrsur_ours_vs_lal']:.1e}", flush=True)
        out.append(record)
    write("crossmodel.json", dict(
        description="IMRPhenomXPNR against NRSur7dq4 on the benchmark binaries (report/crossmodel.py)",
        settings=dict(delta_f=DELTA_F, f_low=F_LOW, f_max=F_MAX, f_ref=F_REF, starts=STARTS.tolist(), keep=KEEP,
                      points=points_file),
        environment=environment(), seconds=time.perf_counter() - start, records=out,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--select", action="store_true", help="draw and select the benchmark binaries")
    parser.add_argument("--n-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--points", default=POINTS)
    parser.add_argument("--no-gradient", action="store_true", help="skip the gradient optimisation check")
    args = parser.parse_args()
    if args.select:
        select(args.n_draws, args.seed, args.workers)
    else:
        measure(args.points, not args.no_gradient)


if __name__ == "__main__":
    main()
