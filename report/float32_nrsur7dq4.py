#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 in float32 against float64: the precision of the inner
products between different waveforms, of the waveforms and of their
derivatives, and the cost on the CPU.

Pairs of binaries are drawn as in ``validate_nrsur7dq4.py`` (mass ratios
1-6 with either body first, total masses 20-300 solar masses, isotropic
spins up to 0.99): half as two independent draws ("random" pairs, whose
overlaps are near zero) and half as a draw and a small perturbation of its
masses, spins and angles ("nearby" pairs, overlaps mostly above 0.9). Half
of the pairs have ``f_ref = 0`` (the start of the surrogate), the others a
common ``f_ref`` of at least 20 Hz.

Both binaries of a pair are evaluated by the pure function on one time
grid (4096, 2048 or 1024 Hz by the lighter binary, covering both), in
float64 and in float32 (a ``Binary`` of float32 parameters), at the
default scale (the strain times 1e21), projected on a detector (``F_+ =
cos 2 psi``, ``F_x = sin 2 psi``) and tapered over the first 400 M. Inner
products are weighted by the aLIGO design sensitivity from 20 Hz, without
maximisation. They are computed in float64 from both, and in float32
throughout (whitened Fourier transforms and sums) for the float32
waveforms.

Recorded: for each pair the change of the normalised overlap and of the
inner product over the product of the norms; for each waveform the
noise-weighted and the largest relative error; the same with the
dynamics, or everything up to the polarizations on the surrogate's own
times, computed in float64 (which stage the error comes from); the float32
gradients of the overlap and of a noise-free log-likelihood for some
nearby pairs; and the time of one waveform, of a batch of 64 under
``jax.vmap`` and of a gradient in each precision.

Usage::

    python report/float32_nrsur7dq4.py [--n-pairs 150] [--n-gradients 40]
    python report/float32_nrsur7dq4.py --only-timings   # re-time, keep the pairs
"""

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from common import DATA, NAMES, environment, median_time, psd, wf, write
from waveformjax.models.nrsur7dq4 import _data, _driver, _waveform
from waveformjax.models.nrsur7dq4._dynamics import START_TIME, integrate
from waveformjax.models.nrsur7dq4._frames import coorbital_to_coprecessing, coprecessing_to_inertial

F_LOW = 20.0
SINGLE, DOUBLE = jnp.float32, jnp.float64
POLARIZATIONS = wf.td_polarizations_function("NRSur7dq4")


# ---------------------------------------------------------------------------
# the pairs
# ---------------------------------------------------------------------------

def isotropic_spin(rng):
    magnitude = rng.uniform(0, 0.99)
    cos_theta = rng.uniform(-1, 1)
    phi = rng.uniform(0, 2 * np.pi)
    sin_theta = math.sqrt(1 - cos_theta ** 2)
    return magnitude * np.array([sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta])


def draw(rng):
    total = float(np.exp(rng.uniform(np.log(20.0), np.log(300.0))))
    q = rng.uniform(1.0, 6.0)
    heavy, light = total * q / (1 + q), total / (1 + q)
    mass_1, mass_2 = (light, heavy) if rng.uniform() < 1 / 3 else (heavy, light)
    return dict(mass_1=mass_1, mass_2=mass_2, spin_1=isotropic_spin(rng), spin_2=isotropic_spin(rng),
                luminosity_distance=float(np.exp(rng.uniform(np.log(100), np.log(3000)))),
                inclination=float(np.arccos(rng.uniform(-1, 1))), phi_ref=float(rng.uniform(0, 2 * np.pi)))


def clip_spin(chi):
    norm = np.linalg.norm(chi)
    return chi * (0.99 / norm) if norm > 0.99 else chi


def perturb(rng, p, s):
    """``p`` with masses moved by a fraction ``s`` and spins and angles by
    ``3 s`` (standard deviations)."""
    return dict(p, mass_1=p["mass_1"] * (1 + s * rng.normal()), mass_2=p["mass_2"] * (1 + s * rng.normal()),
                spin_1=clip_spin(p["spin_1"] + 3 * s * rng.normal(size=3)),
                spin_2=clip_spin(p["spin_2"] + 3 * s * rng.normal(size=3)),
                inclination=p["inclination"] + 3 * s * rng.normal(), phi_ref=p["phi_ref"] + 3 * s * rng.normal())


def pair(rng, kind, binaries, data, scale=None):
    """The pair with its ``f_ref``, detector and time grid."""
    totals = [b["mass_1"] + b["mass_2"] for b in binaries]
    if rng.uniform() < 0.5:
        f_ref = 0.0
    else:
        start = max(_driver.start_frequency(wf.Binary(b["mass_1"], b["mass_2"], b["spin_1"], b["spin_2"]), data)
                    for b in binaries)
        f_ref = max(F_LOW, 1.05 * start)
        # LAL refuses omega_ref > 0.201 (the heavier binary of a random pair)
        if f_ref > 0.9 * min(0.201 / (np.pi * m * _waveform.MTSUN_SI) for m in totals):
            f_ref = 0.0
    seconds = [m * _waveform.MTSUN_SI for m in totals]
    lightest = min(totals)
    rate = 4096.0 if lightest < 60 else 2048.0 if lightest < 150 else 1024.0
    t_lo = min(float(data.t_coorb[0]) * m for m in seconds) - 0.01
    t_hi = max(float(data.t_coorb[-1]) * m for m in seconds) + 0.01
    n = int(math.ceil((t_hi - t_lo) * rate))
    n = max(1024, 1 << (n - 1).bit_length())
    binaries = [dict(b, f_ref=f_ref, spin_1=[float(x) for x in b["spin_1"]], spin_2=[float(x) for x in b["spin_2"]])
                for b in binaries]
    return dict(kind=kind, binaries=binaries, psi=float(rng.uniform(0, np.pi)), scale=scale, t_lo=t_lo,
                dt=1 / rate, n=n)


def draw_pairs(rng, n, data):
    pairs = [pair(rng, "random", [draw(rng), draw(rng)], data) for _ in range(n)]
    for _ in range(n):
        p = draw(rng)
        s = float(10 ** rng.uniform(-3.5, -1.5))
        pairs.append(pair(rng, "nearby", [p, perturb(rng, p, s)], data, s))
    return pairs


# ---------------------------------------------------------------------------
# the waveforms and inner products (in the precision of the binary)
# ---------------------------------------------------------------------------

def binary_of(keywords, dtype):
    return wf.Binary(*(jnp.asarray(keywords[name], dtype) for name in wf.Binary._fields))


def detector(b, times, psi, hp, hc):
    """``F_+ h_+ + F_x h_x``, tapered over the first 400 M of the
    surrogate."""
    m = (b.mass_1 + b.mass_2) * _waveform.MTSUN_SI
    x = jnp.clip((times - START_TIME * m) / (400.0 * m), 0.0, 1.0)
    return (hp * jnp.cos(2 * psi) + hc * jnp.sin(2 * psi)) * (0.5 - 0.5 * jnp.cos(jnp.pi * x))


def strain(b, times, psi):
    return detector(b, times, psi, *POLARIZATIONS(b, times))


def float64_stages(b):
    """The dynamics states at the nodes and the polarizations on the
    coorbital times, in float64."""
    data = _data.default_data()
    labels = _waveform.labelled(b)
    states, _ = integrate(data, labels.q, labels.chi_a, labels.chi_b, _waveform.omega_ref(b, labels))
    h, labels, _ = _waveform.inertial_modes(data, b)
    return states, jnp.stack(_waveform.polarizations_from_modes(b, h, labels), axis=-1)


def strain_from_states(b, times, psi, states):
    """The waveform from given dynamics states, in the precision of ``b``."""
    data = _data.default_data(b.mass_1.dtype)
    labels = _waveform.labelled(b)
    quat, phase, chi_a, chi_b = _waveform._coorbital_dynamics(data, states, labels.chi_a, labels.chi_b)
    re, im = _waveform._coorbital_modes(data, labels.q, chi_a, chi_b)
    re, im = coorbital_to_coprecessing(re, im, phase)
    h = coprecessing_to_inertial(re, im, quat.T)
    hpc = jnp.stack(_waveform.polarizations_from_modes(b, h, labels), axis=-1)
    return strain_from_coorbital(b, times, psi, hpc)


def strain_from_coorbital(b, times, psi, hpc):
    """The waveform from given polarizations on the coorbital times."""
    data = _data.default_data(b.mass_1.dtype)
    out = _waveform._resample(data, hpc, times / _waveform.labelled(b).m_tot_sec)
    return detector(b, times, psi, out[:, 0], out[:, 1])


def whiten(h, inv_asd, dt):
    return jnp.fft.rfft(h, 2 * h.shape[0]) * dt * inv_asd


def inner(a, b, df):
    return 4 * df * jnp.sum(a.real * b.real + a.imag * b.imag)


def products(h1, h2, inv_asd, dt, df):
    w1, w2 = whiten(h1, inv_asd, dt), whiten(h2, inv_asd, dt)
    return jnp.stack([inner(w1, w1, df), inner(w2, w2, df), inner(w1, w2, df)])


def overlap(b1, w2, times, psi, inv_asd, dt, df):
    w1 = whiten(strain(b1, times, psi), inv_asd, dt)
    return inner(w1, w2, df) / jnp.sqrt(inner(w1, w1, df) * inner(w2, w2, df))


def log_likelihood(b1, w2, times, psi, inv_asd, dt, df):
    """``<d, h1> - <h1, h1> / 2`` for the noise-free data ``d = h2``."""
    w1 = whiten(strain(b1, times, psi), inv_asd, dt)
    return inner(w1, w2, df) - 0.5 * inner(w1, w1, df)


strain_jit = jax.jit(strain)
stages_jit = jax.jit(float64_stages)
from_states_jit = jax.jit(strain_from_states)
from_coorbital_jit = jax.jit(strain_from_coorbital)
products_jit = jax.jit(products)
whiten_jit = jax.jit(whiten)
grad_overlap = jax.jit(jax.grad(overlap))
grad_log_likelihood = jax.jit(jax.grad(log_likelihood))

_INV_ASD = {}


def inv_asd(n, dt):
    """``1 / sqrt(S_n)`` on the frequencies of the zero-padded grid, zero
    below ``F_LOW``, divided by 1e21 for the scaled strain."""
    if (n, dt) not in _INV_ASD:
        f = np.fft.rfftfreq(2 * n, dt)
        out = np.zeros_like(f)
        keep = f >= F_LOW
        out[keep] = 1 / np.sqrt(psd(f[keep]))
        _INV_ASD[n, dt] = out / _waveform.STRAIN_SCALE
    return _INV_ASD[n, dt]


# ---------------------------------------------------------------------------
# the measurements
# ---------------------------------------------------------------------------

def flat(gradient):
    return np.concatenate([np.atleast_1d(np.asarray(x, np.float64)) for x in gradient])


def measure(case, gradients):
    n, dt = case["n"], case["dt"]
    ia = inv_asd(n, dt)
    out = {}
    for name, dtype in (("double", DOUBLE), ("single", SINGLE)):
        times = jnp.asarray(case["t_lo"], dtype) + jnp.asarray(dt, dtype) * jnp.arange(n, dtype=dtype)
        psi = jnp.asarray(case["psi"], dtype)
        args = (jnp.asarray(ia, dtype), jnp.asarray(dt, dtype), jnp.asarray(1 / (2 * n * dt), dtype))
        bs = [binary_of(b, dtype) for b in case["binaries"]]
        hs = [strain_jit(b, times, psi) for b in bs]
        out[name] = dict(h=[np.asarray(h, np.float64) for h in hs],
                         products=np.asarray(products_jit(*hs, *args), np.float64))
        if name == "double":
            out["stages"] = [stages_jit(b) for b in bs]
        else:
            out["dynamics64"] = [np.asarray(from_states_jit(b, times, psi, s.astype(dtype)), np.float64)
                                 for b, (s, _) in zip(bs, out["stages"])]
            out["coorbital64"] = [np.asarray(from_coorbital_jit(b, times, psi, h.astype(dtype)), np.float64)
                                  for b, (_, h) in zip(bs, out["stages"])]
        if gradients:
            w2 = whiten_jit(hs[1], *args[:2])
            out[name]["grad_overlap"] = flat(grad_overlap(bs[0], w2, times, psi, *args))
            out[name]["grad_log_likelihood"] = flat(grad_log_likelihood(bs[0], w2, times, psi, *args))

    # everything below in float64
    def white(h):
        return np.fft.rfft(h, 2 * n) * dt * ia

    def dot(a, b):
        return 4 / (2 * n * dt) * np.sum(a.real * b.real + a.imag * b.imag)

    def normalised(a, b):
        return dot(a, b) / np.sqrt(dot(a, a) * dot(b, b))

    ref = [white(h) for h in out["double"]["h"]]
    norm = np.sqrt(dot(ref[0], ref[0]) * dot(ref[1], ref[1]))
    o64 = normalised(*ref)
    record = dict(kind=case["kind"], totals=[b["mass_1"] + b["mass_2"] for b in case["binaries"]],
                  f_ref=case["binaries"][0]["f_ref"], samples=n, overlap=o64)
    variants = dict(single=out["single"]["h"], dynamics64=out["dynamics64"], coorbital64=out["coorbital64"])
    for name, hs in variants.items():
        w = [white(h) for h in hs]
        record[f"d_overlap_{name}"] = normalised(*w) - o64
        record[f"epsilon_{name}"] = (dot(*w) - dot(*ref)) / norm
        record[f"noise_weighted_{name}"] = [float(np.sqrt(dot(a - b, a - b) / dot(b, b))) for a, b in zip(w, ref)]
    w32 = [white(h) for h in out["single"]["h"]]
    p32, p64 = out["single"]["products"], out["double"]["products"]
    record["d_overlap_single_inner32"] = p32[2] / np.sqrt(p32[0] * p32[1]) - o64
    record["epsilon_single_inner32"] = (p32[2] - p64[2]) / np.sqrt(p64[0] * p64[1])
    record["d_overlap_inner32_only"] = p32[2] / np.sqrt(p32[0] * p32[1]) - normalised(*w32)
    record["norm_error"] = [float(p32[k] / p64[k] - 1) for k in range(2)]
    record["snr"] = [float(np.sqrt(p64[k])) for k in range(2)]
    record["log_likelihood_error"] = float((p32[2] - 0.5 * p32[0]) - (p64[2] - 0.5 * p64[0]))
    record["max_relative"] = [float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
                              for a, b in zip(out["single"]["h"], out["double"]["h"])]
    record["finite"] = bool(all(np.all(np.isfinite(h)) for h in out["single"]["h"]))
    if gradients:
        for key in ("grad_overlap", "grad_log_likelihood"):
            record[key] = dict(double=out["double"][key].tolist(), single=out["single"][key].tolist())
    return record


def percentiles(values):
    x = np.abs(np.asarray(values, dtype=float))
    return dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(x.max()), n=int(x.size))


def summarise(records):
    out = {}
    for kind in ("random", "nearby", "all"):
        chosen = [r for r in records if kind == "all" or r["kind"] == kind]
        entry = dict(n=len(chosen), overlap_median=float(np.median([r["overlap"] for r in chosen])))
        for key in ("d_overlap_single", "d_overlap_single_inner32", "d_overlap_inner32_only",
                    "d_overlap_dynamics64", "d_overlap_coorbital64", "epsilon_single", "epsilon_single_inner32"):
            entry[key] = percentiles([r[key] for r in chosen])
        out[kind] = entry
    waveforms = {}
    for key in ("noise_weighted_single", "noise_weighted_dynamics64", "noise_weighted_coorbital64", "max_relative",
                "norm_error"):
        waveforms[key] = percentiles([x for r in records for x in r[key]])
    out["waveforms"] = waveforms
    nearby = [r for r in records if r["kind"] == "nearby"]
    out["log_likelihood_over_snr2"] = percentiles([r["log_likelihood_error"] / r["snr"][1] ** 2 for r in nearby])
    gradients = {}
    for key in ("grad_overlap", "grad_log_likelihood"):
        chosen = [r[key] for r in records if key in r]
        if not chosen:
            continue
        double = np.array([g["double"] for g in chosen])
        single = np.array([g["single"] for g in chosen])
        # columns that vary: not f_ref (zero at f_ref = 0), nor the distance
        # for the overlap (independent of it)
        columns = [i for i, name in enumerate(NAMES) if name != "f_ref"
                   and not (key == "grad_overlap" and name == "luminosity_distance")]
        relative = np.abs(single - double)[:, columns] / np.abs(double)[:, columns]
        typical = np.abs(single - double)[:, columns] / np.sqrt(np.mean(double[:, columns] ** 2, axis=0))
        gradients[key] = dict(n=len(chosen), relative=percentiles(relative.ravel()),
                              over_typical=percentiles(typical.ravel()),
                              finite=bool(np.all(np.isfinite(single))))
    out["gradients"] = gradients
    return out


def timings():
    """Median times (s) on the CPU of one waveform on 8192 samples at 4096
    Hz, its dynamics alone, a vmapped batch of 64 (and its dynamics) and
    the gradient of the power, in each precision."""
    rng = np.random.default_rng(1)
    out = {}
    base = dict(mass_1=40.0, mass_2=20.0, spin_1=[0.4, 0.2, 0.3], spin_2=[-0.2, 0.3, -0.1],
                luminosity_distance=500.0, inclination=0.7, phi_ref=1.2, f_ref=20.0)
    others = [dict(draw(rng), f_ref=0.0) for _ in range(64)]
    for name, dtype in (("double", DOUBLE), ("single", SINGLE)):
        times = jnp.asarray(-1.9, dtype) + jnp.asarray(1 / 4096, dtype) * jnp.arange(8192, dtype=dtype)
        b = binary_of(base, dtype)
        batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[binary_of(o, dtype) for o in others])

        def waveform(b):
            return POLARIZATIONS(b, times)[0]

        def dynamics(b):
            data = _data.default_data(dtype)
            labels = _waveform.labelled(b)
            return integrate(data, labels.q, labels.chi_a, labels.chi_b, _waveform.omega_ref(b, labels))[0]

        functions = dict(single=(jax.jit(waveform), b), dynamics=(jax.jit(dynamics), b),
                         batch64=(jax.jit(jax.vmap(waveform)), batch),
                         batch64_dynamics=(jax.jit(jax.vmap(dynamics)), batch),
                         grad=(jax.jit(jax.grad(lambda b: jnp.sum(waveform(b) ** 2))), b))
        out[name] = {key: median_time(lambda: f(x), repeats=7 if "batch" in key else 31, number=1 if "batch" in key
                                      else 5) for key, (f, x) in functions.items()}
        print(name, {k: f"{v * 1e3:.2f} ms" for k, v in out[name].items()}, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-pairs", type=int, default=150, help="pairs of each kind")
    parser.add_argument("--n-gradients", type=int, default=40, help="nearby pairs whose gradients are compared")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output", default="float32_nrsur7dq4.json")
    parser.add_argument("--only-timings", action="store_true",
                        help="measure the timings again and keep the pairs of the existing output")
    args = parser.parse_args()
    if wf.nrsur7dq4_data_file() is None:
        raise SystemExit("NRSur7dq4_v1.0.h5 not found: set NRSUR7DQ4_DATA")
    if args.only_timings:
        payload = json.loads((DATA / args.output).read_text())
        payload.update(timings=timings(), timings_environment=environment())
        write(args.output, payload)
        return
    data = _data.default_data()
    rng = np.random.default_rng(args.seed)
    cases = draw_pairs(rng, args.n_pairs, data)
    records, compared = [], 0
    start = time.perf_counter()
    for index, case in enumerate(cases):
        gradients = case["kind"] == "nearby" and compared < args.n_gradients
        compared += gradients
        records.append(dict(measure(case, gradients), index=index))
        if index % 25 == 0:
            r = records[-1]
            print(f"{index:4d} {r['kind']:6s} overlap {r['overlap']: .3f} change {r['d_overlap_single']: .1e} "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)
    payload = dict(environment=environment(), seed=args.seed, n_pairs=args.n_pairs, f_low=F_LOW,
                   summary=summarise(records), timings=timings(), records=records)
    s = payload["summary"]
    print("change of the overlap:", s["all"]["d_overlap_single_inner32"])
    print("per waveform:", s["waveforms"]["noise_weighted_single"])
    write(args.output, payload)


if __name__ == "__main__":
    main()
