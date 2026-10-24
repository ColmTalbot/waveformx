# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Shared set-up of the measurements of the report on IMRPhenomXPNR without
fallbacks and with the extended SpinTaylor angles (``report/extended``).

The model is called through :func:`waveformx.models.xpnr.model._decide`
with the static flags of the three variants: LAL's (``fallback=True``),
LAL's SpinTaylor angles only (``fallback=False``) and the extended, smooth
angles (``extended=True``), as ``WaveformOptions`` selects them, and the
lean model (``lean=True``).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import MTSUN_SI, environment, mismatch, psd  # noqa: E402, F401  (sets OMP_NUM_THREADS)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import waveformx as wf  # noqa: E402
from waveformx.models.xpnr import model, pnr  # noqa: E402

DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
MODES = model.DEFAULT_MODES
VARIANTS = {"lal": dict(fallback=True), "strict": dict(fallback=False), "extended": dict(fallback=False, extended=True),
            "lean": dict(fallback=False, lean=True)}


def polarizations_fn(variant):
    """``f(binary, frequencies) -> (h_+, h_x)`` of a variant, NaN where it
    fails (unjitted)."""
    flags = VARIANTS[variant]

    def f(b, frequencies):
        with model.lean_mode(flags.get("lean", False)):
            pnr.clear_cache()
            data = model._decide(b, frequencies, 0.0, frequencies[0], frequencies[-1], MODES, True, **flags)
            hp, hc = model._twist_polarizations(data, MODES, frequencies)
        nan = jnp.asarray(jnp.nan + 1j * jnp.nan, jnp.complex128)
        return jnp.where(data.error, nan, hp), jnp.where(data.error, nan, hc)

    return f


def setup_fn(extended=False, smooth=None):
    """``f(binary, frequencies) -> (prec, wf, grid, flags)``: the set-up of
    version 330 (:func:`.model._spintaylor_setup`) as the model does it."""
    smooth = extended if smooth is None else smooth

    def f(b, frequencies):
        pnr.clear_cache()
        w, m1_si, m2_si, chi1, chi2 = model.setup_waveform(b, 0.0, frequencies[0], frequencies[-1])
        w.hm_release = 122019
        base, wf0 = pnr.common_setup(w, m1_si, m2_si, chi1, chi2, pnr.mode_range(MODES), True, extended, smooth)
        return model._spintaylor_setup(base, wf0, m1_si, m2_si, MODES, jnp.asarray(True))

    return f


def binary(d, **changes):
    """A :class:`waveformx.Binary` from a dict with ``m1``, ``m2``, ``s1``,
    ``s2``, ``fref`` (and optionally ``inclination``, ``phi_ref``)."""
    d = dict(d, **changes)
    return wf.Binary(mass_1=jnp.float64(d["m1"]), mass_2=jnp.float64(d["m2"]), spin_1=jnp.asarray(d["s1"], jnp.float64),
                     spin_2=jnp.asarray(d["s2"], jnp.float64), luminosity_distance=jnp.float64(500.0),
                     inclination=jnp.float64(d.get("inclination", 0.5)), phi_ref=jnp.float64(d.get("phi_ref", 0.3)),
                     f_ref=jnp.float64(d["fref"]))


def stack(binaries):
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *binaries)


def f_max(total_mass):
    return min(0.3 / (total_mass * MTSUN_SI), 1024.0)


def frequencies(d, n=1024):
    return jnp.linspace(d["fmin"], f_max(d["m1"] + d["m2"]), n)


def weighted_mismatch(a, b, f):
    """The mismatch of :mod:`common` with aLIGO design weights."""
    return mismatch(np.asarray(a), np.asarray(b), 1.0 / psd(np.asarray(f)))


def relative_change(a, b, f):
    """``|a - b| / |b|`` with aLIGO design weights."""
    w = 1.0 / psd(np.asarray(f))
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.sum(w * np.abs(a - b) ** 2) / np.sum(w * np.abs(b) ** 2)))


def draws(n, seed):
    """Random binaries: total masses log-uniform in 5-500 solar masses, mass
    ratios uniform in 1/20-1, isotropic spins of magnitude up to 0.99,
    ``f_min`` among 10, 15 and 20 Hz, ``f_ref = f_min`` (30%) or uniform up
    to ``min(5 f_min, f_max)``; binaries with ``f_max <= 1.2 f_min`` are
    redrawn."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        total = np.exp(rng.uniform(np.log(5.0), np.log(500.0)))
        q = rng.uniform(1 / 20, 1.0)
        m1 = total / (1 + q)
        fmin = float(rng.choice([10.0, 15.0, 20.0]))
        spins = []
        for _ in range(2):
            a = rng.uniform(0, 0.99)
            c = rng.uniform(-1, 1)
            p = rng.uniform(0, 2 * np.pi)
            s = np.sqrt(1 - c * c)
            spins.append([a * s * np.cos(p), a * s * np.sin(p), a * c])
        u = rng.uniform()
        if f_max(total) <= 1.2 * fmin:
            continue
        fref = fmin if u < 0.3 else rng.uniform(fmin, min(5 * fmin, f_max(total)))
        out.append(dict(M=total, q=q, m1=m1, m2=total - m1, fmin=fmin, fref=float(fref), s1=spins[0], s2=spins[1]))
    return out


def write(name, payload):
    path = DATA / name
    path.write_text(json.dumps(dict(environment=environment(), **payload), separators=(",", ":")))
    print(f"wrote {path}")


def read(name):
    return json.loads((DATA / name).read_text())
