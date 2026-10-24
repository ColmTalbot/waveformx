#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The figures of the report on the extended SpinTaylor angles, from
``report/extended/data``, in the style of the main report (SVG whose text
and axis colours follow the page's light and dark themes).

Colours: LAL's model is always blue (circles), the extended one orange
(squares); a third and fourth series take green and purple.

Usage::

    python report/extended/figures.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import make_figures as base  # noqa: E402
from make_figures import BLUE, GREEN, ORANGE, PURPLE, plt  # noqa: E402

import numpy as np  # noqa: E402

DATA = HERE / "data"
FIGURES = HERE / "figures"
LAL = dict(color=BLUE, marker="o", label="LAL (with its fallbacks)")
EXTENDED = dict(color=ORANGE, marker="s", label="extended")
BEFORE = dict(color=PURPLE, marker="D", label="extended, before the smoothness fixes")


def load(name):
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else None


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    svg = path.read_text()
    for colour, variable in base.REPLACEMENTS.items():
        svg = re.sub(colour, variable, svg, flags=re.IGNORECASE)
    path.write_text(svg)
    print(f"wrote {path}")


def failure_map():
    data = load("failures.json")
    if data is None:
        return
    draws = data["draws"]
    ok = [d for d in draws if d["status"] == "ok"]
    fail = [d for d in draws if d["status"] != "ok"]
    msun = 4.925490947641267e-06
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ax = axes[0]
    rng = np.random.default_rng(0)
    shown = [ok[i] for i in rng.choice(len(ok), min(3000, len(ok)), replace=False)]
    ax.scatter([d["M"] for d in shown], [d["M"] * msun * d["fref"] for d in shown], s=4, color=BLUE, alpha=0.5,
               linewidths=0, label="SpinTaylor angles")
    ax.scatter([d["M"] for d in fail], [d["M"] * msun * d["fref"] for d in fail], s=5, color=ORANGE, marker="s",
               alpha=0.6, linewidths=0, label="LAL falls back")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total mass (M$_\\odot$)")
    ax.set_ylabel("$M f_\\mathrm{ref}$")
    ax.legend(loc="lower right", markerscale=3)
    ax = axes[1]
    with_test = [d for d in draws if "energy_test" in d]
    for group, style in (("ok", dict(color=BLUE, label="SpinTaylor angles")),
                         ("fail", dict(color=ORANGE, label="LAL falls back"))):
        values = np.array([d["energy_test"] for d in with_test if (d["status"] == "ok") == (group == "ok")])
        bins = np.linspace(-6, 3, 61)
        ax.hist(np.clip(values, -6, 3), bins=bins, histtype="step", linewidth=1.5, **style)
    ax.axvline(0.0, color=base.MUTED, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("PN energy test at $f_\\mathrm{ref}$ (clipped to [−6, 3])")
    ax.set_ylabel("binaries")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "failure_map")


def failure_fraction():
    data = load("failures.json")
    if data is None:
        return
    msun = 4.925490947641267e-06
    draws = data["draws"]
    x = np.array([d["M"] * msun * d["fref"] for d in draws])
    failed = np.array([d["status"] != "ok" for d in draws])
    edges = np.geomspace(x.min(), x.max(), 30)
    centres = np.sqrt(edges[1:] * edges[:-1])
    fraction = [failed[(x >= a) & (x < b)].mean() if np.any((x >= a) & (x < b)) else np.nan
                for a, b in zip(edges[:-1], edges[1:])]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(centres, fraction, color=ORANGE, marker="s", markersize=4, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("$M f_\\mathrm{ref}$")
    ax.set_ylabel("fraction where LAL falls back")
    save(fig, "failure_fraction")


def jumps():
    data = load("jumps.json")
    if data is None:
        return
    groups = {}
    for variant in ("lal", "extended"):
        for r in data["jumps"][variant]:
            if not np.isfinite(r["jump"]):
                continue
            key = "+".join(sorted(r["changed"])) or "none"
            label = ("end of the fine run" if "n_fine" in key else
                     "first backward sample" if "n_b" in key else
                     "PNR grid" if "grid_n" in key or "grid_start" in key else
                     "β connection" if any(k in key for k in ("upper_none", "attach")) else
                     "forward run" if "n_fwd" in key else "other")
            groups.setdefault(label, {"lal": [], "extended": []})[variant].append(max(r["jump"], 1e-16))
    labels = [k for k in ("end of the fine run", "first backward sample", "forward run", "PNR grid", "β connection",
                          "other") if k in groups]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, label in enumerate(labels):
        for offset, variant, style in ((-0.15, "lal", LAL), (0.15, "extended", EXTENDED)):
            values = groups[label][variant]
            if values:
                jitter = np.random.default_rng(i).uniform(-0.06, 0.06, len(values))
                ax.scatter(i + offset + jitter, values, s=12, color=style["color"], marker=style["marker"], alpha=0.75,
                           linewidths=0, label=style["label"] if i == 0 else None)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yscale("log")
    ax.set_ylabel("|Δh| / |h| across the change")
    ax.legend(loc="upper right")
    save(fig, "jumps")


def switches():
    data = load("switches.json")
    if data is None:
        return
    rows = data["boundaries"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(rows))
    for offset, variant, style in ((-0.15, "lal", LAL), (0.15, "extended", EXTENDED)):
        ax.scatter([max(r[variant]["mismatch"], 1e-17) for r in rows], y + offset, color=style["color"],
                   marker=style["marker"], s=22, label=style["label"], zorder=3)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("mismatch across the switch")
    ax.axvline(1e-3, color=base.MUTED, linewidth=0.8, linestyle="--")
    ax.legend(loc="lower right")
    save(fig, "switches")


def two_spin_and_beta():
    data = load("switches.json")
    if data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axes[0]
    ts = data["two_spin"]
    ax.scatter([r["q"] for r in ts], [max(r["lal"], 1e-17) for r in ts], s=10, **LAL)
    ax.scatter([r["q"] for r in ts], [max(r["extended"], 1e-17) for r in ts], s=10, **EXTENDED)
    ax.set_yscale("log")
    ax.set_xlabel("mass ratio $q$")
    ax.set_ylabel("mismatch across $|\\chi_2| = 10^{-3}$")
    ax.axhline(1e-3, color=base.MUTED, linewidth=0.8, linestyle="--")
    # (in the empty band between the two groups)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.3), markerscale=1.5)
    ax = axes[1]
    beta = data["beta"]
    ax.scatter([max(r["lal"], 1e-17) for r in beta], [max(r["extended_1e4"], 1e-17) for r in beta], s=12,
               color=ORANGE, marker="s", linewidths=0)
    lim = [1e-17, 1.0]
    ax.plot(lim, lim, color=base.MUTED, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("LAL: mismatch across the decision")
    ax.set_ylabel("extended: mismatch across ±5e-5 of spin")
    fig.tight_layout()
    save(fig, "two_spin_beta")


def q12():
    data = load("switches.json")
    if data is None or "q12" not in data:
        return
    d = data["q12"]
    q = np.array(d["q"])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(q, np.maximum(d["difference"], 1e-17), color=ORANGE, marker="s", markersize=3, linewidth=1.2,
            label="extended against LAL")
    ax.axvspan(12.0, 14.0, color=base.GRID, alpha=0.6, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("mass ratio $q$ ($M = 120\\,M_\\odot$)")
    ax.set_ylabel("mismatch")
    ax.legend(loc="upper right")
    save(fig, "q12")


def steps():
    data = load("switches.json")
    if data is None:
        return
    s = data["steps"]["steps"]
    h = np.array([r["step"] for r in s])
    mm = np.array([max(r["mismatch"], 1e-18) for r in s])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(h, mm, color=ORANGE, marker="s", markersize=4, linewidth=1.5, label="extended")
    ax.plot(h, mm[0] * (h / h[0]) ** 2, color=base.MUTED, linewidth=0.8, linestyle="--", label="∝ step²")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step in the spin magnitude")
    ax.set_ylabel("mismatch between the two ends")
    ax.legend(loc="upper left")
    save(fig, "steps")


def noise():
    data = load("noise.json")
    if data is None:
        return
    entries = [("waveformrs IMRPhenomXAS", data["waveformrs"]["IMRPhenomXAS"], PURPLE, "^"),
               ("waveformrs IMRPhenomXHM", data["waveformrs"]["IMRPhenomXHM"], GREEN, "D"),
               ("waveformrs IMRPhenomXPNR", data["waveformrs"]["IMRPhenomXPNR"], BLUE, "v"),
               ("JAX, LAL's model", data["jax"]["lal"]["polarizations"], BLUE, "o"),
               ("JAX, extended", data["jax"]["extended"]["polarizations"], ORANGE, "s")]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (label, values, colour, marker) in enumerate(entries):
        ax.scatter(np.maximum(values, 1e-17), np.full(len(values), i), color=colour, marker=marker, s=24, zorder=3)
    ax.set_yticks(range(len(entries)), [e[0] for e in entries])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("|Δh| / |h| for a change of $m_1$ in its last bit")
    save(fig, "noise")


def convergence():
    data = load("convergence.json")
    if data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), sharey=True)
    for ax, name in zip(axes, ("spin_2x", "mass_1")):
        for variant, style in (("lal", LAL), ("extended", EXTENDED)):
            r = data["results"][variant][name]
            steps = [s["step"] for s in r["steps"]]
            errors = [max(s["relative_error"], 1e-12) for s in r["steps"]]
            ax.plot(steps, errors, color=style["color"], marker=style["marker"], markersize=4, linewidth=1.5,
                    label=style["label"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        label = "step in $\\chi_{2x}$" if name == "spin_2x" else "relative step in $m_1$"
        ax.set_xlabel(label)
    axes[0].set_ylabel("|finite difference − jvp| / |jvp|")
    axes[0].legend(loc="upper center")
    fig.tight_layout()
    save(fig, "convergence")


def validation():
    data = load("validation.json")
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    colours = [ORANGE, GREEN, PURPLE, BLUE]
    for colour, (width, values) in zip(colours, data["mismatches"].items()):
        v = np.sort(np.maximum([m for m in values if m is not None], 1e-17))
        ax.plot(v, 1.0 - np.arange(len(v)) / len(v), color=colour, linewidth=1.5, label=f"blend half-width {width}")
    ax.axvline(1e-3, color=base.MUTED, linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mismatch against LAL")
    ax.set_ylabel("fraction of binaries above")
    ax.legend(loc="lower left")
    save(fig, "validation")


#: The parameters of the overlap scans, as labelled in the figures.
SCAN_LABELS = {"mass_1": "$m_1$", "mass_2": "$m_2$", "spin_1x": "$\\chi_{1x}$", "spin_1y": "$\\chi_{1y}$",
               "spin_1z": "$\\chi_{1z}$", "spin_2x": "$\\chi_{2x}$", "spin_2y": "$\\chi_{2y}$",
               "spin_2z": "$\\chi_{2z}$", "inclination": "$\\iota$", "phi_ref": "$\\phi_\\mathrm{ref}$",
               "f_ref": "$f_\\mathrm{ref}$"}
#: The causes of the located jumps, in the order of the figure.
CAUSES = ["higher-mode choice", "alpha branch at the pole", "PNR grid", "SpinTaylor knots", "integration steps",
          "none (numerical noise)"]


def smooth_residual(scan, degree=20):
    """``1 - O`` along a scan, and its residual from a Chebyshev fit of
    ``degree`` relative to the largest ``1 - O``."""
    eps = np.linspace(scan["lo"], scan["hi"], len(scan["overlap"]))
    y = 1 - np.asarray(scan["overlap"])
    fit = np.polynomial.chebyshev.Chebyshev.fit(eps, y, degree)
    return eps, y, (y - fit(eps)) / np.max(y)


def smoothness():
    data = load("smoothness.json")
    if data is None:
        return
    before = load("smoothness_before.json")
    scans = dict(data["scans"])
    series = [("lal", LAL), ("extended", EXTENDED)]
    if before is not None:
        scans["before"] = before["scans"]["extended"]
        series.insert(1, ("before", BEFORE))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, name in zip(axes[:2], ("mass_1", "spin_2y")):
        for variant, style in series:
            eps, y, r = smooth_residual(scans[variant][name])
            ax.plot(eps, np.abs(r), color=style["color"], linewidth=1.0, label=style["label"])
        ax.set_yscale("log")
        ax.set_ylim(1e-13, 1e-1)
        ax.set_xlabel(f"offset in {SCAN_LABELS[name]}")
    axes[0].set_ylabel("|1 − O − smooth fit| / max(1 − O)")
    axes[0].legend(loc="upper left")
    ax = axes[2]
    names = [n for n in SCAN_LABELS if n in scans["lal"]]
    offsets = np.linspace(-0.2, 0.2, len(series))
    for offset, (variant, style) in zip(offsets, series):
        values = [np.max(np.abs(smooth_residual(scans[variant][n])[2])) for n in names]
        ax.scatter(np.arange(len(names)) + offset, values, s=16, color=style["color"], marker=style["marker"],
                   linewidths=0)
    ax.set_xticks(range(len(names)), [SCAN_LABELS[n] for n in names])
    ax.set_yscale("log")
    ax.set_ylim(1e-13, 1e-1)
    ax.set_ylabel("largest residual along the scan")
    save(fig, "smoothness_scans")

    def events_of(source):
        return [e for r in source["survey"] if r["variant"] == "extended" for s in r["scans"].values()
                for e in s.get("events", [])]

    groups = [(events_of(data), EXTENDED, 0.15)]
    if before is not None:
        groups.insert(0, (events_of(before), BEFORE, -0.15))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    causes = [c for c in CAUSES if any(e["cause"] == c for events, _, _ in groups for e in events)]
    for events, style, offset in groups:
        for i, cause in enumerate(causes):
            values = [max(e["jump"], 1e-16) for e in events if e["cause"] == cause]
            jitter = np.random.default_rng(i).uniform(-0.08, 0.08, len(values))
            ax.scatter(i + offset + jitter, values, s=14, color=style["color"], marker=style["marker"], alpha=0.8,
                       linewidths=0, label=style["label"] if i == 0 else None)
    labels = {"higher-mode choice": "higher-mode\nchoice", "alpha branch at the pole": "α branch\nat the pole",
              "PNR grid": "PNR grid", "SpinTaylor knots": "SpinTaylor\nknots", "integration steps": "integration\nsteps",
              "none (numerical noise)": "nothing\n(noise)"}
    ax.set_xticks(range(len(causes)), [labels[c] for c in causes])
    ax.set_yscale("log")
    ax.set_ylabel("|Δh| / |h| across the point")
    ax.legend(loc="upper right")
    save(fig, "smoothness_survey")


def main():
    base.style()
    for figure in (failure_map, failure_fraction, jumps, switches, two_spin_and_beta, q12, steps, noise, validation,
                   convergence, smoothness):
        figure()


if __name__ == "__main__":
    main()
