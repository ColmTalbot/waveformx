#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The figures of the report, from the measurements in ``report/data``.

They are written as SVG with the text and axis colours replaced by CSS
variables, so that they follow the light and dark themes of the page (as in
waveformrs's report).

Usage::

    python report/make_figures.py
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"

# sentinel colours replaced by CSS variables after rendering
FOREGROUND = "#010203"
MUTED = "#040506"
GRID = "#070809"
REPLACEMENTS = {FOREGROUND: "var(--fg)", MUTED: "var(--muted)", GRID: "var(--rule)"}
# mid-tone colours that read on light and dark backgrounds
BLUE, ORANGE, GREEN, PURPLE, GOLD, PINK = "#3b7dd8", "#d0612f", "#2f9e6e", "#9b59c9", "#c79a1c", "#d1477a"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "text.color": FOREGROUND, "axes.labelcolor": FOREGROUND,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": FOREGROUND, "ytick.labelcolor": FOREGROUND, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "legend.frameon": False,
        "svg.hashsalt": "waveformx",
    })


def load(name):
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else None


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    svg = path.read_text()
    for colour, variable in REPLACEMENTS.items():
        svg = re.sub(colour, variable, svg, flags=re.IGNORECASE)
    path.write_text(svg)
    print(f"wrote {path}")


def accuracy():
    xpnr = load("validation_xpnr.json")
    nrsur = load("validation_nrsur7dq4.json")
    if xpnr is None or nrsur is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    ax = axes[0]
    for key, label, colour, marker in (
        ("jax_vs_lal", "without the fix, against LAL 6.2.0", BLUE, "o"),
        ("jax_fix_vs_lal_fix", "with the fix, against 6.2.0 + !2593", ORANGE, "s"),
    ):
        points = [(r["mass_1"] + r["mass_2"], r[key]["relative_error"]) for r in xpnr["records"] if r.get(key)]
        if points:
            mass, error = np.array(points).T
            ax.scatter(mass, error, s=9, color=colour, marker=marker, alpha=0.75, linewidths=0, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total mass (M$_\\odot$)")
    ax.set_ylabel("max |h − h$_{LAL}$| / max |h$_{LAL}$|")
    ax.set_title("IMRPhenomXPNR", fontsize=9)
    ax.legend(loc="lower left", fontsize=7.5, markerscale=1.5)
    ax = axes[1]
    points = [(r["mass_1"] + r["mass_2"], r["jax_vs_lal"]["relative_error"], r["f_min"] == 0.0)
              for r in nrsur["records"] if r.get("jax_vs_lal") and "relative_error" in r["jax_vs_lal"]]
    if points:
        mass, error, whole = (np.array(column) for column in zip(*points))
        ax.scatter(mass[~whole], error[~whole], s=9, color=GREEN, alpha=0.75, linewidths=0, label="from f_min > 0")
        ax.scatter(mass[whole], error[whole], s=12, color=PURPLE, marker="^", alpha=0.85, linewidths=0,
                   label="whole surrogate (f_min = 0)")
    ax.set_xscale("log")
    ax.set_xlabel("Total mass (M$_\\odot$)")
    ax.set_title("NRSur7dq4", fontsize=9)
    ax.legend(loc="upper left", fontsize=7.5, markerscale=1.5)
    for ax, ticks in zip(axes, ([5, 10, 20, 50, 100, 200], [20, 50, 100, 200, 300])):
        ax.set_ylim(1e-16, 1e-3)
        ax.set_xticks(ticks)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.tight_layout()
    save(fig, "accuracy")


def batch():
    timings = load("timings.json")
    if timings is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, model in zip(axes, ("IMRPhenomXPNR", "NRSur7dq4")):
        rows = timings["batches"].get(model) or {}
        if not rows:
            continue
        sizes = np.array([int(n) for n in rows])
        rates = np.array([rows[str(n)]["per_second"] for n in sizes])
        ax.plot(sizes, rates, "o-", color=BLUE, ms=4, lw=1.2, label="waveformx, jax.vmap")
        public = timings["public_batches"].get(model, {})
        single = timings["single"].get(model, {})
        lines = [
            (public.get("waveformrs_0"), ORANGE, "--", "waveformrs, all threads"),
            (public.get("waveformrs_1"), ORANGE, ":", "waveformrs, one thread"),
            (public.get("lal"), GREEN, "-.", "LALSimulation, one thread"),
        ]
        for value, colour, dash, label in lines:
            if value:
                ax.axhline(value, color=colour, ls=dash, lw=1.1, label=label)
        if single.get("pure"):
            ax.plot([1], [1 / single["pure"]], "D", color=PURPLE, ms=4, label="waveformx, one binary")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Binaries per call")
        ax.set_title(model, fontsize=9)
    axes[0].set_ylabel("Waveforms per second")
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "batch")


def gradients():
    data = load("gradients.json")
    if data is None:
        return
    names = data["parameters"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    colours = [BLUE, ORANGE, GREEN, PURPLE, GOLD, PINK]
    for ax, model, title in zip(axes, ("xpnr", "nrsur7dq4"), ("IMRPhenomXPNR", "NRSur7dq4")):
        if model not in data:
            continue
        for k, (label, result) in enumerate(data[model]["binaries"].items()):
            errors = np.maximum([result["errors"][name] for name in names], 1e-16)
            offset = (k - 1.5) * 0.12
            ax.scatter(np.arange(len(names)) + offset, errors, s=12, color=colours[k % len(colours)], label=label,
                       linewidths=0)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([n.replace("luminosity_distance", "distance") for n in names], rotation=60,
                           ha="right", fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper left", fontsize=7, ncol=1)
    axes[0].set_ylabel("max |J − J$_{FD}$| / max |J|")
    fig.tight_layout()
    save(fig, "gradients")


def mlx():
    data = load("mlx_nrsur7dq4.json")
    if data is None:
        return
    pure = data["timings"]["pure"]
    device = data["mlx_environment"]["device"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, key, title in zip(axes, ("value", "grad"), ("Waveform (h$_+$ on 8192 samples)", "Gradient of its power")):
        for name, label, colour, marker in (("mlx", f"MLX, float32, {device} GPU", BLUE, "o"),
                                            ("jax_float32", "JAX, float32, CPU", ORANGE, "s"),
                                            ("jax_float64", "JAX, float64, CPU", GREEN, "^")):
            rows = pure[name][key]
            sizes = np.array([int(n) for n in rows])
            ax.plot(sizes, sizes / np.array([rows[str(n)] for n in sizes]), marker=marker, color=colour, ms=4, lw=1.2,
                    label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Binaries per call")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Per second")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "mlx")


def main():
    style()
    accuracy()
    batch()
    gradients()
    mlx()


if __name__ == "__main__":
    main()
