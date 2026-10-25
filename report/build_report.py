#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Build the report from the measurements in ``report/data``.

Writes

* ``report/waveformx-report.html``: a standalone HTML document;
* ``report/artifact/index.html``: the same page as a fragment (no html,
  head or body);
* ``report/waveformx-report.pdf``: printed with headless Chrome (optional).

Run the measurements first (``report/measure_all.sh``).

Usage::

    python report/build_report.py [--chrome PATH] [--no-pdf]
"""

import argparse
import datetime
import html
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))

import make_figures  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TEST_FILES = {
    "test_quick.py": "The quick tier (<code>pytest</code>): IMRPhenomXPNR with its derivative, NRSur7dq4",
    "test_api.py": "The interface: options, arguments, errors, spherical harmonics",
    "test_xpnr_xas.py": "IMRPhenomXPNR's (2,2) mode without precession against <code>SimIMRPhenomXASFrequencySequence</code>",
    "test_xas.py": "IMRPhenomXAS against LAL through the public functions, float32 (binary neutron stars too), batches, LAL's errors, the QNM fits",
    "test_transforms_xas.py": "IMRPhenomXAS under <code>jit</code>, <code>vmap</code>, <code>grad</code> and <code>jvp</code>, in float64 and float32",
    "test_xhm.py": "IMRPhenomXHM against LAL (polarizations, one-mode functions, <code>ChooseFDModes</code>), the exact derivatives of the (3,2), float32, batches, errors",
    "test_transforms_xhm.py": "IMRPhenomXHM under <code>jit</code>, <code>vmap</code>, <code>grad</code> and <code>jvp</code>, in float64 and float32, at equal masses",
    "test_xhm_mlx.py": "IMRPhenomXHM in MLX against JAX: polarizations and modes on non-uniform and uniform grids, Jacobian, gradients, batches",
    "test_xas_mlx.py": "IMRPhenomXAS in MLX against JAX and LAL: waveform on non-uniform and large uniform grids, Jacobian, gradients, batches",
    "test_taylorf2.py": "TaylorF2 against LAL and waveformrs through the public functions, float32 (binary neutron stars too), batches, errors",
    "test_transforms_taylorf2.py": "TaylorF2 under <code>jit</code>, <code>vmap</code>, <code>grad</code> and <code>jvp</code>, in float64 and float32",
    "test_taylorf2_mlx.py": "TaylorF2 in MLX against JAX and LAL: waveform on non-uniform and large uniform grids, Jacobian, gradients, batches",
    "test_phenomd.py": "IMRPhenomD against LAL (also with LAL's QNM spline in place of the fits) and waveformrs, the QNM fits, float32, batches, errors",
    "test_transforms_phenomd.py": "IMRPhenomD under <code>jit</code>, <code>vmap</code>, <code>grad</code> and <code>jvp</code>, in float64 and float32",
    "test_phenomd_mlx.py": "IMRPhenomD in MLX against JAX and LAL: waveform on non-uniform and large uniform grids, Jacobian, gradients, batches",
    "test_xpnr_hm.py": "Higher modes (release 122019) against <code>SimIMRPhenomXHMFrequencySequenceOneMode</code>",
    "test_xpnr_msa.py": "MSA and NNLO angles and the precession set-up, GSL's elliptic functions",
    "test_xpnr_pnr_helpers.py": "PNR fits and angle helpers against LAL's internal functions (ctypes)",
    "test_xpnr.py": "IMRPhenomXPNR polarizations against LAL with the parity fix: waveformrs's reference suite, mirror symmetry",
    "test_xpnr_modes.py": "IMRPhenomXPNR J-frame modes, <code>ChooseFDModes</code>, checks",
    "test_xpnr_model.py": "IMRPhenomXPNR at special spins, in float32, through the public functions, under <code>vmap</code> and <code>grad</code>",
    "test_xpnr_smooth.py": "IMRPhenomXPNR across LAL's switches, the interpolants of the PNR angles, the phase grid, gradients",
    "test_xpnr_reference.py": "IMRPhenomXPNR with the mass-scaled reference frequency, the converters, invariance, against LAL",
    "test_xpnr_float32.py": "IMRPhenomXPNR in float32 (no float64 operation, accuracy, gradients) and the scale of its output",
    "test_xpnr_random.py": "IMRPhenomXPNR, 64 random binaries against LAL with the parity fix (slow)",
    "test_transforms_xpnr.py": "IMRPhenomXPNR under <code>jit</code>, <code>vmap</code>, <code>grad</code> and <code>jvp</code>",
    "test_nrsur7dq4.py": "NRSur7dq4 polarizations, modes, dynamics, errors, data file, batches",
    "test_nrsur7dq4_random.py": "NRSur7dq4, 64 random binaries against LAL (slow)",
    "test_transforms_nrsur7dq4.py": "NRSur7dq4 under <code>jit</code>, <code>vmap</code> and <code>grad</code>, removable singularities",
    "test_nrsur7dq4_float32.py": "NRSur7dq4 in float32 (no float64 operation, accuracy, <code>jit</code>, <code>vmap</code>, <code>grad</code>) and the scale of its output",
    "test_nrsur7dq4_mlx.py": "NRSur7dq4 in MLX against JAX: waveform, modes, batches, gradients, the Metal kernels, LAL's grids and the conditioned waveforms",
    "test_xpnr_mlx.py": "IMRPhenomXPNR in MLX against JAX: the stages, the generated kernels, the SpinTaylor kernel, waveform, Jacobian and gradients",
    "test_conversions.py": "<code>simulate_td</code> and <code>simulate_fd</code>: lengths, epochs, filters, tapers against LAL",
    "test_transforms_conversions.py": "The conversions as pure functions under <code>jit</code>, <code>vmap</code> and <code>grad</code>",
}


def sci(value, digits=1):
    """Scientific notation with a real superscript, e.g. 2.2×10⁻¹¹."""
    if value is None:
        return "–"
    if value <= 0:
        return "0"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}×10<sup>{int(exponent)}</sup>"


def duration(seconds):
    if seconds is None:
        return "–"
    if seconds >= 1.0:
        return f"{seconds:.1f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms" if seconds < 0.1 else f"{seconds * 1e3:.0f} ms"
    if seconds >= 1e-5:
        return f"{seconds * 1e6:.0f} µs"
    return f"{seconds * 1e6:.2f} µs"


def factor(value):
    return "–" if value is None else (f"{value:.1f}×" if value < 100 else f"{value:.0f}×")


def rate(value):
    return "–" if value is None else (f"{value:,.0f}" if value >= 10 else f"{value:.1f}")


def load(name):
    return json.loads((DATA / name).read_text())


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def commit_rows():
    kinds = {"FEAT": "info", "FIX": "bad", "BUG": "bad", "PERF": "good", "TEST": "", "TST": "", "DOC": "",
             "MAINT": "", "STY": ""}
    rows = []
    for line in git("log", "--reverse", "--format=%h%x09%s").strip().splitlines():
        sha, subject = line.split("\t", 1)
        if subject.startswith("DOC: the report"):
            continue
        kind, _, rest = subject.partition(": ")
        chip = f'<span class="chip {kinds.get(kind, "")}">{html.escape(kind)}</span>' if kind in kinds else ""
        text = html.escape(rest if chip else subject)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        rows.append(f"<tr><td class=\"mono\">{sha}</td><td>{chip}</td><td>{text}</td></tr>")
    return rows


def _junit(path):
    """Per test file: (tests, passed, failed, skipped, seconds), and the
    time of the run, from a JUnit XML file."""
    tree = ElementTree.parse(path)
    files = {}
    for case in tree.iter("testcase"):
        name = case.get("classname", "").split(".")[-1] + ".py"
        entry = files.setdefault(name, dict(n=0, passed=0, failed=0, skipped=0, seconds=0.0))
        entry["n"] += 1
        entry["seconds"] += float(case.get("time", 0.0))
        if case.find("failure") is not None or case.find("error") is not None:
            entry["failed"] += 1
        elif case.find("skipped") is not None:
            entry["skipped"] += 1
        else:
            entry["passed"] += 1
    suite = next(tree.iter("testsuite"))
    return files, float(suite.get("time", 0.0))


def test_results():
    """Per test file: (tests, passed, failed, skipped, seconds) from the
    JUnit XML of the last full run, with the MLX tests from their own run
    (``pytest_mlx.xml``, in a Python with MLX) in place of their skips."""
    files, seconds = _junit(DATA / "pytest.xml")
    if (DATA / "pytest_mlx.xml").exists():
        mlx_files, mlx_seconds = _junit(DATA / "pytest_mlx.xml")
        replaced = sum(files[name]["seconds"] for name in mlx_files if name in files)
        files.update(mlx_files)
        seconds += mlx_seconds - replaced
    return files, seconds


def source_lines():
    sources = list((ROOT / "src" / "waveformx").rglob("*.py"))
    generated = [p for p in sources if "_generated" in p.parts]
    count = lambda paths: sum(len(p.read_text().splitlines()) for p in paths)  # noqa: E731
    return count(sources), count(generated), len(sources)


def svg(name, label):
    path = HERE / "figures" / f"{name}.svg"
    text = path.read_text()
    text = re.sub(r"<metadata>.*?</metadata>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\?xml[^>]*\?>\s*", "", text)
    text = re.sub(r"<!DOCTYPE[^>]*>\s*", "", text)
    return text.replace("<svg ", f'<svg role="img" aria-label="{html.escape(label)}" ', 1)


def failure_text(data, how):
    """Whether waveformx fails where LAL does, as a sentence."""
    n, failures, agree = data["n_draws"], data["n_lal_failures"], data["n_failures_agree"]
    if failures == 0:
        return (f"LAL evaluates all {n} draws, and so does waveformx." if agree == n else
                f"LAL evaluates all {n} draws; waveformx fails on {n - agree} of them.")
    return (f"LAL fails for {failures} of the {n} draws, and waveformx {how} for the same ones "
            f"({agree}/{n} agree).")


def cost_values(values):
    """The cost of compiling and running IMRPhenomXPNR (``report/xpnr_cost.py``)."""
    names = [f"cost_compile_{g}_{k}" for g in ("uniform", "phase") for k in ("eval", "vmap", "grad")]
    names += [f"cost_runtime_{c}_{g}" for c in ("benchmark", "light") for g in ("uniform", "phase")]
    if not (DATA / "xpnr_cost.json").exists():
        values.update({name: "–" for name in names})
        return
    cost = load("xpnr_cost.json")
    for key, seconds in cost["compile_seconds"].items():
        values[f"cost_compile_{key}"] = duration(seconds)
    for key, seconds in cost["runtime"].items():
        values[f"cost_runtime_{key}"] = duration(seconds)


def reference_values(values):
    """The mass-scaled reference frequency (``report/reference_xpnr.py``)."""
    names = ["ref_n", "ref_finite_lal_21", "ref_finite_lal_all", "ref_finite_scaled_min", "ref_lal_n",
             "ref_lean_vs_lal_median", "ref_lean_vs_lal_max", "ref_lal_vs_lal_median", "ref_lal_vs_lal_max",
             "ref_lal_vs_lal_above", "ref_compile_lal", "ref_compile_scaled", "ref_lal_20_vs_lean_median",
             "ref_lal_20_vs_lean_max", "ref_lal_20_vs_lean_above", "ref_lal_near_vs_lean_median",
             "ref_lal_near_vs_lean_max", "ref_lal_near_vs_lean_above"]
    conventions = ("lal", "scaled", "scaled_0.006", "scaled_0.002")
    names += [f"ref_compile_{c.replace('.', '')}" for c in conventions[2:]]
    names += [f"ref_runtime_{c.replace('.', '')}_{m}" for c in conventions for m in ("10", "60", "150", "400")]
    names += [f"ref_near_{b}_{x}" for b in ("2", "5", "20") for x in ("n", "median", "max")]
    if not (DATA / "reference_xpnr.json").exists():
        values.update({name: "–" for name in names})
        return
    data = load("reference_xpnr.json")
    finite = data["finite"]
    values["ref_n"] = str(data["n"])
    values["ref_finite_lal_21"] = str(finite["lal_21"])
    values["ref_finite_lal_all"] = str(finite["lal_all"])
    values["ref_finite_scaled_min"] = str(min(v for k, v in finite.items() if k.startswith("scaled")))
    records = data.get("lal", [])
    values["ref_lal_n"] = str(len(records)) if records else "–"
    for key in ("lean_vs_lal", "lal_vs_lal", "lal_20_vs_lean", "lal_near_vs_lean"):
        mm = sorted(r[key] for r in records if key in r)
        values[f"ref_{key}_median"] = sci(mm[len(mm) // 2]) if mm else "–"
        values[f"ref_{key}_max"] = sci(mm[-1]) if mm else "–"
    for key in ("lal_vs_lal", "lal_20_vs_lean", "lal_near_vs_lean"):
        above = [r[key] > 1e-3 for r in records if key in r]
        values[f"ref_{key}_above"] = str(sum(above)) if above else "–"
    # LAL near the canonical reference, by the relative offset of its reference
    for label, lo, hi in (("2", 0.0, 0.02), ("5", 0.02, 0.05), ("20", 0.05, 0.2)):
        mm = sorted(r["lal_near_vs_lean"] for r in records if "offset" in r and lo <= abs(r["offset"]) < hi + 1e-12)
        values[f"ref_near_{label}_n"] = str(len(mm)) if mm else "–"
        values[f"ref_near_{label}_median"] = sci(mm[len(mm) // 2]) if mm else "–"
        values[f"ref_near_{label}_max"] = sci(mm[-1]) if mm else "–"
    for convention in conventions:
        name = convention.replace(".", "")
        if convention not in data["timings"]:
            values[f"ref_compile_{name}"] = "–"
            values.update({f"ref_runtime_{name}_{m}": "–" for m in ("10", "60", "150", "400")})
            continue
        t = data["timings"][convention]
        values[f"ref_compile_{name}"] = duration(t["compile"])
        for m in ("10", "60", "150", "400"):
            values[f"ref_runtime_{name}_{m}"] = duration(t["runtime"][f"{float(m)}"])


def float32_values(values):
    """The float32 study of NRSur7dq4 (``float32_nrsur7dq4.json``)."""
    data = load("float32_nrsur7dq4.json")
    s = data["summary"]
    values["f32_pairs"] = str(s["all"]["n"])
    values["f32_pairs_each"] = str(s["random"]["n"])
    values["f32_nearby_overlap"] = f"{s['nearby']['overlap_median']:.2f}"
    gradients = s["gradients"]["grad_log_likelihood"]
    rows = [
        ("Inner product of the two waveforms, change over the product of their norms", s["all"]["epsilon_single"]),
        ("Normalised overlap, independent binaries", s["random"]["d_overlap_single"]),
        ("Normalised overlap, nearby binaries", s["nearby"]["d_overlap_single"]),
        ("Normalised overlap, with the inner products in float32 too", s["all"]["d_overlap_single_inner32"]),
        ("Normalised overlap, with the dynamics in float64", s["all"]["d_overlap_dynamics64"]),
        ("One waveform, noise-weighted relative error ‖δh‖ / ‖h‖", s["waveforms"]["noise_weighted_single"]),
        ("One waveform, largest relative error max |δh| / max |h|", s["waveforms"]["max_relative"]),
        ("One waveform, relative error of ⟨h, h⟩", s["waveforms"]["norm_error"]),
        (f"Derivatives of a log-likelihood, relative error ({gradients['n']} nearby pairs)", gradients["relative"]),
        ("Derivatives of a log-likelihood, error over the typical size of the component", gradients["over_typical"]),
    ]
    values["float32_rows"] = "\n".join(
        f"<tr><td>{label}</td><td class=\"num\">{sci(e['median'])}</td><td class=\"num\">{sci(e['p90'])}</td>"
        f"<td class=\"num\">{sci(e['max'])}</td></tr>" for label, e in rows)
    eps, overlap = s["all"]["epsilon_single"], s["all"]["d_overlap_single"]
    values["f32_eps_median"], values["f32_eps_max"] = sci(eps["median"]), sci(eps["max"])
    values["f32_do_median"], values["f32_do_p90"] = sci(overlap["median"]), sci(overlap["p90"])
    dynamics = s["all"]["d_overlap_dynamics64"]
    values["f32_dyn_median"], values["f32_dyn_p90"] = sci(dynamics["median"]), sci(dynamics["p90"])
    values["f32_inner_only_max"] = sci(s["all"]["d_overlap_inner32_only"]["max"])
    values["f32_dh_median"] = sci(s["waveforms"]["noise_weighted_single"]["median"])
    values["f32_dh_coorb_median"] = sci(s["waveforms"]["noise_weighted_coorbital64"]["median"])
    lnl = s["log_likelihood_over_snr2"]
    values["f32_lnl_median"], values["f32_lnl_p90"] = sci(lnl["median"]), sci(lnl["p90"])
    values["f32_lnl_20"] = sci(lnl["median"] * 20 ** 2)
    values["f32_lnl_100"] = sci(lnl["median"] * 100 ** 2)
    values["f32_lnl_100_p90"] = sci(lnl["p90"] * 100 ** 2)
    values["f32_grad_median"], values["f32_grad_p90"] = sci(gradients["relative"]["median"]), sci(
        gradients["relative"]["p90"])
    values["f32_grad_typical_max"] = sci(gradients["over_typical"]["max"])
    t = data["timings"]
    values["f32_single"], values["f64_single"] = duration(t["single"]["single"]), duration(t["double"]["single"])
    values["f32_batch"], values["f64_batch"] = duration(t["single"]["batch64"]), duration(t["double"]["batch64"])
    values["f32_speed_single"] = factor(t["double"]["single"] / t["single"]["single"])
    values["f32_speed_batch"] = factor(t["double"]["batch64"] / t["single"]["batch64"])
    values["f32_speed_grad"] = factor(t["double"]["grad"] / t["single"]["grad"])
    values["f32_dyn_share"] = f"{t['double']['batch64_dynamics'] / t['double']['batch64']:.0%}"


def mlx_xpnr_values(values):
    """The lean IMRPhenomXPNR in MLX (``mlx_xpnr.json``)."""
    data = load("mlx_xpnr.json")
    s = data["summary"]
    values["mx_n"] = str(data["n"])
    values["mx_mm_median"], values["mx_mm_p90"], values["mx_mm_max"] = (
        sci(s["mlx"][k]) for k in ("median", "p90", "max"))
    values["mx32_mm_median"], values["mx32_mm_p90"], values["mx32_mm_max"] = (
        sci(s["jax32"][k]) for k in ("median", "p90", "max"))
    records = data["records"]
    values["mx_above"] = str(sum(r.get("mismatch_mlx", 0.0) > 1e-6 for r in records))
    values["mx32_above"] = str(sum(r.get("mismatch_jax32", 0.0) > 1e-6 for r in records))
    worst = max(records, key=lambda r: r.get("mismatch_mlx", 0.0))
    values["mx_worst_total"] = f"{worst['total']:.0f}"
    values["mx_worst_jax32"] = sci(worst.get("mismatch_jax32"))
    finite = s["finite"]
    values["mx_finite_both"] = str(finite["both"])
    values["mx_finite_double_only"] = str(finite["double_only"])
    values["mx_finite_mlx_only"] = str(finite["mlx_only"])
    values["mx_jax32_not_finite"] = str(finite["jax32_not_finite"])
    g = data["gradient_summary"]
    values["mx_grad_n"] = str(g["n"])
    values["mx_grad_within"] = str(g["within_mlx"])
    values["mx32_grad_within"] = str(g["within_jax32"])
    values["mx_grad_ratio_median"], values["mx_grad_ratio_max"] = sci(g["ratio_mlx"]["median"]), sci(g["ratio_mlx"]["max"])
    values["mx32_grad_ratio_median"], values["mx32_grad_ratio_max"] = (sci(g["ratio_jax32"]["median"]),
                                                                       sci(g["ratio_jax32"]["max"]))
    t = data["timings"]
    mlx_t = t["mlx"]

    def cell(name, key, size):
        value = t[name][key].get(size)
        return f'<td class="num">{duration(value / int(size)) if value is not None else "–"}</td>'

    rows = []
    for size in mlx_t["value"]:
        rows.append(f'<tr><td class="num">{size}</td><td class="num">{duration(mlx_t["value"][size])}</td>'
                    + "".join(cell(name, "value", size) for name in ("mlx", "jax_float32", "jax_float64"))
                    + "".join(cell(name, "grad", size) for name in ("mlx", "jax_float32", "jax_float64"))
                    + "</tr>")
    values["mx_timing_rows"] = "\n".join(rows)

    def best(name, key):
        return max((int(n) / v, n) for n, v in t[name][key].items())

    values["mx_single"] = duration(mlx_t["value"]["1"])
    values["mx_jax_single"] = duration(t["jax_float64"]["value"]["1"])
    values["mx_grad_single"] = duration(mlx_t["grad"]["1"])
    values["mx_jax_grad_single"] = duration(t["jax_float64"]["grad"]["1"])
    (rate_mlx, size_mlx), (rate_jax, size_jax) = best("mlx", "value"), max(best("jax_float64", "value"),
                                                                            best("jax_float32", "value"))
    values["mx_best_rate"], values["mx_best_size"] = rate(rate_mlx), size_mlx
    values["mx_jax_best_rate"] = rate(rate_jax)
    values["mx_best_speedup"] = factor(rate_mlx / rate_jax)
    (rate_mlx, size_mlx), (rate_jax, _) = best("mlx", "grad"), max(best("jax_float64", "grad"),
                                                                   best("jax_float32", "grad"))
    values["mx_grad_best_rate"], values["mx_grad_best_size"] = rate(rate_mlx), size_mlx
    values["mx_jax_grad_best_rate"] = rate(rate_jax)
    values["mx_grad_speedup"] = factor(rate_mlx / rate_jax)
    first = mlx_t["first_calls"]
    values["mx_first_imports"] = duration(first["imports"])
    values["mx_first_value"] = duration(first["value"])
    values["mx_first_grad"] = duration(first["grad"])
    values["mx_frequencies"] = str(t["frequencies"])


def xas_values(values):
    """IMRPhenomXAS (``xas.json``)."""
    data = load("xas.json")
    values["xas_table_size"] = str(data["table_size"])
    lal = data["lal"]
    values["xas_lal_n"] = str(lal["relative_error"]["n"])
    values["xas_lal_max_rel"] = sci(lal["relative_error"]["max"])
    values["xas_lal_median_rel"] = sci(lal["relative_error"]["median"])
    values["xas_lal_max_mm"] = sci(lal["mismatch"]["max"])
    s = data["float32"]["summary"]
    for group, prefix in (("population", "pop"), ("bns", "bns"), ("bns_uniform", "bnsu"), ("light", "light")):
        for name in ("jax32", "mlx"):
            stats = s[group].get(name) or {}
            for key in ("median", "p90", "max"):
                values[f"xas_{prefix}_{name}_{key}"] = sci(stats.get(key))
        values[f"xas_{prefix}_n"] = str((s[group].get("jax32") or {}).get("n", 0))
    g = data["gradients"]
    values["xas_grad_n"] = str(g["n"])
    values["xas_grad_within_jax32"] = str(g["within_jax32"])
    values["xas_grad_within_mlx"] = str(g["within_mlx"])
    values["xas_grad_ratio_jax32"] = sci(g["ratio_jax32"]["max"])
    values["xas_grad_ratio_mlx"] = sci((g["ratio_mlx"] or {}).get("max"))
    t = data["timings"]
    f64, f32, old, lal_t, mlx_t = t["float64"], t["float32"], t["lal_faithful_float64"], t["lal"], t.get("mlx")

    def per_binary(entry, size):
        return entry["batch"][size] / int(size)

    def mlx_cell(key, size=None, large=False):
        if mlx_t is None:
            return "–"
        table = mlx_t[("large_" if large else "") + key]
        return duration(table["1"] if size is None else table[size] / int(size))

    rows = [
        ("One binary, 1000 frequencies", [f64["single"], f32["single"], old["single"], lal_t["single"]],
         mlx_cell("value")),
        (f"One binary, 2<sup>20</sup> uniform frequencies", [f64["large"], f32["large"], old["large"], lal_t["large"]],
         mlx_cell("value", large=True)),
        ("256 binaries × 1000 frequencies, per binary",
         [per_binary(f64, "256"), per_binary(f32, "256"), per_binary(old, "256"), lal_t["single"]],
         mlx_cell("value", "256")),
        ("4096 binaries × 1000 frequencies, per binary",
         [per_binary(f64, "4096"), per_binary(f32, "4096"), per_binary(old, "4096"), lal_t["single"]],
         mlx_cell("value", "4096")),
        ("Gradient, one binary, 1000 frequencies", [f64["grad"], f32["grad"], old["grad"], None], mlx_cell("grad")),
        ("Gradient, one binary, 2<sup>20</sup> frequencies", [None, None, None, None],
         mlx_cell("grad", large=True)),
    ]
    values["xas_timing_rows"] = "\n".join(
        f"<tr><td>{label}</td>" + "".join(f'<td class="num">{duration(v)}</td>' for v in cells)
        + f'<td class="num">{last}</td></tr>' for label, cells, last in rows)
    values["xas_single"] = duration(f64["single"])
    values["xas_lal_single"] = duration(lal_t["single"])
    values["xas_old_single"] = duration(old["single"])
    values["xas_large"] = duration(f64["large"])
    values["xas_large32"] = duration(f32["large"])
    values["xas_old_large"] = duration(old["large"])
    values["xas_lal_large"] = duration(lal_t["large"])
    values["xas_large_speedup"] = factor(lal_t["large"] / f64["large"])
    values["xas_batch_speedup"] = factor(per_binary(old, "4096") / per_binary(f64, "4096"))
    for name, entry in (("", f64), ("old_", old)):
        values[f"xas_{name}compile"] = duration(entry["compile"])
        values[f"xas_{name}compile_vmap"] = duration(entry["compile_vmap"])
        values[f"xas_{name}compile_grad"] = duration(entry["compile_grad"])
    if mlx_t is not None:
        best = max((int(n) / v, n) for n, v in mlx_t["value"].items())
        jax_best = max(int(n) / v for n, v in f32["batch"].items())
        values["xas_mlx_best_rate"], values["xas_mlx_best_size"] = rate(best[0]), best[1]
        values["xas_mlx_best_speedup"] = factor(best[0] / max(jax_best, max(int(n) / v for n, v in f64["batch"].items())))
        values["xas_mlx_single"] = mlx_cell("value")
        values["xas_mlx_large"] = mlx_cell("value", large=True)
        values["xas_mlx_large16"] = duration(mlx_t["large_value"]["16"])
        values["xas_mlx_grad"] = mlx_cell("grad")
        first = mlx_t["first_calls"]
        values["xas_mlx_first_imports"] = duration(first["imports"])
        values["xas_mlx_first_value"] = duration(first["value"])
        values["xas_mlx_first_grad"] = duration(first["grad"])


def crossmodel_values(values):
    """IMRPhenomXPNR against NRSur7dq4 on the benchmark binaries
    (``report/crossmodel.py``: ``crossmodel_points.json``, ``crossmodel.json``
    and the checks of ``--thorough``, ``crossmodel_checks.json``)."""
    points, data, checks = load("crossmodel_points.json"), load("crossmodel.json"), load("crossmodel_checks.json")
    settings = points["settings"]
    values["cm_draws"] = str(settings["n_draws"])
    values["cm_valid"] = str(points["n_valid"])
    values["cm_agree_n"] = str(points["n_agree"])
    values["cm_disagree_n"] = str(points["n_disagree"])
    values["cm_agree_below"] = sci(settings["agree_below"], 0)
    values["cm_disagree_above"] = sci(settings["disagree_above"], 0)
    values["cm_select_seconds"] = duration(points["seconds"])
    records = data["records"]
    values["cm_n"] = str(len(records))
    values["cm_seconds"] = duration(data["seconds"])
    values["cm_workers"] = str(data["settings"]["workers"])
    values["cm_nrsur_max"] = sci(max(abs(r["nrsur_ours_vs_lal"]) for r in records))
    pairs = (("lal_xpnr_vs_lal_nrsur", "LAL IMRPhenomXPNR", "NRSur7dq4"),
             ("our_xpnr_vs_lal_nrsur", "our IMRPhenomXPNR", "NRSur7dq4"))
    rows = []
    for key, template, signal in pairs:
        cells = []
        for label in ("agree", "disagree"):
            mm = [r[key]["mismatch"] for r in records if r["label"] == label]
            cells += [sci(float(np.median(mm))), sci(max(mm))]
        rows.append(f"<tr><td>{signal}</td><td>{template}</td>" + "".join(f'<td class="num">{c}</td>' for c in cells)
                    + "</tr>")
    cells = []
    for label in ("agree", "disagree"):
        mm = [r["our_xpnr_vs_lal_xpnr"] for r in records if r["label"] == label]
        cells += [sci(float(np.median(mm))), sci(max(mm))]
    rows.append("<tr><td>LAL IMRPhenomXPNR</td><td>our IMRPhenomXPNR, same binary*</td>"
                + "".join(f'<td class="num">{c}</td>' for c in cells) + "</tr>")
    values["cm_rows"] = "\n".join(rows)
    for label in ("agree", "disagree"):
        group = [r for r in records if r["label"] == label]
        ratio = [r["our_xpnr_vs_lal_nrsur"]["mismatch"] / r["lal_xpnr_vs_lal_nrsur"]["mismatch"] for r in group]
        values[f"cm_{label}_closer"] = str(sum(q < 1 for q in ratio))
        values[f"cm_{label}_count"] = str(len(group))
        values[f"cm_{label}_ratio_median"] = f"{float(np.median(ratio)):.2f}"
        values[f"cm_{label}_ratio_min"] = f"{min(ratio):.2f}"
        values[f"cm_{label}_ratio_max"] = f"{max(ratio):.1f}"
        values[f"cm_{label}_unopt_median"] = sci(float(np.median(
            [r["lal_xpnr_vs_lal_nrsur"]["unoptimised"] for r in group])))
    values["cm_spintaylor"] = str(sum(r["lal_spintaylor"] for r in records))
    # the checks: the procedure against the thorough one, the gradient
    # optimisation, our NRSur7dq4 as the signal, the start of the surrogate
    by_key = {json.dumps(r["features"], sort_keys=True): r for r in checks["records"]}
    ratios = [r[k]["mismatch"] / by_key[json.dumps(r["features"], sort_keys=True)][k]["mismatch"]
              for r in records for k, _, _ in pairs]
    values["cm_fast_vs_thorough"] = sci(max(abs(q - 1) for q in ratios))
    values["cm_evaluations"] = str(int(np.median([r[k]["evaluations"] for r in records for k, _, _ in pairs])))
    values["cm_thorough_evaluations"] = str(int(np.median(
        [r[k]["evaluations"] for r in checks["records"] for k, _, _ in pairs])))
    values["cm_thorough_seconds"] = duration(checks["seconds"])
    c = checks["records"]
    values["cm_conditioning_max"] = sci(max(r["nrsur_conditioning"] for r in c))
    values["cm_ours_signal"] = sci(max(abs(r["our_xpnr_vs_our_nrsur"]["mismatch"] / r["our_xpnr_vs_lal_nrsur"]["mismatch"] - 1)
                                       for r in c))
    gradient = [(r["our_xpnr_vs_lal_nrsur_gradient"]["mismatch"], r["our_xpnr_vs_lal_nrsur"]["mismatch"]) for r in c]
    values["cm_gradient_n"] = str(len(gradient))
    values["cm_gradient_worst"] = sci(max(abs(g / n - 1) for g, n in gradient))
    departure = [r["departure"] for r in c]
    values["cm_nofix_vs_lal"] = sci(max(r["default_without_fix_vs_lal"] for r in departure))
    values["cm_parity_fix_max"] = sci(max(r["parity_fix"] for r in departure))
    values["cm_lean_vs_default"] = sci(max(r["lean_vs_default"] for r in departure))
    for label in ("agree", "disagree"):
        mm = [r["our_xpnr_vs_lal_xpnr"] for r in records if r["label"] == label]
        values[f"cm_{label}_self_median"] = sci(float(np.median(mm)))
    values["crossmodel_svg"] = svg("crossmodel", "Our IMRPhenomXPNR against LAL's, each against NRSur7dq4")


def xhm_values(values):
    """IMRPhenomXHM (``xhm.json``)."""
    data = load("xhm.json")
    lal = data["lal"]
    values["xhm_lal_n"] = str(lal["relative_error"]["n"])
    values["xhm_lal_max_rel"] = sci(lal["relative_error"]["max"])
    values["xhm_lal_median_rel"] = sci(lal["relative_error"]["median"])
    values["xhm_lal_max_mm"] = sci(lal["mismatch"]["max"])
    values["xhm_lal_modes_max"] = sci(lal["modes_relative_error"]["max"])
    values["xhm_lal_32_median"] = sci(lal["mismatch_32"]["median"])
    values["xhm_lal_32_max"] = sci(lal["mismatch_32"]["max"])
    s = data["float32"]["summary"]
    for group, prefix in (("population", "pop"), ("bns", "bns"), ("bns_uniform", "bnsu")):
        for name in ("jax32", "mlx", "jax32_modes", "mlx_modes"):
            stats = s[group].get(name) or {}
            for key in ("median", "max"):
                values[f"xhm_{prefix}_{name}_{key}"] = sci(stats.get(key))
        values[f"xhm_{prefix}_n"] = str((s[group].get("jax32") or {}).get("n", 0))
    g = data["gradients"]
    values["xhm_grad_n"] = str(g["n"])
    values["xhm_grad_within_jax32"] = str(g["within_jax32"])
    values["xhm_grad_within_mlx"] = str(g["within_mlx"])
    values["xhm_grad_ratio_jax32"] = sci(g["ratio_jax32"]["max"])
    values["xhm_grad_ratio_mlx"] = sci((g["ratio_mlx"] or {}).get("max"))
    t = data["timings"]
    f64, f32, lal_t, mlx_t = t["float64"], t["float32"], t["lal"], t.get("mlx")

    def per_binary(entry, size):
        return entry["batch"][size] / int(size)

    def mlx_cell(key, size=None, large=False):
        if mlx_t is None:
            return "–"
        table = mlx_t[("large_" if large else "") + key]
        return duration(table["1"] if size is None else table[size] / int(size))

    rows = [
        ("One binary, 1000 frequencies", [f64["single"], f32["single"], lal_t["single"]], mlx_cell("value")),
        ("One binary, 2<sup>20</sup> uniform frequencies", [f64["large"], f32["large"], lal_t["large"]],
         mlx_cell("value", large=True)),
        ("256 binaries × 1000 frequencies, per binary", [per_binary(f64, "256"), per_binary(f32, "256"),
                                                         lal_t["single"]], mlx_cell("value", "256")),
        ("4096 binaries × 1000 frequencies, per binary", [per_binary(f64, "4096"), per_binary(f32, "4096"),
                                                          lal_t["single"]], mlx_cell("value", "4096")),
        ("Gradient, one binary, 1000 frequencies", [f64["grad"], f32["grad"], None], mlx_cell("grad")),
        ("Gradient, 4096 binaries × 1000, per binary", [None, None, None], mlx_cell("grad", "4096")),
        ("Gradient, one binary, 2<sup>20</sup> frequencies", [None, None, None], mlx_cell("grad", large=True)),
    ]
    values["xhm_timing_rows"] = "\n".join(
        f"<tr><td>{label}</td>" + "".join(f'<td class="num">{duration(v)}</td>' for v in cells)
        + f'<td class="num">{last}</td></tr>' for label, cells, last in rows)
    values["xhm_single"] = duration(f64["single"])
    values["xhm_lal_single"] = duration(lal_t["single"])
    values["xhm_large"] = duration(f64["large"])
    values["xhm_large32"] = duration(f32["large"])
    values["xhm_lal_large"] = duration(lal_t["large"])
    values["xhm_large_speedup"] = factor(lal_t["large"] / f64["large"])
    values["xhm_compile"] = duration(f64["compile"])
    values["xhm_compile_vmap"] = duration(f64["compile_vmap"])
    values["xhm_compile_grad"] = duration(f64["compile_grad"])
    if mlx_t is not None:
        best = max((int(n) / v, n) for n, v in mlx_t["value"].items())
        jax_best = max(max(int(n) / v for n, v in e["batch"].items()) for e in (f32, f64))
        values["xhm_mlx_best_rate"], values["xhm_mlx_best_size"] = rate(best[0]), best[1]
        values["xhm_mlx_best_speedup"] = factor(best[0] / jax_best)
        values["xhm_mlx_single"] = mlx_cell("value")
        values["xhm_mlx_large"] = mlx_cell("value", large=True)
        values["xhm_mlx_grad"] = mlx_cell("grad")
        values["xhm_mlx_grad_batch"] = duration(mlx_t["grad"]["4096"] / 4096)
        first = mlx_t["first_calls"]
        values["xhm_mlx_first_value"] = duration(first["value"])
        values["xhm_mlx_first_grad"] = duration(first["grad"])


def inspiral_values(values, pending):
    """TaylorF2 and IMRPhenomD (``taylorf2.json``, ``phenomd.json``,
    ``phenomd_qnm.json``): placeholders ``tf2_*`` and ``pd_*``."""
    busy = []
    for prefix, name, label in (("tf2", "taylorf2.json", "TaylorF2"), ("pd", "phenomd.json", "IMRPhenomD")):
        data = load(name)
        lal = data["lal"]
        values[f"{prefix}_table_size"] = str(data["table_size"])
        values[f"{prefix}_lal_n"] = str(lal["relative_error"]["n"])
        for key in ("max", "median", "p90"):
            values[f"{prefix}_lal_{key}_rel"] = sci(lal["relative_error"][key])
            values[f"{prefix}_lal_{key}_mm"] = sci(lal["mismatch"][key])
        s = data["float32"]["summary"]
        for group, short in (("population", "pop"), ("bns", "bns"), ("bns_uniform", "bnsu")):
            for kind in ("jax32", "mlx"):
                stats = s[group].get(kind) or {}
                for key in ("median", "max"):
                    values[f"{prefix}_{short}_{kind}_{key}"] = sci(stats.get(key))
            values[f"{prefix}_{short}_n"] = str((s[group].get("jax32") or {}).get("n", 0))
        g = data["gradients"]
        values[f"{prefix}_grad_n"] = str(g["n"])
        values[f"{prefix}_grad_within_jax32"] = str(g["within_jax32"])
        values[f"{prefix}_grad_within_mlx"] = str(g["within_mlx"])
        values[f"{prefix}_grad_ratio_jax32"] = sci(g["ratio_jax32"]["max"])
        values[f"{prefix}_grad_ratio_mlx"] = sci((g["ratio_mlx"] or {}).get("max"))
        t = data["timings"]
        f64, f32, lal_t, mlx_t = t["float64"], t["float32"], t["lal"], t.get("mlx")

        def per_binary(entry, size):
            return entry["batch"][size] / int(size)

        def mlx_cell(key, size=None, large=False):
            if mlx_t is None:
                return "–"
            table = mlx_t[("large_" if large else "") + key]
            return duration(table["1"] if size is None else table[size] / int(size))

        rows = [
            ("One binary, 1000 frequencies", [f64["single"], f32["single"], lal_t["single"]], mlx_cell("value")),
            ("One binary, 2<sup>20</sup> uniform frequencies", [f64["large"], f32["large"], lal_t["large"]],
             mlx_cell("value", large=True)),
            ("256 binaries × 1000 frequencies, per binary", [per_binary(f64, "256"), per_binary(f32, "256"),
                                                             lal_t["single"]], mlx_cell("value", "256")),
            ("4096 binaries × 1000 frequencies, per binary", [per_binary(f64, "4096"), per_binary(f32, "4096"),
                                                              lal_t["single"]], mlx_cell("value", "4096")),
            ("Gradient, one binary, 1000 frequencies", [f64["grad"], f32["grad"], None], mlx_cell("grad")),
            ("Gradient, 4096 binaries × 1000, per binary", [None, None, None], mlx_cell("grad", "4096")),
            ("Jacobian, 4096 binaries × 1000, per binary", [None, None, None], mlx_cell("jacobian", "4096")),
            ("Compilation (first call)", [f64["compile"], f32["compile"], None], "–"),
            ("Compilation under <code>jax.vmap</code>", [f64["compile_vmap"], f32["compile_vmap"], None], "–"),
            ("Compilation of the gradient", [f64["compile_grad"], f32["compile_grad"], None], "–"),
        ]
        values[f"{prefix}_timing_rows"] = "\n".join(
            f"<tr><td>{row}</td>" + "".join(f'<td class="num">{duration(v)}</td>' for v in cells)
            + f'<td class="num">{last}</td></tr>' for row, cells, last in rows)
        values[f"{prefix}_single"] = duration(f64["single"])
        values[f"{prefix}_lal_single"] = duration(lal_t["single"])
        values[f"{prefix}_large"] = duration(f64["large"])
        values[f"{prefix}_large32"] = duration(f32["large"])
        values[f"{prefix}_lal_large"] = duration(lal_t["large"])
        values[f"{prefix}_large_speedup"] = factor(lal_t["large"] / f64["large"])
        values[f"{prefix}_batch4096"] = duration(per_binary(f64, "4096"))
        values[f"{prefix}_batch_speedup"] = factor(lal_t["single"] / per_binary(f64, "4096"))
        for key in ("compile", "compile_vmap", "compile_grad"):
            values[f"{prefix}_{key}"] = duration(f64[key])
        if mlx_t is not None:
            best = max((int(n) / v, n) for n, v in mlx_t["value"].items())
            jax_best = max(max(int(n) / v for n, v in e["batch"].items()) for e in (f32, f64))
            values[f"{prefix}_mlx_best_rate"], values[f"{prefix}_mlx_best_size"] = rate(best[0]), best[1]
            values[f"{prefix}_mlx_best_speedup"] = factor(best[0] / jax_best)
            values[f"{prefix}_mlx_single"] = mlx_cell("value")
            values[f"{prefix}_mlx_large"] = mlx_cell("value", large=True)
            values[f"{prefix}_mlx_first_value"] = duration(mlx_t["first_calls"]["value"])
            values[f"{prefix}_mlx_first_grad"] = duration(mlx_t["first_calls"]["grad"])
        if max(data.get("load_average") or [0.0]) > 4.0:
            busy.append(label)
    qnm = load("phenomd_qnm.json")
    values["pd_qnm_degree"] = str(qnm["degree"])
    rows = []
    for key, label in (("fring", "<i>Mf</i><sub>ring</sub>"), ("fdamp", "<i>Mf</i><sub>damp</sub>")):
        for row in qnm[key]:
            lo, hi = row["range"]
            rows.append(f"<tr><td>{label}</td><td>[{lo:+.2f}, {hi:+.2f}]</td>"
                        + "".join(f'<td class="num">{sci(row[k])}</td>' for k in
                                  ("fit_median", "fit_max", "loo_median", "loo_max", "spline_median", "spline_max"))
                        + "</tr>")
    values["pd_qnm_rows"] = "\n".join(rows)
    middle = {key: next(r for r in qnm[key] if r["range"] == [-0.9, 0.5]) for key in ("fring", "fdamp")}
    values["pd_qnm_fit_typical"] = sci(max(middle[k]["fit_median"] for k in middle))
    low = next(r for r in qnm["fdamp"] if r["range"][0] == -1.0)
    values["pd_qnm_low_fit"], values["pd_qnm_low_loo"] = sci(low["fit_max"]), sci(low["loo_max"])
    top = {k: next(r for r in qnm[k] if r["range"][1] == 1.0) for k in ("fring", "fdamp")}
    values["pd_qnm_top_loo"] = f'{100 * max(r["loo_max"] for r in top.values()):.0f}%'
    values["pd_qnm_top_spline"] = f'{100 * max(r["spline_max"] for r in top.values()):.0f}%'
    values["pd_qnm_top_fit"] = sci(max(r["fit_max"] for r in top.values()))
    values["pd_qnm_spline_max"] = sci(max(r["spline_max"] for k in ("fring", "fdamp") for r in qnm[k]
                                          if r["range"][0] >= -0.9 and r["range"][1] < 1.0))
    if busy:
        pending.append(f"the timings of {' and '.join(busy)} on an idle machine: they were measured while other "
                       "jobs ran (<code>report/taylorf2_phenomd.py</code>; "
                       '<a href="#phenomd">the TaylorF2 and IMRPhenomD section</a>)')


def mlx_values(values):
    """NRSur7dq4 in MLX (``mlx_nrsur7dq4.json``)."""
    data = load("mlx_nrsur7dq4.json")
    env = data["mlx_environment"]
    values["mlx_version"] = env["mlx"]
    values["mlx_device"] = html.escape(env["device"])
    values["mlx_memory"] = f"{env['memory_gb']} GB"
    s = data["summary"]
    values["mlx_pairs"] = str(s["all"]["n"])

    def cells(entry):
        if entry is None:
            return '<td class="num">–</td>' * 3
        return "".join(f'<td class="num">{sci(entry[k])}</td>' for k in ("median", "p90", "max"))

    rows = [(label, s["all"][f"{key}_mlx"], s["all"][f"{key}_jax32"]) for label, key in (
        ("Normalised overlap of the two binaries, change", "d_overlap"),
        ("Inner product over the product of the norms, change", "epsilon"),
        ("One waveform, noise-weighted relative error ‖δh‖ / ‖h‖", "noise_weighted"),
        ("One waveform, largest relative error max |δh| / max |h|", "max_relative"))]
    gradients = s["gradients"]
    # JAX's float32 derivatives, from the float32 study of the same pairs
    study = load("float32_nrsur7dq4.json")
    same_pairs = (study.get("seed"), study.get("n_pairs")) == (data["seed"], data["n_pairs"])
    jax_gradients = study["summary"]["gradients"]["grad_log_likelihood"] if same_pairs else {}
    rows += [
        (f"Derivatives of a log-likelihood, relative error ({gradients['n']} nearby pairs)", gradients["relative"],
         jax_gradients.get("relative")),
        ("Derivatives of a log-likelihood, error over the typical size of the component",
         gradients["over_typical"], jax_gradients.get("over_typical")),
    ]
    values["mlx_accuracy_rows"] = "\n".join(f"<tr><td>{label}</td>{cells(a)}{cells(b)}</tr>" for label, a, b in rows)
    for name in ("mlx", "jax32"):
        overlap = s["all"][f"d_overlap_{name}"]
        values[f"{name}_do_median"], values[f"{name}_do_p90"] = sci(overlap["median"]), sci(overlap["p90"])
        values[f"{name}_dh_median"] = sci(s["all"][f"noise_weighted_{name}"]["median"])
    values["mlx_grad_median"] = sci(gradients["relative"]["median"])
    values["mlx_grad_typical_max"] = sci(gradients["over_typical"]["max"])
    grids = data["grids"]
    values["mlx_grid_n"] = str(sum(g["n"] for g in grids.values()))
    values["mlx_grid_items"] = "\n".join(
        f"        <li>{'from ' + key + ' Hz' if key != '0' else 'from the start of the surrogate'}: the same length "
        f"and epoch for {g['same_length_and_epoch']} of the {g['valid']} binaries LAL accepts"
        + (f", and LAL's error for {g['same_errors']} of the {g['lal_errors']} it refuses" if g["lal_errors"] else "")
        + ";</li>" for key, g in grids.items())
    values["mlx_grid_t0"] = sci(max(g["max_t0_difference"] for g in grids.values()))
    conditioned = data["conditioned"]
    values["mlx_conditioned_n"] = str(len(conditioned))
    values["mlx_conditioned_max"] = sci(max(r["relative_error"] for r in conditioned))
    values["mlx_conditioned_same"] = f"{sum(r['same_grid'] for r in conditioned)}/{len(conditioned)}"

    pure = data["timings"]["pure"]
    mlx_value, mlx_grad = pure["mlx"]["value"], pure["mlx"]["grad"]

    def per_binary(name, key, size):
        t = pure[name][key].get(size)
        return f'<td class="num">{duration(t / int(size)) if t is not None else "–"}</td>'

    rows = []
    for size in mlx_value:
        rows.append(f'<tr><td class="num">{size}</td><td class="num">{duration(mlx_value[size])}</td>'
                    + "".join(per_binary(name, "value", size) for name in ("mlx", "jax_float32", "jax_float64"))
                    + "".join(per_binary(name, "grad", size) for name in ("mlx", "jax_float32", "jax_float64"))
                    + "</tr>")
    values["mlx_timing_rows"] = "\n".join(rows)

    def best(name, key):
        return max((int(n) / t, n) for n, t in pure[name][key].items())

    values["mlx_single"] = duration(mlx_value["1"])
    values["jax32_single"] = duration(pure["jax_float32"]["value"]["1"])
    values["mlx_grad_single"] = duration(mlx_grad["1"])
    values["jax32_grad_single"] = duration(pure["jax_float32"]["grad"]["1"])
    first = pure["mlx"]["first_calls"]
    values["mlx_first_data"] = duration(first["data"])
    values["mlx_first_value"] = duration(first["value"])
    values["mlx_first_grad"] = duration(first["grad"])
    (mlx_rate, mlx_size), (jax_rate, _) = best("mlx", "value"), best("jax_float32", "value")
    values["mlx_best_rate"], values["mlx_best_size"] = rate(mlx_rate), mlx_size
    values["jax32_best_rate"] = rate(jax_rate)
    values["mlx_best_speedup"] = factor(mlx_rate / jax_rate)
    (mlx_rate, mlx_size), (jax_rate, _) = best("mlx", "grad"), best("jax_float32", "grad")
    values["mlx_grad_best_rate"], values["mlx_grad_best_size"] = rate(mlx_rate), mlx_size
    values["jax32_grad_best_rate"] = rate(jax_rate)
    values["mlx_grad_speedup"] = factor(mlx_rate / jax_rate)
    public = data["timings"]["public"]
    batch = public["batch"]
    values["mlx_batch_n"] = str(batch["n"])
    values["mlx_batch_samples"] = f"{batch['samples'] / batch['n']:,.0f}"
    values["mlx_batch_time"] = duration(batch["mlx"])
    values["mlx_batch_grids"] = duration(batch["mlx_grids"])
    values["mlx_batch_rate"] = rate(batch["n"] / batch["mlx"])
    values["jax_batch_rate"] = rate(batch["n"] / batch["jax"])
    values["mlx_batch_speedup"] = factor(batch["jax"] / batch["mlx"])
    values["mlx_td_single"] = duration(public["td_polarizations"]["mlx"])
    values["jax_td_single"] = duration(public["td_polarizations"]["jax"])
    values["mlx_fd_single"] = duration(public["simulate_fd"]["mlx"])
    values["jax_fd_single"] = duration(public["simulate_fd"]["jax"])
    values["mlx_svg"] = svg("mlx", "Waveforms and gradients per second against the number of binaries per call, "
                                   "for NRSur7dq4 in MLX on the GPU and in JAX on the CPU")


def build(args):
    make_figures.main()
    values = {}
    values["css"] = (HERE / "report.css").read_text()
    values["date"] = datetime.date.today().strftime("%-d %B %Y")

    xpnr = load("validation_xpnr.json")
    nrsur = load("validation_nrsur7dq4.json")
    conversions = load("validation_conversions.json")
    gradients = load("gradients.json") if (DATA / "gradients.json").exists() else None
    timings = load("timings.json")
    env = timings["environment"]
    values["jax_version"] = env["jax"]
    values["lalsim_version"] = env["lalsimulation"]
    values["lal_version"] = env["lal"]
    values["python_version"] = env["python"]
    values["machine"] = html.escape(f"{env['cpu'] or env['machine']}, {env['cpus']} cores")
    values["commit"] = git("rev-parse", "--short", "HEAD").strip()
    total, generated, n_files = source_lines()
    values["n_lines"] = f"{total:,}"
    values["n_lines_generated"] = f"{generated:,}"
    values["n_files"] = str(n_files)
    values["n_commits"] = str(len(commit_rows()))
    values["commit_rows"] = "\n".join(commit_rows())

    # --- IMRPhenomXPNR random draws
    s = xpnr["summary"]
    values["xpnr_draws"] = str(xpnr["n_draws"])
    values["xpnr_nf"] = f"{xpnr['n_frequencies']:,}"
    main = s["jax_vs_rs_fix"]
    values["xpnr_n"] = str(main["n"])
    values["xpnr_n_st"] = str(xpnr["n_lal_spintaylor"])
    values["xpnr_max_rel"] = sci(main["max_relative_error"])
    values["xpnr_median_rel"] = sci(main["median_relative_error"])
    values["xpnr_p90_rel"] = sci(main["p90_relative_error"])
    values["xpnr_max_mm"] = sci(main["max_mismatch"])
    values["xpnr_median_mm"] = sci(main["median_mismatch"])
    fallback = s.get("jax_vs_rs_fix_fallback")
    values["xpnr_fallback_n"] = str(fallback["n"]) if fallback else "0"
    values["xpnr_fallback_median_mm"] = sci(fallback["median_mismatch"]) if fallback else "–"
    values["xpnr_fallback_max_mm"] = sci(fallback["max_mismatch"]) if fallback else "–"
    unfixed = s.get("jax_vs_lal")
    values["xpnr_unfixed_median_mm"] = sci(unfixed["median_mismatch"]) if unfixed else "–"
    values["xpnr_unfixed_max_mm"] = sci(unfixed["max_mismatch"]) if unfixed else "–"
    rsfix = s.get("rs_fix_vs_lal_fix")
    values["xpnr_rs_fix_max_rel"] = sci(rsfix["max_relative_error"]) if rsfix else "–"
    build_info = xpnr.get("patched_build") or {}
    values["xpnr_patched_build"] = html.escape(build_info.get("vcsStatus", "") or "not measured")
    n_failed = xpnr["n_lal_failures"]
    values["xpnr_failure_text"] = (f"LAL evaluates all {xpnr['n_draws']} draws." if n_failed == 0 else
                                   f"LAL fails for {n_failed} of the {xpnr['n_draws']} draws.")
    compared = [r for r in xpnr["records"] if r.get("jax_vs_rs_fix")]
    masses = np.array([r["mass_1"] + r["mass_2"] for r in compared])
    mismatches = np.array([r["jax_vs_rs_fix"]["mismatch"] for r in compared])
    values["xpnr_median_light"] = sci(float(np.median(mismatches[masses < 20])))
    values["xpnr_median_heavy"] = sci(float(np.median(mismatches[masses >= 150])))
    share = float(np.mean(mismatches > 1e-4))
    values["xpnr_frac_1e4"] = f"{share * 100:.0f}%" if share > 0 else "None"
    seconds = xpnr["median_seconds"]
    values["xpnr_draw_jax"] = duration(seconds.get("jax"))
    values["xpnr_draw_lal"] = duration(seconds.get("lal"))
    values["xpnr_draw_rs"] = duration(seconds.get("rs_fix"))

    # --- NRSur7dq4 random draws
    s = nrsur["summary"]["jax_vs_lal"]
    values["nrsur_draws"] = str(nrsur["n_draws"])
    values["nrsur_n"] = str(s["n"])
    values["nrsur_same_grid"] = str(s["n_same_grid"])
    values["nrsur_max_rel"] = sci(s["max_relative_error"])
    values["nrsur_median_rel"] = sci(s["median_relative_error"])
    values["nrsur_max_mm"] = sci(s["max_mismatch"])
    values["nrsur_failure_text"] = failure_text(nrsur, "raises")
    rs_nrsur = nrsur["summary"].get("jax_vs_rs")
    values["nrsur_rs_max_rel"] = sci(rs_nrsur["max_relative_error"]) if rs_nrsur else "–"
    float32_values(values)
    cost_values(values)
    reference_values(values)
    mlx_values(values)
    mlx_xpnr_values(values)
    xas_values(values)
    xhm_values(values)
    crossmodel_values(values)

    rows = []
    for label, data, key in (
        ("IMRPhenomXPNR, where LAL keeps the SpinTaylor angles, against 6.2.0 + lalsuite !2593 (waveformrs)", xpnr,
         "jax_vs_rs_fix"),
        ("IMRPhenomXPNR, where LAL leaves them, against 6.2.0 + lalsuite !2593 (waveformrs)", xpnr,
         "jax_vs_rs_fix_fallback"),
        ("IMRPhenomXPNR, where LAL keeps the SpinTaylor angles, against 6.2.0 + lalsuite !2593", xpnr,
         "jax_vs_lal_fix"),
        ("NRSur7dq4, against LALSimulation 6.2.0", nrsur, "jax_vs_lal"),
        ("NRSur7dq4, against waveformrs", nrsur, "jax_vs_rs"),
    ):
        entry = data["summary"].get(key)
        if not entry:
            continue
        rows.append(
            f"<tr><td>{label}</td><td class=\"num\">{entry['n']}/{data['n_draws']}</td>"
            f"<td class=\"num\">{sci(entry['max_relative_error'])}</td>"
            f"<td class=\"num\">{sci(entry['median_relative_error'])}</td>"
            f"<td class=\"num\">{sci(entry['max_mismatch'])}</td>"
            f"<td class=\"num\">{sci(entry['median_mismatch'])}</td></tr>")
    values["random_rows"] = "\n".join(rows)

    # --- conversions
    rows, exact = [], 0
    for record in conversions["records"]:
        cmp = record["jax_vs_lal"]
        grid_ok = cmp["same_length"] and cmp["same_epoch"] and cmp["same_step"]
        exact += grid_ok
        mb = record.get("lal_multibanding_vs_lal")
        arguments = record["arguments"]
        if record["function"] == "simulate_td":
            setting = f"{1 / arguments[0]:.0f} Hz from {arguments[1]:g} Hz"
        else:
            setting = (f"Δf = {arguments[0]:g}" if arguments[0] else "Δf from LAL") + \
                      f", {arguments[1]:g}–{arguments[2]:g} Hz"
        rows.append(
            f"<tr><td><code>{record['function']}</code></td><td>{record['approximant']}</td>"
            f"<td>{record['label']}</td><td>{setting}</td><td>{record['modes']}</td>"
            f"<td class=\"num\">{record['samples']:,}</td>"
            f"<td>{'<span class=\"chip good\">exact</span>' if grid_ok else '<span class=\"chip bad\">differs</span>'}</td>"
            f"<td class=\"num\">{sci(cmp.get('relative_error'))}</td>"
            f"<td class=\"num\">{sci(record['jax_vs_rs'].get('relative_error')) if 'jax_vs_rs' in record else '–'}</td>"
            f"<td class=\"num\">{sci(mb.get('relative_error')) if mb else '–'}</td>"
            f"<td class=\"num\">{duration(record['seconds_jax'])}</td>"
            f"<td class=\"num\">{duration(record.get('seconds_rs'))}</td>"
            f"<td class=\"num\">{duration(record['seconds_lal'])}</td></tr>")
    values["conversion_rows"] = "\n".join(rows)
    values["conversion_exact"] = f"{exact}/{len(conversions['records'])}"
    nrsur_conv = [r["jax_vs_lal"].get("relative_error", 0) for r in conversions["records"]
                  if r["approximant"] == "NRSur7dq4"]
    values["conv_nrsur_max"] = sci(max(nrsur_conv)) if nrsur_conv else "–"
    dominant = [r["jax_vs_lal"].get("relative_error", 0) for r in conversions["records"]
                if r["approximant"] == "IMRPhenomXPNR" and r["modes"] != "all"]
    values["conv_dominant_max"] = sci(max(dominant)) if dominant else "–"
    allmodes = [r["jax_vs_lal"].get("relative_error", 0) for r in conversions["records"]
                if r["approximant"] == "IMRPhenomXPNR" and r["modes"] == "all"]
    values["conv_all_max"] = sci(max(allmodes)) if allmodes else "–"
    banded = [r["lal_multibanding_vs_lal"]["relative_error"] for r in conversions["records"]
              if r.get("lal_multibanding_vs_lal")]
    values["conv_mb_range"] = f"{sci(min(banded))}–{sci(max(banded))}" if banded else "–"

    # --- derivatives (gradients.json; or the preliminary results of an
    # interrupted run, gradients_preliminary.json)
    preliminary = gradients is None
    pending = []
    if preliminary:
        partial = json.loads((DATA / "gradients_preliminary.json").read_text())["models"]
        sections = [(name, partial.get(name)) for name in ("IMRPhenomXPNR", "NRSur7dq4", "simulate_td IMRPhenomXPNR (2,2)")]
        pending.append("the derivative checks: the run was interrupted, so only the largest error per binary is "
                       "shown, the per-parameter figure is missing, and the conversions were not checked")
    else:
        names = gradients["parameters"]
        sections = [("IMRPhenomXPNR", gradients.get("xpnr")), ("NRSur7dq4", gradients.get("nrsur7dq4"))]
        sections += list((gradients.get("conversions") or {}).items())
    rows, worst_by_model = [], {}
    for model, result in sections:
        if not result:
            continue
        for binary, entry in result["binaries"].items():
            singular = entry.get("not_differentiable", [])
            note = (", ".join(f"<code>{n}</code>" for n in singular) + " (zero spin)") if singular else "–"
            if preliminary:
                worst, error = entry["worst"], entry["max_error"]
                cells = f"<td class=\"num\">{sci(error)}</td>"
            else:
                smooth = [n for n in names if n not in singular]
                worst = max(smooth, key=lambda n: entry["errors"][n])
                error = entry["errors"][worst]
                spins = [entry["errors"][n] for n in smooth if n.startswith("spin")]
                others = max(entry["errors"][n] for n in smooth if not n.startswith("spin"))
                cells = (f"<td class=\"num\">{sci(others)}</td>"
                         f"<td class=\"num\">{sci(max(spins)) if spins else '–'}</td>")
            worst_by_model[model] = max(worst_by_model.get(model, 0.0), error)
            rows.append(f"<tr><td>{html.escape(model)}</td><td>{html.escape(binary)}</td>{cells}"
                        f"<td><code>{worst}</code></td><td>{note}</td>"
                        f"<td>{'yes' if entry['finite'] else '<strong>no</strong>'}</td></tr>")
    if preliminary:
        header = ("<th class=\"num\">Largest column error</th>")
        caption = ("Preliminary: largest disagreement over the twelve columns, from an interrupted run of "
                   "<code>report/gradients.py</code> (data: <code>report/data/gradients_preliminary.json</code>).")
    else:
        header = ("<th class=\"num\">Masses, distance, angles, f<sub>ref</sub></th><th class=\"num\">Spins</th>")
        caption = ("Largest column error by group of parameters. The conversions are <code>simulate_td_function</code> "
                   "of IMRPhenomXPNR's (2,2) mode at 1024 Hz from 25 Hz and <code>simulate_fd_function</code> of "
                   "NRSur7dq4 up to 1024 Hz from 20 Hz.")
    values["gradient_table"] = (
        '<div class="table-wrap wide">\n    <table>\n      <thead><tr><th>Model</th><th>Binary</th>' + header
        + "<th>Worst parameter</th><th>Not differentiable</th><th>Finite</th></tr></thead>\n      <tbody>\n"
        + "\n".join(rows) + f"\n      </tbody>\n      <caption>{caption}</caption>\n    </table>\n  </div>")
    if preliminary:
        values["gradients_figure"] = ('<div class="callout"><p><strong>Pending.</strong> The figure of the error of '
                                      "every column, per parameter and binary, needs the full run.</p></div>")
    else:
        values["gradients_figure"] = (
            '<figure class="wide">\n    <div class="plot">' + svg("gradients", "Agreement of each column of the "
            "Jacobian with finite differences") + "</div>\n    <figcaption>Disagreement of each column of the "
            "Jacobian with finite differences, per parameter and binary. Data: <code>report/data/gradients.json"
            "</code>.</figcaption>\n  </figure>")
    values["nrsur_gradient_worst"] = sci(worst_by_model.get("NRSur7dq4"))
    values["xpnr_gradient_worst"] = sci(worst_by_model.get("IMRPhenomXPNR"))
    rows = []
    for model, result in sections:
        if not result or "forward" not in result["timings"]:
            continue
        t = result["timings"]
        batch = t.get("batch", 4 if model.startswith("simulate") else 8)
        vmap_first = t.get(f"vmap_grad_{batch}_compile")
        vmap = t.get(f"vmap_grad_{batch}")
        consistency = t.get("grad_vs_jacobian")
        rows.append(
            f"<tr><td>{html.escape(model)}</td>"
            f"<td class=\"num\">{duration(t['forward_compile'])}</td><td class=\"num\">{duration(t['forward'])}</td>"
            f"<td class=\"num\">{duration(t['jvp'])}</td>"
            f"<td class=\"num\">{duration(t['jacfwd_compile'])}</td><td class=\"num\">{duration(t['jacfwd'])}</td>"
            f"<td class=\"num\">{duration(t['grad_compile'])}</td><td class=\"num\">{duration(t['grad'])}</td>"
            f"<td class=\"num\">{duration(vmap_first) if vmap_first else 'pending'}</td>"
            f"<td class=\"num\">{duration(vmap / batch) if vmap else 'pending'}</td>"
            f"<td class=\"num\">{sci(consistency) if consistency is not None else 'pending'}</td></tr>")
    values["derivative_rows"] = "\n".join(rows)
    xt = dict(sections)["IMRPhenomXPNR"]["timings"]
    values["xpnr_grad_compile"] = duration(xt["grad_compile"])
    values["xpnr_grad_time"] = duration(xt["grad"])
    values["xpnr_grad_ratio"] = factor(xt["grad"] / xt["forward"])
    nt = dict(sections)["NRSur7dq4"]["timings"]
    values["nrsur_grad_time"] = duration(nt["grad"])
    values["nrsur_grad_ratio"] = factor(nt["grad"] / nt["forward"])
    values["nrsur_grad_compile"] = duration(nt["grad_compile"])

    # --- timings
    single = timings["single"]
    rows = []
    for label, key in (("IMRPhenomXPNR, 8033 frequencies", "IMRPhenomXPNR"),
                       ("NRSur7dq4, 4096 Hz from 20 Hz", "NRSur7dq4"),
                       ("<code>simulate_td</code> of IMRPhenomXPNR, 4096 Hz from 20 Hz", "simulate_td IMRPhenomXPNR"),
                       ("<code>simulate_fd</code> of NRSur7dq4, up to 2048 Hz from 20 Hz", "simulate_fd NRSur7dq4")):
        entry = single.get(key)
        if not entry:
            continue
        best_rs = min(v for k, v in entry.items() if k.startswith("waveformrs")) if "waveformrs" in entry else None
        cycled = "checks" in single.get("IMRPhenomXPNR", {})
        if not cycled and key in ("NRSur7dq4", "simulate_fd NRSur7dq4"):
            # repeated calls of one binary reuse waveformrs's last NRSur7dq4 evaluation
            best_rs = None
        rows.append(
            f"<tr><td>{label}</td><td class=\"num\">{duration(entry['first_call'])}</td>"
            f"<td class=\"num\">{duration(entry['public'])}</td><td class=\"num\">{duration(entry.get('pure'))}</td>"
            f"<td class=\"num\">{duration(best_rs) if best_rs else 'pending'}</td>"
            f"<td class=\"num\">{duration(entry['lal'])}{'*' if key == 'simulate_td IMRPhenomXPNR' and not cycled else ''}</td>"
            f"<td class=\"num\">{factor(entry['lal'] / entry.get('pure', entry['public']))}</td></tr>")
    values["single_rows"] = "\n".join(rows)
    if "checks" not in single.get("IMRPhenomXPNR", {}):
        pending.append("the single-waveform timings over a cycle of different binaries (waveformrs's NRSur7dq4 "
                       "times are left out, since repeated calls of one binary reuse its last result; LAL's "
                       "<code>simulate_td</code> of IMRPhenomXPNR, marked *, used its default multibanding) and "
                       "JAX on one core")
        values["single_note"] = ("Preliminary: one binary called repeatedly. waveformrs's NRSur7dq4 times are pending, "
                                 "and the * marks LAL with its default multibanding, which does less work.")
    else:
        values["single_note"] = "Every call is for a different binary of a cycle of 16."
    x = single["IMRPhenomXPNR"]
    values["xpnr_compile"] = duration(x["first_call"])
    values["xpnr_pure"] = duration(x["pure"])
    values["xpnr_public"] = duration(x["public"])
    values["xpnr_lal"] = duration(x["lal"])
    values["xpnr_rs"] = duration(x.get("waveformrs"))
    values["xpnr_rs_grid"] = duration(x.get("waveformrs_grid"))
    values["xpnr_vs_lal"] = factor(x["lal"] / x["pure"])
    values["xpnr_vs_rs"] = factor(x["pure"] / x["waveformrs_grid"]) if x.get("waveformrs_grid") else "–"
    values["xpnr_cpu_share"] = f"{x['cpu_share']:.1f}"
    values["xpnr_checks"] = duration(x.get("checks"))
    td = single.get("simulate_td IMRPhenomXPNR", {})
    values["td_lal"] = duration(td.get("lal"))
    values["td_lal_mb"] = duration(td.get("lal_multibanding"))
    values["td_jax"] = duration(td.get("public"))
    values["td_rs"] = duration(td.get("waveformrs"))
    n = single.get("NRSur7dq4", {})
    values["nrsur_compile"] = duration(n.get("first_call"))
    values["nrsur_pure"] = duration(n.get("pure"))
    values["nrsur_public"] = duration(n.get("public"))
    values["nrsur_lal"] = duration(n.get("lal"))
    one_thread = timings["public_batches"].get("NRSur7dq4", {}).get("waveformrs_1")
    values["nrsur_rs"] = duration(n.get("waveformrs") if "checks" in n else (1 / one_thread if one_thread else None))
    values["nrsur_cpu_share"] = f"{n.get('cpu_share', 0):.1f}"

    sizes = list(timings["batches"]["IMRPhenomXPNR"])
    values["batch_header"] = "".join(f"<th class=\"num\">{s}</th>" for s in sizes)
    rows = []
    for model in ("IMRPhenomXPNR", "NRSur7dq4"):
        entries = timings["batches"].get(model)
        if not entries:
            continue
        rows.append(f"<tr><td>{model}, waveforms/s</td>" + "".join(
            f"<td class=\"num\">{rate(entries[s]['per_second'])}</td>" for s in sizes) + "</tr>")
        rows.append(f"<tr><td>{model}, compilation</td>" + "".join(
            f"<td class=\"num\">{duration(entries[s]['compile'])}</td>" for s in sizes) + "</tr>")
        rows.append(f"<tr><td>{model}, cores busy</td>" + "".join(
            f"<td class=\"num\">{entries[s]['cpu_share']:.1f}</td>" for s in sizes) + "</tr>")
    values["batch_rows"] = "\n".join(rows)
    largest = sizes[-1]
    values["batch_largest"] = largest
    values["xpnr_batch_best"] = rate(max(e["per_second"] for e in timings["batches"]["IMRPhenomXPNR"].values()))
    values["xpnr_batch_compile"] = duration(timings["batches"]["IMRPhenomXPNR"]["64"]["compile"]) \
        if "64" in timings["batches"]["IMRPhenomXPNR"] else "–"
    rows = []
    for model in ("IMRPhenomXPNR", "NRSur7dq4"):
        entry = timings["public_batches"].get(model)
        if not entry:
            continue
        rows.append(
            f"<tr><td>{model}</td><td class=\"num\">{rate(entry['lal'])}</td>"
            f"<td class=\"num\">{rate(entry.get('waveformrs_1'))}</td><td class=\"num\">{rate(entry.get('waveformrs_0'))}</td>"
            f"<td class=\"num\">{rate(entry['waveformx'])}</td><td class=\"num\">{duration(entry['first_call'])}</td></tr>")
    values["public_batch_rows"] = "\n".join(rows)
    pb = timings["public_batches"]["IMRPhenomXPNR"]
    values["xpnr_public_batch"] = rate(pb["waveformx"])
    values["xpnr_rs_batch_all"] = rate(pb.get("waveformrs_0"))
    values["xpnr_lal_batch"] = rate(pb["lal"])

    # --- tests
    files, total_seconds = test_results()
    rows = []
    totals = dict(n=0, passed=0, failed=0, skipped=0)
    not_run = []
    for name, label in TEST_FILES.items():
        entry = files.get(name)
        if entry is None:
            # a file added since the last full run
            not_run.append(f"<code>{name}</code>")
            rows.append(f"<tr><td><code>{name}</code><br>{label}</td><td class=\"num\">–</td>"
                        f'<td><span class="chip warn">not in the last run</span></td><td class="num">–</td></tr>')
            continue
        for key in totals:
            totals[key] += entry[key]
        status = (f'<span class="chip good">{entry["passed"]} passed</span>' if entry["failed"] == 0 else
                  f'<span class="chip bad">{entry["failed"]} failed</span>')
        if entry["skipped"]:
            status += f' <span class="chip warn">{entry["skipped"]} skipped</span>'
        rows.append(f"<tr><td><code>{name}</code><br>{label}</td><td class=\"num\">{entry['n']}</td>"
                    f"<td>{status}</td><td class=\"num\">{duration(entry['seconds'])}</td></tr>")
    values["test_rows"] = "\n".join(rows)
    values["n_tests"] = str(totals["n"])
    values["n_passed"] = str(totals["passed"])
    values["n_failed"] = str(totals["failed"])
    values["n_skipped"] = str(totals["skipped"])
    values["test_minutes"] = f"{total_seconds / 60:.0f}"
    if not_run:
        pending.append(f"a full test run with {', '.join(not_run)}, added since the last one "
                       '(<a href="#testing">the tests section</a>)')

    # open questions that the measurements do not settle yet
    pending.append("why finite differences of the full IMRPhenomXPNR waveform disagree with its derivatives in the "
                   'in-plane spins (<a href="#gradients">the derivatives section</a>)')
    pending.append("JAX on one core: setting XLA's thread flags (<code>--xla_cpu_multi_thread_eigen=false "
                   "intra_op_parallelism_threads=1</code>) did not change the ~3 cores it keeps busy, so the "
                   "per-core comparison with LAL and waveformrs is still to be made")
    inspiral_values(values, pending)
    if "timings_environment" not in load("float32_nrsur7dq4.json"):
        pending.append("NRSur7dq4's float32 timings on an idle machine: they were measured while other jobs ran "
                       '(<code>report/float32_nrsur7dq4.py --only-timings</code>; <a href="#nrsur">the NRSur7dq4 '
                       "section</a>)")
    items ="".join(f"<li>{item}</li>" for item in pending)
    values["preliminary_note"] = (
        '<div class="callout"><p><strong>Open items.</strong> The tests, the accuracy comparisons, the derivative '
        "checks and the timings below are complete. Still open:</p><ul>" + items + "</ul></div>")
    values["accuracy_svg"] = svg("accuracy", "Relative error against total mass for IMRPhenomXPNR and NRSur7dq4")
    values["batch_svg"] = svg("batch", "Waveforms per second against the number of binaries per call")

    page = (HERE / "template.html").read_text()
    sections = re.findall(r'<section id="([a-z]+)">', page)
    for number, section in enumerate(sections, start=1):
        page = page.replace("{{sec:" + section + "}}", str(number))
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", str(value))
    missing = re.findall(r"\{\{([\w:]+)\}\}", page)
    if missing:
        raise SystemExit(f"unfilled placeholders: {missing}")

    (HERE / "artifact").mkdir(exist_ok=True)
    (HERE / "artifact" / "index.html").write_text(page)
    document = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
    out = HERE / "waveformx-report.html"
    out.write_text(document)
    print(f"wrote {out} and {HERE / 'artifact' / 'index.html'}")
    if not args.no_pdf:
        pdf = HERE / "waveformx-report.pdf"
        subprocess.run([args.chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=20000", f"--user-data-dir={tempfile.mkdtemp()}",
                        f"--print-to-pdf={pdf}", out.as_uri()], check=True, capture_output=True)
        print(f"wrote {pdf}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chrome", default=CHROME)
    parser.add_argument("--no-pdf", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
