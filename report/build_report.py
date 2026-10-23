#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Build the report from the measurements in ``report/data``.

Writes

* ``report/waveformjax-report.html``: a standalone HTML document;
* ``report/artifact/index.html``: the same page as a fragment (no html,
  head or body);
* ``report/waveformjax-report.pdf``: printed with headless Chrome (optional).

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
    "test_api.py": "The interface: options, arguments, errors, spherical harmonics",
    "test_xpnr_xas.py": "IMRPhenomXAS (2,2) mode against <code>SimIMRPhenomXASFrequencySequence</code>",
    "test_xpnr_hm.py": "Higher modes (release 122019) against <code>SimIMRPhenomXHMFrequencySequenceOneMode</code>",
    "test_xpnr_msa.py": "MSA and NNLO angles and the precession set-up, GSL's elliptic functions",
    "test_xpnr_pnr_helpers.py": "PNR fits and angle helpers against LAL's internal functions (ctypes)",
    "test_xpnr.py": "IMRPhenomXPNR polarizations: waveformrs's reference suite, with and without the parity fix",
    "test_xpnr_modes.py": "IMRPhenomXPNR J-frame modes, <code>ChooseFDModes</code>, version fallbacks, checks",
    "test_xpnr_random.py": "IMRPhenomXPNR, 64 random binaries against LAL (slow)",
    "test_transforms_xpnr.py": "IMRPhenomXPNR under <code>jit</code>, <code>vmap</code>, <code>grad</code> and <code>jvp</code>",
    "test_nrsur7dq4.py": "NRSur7dq4 polarizations, modes, dynamics, errors, data file, batches",
    "test_nrsur7dq4_random.py": "NRSur7dq4, 64 random binaries against LAL (slow)",
    "test_transforms_nrsur7dq4.py": "NRSur7dq4 under <code>jit</code>, <code>vmap</code> and <code>grad</code>, removable singularities",
    "test_nrsur7dq4_float32.py": "NRSur7dq4 in float32 (no float64 operation, accuracy, <code>jit</code>, <code>vmap</code>, <code>grad</code>) and the scale of its output",
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
    return f"{seconds * 1e6:.0f} µs"


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


def test_results():
    """Per test file: (tests, passed, failed, skipped, seconds) from the
    JUnit XML of the last full run."""
    path = DATA / "pytest.xml"
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


def source_lines():
    sources = list((ROOT / "src" / "waveformjax").rglob("*.py"))
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
    """Whether waveformjax fails where LAL does, as a sentence."""
    n, failures, agree = data["n_draws"], data["n_lal_failures"], data["n_failures_agree"]
    if failures == 0:
        return (f"LAL evaluates all {n} draws, and so does waveformjax." if agree == n else
                f"LAL evaluates all {n} draws; waveformjax fails on {n - agree} of them.")
    return (f"LAL fails for {failures} of the {n} draws, and waveformjax {how} for the same ones "
            f"({agree}/{n} agree).")


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
    values["xpnr_n"] = str(s["jax_vs_lal"]["n"])
    values["xpnr_max_rel"] = sci(s["jax_vs_lal"]["max_relative_error"])
    values["xpnr_median_rel"] = sci(s["jax_vs_lal"]["median_relative_error"])
    values["xpnr_p90_rel"] = sci(s["jax_vs_lal"]["p90_relative_error"])
    values["xpnr_max_mm"] = sci(s["jax_vs_lal"]["max_mismatch"])
    values["xpnr_median_mm"] = sci(s["jax_vs_lal"]["median_mismatch"])
    fix = s.get("jax_fix_vs_lal_fix")
    values["xpnr_fix_n"] = str(fix["n"]) if fix else "0"
    values["xpnr_fix_max_rel"] = sci(fix["max_relative_error"]) if fix else "–"
    values["xpnr_fix_median_rel"] = sci(fix["median_relative_error"]) if fix else "–"
    values["xpnr_fix_max_mm"] = sci(fix["max_mismatch"]) if fix else "–"
    rsfix = s.get("rs_fix_vs_lal_fix")
    values["xpnr_rs_fix_max_rel"] = sci(rsfix["max_relative_error"]) if rsfix else "–"
    build_info = xpnr.get("patched_build") or {}
    values["xpnr_patched_build"] = html.escape(build_info.get("vcsStatus", "") or "not measured")
    values["xpnr_failure_text"] = failure_text(xpnr, "raises or returns NaN")
    compared = [r for r in xpnr["records"] if r.get("jax_vs_lal")]
    masses = np.array([r["mass_1"] + r["mass_2"] for r in compared])
    errors = np.array([r["jax_vs_lal"]["relative_error"] for r in compared])
    values["xpnr_median_light"] = sci(float(np.median(errors[masses < 20])))
    values["xpnr_median_heavy"] = sci(float(np.median(errors[masses >= 150])))
    share = float(np.mean(errors > 1e-5))
    values["xpnr_frac_1e5"] = f"{share * 100:.0f}%" if share > 0 else "None"
    seconds = xpnr["median_seconds"]
    values["xpnr_draw_jax"] = duration(seconds.get("jax"))
    values["xpnr_draw_lal"] = duration(seconds.get("lal"))
    values["xpnr_draw_rs"] = duration(seconds.get("rs"))

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

    rows = []
    for label, data, key in (
        ("IMRPhenomXPNR without the parity fix, against LALSimulation 6.2.0", xpnr, "jax_vs_lal"),
        ("IMRPhenomXPNR with the parity fix, against 6.2.0 + lalsuite !2593", xpnr, "jax_fix_vs_lal_fix"),
        ("IMRPhenomXPNR without the fix, against waveformrs", xpnr, "jax_vs_rs"),
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
            f"<td class=\"num\">{rate(entry['waveformjax'])}</td><td class=\"num\">{duration(entry['first_call'])}</td></tr>")
    values["public_batch_rows"] = "\n".join(rows)
    pb = timings["public_batches"]["IMRPhenomXPNR"]
    values["xpnr_public_batch"] = rate(pb["waveformjax"])
    values["xpnr_rs_batch_all"] = rate(pb.get("waveformrs_0"))
    values["xpnr_lal_batch"] = rate(pb["lal"])

    # --- tests
    files, total_seconds = test_results()
    rows = []
    totals = dict(n=0, passed=0, failed=0, skipped=0)
    for name, label in TEST_FILES.items():
        entry = files.get(name)
        if entry is None:
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

    # open questions that the measurements do not settle yet
    pending.append("why finite differences of the full IMRPhenomXPNR waveform disagree with its derivatives in the "
                   'in-plane spins (<a href="#gradients">the derivatives section</a>)')
    pending.append("JAX on one core: setting XLA's thread flags (<code>--xla_cpu_multi_thread_eigen=false "
                   "intra_op_parallelism_threads=1</code>) did not change the ~3 cores it keeps busy, so the "
                   "per-core comparison with LAL and waveformrs is still to be made")
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
    out = HERE / "waveformjax-report.html"
    out.write_text(document)
    print(f"wrote {out} and {HERE / 'artifact' / 'index.html'}")
    if not args.no_pdf:
        pdf = HERE / "waveformjax-report.pdf"
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
