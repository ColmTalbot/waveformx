#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Build the report on IMRPhenomXPNR without fallbacks and with the
extended SpinTaylor angles from the measurements in ``report/extended/data``.

Writes ``report/extended/extended-spintaylor-report.html`` (standalone),
``report/extended/artifact/index.html`` (a fragment) and
``report/extended/extended-spintaylor-report.pdf`` (headless Chrome).

Run the measurements first (``report/extended/measure_all.sh``).

Usage::

    python report/extended/build.py [--chrome PATH] [--no-pdf]
"""

import argparse
import datetime
import html
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np

import figures

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MTSUN_SI = 4.925490947641267e-06


def load(name):
    return json.loads((DATA / name).read_text())


def sci(value, digits=1):
    """``1.2e-05`` as ``1.2 × 10⁻⁵`` in HTML."""
    if value is None or not np.isfinite(value):
        return "–"
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10 ** exponent
    if round(mantissa, digits) >= 10:
        mantissa /= 10
        exponent += 1
    if -2 <= exponent <= 2:
        return f"{value:.{max(digits, -exponent + digits)}g}" if exponent < 0 else f"{value:.{digits + 1}g}"
    return f"{mantissa:.{digits}f} × 10<sup>{exponent}</sup>"


def mm(value):
    """A mismatch (rounding can make it slightly negative)."""
    return sci(max(float(value), 0.0))


def percent(value):
    return f"{100 * value:.1f}%"


def svg(name, label):
    text = (HERE / "figures" / f"{name}.svg").read_text()
    text = re.sub(r"<metadata>.*?</metadata>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\?xml[^>]*\?>\s*", "", text)
    text = re.sub(r"<!DOCTYPE[^>]*>\s*", "", text)
    return text.replace("<svg ", f'<svg role="img" aria-label="{html.escape(label)}" ', 1)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def failure_values(values):
    data = load("failures.json")
    draws = data["draws"]
    fail = [d for d in draws if d["status"] != "ok"]
    ok = [d for d in draws if d["status"] == "ok"]
    values["n_draws"] = f"{len(draws):,}"
    values["n_fail"] = f"{len(fail):,}"
    values["frac_fail"] = percent(len(fail) / len(draws))
    values["frac_ok"] = percent(len(ok) / len(draws))
    tests_fail = np.array([d["energy_test"] for d in fail])
    tests_ok = np.array([d["energy_test"] for d in ok if "energy_test" in d])
    values["n_ok_tested"] = f"{len(tests_ok):,}"
    values["max_fail_test"] = f"{tests_fail.max():.4f}"
    values["min_ok_test"] = f"{tests_ok.min():.4f}"
    codes = {int(d["back_code"]) for d in fail}
    values["fail_codes"] = ", ".join(str(c) for c in sorted(codes))
    values["n_single_sample"] = f"{sum(d['back_samples'] <= 1 for d in fail):,}"
    values["ext_finite"] = f"{sum(d['extended_finite'] for d in fail):,}"
    mm = np.array([d["mismatch_to_lal_msa"] for d in fail if d.get("extended_finite")])
    values["msa_median"] = sci(np.median(mm))
    values["msa_p90"] = sci(np.percentile(mm, 90))
    mf = np.array([d["M"] * MTSUN_SI * d["fref"] for d in draws])
    failed = np.array([d["status"] != "ok" for d in draws])
    values["min_mfref_fail"] = f"{mf[failed].min():.4f}"
    values["min_mass_fail"] = f"{min(d['M'] for d in fail):.0f}"
    values["median_mass_fail"] = f"{np.median([d['M'] for d in fail]):.0f}"
    rows = []
    for lo, hi in ((0.0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 1.0)):
        sel = (mf >= lo) & (mf < hi)
        label = f"{lo:g}–{hi:g}" if hi < 1 else f"≥ {lo:g}"
        fraction = percent(failed[sel].mean()) if sel.any() else "–"
        rows.append(f"<tr><td>{label}</td><td class=\"num\">{sel.sum():,}</td>"
                    f"<td class=\"num\">{failed[sel].sum():,}</td><td class=\"num\">{fraction}</td></tr>")
    values["failure_rows"] = "\n".join(rows)
    masses = np.array([d["M"] for d in draws])
    rows = []
    for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 300), (300, 500)):
        sel = (masses >= lo) & (masses < hi)
        rows.append(f"<tr><td>{lo}–{hi} M<sub>☉</sub></td><td class=\"num\">{sel.sum():,}</td>"
                    f"<td class=\"num\">{percent(failed[sel].mean())}</td></tr>")
    values["mass_rows"] = "\n".join(rows)


def switch_values(values):
    data = load("switches.json")
    rows = []
    for r in data["boundaries"]:
        rows.append(f"<tr><td>{html.escape(r['label'])}</td><td class=\"num\">{sci(r['lal']['jump'])}</td>"
                    f"<td class=\"num\">{mm(r['lal']['mismatch'])}</td><td class=\"num\">{sci(r['extended']['jump'])}</td>"
                    f"<td class=\"num\">{mm(r['extended']['mismatch'])}</td></tr>")
    values["switch_rows"] = "\n".join(rows)
    values["switch_lal_max"] = mm(max(r["lal"]["mismatch"] for r in data["boundaries"]))
    values["switch_ext_max"] = mm(max(r["extended"]["mismatch"] for r in data["boundaries"]))
    values["switch_ext_jump"] = sci(max(r["extended"]["jump"] for r in data["boundaries"]))
    values["switch_lal_jump"] = sci(max(r["lal"]["jump"] for r in data["boundaries"]))
    beyond = [r for r in data["boundaries"] if r["label"].startswith("f_ref")][0]
    values["fref_boundary"] = f"{beyond['at']:.2f}"
    values["fref_jump_lal"] = mm(beyond["lal"]["mismatch"])
    values["fref_jump_ext"] = sci(beyond["extended"]["jump"])
    ts = data["two_spin"]
    lal = np.array([r["lal"] for r in ts])
    ext = np.array([r["extended"] for r in ts])
    q = np.array([r["q"] for r in ts])
    values["ts_n"] = str(len(ts))
    values["ts_median"] = sci(np.median(lal))
    values["ts_p90"] = sci(np.percentile(lal, 90))
    values["ts_max"] = sci(lal.max())
    values["ts_frac"] = percent(np.mean(lal > 1e-3))
    values["ts_ext_max"] = sci(ext.max())
    values["ts_mid_median"] = sci(np.median(lal[(q >= 4) & (q < 12)]))
    values["ts_low_max"] = sci(lal[q < 4].max())
    beta = data["beta"]
    bl = np.array([r["lal"] for r in beta])
    be = np.array([r["extended_1e4"] for r in beta])
    values["beta_n"] = str(len(beta))
    values["beta_visible"] = str(int(np.sum(bl > 1e-10)))
    values["beta_big"] = str(int(np.sum(bl > 1e-3)))
    values["beta_lal_max"] = sci(bl.max())
    values["beta_ext_max"] = sci(be.max())
    q12 = data["q12"]
    qs = np.array(q12["q"])
    diff = np.array(q12["difference"])
    values["q12_below"] = sci(diff[qs < 12].max())
    values["q12_band"] = sci(diff[(qs > 12) & (qs < 14)].max())
    values["q12_above"] = sci(diff[qs >= 14].max())
    steps = data["steps"]["steps"]
    values["steps_first"] = sci(steps[0]["mismatch"])
    values["steps_last"] = sci(steps[-1]["mismatch"])
    values["steps_first_h"] = sci(steps[0]["step"])
    values["steps_last_h"] = sci(steps[-1]["step"])


def jump_values(values):
    data = load("jumps.json")

    def stats(variant, keys):
        v = [r["jump"] for r in data["jumps"][variant] if np.isfinite(r["jump"]) and any(k in r["changed"] for k in keys)]
        return (max(v), float(np.median(v)), len(v)) if v else (np.nan, np.nan, 0)

    rows = []
    for label, keys in (("End of the fine run", ("n_fine",)), ("First backward sample", ("n_b",)),
                        ("Forward run", ("n_fwd",)), ("PNR grid", ("grid_n", "grid_start")),
                        ("PNR β connection", ("upper_none", "attach"))):
        lal, ext = stats("lal", keys), stats("extended", keys)
        rows.append(f"<tr><td>{label}</td><td class=\"num\">{lal[2]}</td><td class=\"num\">{sci(lal[1])}</td>"
                    f"<td class=\"num\">{sci(lal[0])}</td><td class=\"num\">{ext[2]}</td><td class=\"num\">{sci(ext[1])}</td>"
                    f"<td class=\"num\">{sci(ext[0])}</td></tr>")
    values["jump_rows"] = "\n".join(rows)
    lal = [r["jump"] for r in data["jumps"]["lal"] if np.isfinite(r["jump"])]
    ext = [r["jump"] for r in data["jumps"]["extended"] if np.isfinite(r["jump"])]
    values["jumps_lal_n"] = str(len(lal))
    values["jumps_ext_n"] = str(len(ext))
    values["jumps_lal_max"] = sci(max(lal))
    values["jumps_ext_max"] = sci(max(ext))
    values["jumps_lal_median"] = sci(np.median(lal))
    values["jumps_ext_median"] = sci(np.median(ext))
    fine = stats("lal", ("n_fine",))
    values["fine_lal_max"] = sci(fine[0])
    values["fine_lal_median"] = sci(fine[1])
    fine = stats("extended", ("n_fine",))
    values["fine_ext_max"] = sci(fine[0])


def noise_values(values):
    data = load("noise.json")
    values["noise_rs_xpnr"] = sci(max(data["waveformrs"]["IMRPhenomXPNR"]))
    values["noise_rs_xhm"] = sci(max(data["waveformrs"]["IMRPhenomXHM"]))
    values["noise_rs_xas"] = sci(max(data["waveformrs"]["IMRPhenomXAS"]))
    values["noise_lal"] = sci(max(data["jax"]["lal"]["polarizations"]))
    values["noise_ext"] = sci(max(data["jax"]["extended"]["polarizations"]))
    modes = np.array(data["jax"]["lal"]["modes"]).max(axis=0)
    names = data["jax"]["lal"]["mode_names"]
    values["noise_worst_mode"] = html.escape(names[int(np.argmax(modes))])
    values["noise_worst_mode_value"] = sci(modes.max())


def validation_values(values):
    data = load("validation.json")
    draws = data["draws"]
    q = np.array([d["m1"] / d["m2"] for d in draws])
    values["val_n"] = f"{data['n']:,}"
    values["val_strict"] = sci(max(data["strict_vs_waveformrs"]))
    rows = []
    for width, mis in data["mismatches"].items():
        mis = np.array([np.nan if m is None else m for m in mis])
        ok = np.isfinite(mis)
        low = ok & (q <= 12)
        high = ok & (q > 12)
        rows.append(f"<tr><td>{width}</td><td class=\"num\">{sci(np.median(mis[ok]))}</td>"
                    f"<td class=\"num\">{sci(np.percentile(mis[ok], 99))}</td><td class=\"num\">{sci(mis[low].max())}</td>"
                    f"<td class=\"num\">{percent(np.mean(mis[low] > 1e-3))}</td><td class=\"num\">{sci(mis[high].max())}</td>"
                    f"<td class=\"num\">{int(np.sum(~ok))}</td></tr>")
    values["validation_rows"] = "\n".join(rows)
    default = str(min(data["mismatches"], key=float))
    mis = np.array([np.nan if m is None else m for m in data["mismatches"][default]])
    ok = np.isfinite(mis)
    values["val_median"] = sci(np.median(mis[ok]))
    values["val_p99"] = sci(np.percentile(mis[ok], 99))
    values["val_max_low"] = sci(mis[ok & (q <= 12)].max())
    values["val_max_high"] = sci(mis[ok & (q > 12)].max())
    values["val_n_high"] = str(int(np.sum(q > 12)))
    values["val_frac_1e4"] = percent(np.mean(mis[ok & (q <= 12)] > 1e-4))
    high = mis[ok & (q > 12)]
    values["val_high_over"] = str(int(np.sum(high > 1e-1)))
    jumps = load("q12_jumps.json")["binaries"]
    rows = []
    for r in sorted(jumps, key=lambda r: -r["mismatch"]):
        rows.append(f"<tr><td class=\"num\">{r['q']:.2f}</td><td class=\"num\">{r['M']:.0f} M<sub>☉</sub></td>"
                    f"<td class=\"num\">{mm(r['mismatch'])}</td><td class=\"num\">{mm(r['lal_jump_at_12'])}</td></tr>")
    values["q12_rows"] = "\n".join(rows)
    worst = max(jumps, key=lambda r: r["mismatch"])
    values["q12_worst_mismatch"] = mm(worst["mismatch"])
    values["q12_worst_jump"] = mm(worst["lal_jump_at_12"])


def gradient_values(values):
    data = load("gradients.json")
    rows = []
    worst = 0.0
    for r in data["records"]:
        errors = [p["relative_error"] for p in r["parameters"]]
        worst = max([worst] + errors)
        rows.append(f"<tr><td>{html.escape(r['binary'])}</td><td>{'yes' if r['finite'] else 'no'}</td>"
                    f"<td class=\"num\">{sci(max(errors)) if errors else 'not compared'}</td></tr>")
    values["gradient_rows"] = "\n".join(rows)
    values["gradient_worst"] = sci(worst)


def convergence_values(values):
    data = load("convergence.json")["results"]

    def best(variant, name):
        return min(r["relative_error"] for r in data[variant][name]["steps"])

    def at(variant, name, step):
        return [r["relative_error"] for r in data[variant][name]["steps"] if r["step"] == step][0]

    values["conv_ext_spin_best"] = sci(best("extended", "spin_2x"))
    values["conv_lal_spin_best"] = sci(best("lal", "spin_2x"))
    values["conv_ext_spin_1e6"] = sci(at("extended", "spin_2x", 1e-6))
    values["conv_lal_spin_1e6"] = sci(at("lal", "spin_2x", 1e-6))


def timing_values(values):
    data = load("timing.json")
    rows = []
    for name, label in (("lal", "LAL's model (default)"), ("strict", "without fallbacks"),
                        ("extended", "extended angles")):
        t = data["timings"][name]
        rows.append(f"<tr><td>{label}</td><td class=\"num\">{t['first_call']:.0f} s</td><td class=\"num\">"
                    f"{1e3 * t['median']:.1f} ms</td><td class=\"num\">{t['grad_first_call']:.0f} s</td>"
                    f"<td class=\"num\">{1e3 * t['grad_median']:.0f} ms</td></tr>")
    values["timing_rows"] = "\n".join(rows)
    values["n_timing_frequencies"] = f"{data['n_frequencies']:,}"
    t = data["timings"]
    values["compile_lal"] = f"{t['lal']['first_call']:.0f} s"
    values["compile_ext"] = f"{t['extended']['first_call']:.0f} s"
    values["run_lal"] = f"{1e3 * t['lal']['median']:.0f} ms"
    values["run_ext"] = f"{1e3 * t['extended']['median']:.0f} ms"
    env = data["environment"]
    values["machine"] = html.escape(f"{env['cpu'] or env['machine']}, {env['cpus']} cores")
    values["jax_version"] = env["jax"]
    values["lalsim_version"] = env["lalsimulation"]


def smoothness_values(values):
    data = load("smoothness.json")
    before = load("smoothness_before.json") if (DATA / "smoothness_before.json").exists() else None
    d = data["first"]
    values["sm_first"] = (f"{d['m1']:.1f} + {d['m2']:.1f} M<sub>☉</sub>, χ<sub>1</sub> = ("
                          + ", ".join(f"{s:.2f}" for s in d["s1"]) + "), χ<sub>2</sub> = ("
                          + ", ".join(f"{s:.2f}" for s in d["s2"]) + f"), f<sub>ref</sub> = {d['fref']:.0f} Hz")
    names = [n for n in figures.SCAN_LABELS if n in data["scans"]["lal"]]
    intrinsic = [n for n in names if n not in ("inclination", "phi_ref")]
    for key, variant, source in (("lal", "lal", data), ("ext", "extended", data), ("before", "extended", before)):
        if source is None:
            continue
        residual = {n: float(np.max(np.abs(figures.smooth_residual(source["scans"][variant][n])[2]))) for n in names}
        values[f"sm_{key}_resid"] = sci(max(residual[n] for n in intrinsic))
        values[f"sm_{key}_resid_min"] = sci(min(residual[n] for n in intrinsic))
        values[f"sm_{key}_resid_extrinsic"] = sci(max(residual[n] for n in names if n not in intrinsic))

    def survey(source, variant):
        records = [r for r in source["survey"] if r["variant"] == variant]
        scans = [s for r in records for s in r["scans"].values() if "flags" in s]
        events = [e for s in scans for e in s.get("events", [])]
        return records, scans, events

    records, scans, events = survey(data, "extended")
    _, lal_scans, _ = survey(data, "lal")
    values["sm_n"] = str(len(records))
    values["sm_scans"] = str(len(scans))
    values["sm_lal_flagged"] = str(sum(1 for s in lal_scans if s["flags"]))
    values["sm_lal_flags"] = str(sum(len(s["flags"]) for s in lal_scans))
    values["sm_ext_flagged"] = str(sum(1 for s in scans if s["flags"]))
    values["sm_ext_flags"] = str(sum(len(s["flags"]) for s in scans))
    values["sm_ext_located"] = str(len(events))
    rows = []
    sources = [("before", before), ("after", data)] if before is not None else [("after", data)]
    by_source = {name: survey(source, "extended")[2] for name, source in sources if source is not None}
    for cause in figures.CAUSES:
        cells = []
        for name, _ in sources:
            group = [e for e in by_source[name] if e["cause"] == cause]
            cells.append(f"<td>{len(group)}</td><td>{sci(max(e['jump'] for e in group)) if group else '—'}</td>"
                         f"<td>{mm(max(e['mismatch'] for e in group)) if group else '—'}</td>")
        rows.append(f"<tr><td>{html.escape(cause)}</td>{''.join(cells)}</tr>")
    values["sm_table"] = "\n".join(rows)
    values["sm_table_head"] = "".join(f'<th colspan="3">{name}</th>' for name, _ in sources)
    values["sm_table_sub"] = "".join("<th>located</th><th>largest |Δh|/|h|</th><th>largest mismatch</th>"
                                     for _ in sources)
    hm_events = [e for e in events if e["cause"] == "higher-mode choice"]
    values["sm_hm_n"] = str(len(hm_events))
    values["sm_hm_max"] = sci(max([e["jump"] for e in hm_events] or [0.0]))
    values["sm_hm_mm"] = mm(max([e["mismatch"] for e in hm_events] or [0.0]))
    other = [e for e in events if e["cause"] != "higher-mode choice"]
    values["sm_other_max"] = sci(max([e["jump"] for e in other] or [0.0]))
    values["sm_other_mm"] = mm(max([e["mismatch"] for e in other] or [0.0]))
    if before is not None:
        old = [e for e in by_source["before"] if e["cause"] != "higher-mode choice"]
        values["sm_before_max"] = sci(max([e["jump"] for e in old] or [0.0]))
        values["sm_before_mm"] = mm(max([e["mismatch"] for e in old] or [0.0]))


def test_values(values):
    path = HERE / "data" / "pytest.xml"
    if not path.exists():
        values["test_summary"] = "not run"
        return
    root = ElementTree.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    tests, failures, errors, skipped = (int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    extended = [c for c in suite.iter("testcase") if "test_xpnr_extended" in c.get("classname", "")]
    passed_ext = sum(1 for c in extended if c.find("failure") is None and c.find("error") is None
                     and c.find("skipped") is None)
    values["test_summary"] = (f"{tests - failures - errors - skipped} passed, {failures + errors} failed and {skipped} "
                              f"skipped of {tests}")
    values["n_extended_tests"] = str(len(extended))
    values["n_extended_passed"] = str(passed_ext)


def build(args):
    figures.main()
    values = {"css": (HERE.parent / "report.css").read_text(), "date": datetime.date.today().strftime("%-d %B %Y"),
              "commit": git("rev-parse", "--short", "HEAD").strip(), "branch": git("branch", "--show-current").strip()}
    for fill in (failure_values, switch_values, jump_values, noise_values, validation_values, gradient_values,
                 convergence_values, timing_values, smoothness_values, test_values):
        fill(values)
    labels = {
        "failure_map_svg": ("failure_map", "Total mass against M f_ref for binaries where LAL keeps or leaves the "
                            "SpinTaylor angles, and the PN energy test at f_ref"),
        "failure_fraction_svg": ("failure_fraction", "Fraction of binaries where LAL leaves the SpinTaylor angles "
                                 "against M f_ref"),
        "jumps_svg": ("jumps", "Jumps of the waveform at changes of the discrete quantities, LAL against extended"),
        "switches_svg": ("switches", "Mismatch across LAL's switches for LAL's model and the extended one"),
        "two_spin_beta_svg": ("two_spin_beta", "Jumps at the two-spin threshold and at the PNR beta connection"),
        "q12_svg": ("q12", "Mismatch between the extended model and LAL across q = 12"),
        "steps_svg": ("steps", "Mismatch between nearby points against their distance"),
        "noise_svg": ("noise", "Change of the waveform for a change of the mass in its last bit"),
        "validation_svg": ("validation", "Distribution of the mismatch against LAL"),
        "convergence_svg": ("convergence", "Error of finite differences against the step, LAL and extended"),
        "smoothness_scans_svg": ("smoothness_scans", "Departure of the overlap from a smooth curve along each "
                                 "parameter around a random binary, LAL against extended"),
        "smoothness_survey_svg": ("smoothness_survey", "Jumps at the non-smooth points of the extended model in "
                                  "random binaries, by cause"),
    }
    for key, (name, label) in labels.items():
        values[key] = svg(name, label)
    page = (HERE / "template.html").read_text()
    for number, section in enumerate(re.findall(r'<section id="([a-z0-9-]+)">', page), start=1):
        page = page.replace("{{sec:" + section + "}}", str(number))
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", str(value))
    missing = re.findall(r"\{\{([\w:-]+)\}\}", page)
    if missing:
        raise SystemExit(f"unfilled placeholders: {missing}")
    (HERE / "artifact").mkdir(exist_ok=True)
    (HERE / "artifact" / "index.html").write_text(page)
    document = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
    out = HERE / "extended-spintaylor-report.html"
    out.write_text(document)
    print(f"wrote {out}")
    if not args.no_pdf:
        pdf = HERE / "extended-spintaylor-report.pdf"
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
