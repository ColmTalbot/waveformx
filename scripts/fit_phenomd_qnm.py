# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The closed-form fits of IMRPhenomD's quasi-normal-mode frequencies.

LALSimulation's IMRPhenomD interpolates the ringdown and damping
frequencies of the (2,2) mode, ``Mf_ring(a)`` and ``Mf_damp(a)``, in the
final spin ``a`` with a natural cubic spline (GSL's ``cspline``) through
1003 tabulated values (``QNMData_a``, ``QNMData_fring``, ``QNMData_fdamp``
of ``LALSimIMRPhenomD.h``, given to six or seven digits). waveformx replaces
the spline by a polynomial in

    x = ((1 - a) / 2)^(1/3),

written in Chebyshev polynomials of ``t = 2 x - 1``: the frequencies vary
like powers of ``1 - a`` near ``a = 1`` (the damping frequency vanishes
there), which the cube root makes smooth. The fit minimises the relative
error with weights that are reduced for points far off the others (a
Huber-like iteration), so that the few steps in the table (near ``a = -1``,
where its values come from different computations) do not pull it.

The error of LAL's spline is estimated by leave-one-out: the spline through
the table without one point, at that point. That is the interpolation
error at twice the spacing (an overestimate where the spline is smooth)
and the noise of the table (its rounding and steps), which the spline
passes through. The fit is no worse, region by region.

Run from the repository (it needs numpy and scipy, and LALSuite's sources
next to the repository, or ``--lal-header``)::

    python scripts/fit_phenomd_qnm.py

It writes ``src/waveformx/models/phenomd/_qnm_fit.py`` and the error summary
``report/data/phenomd_qnm.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "waveformx" / "models" / "phenomd" / "_qnm_fit.py"
SUMMARY = ROOT / "report" / "data" / "phenomd_qnm.json"
HEADER = "lalsuite/lalsimulation/lib/LALSimIMRPhenomD.h"
#: The degree of the Chebyshev series of each frequency.
DEGREE = 20
#: The ranges of the final spin of the error summary.
REGIONS = ((-1.0, -0.9), (-0.9, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0))


def find_header():
    """LAL's header next to the repository or next to one of its parents
    (for git worktrees)."""
    for parent in ROOT.parents:
        path = parent / HEADER
        if path.is_file():
            return path
    raise FileNotFoundError(f"{HEADER} not found next to the repository: pass --lal-header")


def read_table(path):
    """``(a, fring, fdamp)`` from LAL's header."""
    source = Path(path).read_text()

    def array(name):
        match = re.search(r"static const double %s\[\]\s*=\s*\{(.*?)\};" % name, source, re.S)
        return np.array([float(x) for x in re.split(r"[,\s\\]+", match.group(1)) if x])

    return array("QNMData_a"), array("QNMData_fring"), array("QNMData_fdamp")


def variable(a):
    """``t = 2 ((1 - a) / 2)^(1/3) - 1``, in ``[-1, 1]`` for ``a`` in ``[-1, 1]``."""
    return 2.0 * np.cbrt((1.0 - a) / 2.0) - 1.0


def fit(a, y, degree=DEGREE, iterations=20):
    """The Chebyshev coefficients of the robust relative least-squares fit."""
    t = variable(a)
    weights = 1.0 / y
    for _ in range(iterations):
        coefficients = chebyshev.chebfit(t, y, degree, w=weights)
        residual = chebyshev.chebval(t, coefficients) / y - 1.0
        scale = 1.4826 * np.median(np.abs(residual))
        weights = np.minimum(1.0, 3.0 * scale / np.maximum(np.abs(residual), 1e-300)) ** 0.5 / y
    return coefficients


def leave_one_out(a, y):
    """The relative error of the natural cubic spline without each interior
    point, at that point (NaN at the ends)."""
    out = np.full(len(a), np.nan)
    for i in range(1, len(a) - 1):
        keep = np.arange(len(a)) != i
        out[i] = CubicSpline(a[keep], y[keep], bc_type="natural")(a[i]) / y[i] - 1.0
    return out


def summary(a, y, coefficients):
    """The errors of the fit and of the spline, per range of the spin: the
    fit at the table's points, the spline left out at them, and the fit
    against LAL's spline between the points (the difference from LAL)."""
    fitted = chebyshev.chebval(variable(a), coefficients) / y - 1.0
    loo = leave_one_out(a, y)
    middle = 0.5 * (a[1:] + a[:-1])
    spline = CubicSpline(a, y, bc_type="natural")(middle)
    against_spline = chebyshev.chebval(variable(middle), coefficients) / spline - 1.0
    rows = []
    for lo, hi in REGIONS:
        at = (a >= lo) & (a <= hi) if hi == 1.0 else (a >= lo) & (a < hi)
        mid = (middle >= lo) & (middle < hi)
        rows.append(dict(
            range=[lo, hi],
            fit_median=float(np.median(np.abs(fitted[at]))), fit_max=float(np.max(np.abs(fitted[at]))),
            loo_median=float(np.nanmedian(np.abs(loo[at]))), loo_max=float(np.nanmax(np.abs(loo[at]))),
            spline_median=float(np.median(np.abs(against_spline[mid]))),
            spline_max=float(np.max(np.abs(against_spline[mid]))),
        ))
    return rows


def _series(name, coefficients):
    body = ",\n".join(f"    {float(c)!r}" for c in coefficients)
    return f"{name} = (\n{body},\n)\n"


def write_module(coefficients, rows, header):
    def table(key):
        lines = ["        spin range      fit median/max   spline left out median/max   fit - spline (between points)"]
        for row in rows[key]:
            form = "        [{:+.2f}, {:+.2f}]  {:.1e}/{:.1e}         {:.1e}/{:.1e}                {:.1e}/{:.1e}"
            lines.append(form.format(
                *row["range"], row["fit_median"], row["fit_max"], row["loo_median"], row["loo_max"],
                row["spline_median"], row["spline_max"]))
        return "\n".join(lines)

    text = f'''# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
#
# GENERATED by scripts/fit_phenomd_qnm.py from {header} (LALSimulation
# 6.2.0's QNMData_a, QNMData_fring, QNMData_fdamp); do not edit.
"""The ringdown and damping frequencies of the (2,2) mode of IMRPhenomD (in
units of the final mass) as closed-form fits in the final spin ``a``:
Chebyshev series of degree {DEGREE} in ``t = 2 ((1 - a) / 2)^(1/3) - 1``, in
place of LAL's cubic spline through its table (see
``scripts/fit_phenomd_qnm.py``).

Relative errors, per range of the spin: the fit at the table's points, the
spline through the table without each point at that point (LAL's
interpolation error, overestimated where it is smooth, and the noise of the
table), and the fit against LAL's spline halfway between the points.

    fring:
{table("fring")}

    fdamp:
{table("fdamp")}
"""

# flake8: noqa

__all__ = ["FRING", "FDAMP", "DEGREE"]

DEGREE = {DEGREE}

#: The Chebyshev coefficients of ``Mf_ring`` (lowest degree first).
{_series("FRING", coefficients["fring"])}
#: The Chebyshev coefficients of ``Mf_damp`` (lowest degree first).
{_series("FDAMP", coefficients["fdamp"])}'''
    MODULE.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lal-header", type=Path, default=None, help=f"path of {HEADER}")
    args = parser.parse_args()
    path = args.lal_header or find_header()
    a, fring, fdamp = read_table(path)
    coefficients = {"fring": fit(a, fring), "fdamp": fit(a, fdamp)}
    rows = {"fring": summary(a, fring, coefficients["fring"]), "fdamp": summary(a, fdamp, coefficients["fdamp"])}
    write_module(coefficients, rows, HEADER)
    SUMMARY.write_text(json.dumps(dict(degree=DEGREE, variable="t = 2 ((1 - a) / 2)^(1/3) - 1", **rows), indent=1)
                       + "\n")
    for key in ("fring", "fdamp"):
        print(key)
        for row in rows[key]:
            print("  [{:+.2f}, {:+.2f}] fit {:.1e}/{:.1e} loo {:.1e}/{:.1e} vs spline {:.1e}/{:.1e}".format(
                *row["range"], row["fit_median"], row["fit_max"], row["loo_median"], row["loo_max"],
                row["spline_median"], row["spline_max"]))
    print(f"wrote {MODULE.relative_to(ROOT)} and {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
