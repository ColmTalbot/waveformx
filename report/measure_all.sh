#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
#
# Re-run every measurement of the report, one after another (on an otherwise
# idle machine: the timings are wall-clock), then rebuild the report.
#
#     bash report/measure_all.sh
#
# PATCHED_LALSIMULATION: a liblalsimulation 6.2.0 with the fix of lalsuite
# !2593 (waveformrs's scripts/build_patched_lalsimulation.sh), for the
# comparison of IMRPhenomXPNR with the parity fix; optional.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
PY="${PYTHON:-python}"
WAVEFORMJAX_SLOW=1 "$PY" -m pytest -q -p no:cacheprovider --junitxml=report/data/pytest.xml
"$PY" report/validate_xpnr.py --n-draws 200
"$PY" report/validate_nrsur7dq4.py --n-draws 200
"$PY" report/validate_conversions.py
"$PY" report/gradients.py
"$PY" report/benchmark.py
"$PY" report/build_report.py
