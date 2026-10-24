#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
#
# Re-run every measurement of the report on the extended SpinTaylor angles,
# then rebuild it (the timings last: run on an otherwise idle machine).
#
#     bash report/extended/measure_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
PY="${PYTHON:-python}"
"$PY" -m pytest -q -p no:cacheprovider --junitxml=data/pytest.xml ../../tests
"$PY" failures.py
"$PY" noise.py
"$PY" switches.py
"$PY" jumps.py
"$PY" validate.py --widths 0.002,0.005,0.01
"$PY" q12_jumps.py
"$PY" gradients.py
"$PY" convergence.py
"$PY" timing.py
"$PY" build.py
