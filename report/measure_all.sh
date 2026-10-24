#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
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
# MLX_PYTHON: a Python with MLX (and the rest of the environment), for the
# MLX tests and measurement of NRSur7dq4; PYTHON by default, and skipped
# where it has no MLX.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
PY="${PYTHON:-python}"
MLX_PY="${MLX_PYTHON:-$PY}"
# (the whole suite: plain pytest runs the quick tier only)
WAVEFORMX_SLOW=1 "$PY" -m pytest -q -p no:cacheprovider --junitxml=report/data/pytest.xml tests
if "$MLX_PY" -c "import mlx.core" 2>/dev/null; then
    "$MLX_PY" -m pytest -q -p no:cacheprovider tests/test_nrsur7dq4_mlx.py tests/test_xpnr_mlx.py tests/test_xas_mlx.py tests/test_xhm_mlx.py \
        --junitxml=report/data/pytest_mlx.xml
fi
"$PY" report/validate_xpnr.py --n-draws 200
"$PY" report/validate_nrsur7dq4.py --n-draws 200
"$PY" report/float32_nrsur7dq4.py
if "$MLX_PY" -c "import mlx.core" 2>/dev/null; then
    "$MLX_PY" report/mlx_nrsur7dq4.py
    "$MLX_PY" report/mlx_xpnr.py
else
    echo "no MLX in $MLX_PY: report/data/mlx_nrsur7dq4.json and mlx_xpnr.json are not measured again"
fi
# (IMRPhenomXAS: its MLX parts are measured only where MLX_PY has MLX)
"$MLX_PY" report/xas.py
"$MLX_PY" report/xhm.py
"$PY" report/validate_conversions.py
"$PY" report/gradients.py
"$PY" report/benchmark.py
"$PY" report/lean_xpnr.py
"$PY" report/reference_xpnr.py
"$PY" report/build_report.py
