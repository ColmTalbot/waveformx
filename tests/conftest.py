# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Shared set-up of the tests.

* LALSimulation runs single-threaded (``OMP_NUM_THREADS=1``): its MSA and PNR
  angles race in the OpenMP loop of IMRPhenomXP.
* The NRSur7dq4 data are found through ``NRSUR7DQ4_DATA`` (the file) or
  ``LAL_DATA_PATH`` (its directory), else in waveformrs's ``lal_data/`` next
  to this repository (or next to one of its parents, for git worktrees).
* Compiled code is kept in JAX's persistent cache (``~/.cache/waveformx/jax``,
  or ``WAVEFORMX_TEST_CACHE``; empty to turn it off), so that a run with
  unchanged models skips their compilation.
"""

import os
from pathlib import Path

# before lalsimulation (and its OpenMP runtime) is loaded
os.environ["OMP_NUM_THREADS"] = "1"

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import waveformx  # noqa: E402, F401 (enables 64-bit floats)

# JAX's persistent compilation cache: a test run with unchanged models loads
# their compiled code rather than compiling it again (IMRPhenomXPNR takes
# ~15 s, its derivatives more). WAVEFORMX_TEST_CACHE="" turns it off.
_CACHE = os.environ.get("WAVEFORMX_TEST_CACHE", str(Path.home() / ".cache" / "waveformx" / "jax"))
if _CACHE:
    jax.config.update("jax_compilation_cache_dir", _CACHE)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

DATA_FILE = "NRSur7dq4_v1.0.h5"
# waveformrs next to this repository, or next to a parent (from a git worktree)
_SIBLING = next(
    (p / "waveformrs" / "lal_data" / DATA_FILE for p in Path(__file__).resolve().parents[2:]
     if (p / "waveformrs" / "lal_data" / DATA_FILE).is_file()),
    Path(__file__).resolve().parents[2] / "waveformrs" / "lal_data" / DATA_FILE,
)
if waveformx.nrsur7dq4_data_file() is None and _SIBLING.is_file():
    os.environ["NRSUR7DQ4_DATA"] = str(_SIBLING)
NRSUR_DATA = waveformx.nrsur7dq4_data_file()
if NRSUR_DATA is not None:
    paths = [p for p in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep) if p]
    if str(Path(NRSUR_DATA).parent) not in paths:
        os.environ["LAL_DATA_PATH"] = os.pathsep.join([str(Path(NRSUR_DATA).parent), *paths])

requires_nrsur = pytest.mark.skipif(NRSUR_DATA is None, reason=f"{DATA_FILE} not found")


@pytest.fixture(scope="session")
def lal():
    """The lal and lalsimulation modules (skips without them)."""
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    return lal, lalsim


def compare(test, reference):
    """``(max |a - b| / max |b|, mismatch)`` with uniform weights."""
    test = np.asarray(test)
    reference = np.asarray(reference)
    assert test.shape == reference.shape
    scale = np.max(np.abs(reference))
    error = np.max(np.abs(test - reference)) / scale if scale > 0 else np.max(np.abs(test))
    aa = np.vdot(test, test).real
    bb = np.vdot(reference, reference).real
    mismatch = 0.0 if aa == 0 and bb == 0 else 1.0 - np.vdot(reference, test).real / np.sqrt(aa * bb)
    return error, mismatch
