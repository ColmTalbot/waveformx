# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Shared set-up of the measurements of the report.

* LALSimulation runs single-threaded (``OMP_NUM_THREADS=1``, set before it is
  imported): its MSA and PNR angles race in the OpenMP loop of IMRPhenomXP.
* The NRSur7dq4 data are found as in the tests: ``NRSUR7DQ4_DATA``,
  ``LAL_DATA_PATH``, or waveformrs's ``lal_data/`` next to this repository.
* waveformrs (its Python bindings) is optional: without it the comparisons
  with waveformrs are left out.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import lal  # noqa: E402
import lalsimulation as lalsim  # noqa: E402
import numpy as np  # noqa: E402

import waveformjax as wf  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
WAVEFORMRS = ROOT.parent / "waveformrs"

_DATA_FILE = WAVEFORMRS / "lal_data" / "NRSur7dq4_v1.0.h5"
if wf.nrsur7dq4_data_file() is None and _DATA_FILE.is_file():
    os.environ["NRSUR7DQ4_DATA"] = str(_DATA_FILE)
if wf.nrsur7dq4_data_file() is not None:
    _paths = [p for p in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep) if p]
    _directory = str(Path(wf.nrsur7dq4_data_file()).parent)
    if _directory not in _paths:
        os.environ["LAL_DATA_PATH"] = os.pathsep.join([_directory, *_paths])

try:
    import waveformrs as rs
except ImportError:  # pragma: no cover
    rs = None

MTSUN_SI = lal.MTSUN_SI

#: the binaries of waveformrs's Python benchmark (python/benchmarks/benchmark.py)
BINARY = dict(mass_1=36.0, mass_2=29.0, spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1), luminosity_distance=500.0,
              inclination=0.7, phi_ref=1.2, f_ref=20.0)
NRSUR = dict(BINARY, mass_1=40.0, mass_2=20.0)
#: the frequencies of its throughput benchmark: 20-1024 Hz in steps of 1/8 Hz
BENCHMARK_FREQUENCIES = np.arange(160, 8193) / 8.0
NAMES = ["mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z", "luminosity_distance",
         "inclination", "phi_ref", "f_ref"]


def binary_of(keywords):
    """A :class:`waveformjax.Binary` of float64 JAX arrays."""
    return wf.Binary(
        mass_1=jnp.float64(keywords["mass_1"]), mass_2=jnp.float64(keywords["mass_2"]),
        spin_1=jnp.asarray(keywords["spin_1"], jnp.float64), spin_2=jnp.asarray(keywords["spin_2"], jnp.float64),
        luminosity_distance=jnp.float64(keywords["luminosity_distance"]),
        inclination=jnp.float64(keywords["inclination"]), phi_ref=jnp.float64(keywords["phi_ref"]),
        f_ref=jnp.float64(keywords["f_ref"]),
    )


def vector_of(keywords):
    """The twelve parameters of :data:`NAMES` as one float64 vector."""
    return jnp.asarray([keywords["mass_1"], keywords["mass_2"], *keywords["spin_1"], *keywords["spin_2"],
                        keywords["luminosity_distance"], keywords["inclination"], keywords["phi_ref"],
                        keywords["f_ref"]], dtype=jnp.float64)


def binary_of_vector(x):
    return wf.Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                     inclination=x[9], phi_ref=x[10], f_ref=x[11])


def keywords_of(case):
    """The binary's keyword arguments in a case dict."""
    return {key: case[key] for key in ("mass_1", "mass_2", "spin_1", "spin_2", "luminosity_distance", "inclination",
                                       "phi_ref", "f_ref")}


def lal_fd_arguments(keywords):
    return (
        keywords["phi_ref"], keywords["mass_1"] * lal.MSUN_SI, keywords["mass_2"] * lal.MSUN_SI,
        *keywords["spin_1"], *keywords["spin_2"], keywords["f_ref"],
        keywords["luminosity_distance"] * 1e6 * lal.PC_SI, keywords["inclination"],
    )


def lal_sequence(keywords, frequencies, approximant="IMRPhenomXPNR", module=None):
    """``SimInspiralChooseFDWaveformSequence`` (multibanding is off for
    frequency sequences): ``(h_+, h_x)`` or raises ``RuntimeError``."""
    module = lalsim if module is None else module
    vector = lal.CreateREAL8Vector(len(frequencies))
    vector.data = np.asarray(frequencies, float)
    hp, hc = module.SimInspiralChooseFDWaveformSequence(
        *lal_fd_arguments(keywords), None, lalsim.GetApproximantFromString(approximant), vector)
    return hp.data.data.copy(), hc.data.data.copy()


def lal_td_arguments(keywords, delta_t, f_min):
    return (
        keywords["mass_1"] * lal.MSUN_SI, keywords["mass_2"] * lal.MSUN_SI, *keywords["spin_1"],
        *keywords["spin_2"], keywords["luminosity_distance"] * 1e6 * lal.PC_SI, keywords["inclination"],
        keywords["phi_ref"], 0.0, 0.0, 0.0, delta_t, f_min, keywords["f_ref"],
    )


def psd(frequencies):
    """aLIGO design sensitivity (``SimNoisePSDaLIGOZeroDetHighPower``)."""
    return np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f)) for f in frequencies])


def mismatch(a, b, weights):
    """``1 - <a, b> / sqrt(<a, a> <b, b>)`` with ``weights`` (no
    maximisation over time or phase)."""
    inner = lambda x, y: np.sum((x * np.conj(y)).real * weights)  # noqa: E731
    aa, bb = inner(a, a), inner(b, b)
    if aa == 0 and bb == 0:
        return 0.0
    return float(1 - inner(a, b) / np.sqrt(aa * bb))


def td_mismatch(a, b, delta_t, f_low=10.0):
    """The mismatch of two time series after a Fourier transform of the
    zero-padded series, aLIGO weights from ``f_low`` (as waveformrs's
    ``scripts/validate_nrsur7dq4.py``)."""
    n = 1 << int(np.ceil(np.log2(2 * len(a))))
    fa = np.fft.rfft(a, n) * delta_t
    fb = np.fft.rfft(b, n) * delta_t
    frequencies = np.fft.rfftfreq(n, delta_t)
    keep = frequencies >= f_low
    return mismatch(fa[keep], fb[keep], 1 / psd(frequencies[keep]))


def max_relative_error(a, b):
    """``max |a - b| / max |b|``."""
    a, b = np.asarray(a), np.asarray(b)
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a - b)) / scale) if scale > 0 else float(np.max(np.abs(a)))


def block(out):
    return jax.block_until_ready(out)


def timed(function, *args):
    """``(result, seconds)`` of one call (blocking on JAX results)."""
    start = time.perf_counter()
    out = block(function(*args))
    return out, time.perf_counter() - start


def median_time(function, repeats=7, number=1):
    """Median over ``repeats`` of the mean time of ``number`` calls (s),
    after one call."""
    block(function())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            block(function())
        times.append((time.perf_counter() - start) / number)
    return statistics.median(times)


def cpu_share(function, repeats=5):
    """Process CPU time over wall time of ``repeats`` calls: about the
    number of cores the calls kept busy."""
    block(function())
    cpu, wall = time.process_time(), time.perf_counter()
    for _ in range(repeats):
        block(function())
    return (time.process_time() - cpu) / (time.perf_counter() - wall)


def environment():
    """What the measurements ran on."""
    def version(command):
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = version(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
    return dict(
        machine=f"{platform.machine()} {platform.system()} {platform.release()}",
        cpu=version(["sysctl", "-n", "machdep.cpu.brand_string"]),
        cpus=os.cpu_count(),
        python=platform.python_version(),
        jax=jax.__version__,
        jaxlib=getattr(sys.modules.get("jaxlib"), "__version__", None),
        backend=jax.default_backend(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        waveformjax_commit=commit,
        lalsimulation=lalsim.__version__,
        lal=lal.__version__,
        waveformrs=None if rs is None else rs.__version__,
    )


def write(name, payload):
    DATA.mkdir(exist_ok=True)
    path = DATA / name
    path.write_text(json.dumps(payload, indent=1, default=float))
    print(f"wrote {path}")
