#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The LAL reference of the NNLO and MSA prescriptions of IMRPhenomXPNR
(``IMRPhenomXPNR_NNLO``, ``IMRPhenomXPNR_MSA``): LALSimulation 6.2.0 with
the parity fix of lalsuite !2593 (a library built by waveformrs's
``scripts/build_patched_lalsimulation.sh``, loaded with its
``scripts/lalsimulation_ctypes.py``) at ``PhenomXPrecVersion`` 102 and 223,
on the binaries, frequencies and mode arrays of
``tests/data/imrphenomxpnr_parityfix.json``. Written to
``tests/data/imrphenomxpnr_prescriptions.json``: for each version, one
entry per case of that file (``None`` where LAL fails). (waveformrs keeps
version 330 for IMRPhenomXPNR whatever ``precession_version`` is.)

Run LAL single-threaded (its PNR and MSA angles race under OpenMP)::

    OMP_NUM_THREADS=1 python scripts/xpnr_prescriptions_reference.py PATCHED_LIBRARY [--waveformrs ../waveformrs]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
VERSIONS = (102, 223)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("library", help="liblalsimulation 6.2.0 with the fix of lalsuite !2593")
    parser.add_argument("--waveformrs", default=str(ROOT.parent / "waveformrs"))
    parser.add_argument("--output", default=str(DATA / "imrphenomxpnr_prescriptions.json"))
    args = parser.parse_args()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("run with OMP_NUM_THREADS=1")
    sys.path.insert(0, str(Path(args.waveformrs) / "scripts"))
    import lal
    import lalsimulation as lalsim
    from lalsimulation_ctypes import Library

    library = Library(args.library)
    source = json.loads((DATA / "imrphenomxpnr_parityfix.json").read_text())
    out = dict(lalsimulation_version=library.__version__, lal_version=lal.__version__,
               lalsimulation_build=library.vcs_info, source="imrphenomxpnr_parityfix.json", versions={})
    for version in VERSIONS:
        results = []
        for case in source["cases"]:
            params = lal.CreateDict()
            lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(params, version)
            if case.get("mode_array") is not None:
                modes = lalsim.SimInspiralCreateModeArray()
                for ell, emm in case["mode_array"]:
                    lalsim.SimInspiralModeArrayActivateMode(modes, ell, emm)
                lalsim.SimInspiralWaveformParamsInsertModeArray(params, modes)
            frequencies = lal.CreateREAL8Vector(len(case["frequencies"]))
            frequencies.data = np.asarray(case["frequencies"], float)
            arguments = (case["phi_ref"], case["mass_1"] * lal.MSUN_SI, case["mass_2"] * lal.MSUN_SI,
                         *case["spin_1"], *case["spin_2"], case["f_ref"],
                         case["luminosity_distance"] * 1e6 * lal.PC_SI, case["inclination"])
            try:
                hp, hc = library.SimInspiralChooseFDWaveformSequence(
                    *arguments, params, lalsim.GetApproximantFromString("IMRPhenomXPNR"), frequencies)
            except RuntimeError as error:
                print(version, len(results), "LAL failed:", error)
                results.append(None)
                continue
            hp, hc = hp.data.data, hc.data.data
            results.append(dict(hplus_real=hp.real.tolist(), hplus_imag=hp.imag.tolist(),
                                hcross_real=hc.real.tolist(), hcross_imag=hc.imag.tolist()))
        out["versions"][str(version)] = results
    Path(args.output).write_text(json.dumps(out))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
