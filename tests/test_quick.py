# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The quick tests: what ``pytest`` runs by default (``testpaths`` in
``pyproject.toml``), within about two minutes, before every commit. The
whole suite is ``pytest tests``.

Each model is compiled as few times as possible (a compilation of
IMRPhenomXPNR takes ~10-20 s, its derivatives more):

* IMRPhenomXPNR (all modes) and its forward-mode derivative along an
  in-plane spin, in one compilation (of the model's own function: the
  public one adds the rule that assembles the full Jacobian, whose
  compilation is much slower): at one binary of waveformrs's suite, close
  to the stored polarizations of LAL with the parity fix, and finite at an
  aligned and a non-spinning one;
* NRSur7dq4 against LAL at one binary (if its data file is found).
"""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import compare, requires_nrsur
from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import model

CASE = json.loads((Path(__file__).parent / "data" / "imrphenomxpnr_parityfix.json").read_text())["cases"][0]
FREQUENCIES = jnp.asarray(CASE["frequencies"])
def binary_of(case, **changes):
    case = dict(case, **changes)
    return Binary(mass_1=jnp.float64(case["mass_1"]), mass_2=jnp.float64(case["mass_2"]),
                  spin_1=jnp.asarray(case["spin_1"], float), spin_2=jnp.asarray(case["spin_2"], float),
                  luminosity_distance=jnp.float64(case["luminosity_distance"]),
                  inclination=jnp.float64(case["inclination"]), phi_ref=jnp.float64(case["phi_ref"]),
                  f_ref=jnp.float64(case["f_ref"]))


@functools.lru_cache(maxsize=None)
def with_derivative():
    """The model and its derivative along ``chi_1x`` (one compilation)."""
    def along_spin(b):
        tangent = jax.tree_util.tree_map(jnp.zeros_like, b)
        tangent = tangent._replace(spin_1=tangent.spin_1.at[0].set(1.0))
        return jax.jvp(lambda b: model._polarizations(b, FREQUENCIES, OPTIONS), (b,), (tangent,))

    return jax.jit(along_spin)


OPTIONS = WaveformOptions(scaled_strain=False)


@pytest.mark.parametrize("spins", ["generic", "aligned", "non-spinning"])
def test_xpnr_and_its_derivative(spins):
    changes = {"generic": {}, "aligned": dict(spin_1=[0.0, 0.0, 0.3], spin_2=[0.0, 0.0, -0.1]),
               "non-spinning": dict(spin_1=[0.0, 0.0, 0.0], spin_2=[0.0, 0.0, 0.0])}[spins]
    (hp, hc), (dhp, dhc) = with_derivative()(binary_of(CASE, **changes))
    for x in (hp, hc, dhp, dhc):
        assert np.all(np.isfinite(np.asarray(x)))
    if spins == "generic":
        # LAL uses the SpinTaylor angles here: a mismatch of ~5e-9
        for ours, name in ((hp, "hplus"), (hc, "hcross")):
            reference = np.asarray(CASE[f"{name}_real"]) + 1j * np.asarray(CASE[f"{name}_imag"])
            error, mismatch = compare(np.asarray(ours), reference)
            assert error < 1e-3 and mismatch < 1e-7, (name, error, mismatch)


@requires_nrsur
def test_nrsur7dq4_matches_lal(lal):
    from test_nrsur7dq4 import BINARY, check_polarizations, lal_polarizations

    import waveformx as wf

    result = wf.td_polarizations("NRSur7dq4", 1 / 2048, 20.0, scaled_strain=False, **BINARY)
    check_polarizations(result, *lal_polarizations(*lal, BINARY, 1 / 2048, 20.0))
