# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``MODEL`` of IMRPhenomXPNR: the contract of :mod:`waveformx.models`.

* :meth:`XPNRModel.polarizations` is pure (``jax.jit`` with ``options``
  static, ``jax.vmap``, ``jax.grad``);
* :meth:`XPNRModel.validate` applies the model's checks to concrete
  values (LALSimulation's, and :class:`~waveformx.exceptions.ModelError`
  where the SpinTaylor angles cannot be computed), the latter after the
  waveform (:meth:`XPNRModel.validate_result`) for the polarizations;
* :meth:`XPNRModel.modes` and :meth:`XPNRModel.choose_fd_modes` return the
  J-frame modes of ``SimIMRPhenomXPHMFrequencySequenceOneMode`` and
  ``SimInspiralChooseFDModes`` with the flags of IMRPhenomXPNR.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ...exceptions import InvalidParameterError, ModelError
from ..._results import FrequencyModes
from . import model, pnr
from ._math import MTSUN_SI

__all__ = ["MODEL", "XPNRModel"]


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def _checks(binary, f_min, f_max, modes, reference_mf=None, integrate=True):
    """The flags of the model's errors, for :meth:`XPNRModel.validate`
    (``f_min``, ``f_max`` the first and last frequencies, ``reference_mf``
    the mass-scaled reference frequency); those of the SpinTaylor
    integration with ``integrate`` only."""
    pnr.clear_cache()
    wf, m1, m2, chi1, chi2 = model.setup_waveform(binary, 0.0, f_min, f_max, reference_mf=reference_mf)
    wf.hm_release = 122019
    base, wf0 = pnr.common_setup(wf, m1, m2, chi1, chi2, pnr.mode_range(modes), reference=reference_mf is not None)
    flags = dict(
        mass_ratio=wf.error_mass_ratio,
        final_mass=wf.error_final_mass,
        final_spin=wf.error_final_spin,
        meco=wf.error_meco,
        f_ref=wf.f_ref,
        cutoff=wf0.f_max_prime <= wf0.f_min,
    )
    if integrate:
        st = pnr.spintaylor_solution(base, wf0, m1, m2)
        flags.update(st_failed=st.failed, st_incomplete=st.incomplete)
    return flags


class XPNRModel:
    """IMRPhenomXPNR (``SimInspiralChooseFDWaveformSequence`` and
    ``SimInspiralChooseFDModes``)."""

    domain = "frequency"
    precessing = True

    def validate(self, binary, frequencies, options, integrate=True):
        """The checks of IMRPhenomXPNR (LALSimulation's) for concrete values
        (those of the SpinTaylor integration with ``integrate`` only: see
        :meth:`validate_result`)."""
        frequencies = np.asarray(frequencies, dtype=float)
        if frequencies.size == 0:
            return
        modes = model.coprecessing_modes(options.mode_array)
        for ell, emm in options.mode_array or ():
            if (int(ell), abs(int(emm))) not in model.DEFAULT_MODES:
                raise InvalidParameterError(f"mode array contains unavailable mode ({ell}, {emm})")
        flags = {k: np.asarray(v) for k, v in _checks(binary, frequencies[0], frequencies[-1], modes,
                                                       options.reference_mf, integrate).items()}
        if flags["mass_ratio"]:
            raise InvalidParameterError("IMRPhenomXPNR is not valid at mass ratios beyond 1000")
        if flags["final_mass"]:
            raise InvalidParameterError("the fitted final mass is not physical")
        if flags["final_spin"]:
            raise InvalidParameterError("the fitted final spin exceeds one")
        if flags["meco"]:
            raise InvalidParameterError("f_MECO cannot be greater than f_ISCO")
        f_ref = float(flags["f_ref"])
        if options.reference_mf is not None:
            if float(binary.f_ref) != 0.0:
                raise InvalidParameterError(
                    "with reference_mf, the reference frequency is reference_mf / M: f_ref must be zero "
                    "(waveformx.to_mass_scaled_reference converts a binary given at f_ref)"
                )
        elif not frequencies[0] <= f_ref <= frequencies[-1]:
            raise InvalidParameterError("f_min <= f_ref <= f_max is required when using tuned angles")
        if flags["cutoff"]:
            raise InvalidParameterError("the cutoff frequency lies below f_min")
        if not integrate:
            return
        if flags["st_failed"]:
            raise ModelError("the SpinTaylor angles cannot be computed")
        if flags["st_incomplete"]:
            raise ModelError("the SpinTaylor integration exceeds the capacities of waveformx")

    def validate_result(self, binary, frequencies, options, result):
        """The checks of the SpinTaylor integration (:meth:`validate`) after
        the polarizations ``result`` were computed without them: where the
        model fails, every frequency is NaN, so they are needed only then
        (the integration is not repeated for a waveform that succeeded)."""
        if np.asarray(result[0]).size and not np.isfinite(np.asarray(result[0][0])):
            self.validate(binary, frequencies, options)

    def polarizations(self, binary, frequencies, options):
        """``SimInspiralChooseFDWaveformSequence`` (see :func:`.model.polarizations`)."""
        return model.polarizations(binary, frequencies, options)

    def modes(self, binary, frequencies, options):
        """The J-frame modes of ``SimIMRPhenomXPHMFrequencySequenceOneMode``
        with the flags of IMRPhenomXPNR at ``frequencies``, on the two-sided
        frequencies ``-f[::-1], f``."""
        frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
        modes, theta, zeta = _jframe_jit(binary, frequencies, options, 0.0, None, None)
        two_sided = jnp.concatenate([-frequencies[::-1], frequencies])
        out = {key: jnp.concatenate([neg[::-1], pos]) for key, (pos, neg) in sorted(modes.items())}
        return FrequencyModes(two_sided, out, float(theta), 0.0, float(zeta))

    def choose_fd_modes(self, binary, delta_f, f_min, f_max, options):
        """``SimInspiralChooseFDModes``: the J-frame modes on LAL's two-sided
        uniform grid (multibanding off)."""
        m1, m2 = float(binary.mass_1), float(binary.mass_2)
        chi1 = np.asarray(binary.spin_1, dtype=float)
        chi2 = np.asarray(binary.spin_2, dtype=float)
        if m1 <= m2:
            m1, m2, chi1, chi2 = m2, m1, chi2, chi1
        eta = m1 * m2 / (m1 + m2) ** 2
        delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
        chi_eff = 0.5 * (1.0 + delta) * chi1[2] + 0.5 * (1.0 - delta) * chi2[2]
        f_cut = (0.33 if chi_eff > 0.99 else 0.3) / ((m1 + m2) * MTSUN_SI)
        f_max_prime = min(f_max if f_max != 0.0 else f_cut, f_cut)
        i_start = int(f_min / delta_f)
        i_stop = int(f_max_prime / delta_f) + 1
        last = f_max if f_max_prime < f_max else f_max_prime
        n_full = int(2.0 ** np.ceil(np.log2(int(last / delta_f)))) + 1
        n = max(n_full, i_stop)
        band = jnp.arange(i_start, i_stop) * delta_f
        modes, theta, zeta = _jframe_jit(binary, band, options, float(delta_f), float(f_min), float(f_max))
        frequencies = jnp.arange(n_full) * delta_f
        out = {}
        for key, (pos, neg) in sorted(modes.items()):
            full_pos = jnp.zeros(n, jnp.complex128).at[i_start:i_stop].set(pos)[:n_full]
            full_neg = jnp.zeros(n, jnp.complex128).at[i_start:i_stop].set(neg)[:n_full]
            # a zero frequency appears once
            out[key] = jnp.concatenate([full_neg[1:][::-1], full_pos])
        two_sided = jnp.concatenate([-frequencies[1:][::-1], frequencies])
        return FrequencyModes(two_sided, out, float(theta), 0.0, float(zeta))


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _jframe_jit(binary, frequencies, options, delta_f, f_min, f_max):
    return model.jframe_modes(binary, frequencies, options, delta_f, f_min, f_max)


MODEL = XPNRModel()
