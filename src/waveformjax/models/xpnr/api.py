# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``MODEL`` of IMRPhenomXPNR: the contract of :mod:`waveformjax.models`.

* :meth:`XPNRModel.polarizations` is pure (``jax.jit`` with ``options``
  static, ``jax.vmap``, ``jax.grad``);
* :meth:`XPNRModel.validate` applies LALSimulation's checks to concrete
  values, and with ``version_fallback=False`` raises
  :class:`~waveformjax.exceptions.PrecessionVersionUnavailableError` where
  LAL would leave the SpinTaylor/PNR angles (version 330) (with
  ``extended_spintaylor``, :class:`~waveformjax.exceptions.ModelError` where
  even the extended angles fail);
* :meth:`XPNRModel.modes` and :meth:`XPNRModel.choose_fd_modes` return the
  J-frame modes of ``SimIMRPhenomXPHMFrequencySequenceOneMode`` and
  ``SimInspiralChooseFDModes`` with the flags of IMRPhenomXPNR.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ...exceptions import InvalidParameterError, ModelError, PrecessionVersionUnavailableError
from ..._results import FrequencyModes
from . import model, pnr
from ._math import MSUN_SI, MTSUN_SI

__all__ = ["MODEL", "XPNRModel"]


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def _prescription(binary, f_min, f_max, modes, parity_fix, extended=False):
    """The flags deciding LAL's prescription and LAL's errors, for
    :meth:`XPNRModel.validate` (``f_min``, ``f_max`` the first and last
    frequencies; ``extended``: those of the extended SpinTaylor angles)."""
    pnr.clear_cache()
    wf, m1_si, m2_si, chi1, chi2 = model.setup_waveform(binary, 0.0, f_min, f_max)
    wf.hm_release = 122019
    base, wf0 = pnr.common_setup(wf, m1_si, m2_si, chi1, chi2, pnr.mode_range(modes), parity_fix, extended)

    def st_flags(_):
        st = pnr.spintaylor_solution(base, wf0, m1_si, m2_si)
        return st.failed, st.incomplete

    failed, incomplete = jax.lax.cond(
        base.pnr_on & ~base.nnlo_from_start, st_flags, lambda _: (jnp.asarray(False), jnp.asarray(False)), None
    )
    return dict(
        mass_ratio=wf.error_mass_ratio,
        final_mass=wf.error_final_mass,
        final_spin=wf.error_final_spin,
        meco=wf.error_meco,
        f_ref=wf.f_ref,
        pnr_on=base.pnr_on,
        nnlo=base.nnlo_from_start,
        st_failed=failed,
        st_incomplete=incomplete,
        cutoff=wf0.f_max_prime <= wf0.f_min,
    )


@functools.partial(jax.jit, static_argnums=(3, 4))
def _msa_fails(binary, f_min, f_max, modes, parity_fix):
    """Whether the MSA system cannot be initialised (which prescription
    LAL falls back to when the SpinTaylor angles fail), compiled apart from
    :func:`_prescription` because it is rarely needed."""
    pnr.clear_cache()
    wf, m1_si, m2_si, chi1, chi2 = model.setup_waveform(binary, 0.0, f_min, f_max)
    wf.hm_release = 122019
    base, wf0 = pnr.common_setup(wf, m1_si, m2_si, chi1, chi2, pnr.mode_range(modes), parity_fix)
    return pnr._msa.msa_error(pnr._msa.system(pnr.msa_inputs(base, wf0, base.chi1, base.chi2)))


class XPNRModel:
    """IMRPhenomXPNR (``SimInspiralChooseFDWaveformSequence`` and
    ``SimInspiralChooseFDModes``)."""

    domain = "frequency"
    precessing = True

    def validate(self, binary, frequencies, options):
        """LALSimulation's checks of IMRPhenomXPNR for concrete values."""
        frequencies = np.asarray(frequencies, dtype=float)
        if frequencies.size == 0:
            return
        modes = model.coprecessing_modes(options.mode_array)
        for ell, emm in options.mode_array or ():
            if (int(ell), abs(int(emm))) not in model.DEFAULT_MODES:
                raise InvalidParameterError(f"mode array contains unavailable mode ({ell}, {emm})")
        flags = {k: np.asarray(v) for k, v in _prescription(binary, frequencies[0], frequencies[-1], modes,
                                                             options.pnr_parity_fix,
                                                             options.extended_spintaylor).items()}
        if flags["mass_ratio"]:
            raise InvalidParameterError("IMRPhenomXPNR is not valid at mass ratios beyond 1000")
        if flags["final_mass"]:
            raise InvalidParameterError("the fitted final mass is not physical")
        if flags["final_spin"]:
            raise InvalidParameterError("the fitted final spin exceeds one")
        if flags["meco"]:
            raise InvalidParameterError("f_MECO cannot be greater than f_ISCO")
        f_ref = float(flags["f_ref"])
        if not frequencies[0] <= f_ref <= frequencies[-1]:
            raise InvalidParameterError("f_min <= f_ref <= f_max is required when using tuned angles")
        if flags["cutoff"]:
            raise InvalidParameterError("the cutoff frequency lies below f_min")
        if flags["st_incomplete"] and not flags["st_failed"]:
            raise ModelError("the SpinTaylor integration exceeds the capacities of waveformjax")
        if options.extended_spintaylor:
            if flags["st_failed"]:
                raise ModelError("the extended SpinTaylor angles cannot be computed")
        elif not options.version_fallback:
            reason = None
            if not flags["pnr_on"]:
                reason, fallback = "the in-plane spins are below 1e-7, which switches the PNR angles off", 102
            elif flags["nnlo"]:
                reason, fallback = "the in-plane spins are below 1e-6", 102
            elif flags["st_failed"]:
                if np.asarray(_msa_fails(binary, frequencies[0], frequencies[-1], modes, options.pnr_parity_fix)):
                    reason, fallback = "the SpinTaylor angles cannot be computed and the MSA system fails", 102
                else:
                    reason, fallback = "the SpinTaylor angles cannot be computed", 223
            if reason is not None:
                raise PrecessionVersionUnavailableError(
                    f"PhenomXPrecVersion 330 cannot be used ({reason}); LALSimulation would switch to "
                    f"{fallback}, but version fallbacks are disabled"
                )

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
        if m1 * MSUN_SI <= m2 * MSUN_SI:
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
