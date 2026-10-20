# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""One higher mode of IMRPhenomXHM (release 122019): the frequency-independent
set-up of ``IMRPhenomXHMGenerateFDOneMode`` (waveformrs's ``HMMode::new``
and ``ModeGenerator::new``), the mode at given frequencies
(``ModeGenerator::h``) and ``XLALSimIMRPhenomXHMFrequencySequenceOneMode``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .. import amp22 as amp22_module
from .. import phase22 as phase22_module
from .. import xas
from .._math import MPC_SI, MSUN_SI, Struct, powers
from ..internals import waveform_struct
from .amplitude import amplitude, amplitude_coefficients, choose_polynomial, finish_amplitude
from .phase import phase, phase_coefficients, spheroidal_coefficients, spheroidal_to_spherical
from .variables import HIGHER_MODES, hm_waveform_struct

__all__ = ["ModeCoefficients", "mode_coefficients", "modes_coefficients", "mode_h", "frequency_sequence_one_mode"]

RELEASE = 122019


@jax.tree_util.register_pytree_node_class
class ModeCoefficients(Struct):
    """The coefficients of one higher mode: a :class:`~.._math.Struct` whose
    leaves are the traced coefficients, with the mode ``(ell, emm)`` as
    static data of the pytree (attributes ``ell`` and ``emm``)."""

    def __init__(self, ell, emm, **fields):
        super().__init__(**fields)
        object.__setattr__(self, "ell", ell)
        object.__setattr__(self, "emm", emm)

    def replace(self, **fields):
        new = dict(self._fields)
        new.update(fields)
        return ModeCoefficients(self.ell, self.emm, **new)

    def copy(self):
        return ModeCoefficients(self.ell, self.emm, **self._fields)

    def __repr__(self):
        return f"ModeCoefficients(ell={self.ell}, emm={self.emm}, " + ", ".join(self._fields) + ")"

    def tree_flatten(self):
        names = tuple(sorted(self._fields))
        return [self._fields[name] for name in names], (self.ell, self.emm, names)

    @classmethod
    def tree_unflatten(cls, aux, values):
        ell, emm, names = aux
        return cls(ell, emm, **dict(zip(names, values)))


def _view(c):
    """The static description of the mode and its ringdown frequencies, as
    the ansatzes expect them."""
    tag = 10 * c.ell + c.emm
    return Struct(ell=c.ell, emm=c.emm, mode_tag=tag, mixing_on=tag == 32, f_ring=c.f_ring, f_damp=c.f_damp)


def mode_coefficients(wf, phase22, ell, emm, defer=False):
    """All frequency-independent quantities of the co-precessing mode
    ``(ell, emm)``, ``emm > 0``, one of (2,1), (3,3), (3,2), (4,4)
    (``IMRPhenomXHM_SetHMWaveformVariables``,
    ``IMRPhenomXHM_GetAmplitudeCoefficients``,
    ``IMRPhenomXHM_GetPhaseCoefficients`` and, for the (3,2),
    ``GetSpheroidalCoefficients`` and the 22 amplitude coefficients), for
    ``PhenomXHMReleaseVersion = 122019``.

    ``wf`` is :func:`~..internals.waveform_struct` (possibly modified by the
    precessing model: final spin, ``coprec``, reference phase ...) and
    ``phase22 = phase22.phase_coefficients(wf)``. ``ell``, ``emm`` are
    Python ints. Returns a :class:`ModeCoefficients` for :func:`mode_h`.

    The returned fields include ``amp_zero`` (the odd modes of equal masses
    and spins vanish), ``error_poly_roots`` (``gsl_poly_complex_solve``
    fails, where LAL aborts) and the region boundaries ``amp.f_amp_match_in``,
    ``amp.f_amp_match_im``, ``phase.f_phase_match_in``,
    ``phase.f_phase_match_im``.
    """
    release = wf.hm_release if "hm_release" in wf else RELEASE
    if isinstance(release, int) and release != RELEASE:
        raise ValueError(f"mode_coefficients implements PhenomXHMReleaseVersion {RELEASE}, not {release}")
    if (ell, emm) not in HIGHER_MODES:
        raise ValueError(f"IMRPhenomXHM: ({ell},{emm}) is not one of the higher modes {HIGHER_MODES}")
    # (optimization barriers keep XLA from fusing, and duplicating, large
    # frequency-independent computations into their many consumers, which
    # makes the compilation of gradients much slower)
    wf, phase22 = lax.optimization_barrier((wf, phase22))
    hm = hm_waveform_struct(wf, ell, emm)
    if 10 * ell + emm in (21, 33):
        # LAL skips the vanishing odd modes of equal masses and spins; here
        # they are computed and discarded, with the inputs detached so that
        # their singular coefficients give no NaN gradients
        amp_zero = hm.amp_zero

        def detach(tree):
            return jax.tree_util.tree_map(lambda x: jnp.where(amp_zero, lax.stop_gradient(x), x), tree)

        wf = detach(wf)
        phase22 = detach(phase22)
        hm = hm_waveform_struct(wf, ell, emm)
        hm.amp_zero = amp_zero
    if hm.mixing_on:
        spheroidal, timeshift, phiref22 = spheroidal_coefficients(wf, hm, phase22)
        mix = Struct(
            amp22=amp22_module.amp_coefficients(wf),
            phase22=phase22,
            f_ring22=wf.f_ring,
            f_damp22=wf.f_damp,
            inveta=1.0 / wf.eta,
            timeshift=timeshift,
            phaseshift=0.0,
            phiref22=phiref22,
            spheroidal=spheroidal,
            mixing_coeffs=hm.mixing_coeffs,
            f_ring=hm.f_ring,
            f_damp=hm.f_damp,
        )
        mix = lax.optimization_barrier(mix)

        def rotate(mf, rd):
            return spheroidal_to_spherical(mf, powers(mf), mix, rd)
    else:
        mix = None
        rotate = None
    amp = lax.optimization_barrier(amplitude_coefficients(wf, hm, rotate, defer=defer))
    if mix is not None:
        mix.rd = amp.rd
    ph = lax.optimization_barrier(phase_coefficients(wf, hm, phase22, mix))
    sign = -1.0 if ell % 2 else 1.0
    coefficients = ModeCoefficients(
        ell,
        emm,
        amp=amp,
        phase=ph,
        mix=mix,
        f_ring=hm.f_ring,
        f_damp=hm.f_damp,
        amp0=sign * wf.amp0,
        amp_zero=hm.amp_zero,
    )
    if not defer:
        coefficients.error_poly_roots = amp.error_poly_roots & ~hm.amp_zero
    return lax.optimization_barrier(coefficients)


def modes_coefficients(wf, phase22, modes):
    """:func:`mode_coefficients` of the higher modes ``modes`` (pairs
    ``(ell, emm)``), with the intermediate polynomials of their amplitudes
    chosen together (:func:`~.amplitude.choose_polynomial` under
    ``jax.vmap``: GSL's root finder and the polynomials compiled once)."""
    partial = [mode_coefficients(wf, phase22, ell, emm, defer=True) for ell, emm in modes]
    if not partial:
        return []
    choices = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[c.amp.choice for c in partial])
    chosen = jax.vmap(choose_polynomial)(choices)
    out = []
    for k, c in enumerate(partial):
        amp = finish_amplitude(c.amp, jax.tree_util.tree_map(lambda x: x[k], chosen))
        c = c.replace(amp=amp)
        c.error_poly_roots = amp.error_poly_roots & ~c.amp_zero
        out.append(c)
    return out


def mode_h(coefficients, mf):
    """``h_{l,-m}(Mf)`` of the mode of ``coefficients``
    (:func:`mode_coefficients`) at the dimensionless frequencies ``mf``:
    ``(-1)^l amp0 A(Mf) exp(i phi(Mf))`` in strain units, as
    ``IMRPhenomXHMGenerateFDOneMode`` computes it (waveformrs's
    ``ModeGenerator::h``), without the cutoff at ``f_max_prime``. Zero where
    the mode vanishes (``amp_zero``). The 22 mode is not recycled in the
    ringdown of the (3,2) (as in LAL's one-mode functions)."""
    c = coefficients
    view = _view(c)
    mf = jnp.asarray(mf, jnp.float64)
    p = powers(mf)
    rotation = spheroidal_to_spherical(mf, p, c.mix, c.amp.rd) if view.mixing_on else None
    amp = amplitude(mf, p, c.amp, view, rotation)
    phi = phase(mf, p, c.phase, view, rotation)
    r = c.amp0 * amp
    h = lax.complex(r * jnp.cos(phi), r * jnp.sin(phi))
    return jnp.where(c.amp_zero, jnp.zeros_like(h), h)


def frequency_sequence_one_mode(frequencies, mass_1, mass_2, chi_1, chi_2, ell, emm, distance, phi_ref, f_ref):
    """``XLALSimIMRPhenomXHMFrequencySequenceOneMode`` with
    ``PhenomXHMReleaseVersion = 122019``: the mode ``(ell, emm)`` at
    ``frequencies`` (Hz, increasing), with the masses in solar masses (any
    order), the aligned spins, the distance in Mpc, the orbital reference
    phase ``phi_ref`` and the reference frequency ``f_ref`` (Hz, 0 for the
    first frequency).

    As LAL: for ``emm < 0`` this is ``h_{l,m}(f)``; for ``emm > 0`` LAL
    returns ``(-1)^l conj(h_{l,-m}(f))``, the mode at the negative frequency
    ``-f``. The mode vanishes above ``f_max_prime = min(f_last, f_cut)``
    (``f_cut`` at ``Mf = 0.3``, 0.33 for ``chi_eff > 0.99``). ``(2, +-2)``
    is ``XLALSimIMRPhenomXASFrequencySequence`` (conjugated for ``emm >
    0``)."""
    frequencies = jnp.asarray(frequencies, jnp.float64)
    if (ell, abs(emm)) == (2, 2):
        h = xas.frequency_sequence(frequencies, mass_1, mass_2, chi_1, chi_2, distance, phi_ref, f_ref)
        return jnp.conj(h) if emm > 0 else h
    wf = waveform_struct(
        mass_1 * MSUN_SI, mass_2 * MSUN_SI, chi_1, chi_2, 0.0, f_ref, phi_ref, frequencies[0], frequencies[-1],
        distance * MPC_SI, 0.0,
    )
    wf.hm_release = RELEASE
    coefficients = mode_coefficients(wf, phase22_module.phase_coefficients(wf), ell, abs(emm))
    mf = wf.m_sec * frequencies
    h = mode_h(coefficients, mf)
    h = jnp.where(mf <= wf.f_max_prime * wf.m_sec, h, jnp.zeros_like(h))
    if emm > 0:
        h = (-1.0 if ell % 2 else 1.0) * jnp.conj(h)
    return h
