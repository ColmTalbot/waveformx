# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The aligned-spin (2,2) mode IMRPhenomXAS (``IMRPhenomXASGenerateFD`` and
``XLALSimIMRPhenomXASFrequencySequence`` of ``LALSimIMRPhenomX.c``), after
waveformrs's ``imrphenomx/xas.rs``."""

from __future__ import annotations

import jax.numpy as jnp

from ._math import FRAC_PI_4, MPC_SI, MSUN_SI, Struct, fma, powers
from .amp22 import amp_coefficients, amp_ansatz
from .internals import waveform_struct
from .phase22 import phase_22, phase_coefficients, phase_xas, time_shift_22

__all__ = ["xas_setup", "phif_ref", "h22", "frequency_sequence"]


def xas_setup(wf):
    """The (2,2) coefficients of ``IMRPhenomXASGenerateFD``: amplitude and
    phase coefficients (with the connection coefficients), ``linb`` and
    ``lina``."""
    amp = amp_coefficients(wf)
    phase = phase_coefficients(wf)
    linb = time_shift_22(phase, wf)
    return Struct(amp=amp, phase=phase, linb=linb, lina=0.0, inveta=1.0 / wf.eta, amp_prefactor=wf.amp0 * wf.amp_norm)


def phif_ref(wf, xas):
    """``phifRef``: the phase offset that makes the (2,2) phase ``2 phi0 +
    pi / 4`` at the reference frequency."""
    mf_ref = wf.mf_ref
    phi = xas.inveta * phase_22(mf_ref, powers(mf_ref), xas.phase)
    return -(fma(xas.linb, mf_ref, phi) + xas.lina) + 2.0 * wf.phi0 + FRAC_PI_4


def h22(mf, wf, xas, phif_ref_value, mf_cut=None):
    """The (2,2) mode ``h22(Mf)`` of ``IMRPhenomXASGenerateFD`` (``h22 = A
    exp(i phi)``), zero above ``mf_cut`` (default ``f_max_prime M``)."""
    if mf_cut is None:
        mf_cut = wf.f_max_prime * wf.m_sec
    p = powers(mf)
    phi = phase_xas(mf, p, xas.phase) * xas.inveta
    phi = phi + (fma(xas.linb, mf, xas.lina) + phif_ref_value)
    amp = xas.amp_prefactor * p.m_seven_sixths * amp_ansatz(mf, p, xas.amp)
    h = amp * jnp.exp(1j * phi)
    return jnp.where(mf <= mf_cut, h, 0.0)


def frequency_sequence(frequencies, m1, m2, chi1, chi2, distance, phi0, f_ref):
    """``XLALSimIMRPhenomXASFrequencySequence`` with the masses in solar
    masses and the distance in Mpc: the (2,2) mode at ``frequencies``."""
    frequencies = jnp.asarray(frequencies)
    wf = waveform_struct(
        m1 * MSUN_SI, m2 * MSUN_SI, chi1, chi2, 0.0, f_ref, phi0, frequencies[0], frequencies[-1],
        distance * MPC_SI, 0.0,
    )
    xas = xas_setup(wf)
    return h22(wf.m_sec * frequencies, wf, xas, phif_ref(wf, xas))

