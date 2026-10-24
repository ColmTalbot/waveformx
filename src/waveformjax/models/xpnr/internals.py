# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The waveform variables of IMRPhenomX (``IMRPhenomXSetWaveformVariables``,
``LALSimIMRPhenomX_internals.c``), following waveformrs's
``imrphenomx/internals.rs``.

:func:`waveform_struct` returns a :class:`~._math.Struct` with LAL's fields
in snake case (``m1``, ``m2`` in units of the total mass with ``m1 >= m2``;
frequencies prefixed with ``mf`` or without a unit, such as ``f_ring``, are
in units of the total mass, ``f_min``, ``f_ref``, ... in Hz). Every field
may be a traced value.

LAL's errors become boolean fields (``error_*``) which the model turns into
NaN and :func:`validate` into exceptions.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._generated import qnm as _qnm
from ._generated import utilities as _utilities
from ._math import (
    MPC_SI, MRSUN_SI, MTSUN_SI, PI, POWERS_OF_LALPI, FencedModule, Struct, fenced, fma, fnms, hz_to_mf, mass_unit,
)

qnm = FencedModule(_qnm)
utilities = FencedModule(_utilities)

__all__ = ["waveform_struct", "set_ringdown_frequencies", "f_max_prime"]

F_CUT_DEF = 0.3
F_CUT_DEF_HIGH_SPIN = 0.33


@fenced
def waveform_struct(m1_si_in, m2_si_in, chi1l_in, chi2l_in, delta_f, f_ref, phi0, f_min, f_max, distance_mpc,
                    inclination, amp_scale=1.0):
    """``IMRPhenomXSetWaveformVariables`` for the default model versions
    (BBH, GR), with the masses in kg (in :func:`._math.mass_unit`) as LAL
    receives them and the distance in Mpc (LAL's is in m, whose square, in
    the derivative of the amplitude, is beyond the range of float32).
    ``delta_f = 0`` denotes a frequency sequence, ``f_ref = 0`` selects
    ``f_min`` and ``f_max = 0`` the default cutoff; the strain is multiplied
    by ``amp_scale``.

    ``delta_f``, ``f_min`` and ``f_max`` may be Python floats or traced
    values; the bodies are ordered heavier first as LAL orders them.
    """
    m1_in = m1_si_in / mass_unit()
    m2_in = m2_si_in / mass_unit()
    mass_ratio = jnp.where(m1_in > m2_in, m1_in / m2_in, m2_in / m1_in)
    swap = m1_in < m2_in
    m1 = jnp.where(swap, m2_in, m1_in)
    m2 = jnp.where(swap, m1_in, m2_in)
    chi1l = jnp.where(swap, chi2l_in, chi1l_in)
    chi2l = jnp.where(swap, chi1l_in, chi2l_in)

    delta = jnp.abs((m1 - m2) / (m1 + m2))
    eta = jnp.abs(0.25 * fnms(delta, delta, 1.0))
    eta = jnp.minimum(eta, 0.25)
    q = jnp.where(m1 > m2, m1 / m2, m2 / m1)
    q = jnp.where(eta == 0.25, 1.0, q)

    m_tot = m1 + m2
    m_sec = m_tot * MTSUN_SI
    chi_eff = utilities.chi_eff(eta, chi1l, chi2l)
    dchi = utilities.dchi(chi1l, chi2l)
    m1_si = m1 * mass_unit()
    m2_si = m2 * mass_unit()
    m1 = m1 / m_tot
    m2 = m2 / m_tot

    f_ref = jnp.where(f_ref == 0.0, f_min, f_ref)
    pi_m = PI * m_sec
    f_cut_def = jnp.where(chi_eff > 0.99, F_CUT_DEF_HIGH_SPIN, F_CUT_DEF)
    f_cut = f_cut_def / m_sec
    f_max_prime_ = jnp.where(f_max != 0.0, f_max, f_cut)
    f_max_prime_ = jnp.minimum(f_max_prime_, f_cut)

    wf = Struct(
        m1_si=m1_si,
        m2_si=m2_si,
        m_tot_si=m1_si + m2_si,
        m1=m1,
        m2=m2,
        m_tot=m_tot,
        q=q,
        eta=eta,
        delta=delta,
        eta2=eta * eta,
        eta3=eta * (eta * eta),
        chi1l=chi1l,
        chi2l=chi2l,
        chi1l2l=chi1l * chi2l,
        chi1l2=chi1l * chi1l,
        chi1l3=chi1l * chi1l * chi1l,
        chi2l2=chi2l * chi2l,
        chi2l3=chi2l * chi2l * chi2l,
        chi_eff=chi_eff,
        chi_pn_hat=utilities.chi_pn_hat(eta, chi1l, chi2l),
        s_tot_r=utilities.s_tot_r(eta, chi1l, chi2l),
        dchi=dchi,
        dchi_half=dchi * 0.5,
        sigma_l=fma(chi2l, m2, -(chi1l * m1)),
        sl=fma(chi1l, m1 * m1, chi2l * (m2 * m2)),
        f_ref=f_ref,
        phi_ref_in=phi0,
        phi0=phi0,
        beta=PI * 0.5 - phi0,
        mf_ref=hz_to_mf(f_ref, m_tot),
        pi_m=pi_m,
        v_ref=jnp.cbrt(pi_m * f_ref),
        m_sec=m_sec,
        delta_f=delta_f,
        delta_mf=hz_to_mf(delta_f, m_tot),
        f_cut_def=f_cut_def,
        f_min=f_min,
        f_max=f_max,
        mf_max=hz_to_mf(f_max, m_tot),
        f_cut=f_cut,
        f_max_prime=f_max_prime_,
        distance=distance_mpc,
        inclination=inclination,
        # (times amp_scale, the factor of the strain: 1e21 keeps the
        # strain and its square within the range of float32)
        amp0=m_tot * (MRSUN_SI / MPC_SI) * (m_tot * MTSUN_SI) * amp_scale / distance_mpc,
        amp_norm=jnp.sqrt(2.0 / 3.0) * jnp.sqrt(eta) * POWERS_OF_LALPI.m_one_sixth,
        dphase0=5.0 / (128.0 * PI ** (5.0 / 3.0)),
        coprec=Struct(
            dev=0.0, mu1=0.0, mu2=0.0, mu3=0.0, nu4=0.0, nu5=0.0, nu6=0.0, zeta1=0.0, zeta2=0.0,
            tuned=False, pnr_window=0.0, f_ring_eff_shift_divided_by_emm=0.0,
        ),
        hm_release=122022,
        error_mass_ratio=(mass_ratio > 1000.0) & (jnp.abs(mass_ratio - 1000.0) > 1e-12),
        error_spin=(jnp.abs(chi1l) > 1.0) | (jnp.abs(chi2l) > 1.0),
    )
    # the final state (XLALSimIMRPhenomXFinalMass2017, ...FinalSpin2017)
    wf.m_final = utilities.final_mass_2017(eta, chi1l, chi2l)
    wf.a_final = utilities.final_spin_2017(eta, chi1l, chi2l)
    wf.a_final_nonprec = wf.a_final
    wf.a_final_prec = wf.a_final
    set_ringdown_frequencies(wf)
    wf.f_meco = utilities.f_meco(eta, chi1l, chi2l)
    wf.f_isco = utilities.f_isco(wf.a_final)
    wf.error_final_mass = wf.m_final > 1.0
    wf.error_final_spin = jnp.abs(wf.a_final) > 1.0
    wf.error_meco = wf.f_meco > wf.f_isco
    return wf


def set_ringdown_frequencies(wf):
    """``fRING`` and ``fDAMP`` of the (2,2) mode from ``a_final`` and
    ``m_final`` (in place)."""
    wf.f_ring = qnm.fring22(wf.a_final) / wf.m_final
    wf.f_damp = qnm.fdamp22(wf.a_final) / wf.m_final
    return wf


def f_max_prime(wf, f_max):
    """``f_max_prime`` of ``IMRPhenomXSetWaveformVariables`` for a requested
    maximum frequency ``f_max`` (Hz)."""
    value = jnp.where(f_max != 0.0, f_max, wf.f_cut)
    return jnp.minimum(value, wf.f_cut)


def errors(wf):
    """Whether LAL refuses the waveform variables (a boolean array)."""
    return wf.error_mass_ratio | wf.error_spin | wf.error_final_mass | wf.error_final_spin | wf.error_meco
