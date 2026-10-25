# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Converters between the reference conventions of IMRPhenomXPNR.

LAL's convention gives the spins, the inclination and ``phi_ref`` of a
binary at a reference frequency ``f_ref`` in Hz (``Binary.f_ref``); the
mass-scaled one (``WaveformOptions.reference_mf``) at ``f_ref =
reference_mf / M``. A binary is converted by evolving its spins and orbital
angular momentum between the two frequencies with the precession equations
of the model (the orbit-averaged SpinTaylorT4 equations of
:mod:`.spintaylor.dynamics`, integrated in ``u = 1 / (M omega)`` with a fixed
number of DOP853 steps).

What a binary describes in either convention: the spins and the orbital
angular momentum ``L`` at the reference (the spins in the frame of ``L``
whose x axis is the orbital separation there), and the line of sight
``N``, through the inclination (the angle between ``N`` and ``L``) and
``phi_ref`` (``N`` lies at the azimuth ``pi / 2 - phi_ref`` in that
frame). The converters keep ``N``, the spins and ``L`` (to the integration
error). The frame at the new reference is the one the equations carry
along (the vector ``E1`` of the SpinTaylor equations) turned about ``L`` by
the orbital phase between the two references: the orbital phase by which
the model's alignment of the (2,2) phase at the reference moves
(:func:`orbital_phase`, from IMRPhenomXAS). For aligned spins the two
descriptions then give the same waveform, and a round trip returns the
binary. The orbital phase is thousands of radians for light binaries, so
convert in float64.

For precessing binaries the waveforms of the two descriptions differ by a
rotation of the polarizations (their basis follows ``L`` at the reference;
``return_angle=True`` gives it) and by the model's own dependence on the
reference: IMRPhenomXPNR takes the aligned spins of the co-precessing
modes, ``J`` and the parameters of its fits at the reference, so LAL
itself at two references differs by up to ~1e-2 in mismatch for strongly
precessing binaries. Given the converted binary at ``f_ref =
reference_mf / M``, LAL agrees with the model with the mass-scaled
reference as closely as with the model in its own convention.

The model with the mass-scaled reference is invariant: whatever its
``reference_mf``, it carries the binary to the canonical reference
``REFERENCE_MF`` (:func:`to_canonical`) and takes everything that depends
on the reference there, its polarization basis included. Conversions
between mass-scaled references pass through the canonical one, with the
orbital phase of both legs from the aligned spins there, so that they
compose, and a binary converted between two references gives the same
waveform to the integration error.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..._binary import Binary
from ..._options import REFERENCE_MF
from ._generated import dop853
from ._math import MTSUN_SI, PI, mass_unit, powers, precision, precision_of, real
from .internals import waveform_struct
from .phase22 import phase_22
from .spintaylor import dynamics
from .spintaylor.angles import STEP_ANGLE
from .xas import xas_setup

__all__ = ["to_mass_scaled_reference", "to_physical_reference", "to_canonical", "orbital_phase", "STEPS"]

#: The sign of the orbital phase in ``phi_ref`` (see :func:`_convert`).
PHASE_SIGN = 1.0

#: The DOP853 steps between the two references: ~0.15 rad of precession of
#: ``L`` per step for a binary of 2 solar masses from 10 Hz.
STEPS = 4096


def _rhs_u(c, u, y):
    """``dy/du`` of the SpinTaylor state at ``u = 1 / (M omega)`` (with ``M
    omega = 1 / u``)."""
    y = y.at[1].set(1.0 / u)
    dydt = dynamics.derivatives(c, y)[0]
    # du/dt = -(d omega / dt) u^2
    return dydt * (-1.0 / (u * u * dydt[1]))


def _dop853_step(c, u, y, h):
    k = [_rhs_u(c, u, y)]
    for s in range(1, dop853.N_STAGES):
        dy = sum(k[j] * a for j, a in dop853.A[s]) * h
        k.append(_rhs_u(c, u + dop853.C[s] * h, y + dy))
    return y + h * sum(k[j] * b for j, b in dop853.B)


def _unit(v):
    return v / jnp.sqrt(jnp.sum(v * v))


def orbital_phase(binary, mf_from, mf_to):
    """Half the change of the phase of the (2,2) mode of IMRPhenomXAS
    (without its alignment) from ``mf_from`` to ``mf_to``, for the bodies
    and the aligned spins of ``binary``: the orbital phase by which the
    model's alignment of the phase at the reference moves with it."""
    m1 = jnp.asarray(binary.mass_1, real())
    m2 = jnp.asarray(binary.mass_2, real())
    chi1 = jnp.asarray(binary.spin_1, real())[2]
    chi2 = jnp.asarray(binary.spin_2, real())[2]
    keep = m1 > m2
    m1, m2, chi1, chi2 = (jnp.where(keep, m1, m2), jnp.where(keep, m2, m1), jnp.where(keep, chi1, chi2),
                          jnp.where(keep, chi2, chi1))
    m_sec = (m1 + m2) * MTSUN_SI
    wf = waveform_struct(m1 * mass_unit(), m2 * mass_unit(), chi1, chi2, 0.0, mf_from / m_sec, 0.0, mf_from / m_sec,
                         mf_to / m_sec, 1.0, 0.0)
    xas = xas_setup(wf)

    def phase(mf):
        return xas.inveta * phase_22(mf, powers(mf), xas.phase) + xas.linb * mf

    return 0.5 * (phase(mf_to) - phase(mf_from))


def _convert(binary, f_from, f_to, f_ref_out, steps, phase_binary):
    """``binary`` given at ``f_from`` (Hz) as a binary given at ``f_to``
    (with ``f_ref = f_ref_out``) after ``steps`` (static) DOP853 steps, and
    the rotation of the polarizations; the orbital phase between the two
    is that of the aligned spins of ``phase_binary`` (at ``f_to`` for
    ``None``)."""
    m1 = jnp.asarray(binary.mass_1, real())
    m2 = jnp.asarray(binary.mass_2, real())
    chi1 = jnp.asarray(binary.spin_1, real())
    chi2 = jnp.asarray(binary.spin_2, real())
    inc = jnp.asarray(binary.inclination, real())
    phi = jnp.asarray(binary.phi_ref, real())
    m_tot = m1 + m2
    msec = m_tot * MTSUN_SI
    norm1, norm2 = (m1 / m_tot) ** 2, (m2 / m_tot) ** 2
    zero, one = jnp.zeros((), real()), jnp.ones((), real())
    u_from = 1.0 / (PI * msec * f_from)
    u_to = 1.0 / (PI * msec * f_to)
    y = jnp.stack([zero, 1.0 / u_from, zero, zero, one, *(norm1 * chi1), *(norm2 * chi2), one, zero, zero])
    c = dynamics.coefficients(m1, m2, f_from, f_to)
    h = (u_to - u_from) / steps

    def step(i, y):
        return _dop853_step(c, u_from + i * h, y, h)

    y = jax.lax.fori_loop(0, steps, step, y)
    # the frame at the new reference: L, the x axis E1 (made orthogonal to
    # L) turned about L by the orbital phase between the two references,
    # and y = L x x, in the frame at the old reference. The orbital phase is
    # that by which the model's alignment of the (2,2) phase at the
    # reference moves (:func:`orbital_phase`), for the aligned spins at the
    # mass-scaled reference both ways (they do not depend on the turn), so
    # that a round trip returns the binary.
    lz = _unit(y[2:5])
    e1 = y[11:14]
    e1 = _unit(e1 - jnp.sum(e1 * lz) * lz)
    e2 = jnp.cross(lz, e1)
    s1z = jnp.sum(y[5:8] * lz) / norm1
    s2z = jnp.sum(y[8:11] * lz) / norm2
    if phase_binary is None:
        phase_binary = binary._replace(spin_1=jnp.stack([zero, zero, s1z]), spin_2=jnp.stack([zero, zero, s2z]))
    turn = PHASE_SIGN * orbital_phase(phase_binary, f_from * msec, f_to * msec)
    ex = jnp.cos(turn) * e1 + jnp.sin(turn) * e2
    ey = jnp.cross(lz, ex)

    def components(v):
        return jnp.stack([jnp.sum(v * ex), jnp.sum(v * ey), jnp.sum(v * lz)])

    spin_1 = components(y[5:8]) / norm1
    spin_2 = components(y[8:11]) / norm2
    # the line of sight, at the azimuth pi/2 - phi_ref
    s = jnp.sin(inc)
    n = jnp.stack([s * jnp.sin(phi), s * jnp.cos(phi), jnp.cos(inc)])
    nx, ny, nz = components(n)
    perp_sq = nx * nx + ny * ny
    has_perp = perp_sq > 0.0
    perp = jnp.sqrt(jnp.where(has_perp, perp_sq, 1.0))
    inclination = jnp.arctan2(jnp.where(has_perp, perp, 0.0), nz)
    # (along L, the azimuth is free: phi_ref turned with the frame, which
    # is exact for aligned spins, whose L does not move)
    azimuth = jnp.arctan2(jnp.where(has_perp, ny, 1.0), jnp.where(has_perp, nx, 0.0))
    phi_ref = jnp.mod(jnp.where(has_perp, 0.5 * PI - azimuth, phi + turn), 2.0 * PI)
    # the rotation of the polarizations: the angle about N from the
    # projection of the old L to that of the new one
    z = jnp.stack([zero, zero, one])

    def projected(v):
        return v - jnp.sum(v * n) * n

    p_old, p_new = projected(z), projected(lz)
    psi = jnp.arctan2(jnp.sum(n * jnp.cross(p_old, p_new)), jnp.sum(p_old * p_new))
    out = Binary(mass_1=binary.mass_1, mass_2=binary.mass_2, spin_1=spin_1.astype(real()),
                 spin_2=spin_2.astype(real()), luminosity_distance=binary.luminosity_distance,
                 inclination=inclination, phi_ref=phi_ref, f_ref=jnp.asarray(f_ref_out, real()))
    return out, psi


def _mass_scaled(binary, mf):
    m_tot = jnp.asarray(binary.mass_1, real()) + jnp.asarray(binary.mass_2, real())
    return mf / (m_tot * MTSUN_SI)


def canonical_steps(reference_mf):
    """The DOP853 steps of :func:`to_canonical` from ``reference_mf``: the
    steps in ``u = 1 / (M omega)`` of the SpinTaylor solution
    (:data:`.spintaylor.angles.STEP_ANGLE` of the precession of ``L`` per
    step, for equal masses, the fastest) between the two references, at
    least :data:`MIN_CANONICAL_STEPS`. Static: both references are numbers
    in units of the total mass."""
    du = abs(1.0 / (math.pi * reference_mf) - 1.0 / (math.pi * REFERENCE_MF))
    rate = 5.0 * (2.0 + 1.5) / 96.0
    return max(MIN_CANONICAL_STEPS, math.ceil(du * rate / STEP_ANGLE))


#: The fewest steps of :func:`to_canonical`.
MIN_CANONICAL_STEPS = 4


def to_canonical(binary, reference_mf):
    """``binary`` given at the mass-scaled reference ``reference_mf / M`` as
    a binary given at the canonical reference ``REFERENCE_MF / M``, where
    the model with the mass-scaled reference takes everything that
    depends on the reference (the aligned spins of the co-precessing
    modes, ``J``, the parameters of the PNR fits, the offsets of the
    angles, the alignment of the phase, the line of sight and the basis of
    the polarizations): the waveform then depends on ``reference_mf`` only
    through the integration error. ``binary`` is returned unchanged for
    ``reference_mf = REFERENCE_MF``; ``f_ref`` is kept (it must be zero).
    The steps are :func:`canonical_steps` (static). Called inside the
    model (traced, in its precision)."""
    if reference_mf == REFERENCE_MF:
        return binary
    out, _ = _convert(binary, _mass_scaled(binary, reference_mf), _mass_scaled(binary, REFERENCE_MF),
                      binary.f_ref, canonical_steps(reference_mf), None)
    return out


def _to_canonical_angle(binary, reference_mf, steps):
    """``binary`` at ``reference_mf`` at the canonical reference and the
    rotation of the polarizations (none for the canonical reference)."""
    if reference_mf == REFERENCE_MF:
        return binary, jnp.zeros((), real())
    return _convert(binary, _mass_scaled(binary, reference_mf), _mass_scaled(binary, REFERENCE_MF), 0.0, steps,
                    None)


def to_mass_scaled_reference(binary, reference_mf=REFERENCE_MF, steps=STEPS, return_angle=False):
    """``binary`` given at its reference frequency ``binary.f_ref`` (Hz,
    LAL's convention; it must be positive) as a binary given at the
    mass-scaled reference ``reference_mf / M`` (``f_ref`` zero), for the
    model with ``WaveformOptions(reference_mf=reference_mf)``: the same spins, orbital angular momentum
    and line of sight (see the module). With ``return_angle``, also the
    angle ``psi`` by which the polarization basis turns: the polarizations
    of the two binaries are related, up to time and phase shifts, by
    ``h_+ + i h_x -> (h_+ + i h_x) exp(2 i psi)``.

    The conversion passes through the canonical reference ``REFERENCE_MF /
    M`` (:func:`to_canonical`), whose aligned spins give the orbital phase
    of both legs, so that conversions between mass-scaled references
    compose: the model gives the same waveform for a binary at any
    two of them.

    Pure: ``jax.jit``, ``jax.vmap`` and ``jax.grad`` work, in float32 for a
    binary of float32 parameters. In float32 the orbital phase between the
    references (thousands of radians for light binaries) costs ~1e-4 in
    the spins and angles: convert in float64. NaN for a zero or negative ``f_ref``; ``steps`` (static)
    is the number of steps of each leg of the integration."""
    with precision(precision_of(binary)):
        f_from = jnp.asarray(binary.f_ref, real())
        f_from = jnp.where(f_from > 0.0, f_from, jnp.nan)
        out, psi = _convert(binary, f_from, _mass_scaled(binary, REFERENCE_MF), 0.0, steps, None)
        if reference_mf != REFERENCE_MF:
            out, psi2 = _convert(out, _mass_scaled(binary, REFERENCE_MF), _mass_scaled(binary, reference_mf), 0.0,
                                 steps, out)
            psi = psi + psi2
    return (out, psi) if return_angle else out


def to_physical_reference(binary, f_ref, reference_mf=REFERENCE_MF, steps=STEPS, return_angle=False):
    """The inverse of :func:`to_mass_scaled_reference`: ``binary`` given at
    the mass-scaled reference ``reference_mf / M`` (its ``f_ref`` ignored)
    as a binary given at the reference frequency ``f_ref`` (Hz, LAL's
    convention), with ``return_angle`` the rotation of the polarizations
    as there."""
    with precision(precision_of(binary)):
        f_to = jnp.asarray(f_ref, real())
        canon, psi = _to_canonical_angle(binary, reference_mf, steps)
        out, psi2 = _convert(canon, _mass_scaled(binary, REFERENCE_MF), f_to, f_to, steps, canon)
    return (out, psi + psi2) if return_angle else out
