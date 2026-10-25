# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Converters between the reference conventions of NRSur7dq4, with the
surrogate's own dynamics.

A binary of NRSur7dq4 is given at a reference orbital frequency ``M
omega_ref``: ``pi M f_ref`` in LAL's convention (``f_ref`` in Hz, zero for
``f_min``, and the start of the surrogate where both are zero), ``pi
reference_mf`` with the mass-scaled reference
(``WaveformOptions.reference_mf``). There the inertial frame is the
coorbital frame of the surrogate (``z`` along the principal axis of the
radiation, ``x`` from the lighter to the heavier black hole, at the orbital
phase zero), the spins are given in it, and the line of sight lies at the
inclination and at the azimuth ``pi / 2 - phi_ref``.

The converters integrate the surrogate's dynamics from the old reference
(as the model does), take the state where the orbital frequency is the new
reference, and give the spins and the line of sight in the coorbital frame
there. The time of the new reference is that of the model's own rule
(``PrecessingNRSur_get_t_ref``: the orbital-frequency fits at the nodes for
the spins at the reference), which depends on the converted spins. The
model starts its integration with an Euler step from the reference to the
nearest node, whose first-order error (up to a few per cent of the
waveform, depending on where the reference falls between the nodes) would
otherwise differ between the two references: the state at the new
reference is chosen so that this step lands on the state of the old
integration at that node. Both are found together by :data:`ITERATIONS`
fixed-point iterations (the fits depend weakly on the spins, the Euler
step is close to the identity). The two descriptions then give the same
waveform up to the difference of the two integrations from their nodes
(RK4 and Adams-Bashforth steps in different directions: up to ~1e-2
radians of orbital phase, the surrogate's own dependence on where its
reference lies) and, for
precessing binaries, a rotation of the polarization basis
(``return_angle``). The time is that of the surrogate (zero at the peak) in
both.

Pure and traceable: ``jax.jit``, ``jax.vmap`` and ``jax.grad`` work, in the
precision of the binary (the data in that precision); NaN where the
references lie outside the surrogate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..._binary import Binary
from ..._options import REFERENCE_MF
from ._data import default_data
from ._dynamics import _initialize_at_node, integrate, reference_omegas, reference_time, safe_norm
from ._spline import evaluate, locate, second_derivatives
from ._waveform import as_precision, labelled, precision

__all__ = ["to_mass_scaled_reference", "to_physical_reference", "convert", "ITERATIONS"]

#: The fixed-point iterations for the time and the state of the new
#: reference.
ITERATIONS = 6


def _rotation(quat):
    """The rotation matrix of the unit quaternion ``(w, x, y, z)``
    (``v -> q v q*``)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.stack([
        jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def _turn(v, angle):
    """``v`` turned by ``-angle`` about ``z`` (its components in a frame
    turned by ``angle``)."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])


def _renormalize(chi, norm):
    square = jnp.sum(chi * chi)
    positive = (square > 0) & (norm > 0)
    return jnp.where(positive, chi * (norm / jnp.sqrt(jnp.where(positive, square, 1.0))), chi)


def convert(binary, omega_from, omega_to, f_ref_out, data=None):
    """``binary`` given at the orbital frequency ``M omega_from`` as a binary
    given at ``M omega_to`` (with ``f_ref = f_ref_out``), and the angle
    ``psi`` by which the polarization basis turns (``h_+ + i h_x`` of the
    old binary is that of the new one times ``exp(2 i psi)``). ``omega =
    0`` is the start of the surrogate. Traced, in the precision of the
    binary."""
    dtype = precision(binary)
    data = default_data(dtype) if data is None else data
    binary = as_precision(binary, dtype)
    labels = labelled(binary)
    q, chi_a, chi_b = labels.q, labels.chi_a, labels.chi_b
    norm_a, norm_b = safe_norm(chi_a), safe_norm(chi_b)
    omegas = reference_omegas(data, q, chi_a, chi_b)
    omega_from = jnp.asarray(omega_from, dtype)
    omega_to = jnp.asarray(omega_to, dtype)
    states, valid = integrate(data, q, chi_a, chi_b, omega_from, omegas)
    knots = data.ds_knots
    c = second_derivatives(knots, states)

    def state_at(t):
        t = jnp.reshape(t, (1,))
        index = locate(knots.x, t)
        return evaluate(knots, states, c, index, t - knots.x[index])[0]

    def spins_at(y):
        return _renormalize(_turn(y[5:8], y[4]), norm_a), _renormalize(_turn(y[8:11], y[4]), norm_b)

    t, valid_to = reference_time(data, omegas, omega_to)
    y = state_at(t)
    for _ in range(ITERATIONS):
        new_a, new_b = spins_at(y)
        t, valid_to = reference_time(data, reference_omegas(data, q, new_a, new_b), omega_to)
        # the state whose Euler step to the nearest node (the model's
        # start) lands on the state of this integration there
        node, stepped = _initialize_at_node(data, q, t, y, norm_a, norm_b)
        y = y - (stepped - states[node])
    new_a, new_b = spins_at(y)
    quat = y[:4] / jnp.sqrt(jnp.sum(y[:4] * y[:4]))
    # the new frame (columns: its axes in the old one), in the labelled
    # frame, which is the binary's turned by pi about z after a switch
    frame = _rotation(quat)
    c_phi, s_phi = jnp.cos(y[4]), jnp.sin(y[4])
    zero, one = jnp.zeros((), dtype), jnp.ones((), dtype)
    frame = frame @ jnp.stack([jnp.stack([c_phi, -s_phi, zero]), jnp.stack([s_phi, c_phi, zero]),
                               jnp.stack([zero, zero, one])])
    flip = jnp.where(labels.switched, jnp.asarray([-1.0, -1.0, 1.0], dtype), jnp.ones(3, dtype))
    inc = jnp.asarray(binary.inclination, dtype)
    phi = jnp.asarray(binary.phi_ref, dtype)
    s = jnp.sin(inc)
    n = jnp.stack([s * jnp.sin(phi), s * jnp.cos(phi), jnp.cos(inc)])
    nx, ny, nz = flip * (frame.T @ (flip * n))
    perp_sq = nx * nx + ny * ny
    has_perp = perp_sq > 0.0
    perp = jnp.sqrt(jnp.where(has_perp, perp_sq, 1.0))
    inclination = jnp.arctan2(jnp.where(has_perp, perp, 0.0), nz)
    azimuth = jnp.arctan2(jnp.where(has_perp, ny, 1.0), jnp.where(has_perp, nx, 0.0))
    # (along z, the azimuth is free: phi_ref turned with the frame)
    turn = jnp.arctan2(frame[1, 0], frame[0, 0])
    phi_ref = jnp.mod(jnp.where(has_perp, 0.5 * jnp.pi - azimuth, phi + turn), 2.0 * jnp.pi)
    # the rotation of the polarizations: the angle about N from the
    # projection of the old z axis to that of the new one
    z_new = flip * frame[:, 2]
    z_old = jnp.stack([zero, zero, one])

    def projected(v):
        return v - jnp.sum(v * n) * n

    p_old, p_new = projected(z_old), projected(z_new)
    psi = jnp.arctan2(jnp.sum(n * jnp.cross(p_old, p_new)), jnp.sum(p_old * p_new))
    spin_1 = jnp.where(labels.switched, flip * new_b, new_a)
    spin_2 = jnp.where(labels.switched, flip * new_a, new_b)
    ok = valid & valid_to
    out = Binary(mass_1=binary.mass_1, mass_2=binary.mass_2, spin_1=jnp.where(ok, spin_1, jnp.nan),
                 spin_2=jnp.where(ok, spin_2, jnp.nan), luminosity_distance=binary.luminosity_distance,
                 inclination=jnp.where(ok, inclination, jnp.nan), phi_ref=jnp.where(ok, phi_ref, jnp.nan),
                 f_ref=jnp.asarray(f_ref_out, dtype))
    return out, jnp.where(ok, psi, jnp.nan)


def _omega(binary, f):
    """``pi M f`` (``f`` in Hz)."""
    return jnp.pi * labelled(binary).m_tot_sec * f


def to_mass_scaled_reference(binary, reference_mf=REFERENCE_MF, f_min=None, return_angle=False, data=None):
    """``binary`` given at its reference frequency ``binary.f_ref`` (Hz,
    LAL's convention: zero for ``f_min``, and for the start of the
    surrogate if ``f_min`` is zero or not given) as a binary given at the
    mass-scaled reference ``pi reference_mf`` (``f_ref`` zero), for
    ``WaveformOptions(reference_mf=reference_mf)``. With ``return_angle``,
    also the angle ``psi`` by which the polarization basis turns (see
    :func:`convert`). Pure; in the precision of the binary."""
    dtype = precision(binary)
    f = jnp.asarray(binary.f_ref, dtype)
    if f_min is not None:
        f = jnp.where(f == 0.0, jnp.asarray(f_min, dtype), f)
    out, psi = convert(binary, _omega(binary, f), jnp.pi * reference_mf, 0.0, data)
    return (out, psi) if return_angle else out


def to_physical_reference(binary, f_ref, reference_mf=REFERENCE_MF, return_angle=False, data=None):
    """The inverse of :func:`to_mass_scaled_reference`: ``binary`` given at
    the mass-scaled reference ``pi reference_mf`` (its ``f_ref`` ignored)
    as a binary given at ``f_ref`` (Hz, LAL's convention; zero for the
    start of the surrogate)."""
    dtype = precision(binary)
    f = jnp.asarray(f_ref, dtype)
    out, psi = convert(binary, jnp.pi * reference_mf, _omega(binary, f), f, data)
    return (out, psi) if return_angle else out
