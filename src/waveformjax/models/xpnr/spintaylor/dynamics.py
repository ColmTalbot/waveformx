# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The orbit-averaged SpinTaylorT4 equations of LALSimulation
(``XLALSimInspiralSpinTaylorT4Setup``, ``XLALSimInspiralSpinTaylorT4DerivativesAvg``,
``XLALSimInspiralSpinDerivativesAvg`` and ``XLALSimInspiralSpinTaylorStoppingTest``
of ``LALSimInspiralSpinTaylor.c``), after waveformrs's
``spintaylor/dynamics.rs``, for black holes with the PN orders that
IMRPhenomXPHM enforces (``phaseO = 7``, ``spinO = 6``, ``tideO = 12`` with
zero deformabilities, ``lscorr = 0``).

The state is LAL's ``[phi, M omega, LNhat (3), S1 (3), S2 (3), E1 (3)]``
with the spins normalised by the total mass, ``S_i = (m_i / M)^2 chi_i``,
and time in units of the total mass.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .._generated import spintaylor_coefficients as _pn
from .._math import MTSUN_SI, PI, REAL4_EPS, FencedModule, Struct, fenced, fma, fnms, mass_unit

pn = FencedModule(_pn)

__all__ = ["coefficients", "derivatives", "stopping_test", "NUM_VARIABLES", "ST_TEST_ENERGY",
           "ST_TEST_OMEGADOUBLEDOT", "ST_TEST_OMEGANAN", "ST_TEST_FREQBOUND", "ST_TEST_LARGEV",
           "ST_DERIVATIVE_OMEGANONPOS"]

NUM_VARIABLES = 14

ST_TEST_ENERGY = 1025
ST_TEST_OMEGADOUBLEDOT = 1026
ST_TEST_OMEGANAN = 1028
ST_TEST_FREQBOUND = 1029
ST_DERIVATIVE_OMEGANONPOS = 1030
ST_TEST_LARGEV = 1031


@fenced
def coefficients(m1_si, m2_si, f_start, f_end):
    """``XLALSimInspiralSpinTaylorT4Setup`` (with
    ``XLALSimSpinTaylorEnergySpinDerivativeSetup``) for black holes:
    the coefficients of the equations of motion and of the stopping test.
    Masses in kg, frequencies (of the gravitational wave) in Hz."""
    m1 = m1_si / mass_unit()
    m2 = m2_si / mass_unit()
    total_mass = m1 + m2
    m1m = m1 / total_mass
    m2m = m2 / total_mass
    eta = m1m * m2m
    c = Struct(
        total_mass=total_mass,
        eta=eta,
        m1_by_m=m1m,
        m2_by_m=m2m,
        f_start=f_start,
        f_end=f_end,
        ecoeff=[1.0, 0.0, pn.pn_energy_2pn(eta), 0.0, pn.pn_energy_4pn(eta), 0.0, pn.pn_energy_6pn(eta), 0.0],
        wdotnewt=pn.taylor_t4_wdot_0pn(eta),
        wdotcoeff=[
            1.0, 0.0, pn.taylor_t4_wdot_2pn(eta), pn.taylor_t4_wdot_3pn(eta), pn.taylor_t4_wdot_4pn(eta),
            pn.taylor_t4_wdot_5pn(eta), pn.taylor_t4_wdot_6pn(eta), pn.taylor_t4_wdot_7pn(eta),
        ],
        wdotlogcoeff=pn.taylor_t4_wdot_6pn_log(eta),
        c_s1=pn.l_3pn_si_coeff_avg(m1m),
        c_s1l=pn.l_3pn_sil_coeff_avg(m1m),
        c_s2=pn.l_3pn_si_coeff_avg(m2m),
        c_s2l=pn.l_3pn_sil_coeff_avg(m2m),
        l1pn=pn.l_2pn(eta),
        omega_start=PI * total_mass * MTSUN_SI * f_start,
        omega_end=PI * total_mass * MTSUN_SI * f_end,
        # spin-dependent energy and spin-precession coefficients
        e6s1s2_avg=pn.pn_energy_6pn_s1s2_avg(eta),
        e6s1os2o_avg=pn.pn_energy_6pn_s1os2o_avg(eta),
        e6s1s1_sum=pn.pn_energy_6pn_s1s1_avg(m1m) + pn.pn_energy_6pn_qms1s1_avg(m1m),
        e6s2s2_sum=pn.pn_energy_6pn_s1s1_avg(m2m) + pn.pn_energy_6pn_qms1s1_avg(m2m),
        e6s1os1o_sum=pn.pn_energy_6pn_s1os1o_avg(m1m) + pn.pn_energy_6pn_qms1os1o_avg(m1m),
        e6s2os2o_sum=pn.pn_energy_6pn_s1os1o_avg(m2m) + pn.pn_energy_6pn_qms1os1o_avg(m2m),
        s1dot6s2_avg=pn.spin_dot_6pn_s2_avg(m1m),
        s2dot6s1_avg=pn.spin_dot_6pn_s2_avg(m2m),
        s1dot6s2o_avg=pn.spin_dot_6pn_s2o_avg(m1m),
        s1dot6s1o_avg=pn.spin_dot_6pn_s1o_avg(m1m),
        s2dot6s1o_avg=pn.spin_dot_6pn_s2o_avg(m2m),
        s2dot6s2o_avg=pn.spin_dot_6pn_s1o_avg(m2m),
        s1dot6qms1o_avg=pn.spin_dot_6pn_qmso_avg(m1m),
        s2dot6qms2o_avg=pn.spin_dot_6pn_qmso_avg(m2m),
        omegashift_s1=pn.ldot_3pn_so(m1m),
        omegashift_s2=pn.ldot_3pn_so(m2m),
        e5s1o=pn.pn_energy_5pn_so(m1m),
        e5s2o=pn.pn_energy_5pn_so(m2m),
        s1dot5=pn.spin_dot_5pn(m1m),
        s2dot5=pn.spin_dot_5pn(m2m),
        e4s1s2_avg=pn.pn_energy_4pn_s1s2_avg(eta),
        e4s1os2o_avg=pn.pn_energy_4pn_s1os2o_avg(eta),
        e4qms1s1_avg=pn.pn_energy_4pn_qms1s1_avg(m1m),
        e4qms1os1o_avg=pn.pn_energy_4pn_qms1os1o_avg(m1m),
        e4qms2s2_avg=pn.pn_energy_4pn_qms1s1_avg(m2m),
        e4qms2os2o_avg=pn.pn_energy_4pn_qms1os1o_avg(m2m),
        s1dot4s2_avg=pn.SPIN_DOT_4PN_S2_AVG,
        s1dot4s2o_avg=pn.SPIN_DOT_4PN_S2O_AVG,
        s1dot4qms1o_avg=pn.spin_dot_4pn_qmso_avg(m1m),
        s2dot4qms2o_avg=pn.spin_dot_4pn_qmso_avg(m2m),
        e3s1o=pn.pn_energy_3pn_so(m1m),
        e3s2o=pn.pn_energy_3pn_so(m2m),
        s1dot3=pn.spin_dot_3pn(m1m),
        s2dot3=pn.spin_dot_3pn(m2m),
        wdot6s1o=pn.taylor_t4_wdot_6pn_so(m1m),
        wdot6s2o=pn.taylor_t4_wdot_6pn_so(m2m),
        wdot6s1s2_avg=pn.taylor_t4_wdot_6pn_s1s2_avg(eta),
        wdot6s1os2o_avg=pn.taylor_t4_wdot_6pn_s1os2o_avg(eta),
        wdot6s1s1_sum=pn.taylor_t4_wdot_6pn_s1s1_avg(m1m) + pn.taylor_t4_wdot_6pn_qms1s1_avg(m1m),
        wdot6s2s2_sum=pn.taylor_t4_wdot_6pn_s1s1_avg(m2m) + pn.taylor_t4_wdot_6pn_qms1s1_avg(m2m),
        wdot6s1os1o_sum=pn.taylor_t4_wdot_6pn_s1os1o_avg(m1m) + pn.taylor_t4_wdot_6pn_qms1os1o_avg(m1m),
        wdot6s2os2o_sum=pn.taylor_t4_wdot_6pn_s1os1o_avg(m2m) + pn.taylor_t4_wdot_6pn_qms1os1o_avg(m2m),
        wdot5s1o=pn.taylor_t4_wdot_5pn_so(m1m),
        wdot5s2o=pn.taylor_t4_wdot_5pn_so(m2m),
        wdot4s1s2_avg=pn.taylor_t4_wdot_4pn_s1s2_avg(eta),
        wdot4s1os2o_avg=pn.taylor_t4_wdot_4pn_s1os2o_avg(eta),
        wdot4s1s1_sum=pn.taylor_t4_wdot_4pn_s1s1_avg(m1m) + pn.taylor_t4_wdot_4pn_qms1s1_avg(m1m),
        wdot4s1os1o_sum=pn.taylor_t4_wdot_4pn_s1os1o_avg(m1m) + pn.taylor_t4_wdot_4pn_qms1os1o_avg(m1m),
        wdot4s2s2_sum=pn.taylor_t4_wdot_4pn_s1s1_avg(m2m) + pn.taylor_t4_wdot_4pn_qms1s1_avg(m2m),
        wdot4s2os2o_sum=pn.taylor_t4_wdot_4pn_s1os1o_avg(m2m) + pn.taylor_t4_wdot_4pn_qms1os1o_avg(m2m),
        wdot3s1o=pn.taylor_t4_wdot_3pn_so(m1m),
        wdot3s2o=pn.taylor_t4_wdot_3pn_so(m2m),
    )
    return c


# (a gather with constant indices, which XLA's CPU backend fuses into the
# arithmetic that uses it, unlike a concatenation of slices)
_YZX = np.array([1, 2, 0])
_ZXY = np.array([2, 0, 1])


def _yzx(a):
    """The components ``(y, z, x)`` of the 3-vectors ``a`` (last axis)."""
    return a[..., _YZX]


def _zxy(a):
    """The components ``(z, x, y)`` of the 3-vectors ``a`` (last axis)."""
    return a[..., _ZXY]


def _cross(a, b):
    """``a x b`` of 3-vectors (last axis), as LAL writes it componentwise."""
    return fma(_yzx(a), _zxy(b), -(_zxy(a) * _yzx(b)))


def _scalar(x):
    """A scalar per 3-vector, broadcast against the vectors' last axis."""
    return x[..., None]


def _spin_derivatives(c, v, lnh, e1, s1, s2, lnh_dot_s1, lnh_dot_s2):
    """``XLALSimInspiralSpinDerivativesAvg``: the derivatives of LNhat, S1,
    S2 and E1 (all PN orders up to spinO = 6, lscorr = 0). The vectors are
    arrays whose last axis holds the three components (each component is
    computed with LAL's expression; as arrays, XLA computes the shared
    scalars once rather than once per component)."""
    eta = c.eta
    ln0mag = eta / v
    v2 = v * v
    omega = v2 * v
    v5 = omega * v2
    omega2 = omega * omega
    v7 = omega2 * v
    v8 = omega2 * v2
    v5, v7, v8, omega2, lnh_dot_s1, lnh_dot_s2 = map(_scalar, (v5, v7, v8, omega2, lnh_dot_s1, lnh_dot_s2))

    lnh_x_s1 = _cross(lnh, s1)
    lnh_x_s2 = _cross(lnh, s2)
    s1_x_s2 = _cross(s1, s2)
    ds1 = c.s1dot3 * v5 * lnh_x_s1
    ds2 = c.s2dot3 * v5 * lnh_x_s2
    dlnhat = -(ds1 + ds2)
    # 2PN
    ds1_nl = omega2 * fma(-c.s1dot4s2_avg, s1_x_s2, c.s1dot4s2o_avg * lnh_dot_s2 * lnh_x_s1)
    ds2_nl = omega2 * fma(c.s1dot4s2_avg, s1_x_s2, c.s1dot4s2o_avg * lnh_dot_s1 * lnh_x_s2)
    ds1_nl = fma(omega2 * c.s1dot4qms1o_avg * lnh_dot_s1, lnh_x_s1, ds1_nl)
    ds2_nl = fma(omega2 * c.s2dot4qms2o_avg * lnh_dot_s2, lnh_x_s2, ds2_nl)
    ds1 = ds1 + ds1_nl
    ds2 = ds2 + ds2_nl
    dlnhat = dlnhat + -(ds1_nl + ds2_nl)
    # 2.5PN
    lnmag = fma(ln0mag * v2, c.l1pn, ln0mag)
    ds1_nnl = c.s1dot5 * v7 * lnh_x_s1
    ds2_nnl = c.s2dot5 * v7 * lnh_x_s2
    ds1 = ds1 + ds1_nnl
    ds2 = ds2 + ds2_nnl
    dlnhat = dlnhat - (ds1_nnl + ds2_nnl)
    # 3PN
    q1 = fma(c.s1dot6s1o_avg, lnh_dot_s1, c.s1dot6s2o_avg * lnh_dot_s2)
    q2 = fma(c.s2dot6s1o_avg, lnh_dot_s1, c.s2dot6s2o_avg * lnh_dot_s2)
    ds1_n3l = v8 * fma(-c.s1dot6s2_avg, s1_x_s2, q1 * lnh_x_s1)
    ds2_n3l = v8 * fma(c.s2dot6s1_avg, s1_x_s2, q2 * lnh_x_s2)
    ds1_n3l = fma(v8, c.s1dot6qms1o_avg * lnh_dot_s1 * lnh_x_s1, ds1_n3l)
    ds2_n3l = fma(v8, c.s2dot6qms2o_avg * lnh_dot_s2 * lnh_x_s2, ds2_n3l)
    ds1 = ds1 + ds1_n3l
    ds2 = ds2 + ds2_n3l
    dlnhat = dlnhat - (ds1_n3l + ds2_n3l)

    dlnhat = dlnhat / _scalar(lnmag)
    om = _cross(lnh, dlnhat)
    # [fma(-om_z, l_y, om_y l_z), fma(-om_x, l_z, om_z l_x), fma(-om_y, l_x, om_x l_y)]
    dlnh = fma(-_zxy(om), _yzx(lnh), _yzx(om) * _zxy(lnh))
    de1 = fma(-_zxy(om), _yzx(e1), _yzx(om) * _zxy(e1))
    return dlnh, ds1, ds2, de1


def _dot(a, b):
    """``a . b`` of 3-vectors (last axis), in LAL's order."""
    return fma(a[..., 2], b[..., 2], fma(a[..., 0], b[..., 0], a[..., 1] * b[..., 1]))


def derivatives(c, y, safe=True):
    """``XLALSimInspiralSpinTaylorT4DerivativesAvg``: ``dy/dt`` (time in
    units of the total mass) for the state ``y`` (an array whose last axis
    has the 14 variables). Returns ``(dydt, ok)`` with ``ok = omega > 0``
    (LAL fails otherwise); with ``safe`` the derivatives are evaluated at
    ``|omega|`` so that they stay finite."""
    omega = y[..., 1]
    ok = omega > 0.0
    omega_eval = jnp.where(ok, omega, jnp.abs(omega) + 1e-300) if safe else omega
    lnh = y[..., 2:5]
    s1 = y[..., 5:8]
    s2 = y[..., 8:11]
    e1 = y[..., 11:14]
    v = jnp.cbrt(omega_eval)
    v2 = v * v
    v11 = omega_eval * omega_eval * omega_eval * v2
    lnh_dot_s1 = _dot(lnh, s1)
    lnh_dot_s2 = _dot(lnh, s2)
    s1_dot_s2 = _dot(s1, s2)
    s1sq = _dot(s1, s1)
    s2sq = _dot(s2, s2)

    w = fma(c.wdot6s1o, lnh_dot_s1, c.wdot6s2o * lnh_dot_s2)
    w = fma(c.wdot6s1os2o_avg * lnh_dot_s1, lnh_dot_s2, w)
    w = fma(c.wdot6s1s2_avg, s1_dot_s2, w)
    w = fma(c.wdot6s1s1_sum, s1sq, w)
    w = fma(c.wdot6s2s2_sum, s2sq, w)
    w = fma(c.wdot6s1os1o_sum * lnh_dot_s1, lnh_dot_s1, w)
    wspin6 = fma(c.wdot6s2os2o_sum * lnh_dot_s2, lnh_dot_s2, w)
    wspin5 = fma(c.wdot5s1o, lnh_dot_s1, c.wdot5s2o * lnh_dot_s2)
    w = fma(c.wdot4s1s2_avg, s1_dot_s2, c.wdot4s1os2o_avg * lnh_dot_s1 * lnh_dot_s2)
    x = fma(c.wdot4s1s1_sum, s1sq, c.wdot4s2s2_sum * s2sq)
    x = fma(c.wdot4s1os1o_sum * lnh_dot_s1, lnh_dot_s1, x)
    x = fma(c.wdot4s2os2o_sum * lnh_dot_s2, lnh_dot_s2, x)
    wspin4 = w + x
    wspin3 = fma(c.wdot3s1o, lnh_dot_s1, c.wdot3s2o * lnh_dot_s2)

    k = c.wdotcoeff
    p = k[7]  # the tidal terms vanish for black holes
    p = fma(v, p, fma(c.wdotlogcoeff, jnp.log(v), k[6] + wspin6))
    p = fma(v, p, k[5] + wspin5)
    p = fma(v, p, k[4] + wspin4)
    p = fma(v, p, k[3] + wspin3)
    p = fma(v, p, k[2])
    p = fma(v, p, k[1])
    p = fma(v, p, k[0])
    domega = c.wdotnewt * v11 * p

    dlnh, ds1, ds2, de1 = _spin_derivatives(c, v, lnh, e1, s1, s2, lnh_dot_s1, lnh_dot_s2)

    a1 = fnms(lnh_dot_s1, lnh_dot_s1, s1sq)
    a2 = fnms(lnh_dot_s2, lnh_dot_s2, s2sq)
    a3 = fnms(lnh_dot_s1, lnh_dot_s2, s1_dot_s2)
    oms1, oms2 = c.omegashift_s1, c.omegashift_s2
    shift = -0.25 * fma(2.0 * oms1 * oms2, a3, fma(oms1 * oms1, a1, oms2 * oms2 * a2))
    dphi = omega_eval * fma(omega_eval * omega_eval, shift, 1.0)
    dydt = jnp.concatenate([jnp.stack([dphi, domega], axis=-1), dlnh, ds1, ds2, de1], axis=-1)
    return dydt, ok


def energy_test(c, y):
    """The PN energy test of ``XLALSimInspiralSpinTaylorStoppingTest``
    (``test < 0``: the energy stopped decreasing)."""
    omega = y[..., 1]
    v = jnp.cbrt(jnp.abs(omega))
    lnh = y[..., 2:5]
    s1 = y[..., 5:8]
    s2 = y[..., 8:11]
    lnh_dot_s1 = _dot(lnh, s1)
    lnh_dot_s2 = _dot(lnh, s2)
    s1sq = _dot(s1, s1)
    s2sq = _dot(s2, s2)
    s1_dot_s2 = _dot(s1, s2)
    e = fma(c.e6s1s2_avg, s1_dot_s2, c.e6s1os2o_avg * lnh_dot_s1 * lnh_dot_s2)
    e = fma(c.e6s1s1_sum, s1sq, e)
    e = fma(c.e6s2s2_sum, s2sq, e)
    e = fma(c.e6s1os1o_sum * lnh_dot_s1, lnh_dot_s1, e)
    espin6 = fma(c.e6s2os2o_sum * lnh_dot_s2, lnh_dot_s2, e)
    espin5 = fma(c.e5s1o, lnh_dot_s1, c.e5s2o * lnh_dot_s2)
    espin4 = fma(c.e4s1s2_avg, s1_dot_s2, c.e4s1os2o_avg * lnh_dot_s1 * lnh_dot_s2)
    e = fma(c.e4qms1s1_avg, s1sq, c.e4qms2s2_avg * s2sq)
    e = fma(c.e4qms1os1o_avg * lnh_dot_s1, lnh_dot_s1, e)
    e = fma(c.e4qms2os2o_avg * lnh_dot_s2, lnh_dot_s2, e)
    espin4 = espin4 + e
    espin3 = fma(c.e3s1o, lnh_dot_s1, c.e3s2o * lnh_dot_s2)
    k = c.ecoeff
    test = 0.0  # tidal terms vanish
    test = fma(9.0, k[7], v * v * v * test)
    test = fma(8.0, k[6] + espin6, v * test)
    test = fma(7.0, k[5] + espin5, v * test)
    test = fma(6.0, k[4] + espin4, v * test)
    test = fma(5.0, k[3] + espin3, v * test)
    test = fma(4.0, k[2], v * test)
    test = fma(v * v, test, 2.0)
    return test, v


def stopping_test(c, y, dydt, prev_domega):
    """``XLALSimInspiralSpinTaylorStoppingTest``: LAL's code (0 to continue)
    and the new ``prev_domega``."""
    omega = y[..., 1]
    test, v = energy_test(c, y)
    backwards = (c.f_end < c.f_start) & (c.f_end != 0.0)
    ddomega = dydt[..., 1] - prev_domega
    ddomega = jnp.where(backwards & (prev_domega != 0.0), -ddomega, ddomega)
    omega_start, omega_end = c.omega_start, c.omega_end
    code = jnp.where(
        (jnp.abs(omega_end) > REAL4_EPS) & (omega_end > omega_start) & (omega > omega_end),
        ST_TEST_FREQBOUND,
        jnp.where(
            (jnp.abs(omega_end) > REAL4_EPS) & (omega_end < omega_start) & (omega < omega_end),
            ST_TEST_FREQBOUND,
            jnp.where(
                test < 0.0,
                ST_TEST_ENERGY,
                jnp.where(
                    jnp.isnan(omega),
                    ST_TEST_OMEGANAN,
                    jnp.where(v >= 1.0, ST_TEST_LARGEV, jnp.where(ddomega <= 0.0, ST_TEST_OMEGADOUBLEDOT, 0)),
                ),
            ),
        ),
    )
    return code, dydt[..., 1]


def mass_in_seconds(m_si):
    return m_si / mass_unit() * MTSUN_SI
