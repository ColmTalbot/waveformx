# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The post-Newtonian phasing of TaylorF2 for aligned spins
(``XLALSimInspiralPNPhasing_F2`` of ``LALSimInspiralPNCoefficients.c``, as
waveformrs's ``src/pn.rs`` writes it), shared by TaylorF2 and IMRPhenomD.

LAL's defaults: every phase and spin order, black holes (the
quadrupole-monopole parameters are one) and no tidal terms. The phase of
the stationary-phase approximation is ``sum_k (v_k + l_k log v) v^(k -
5)``, ``v = (pi M f)^(1/3)``, with :func:`phasing`'s coefficients.

The functions take the masses relative to the total, ``x1 = m1 / M`` and
``x2 = m2 / M`` (either order), and the aligned spins: they are symmetric
under the exchange of the two bodies.
"""

from __future__ import annotations

import numpy as np

from ..xpnr._math import EGAMMA, PI

__all__ = ["phasing", "spin_spin_3pn"]

LN_4 = float(np.log(4.0))


def _phase_3(x1, x2, chi1, chi2):
    return -16.0 * PI + sum(x * (25.0 + 38.0 / 3.0 * x) * chi for x, chi in ((x1, chi1), (x2, chi2)))


def _phase_4(eta, x1, x2, chi1, chi2):
    phase = 15293365.0 / 508032.0 + 27145.0 / 504.0 * eta + 3085.0 / 72.0 * eta * eta
    phase = phase - 395.0 / 4.0 * eta * chi1 * chi2
    for x, chi in ((x1, chi1), (x2, chi2)):
        phase = phase - (50.0 + 5.0 / 8.0) * x * x * chi * chi
    return phase


def _phase_5(eta, x1, x2, chi1, chi2):
    phase = 5.0 / 9.0 * (7729.0 / 84.0 - 13.0 * eta) * PI
    for x, chi in ((x1, chi1), (x2, chi2)):
        phase = phase - chi * x * (13915.0 / 84.0 - x * (1.0 - x) * 10.0 / 3.0
                                   + x * (12760.0 / 81.0 + x * (1.0 - x) * 170.0 / 9.0))
    return phase


def spin_spin_3pn(eta, x1, x2, chi1, chi2):
    """The 3PN spin-spin term of the phasing (before the factor ``3 / (128
    eta)``): ``Subtract3PNSS`` of IMRPhenomD."""
    phase = (32675.0 / 112.0 + 5575.0 / 18.0 * eta) * eta * chi1 * chi2
    for x, chi in ((x1, chi1), (x2, chi2)):
        phase = phase + ((47035.0 / 84.0 + 2935.0 / 6.0 * x - 120.0 * x * x)
                         + (-410825.0 / 672.0 - 1085.0 / 12.0 * x + 1255.0 / 36.0 * x * x)) * x * x * chi * chi
    return phase


_LOG_6 = -6848.0 / 21.0


def _phase_6(eta, x1, x2, chi1, chi2):
    phase = 11583231236531.0 / 4694215680.0 - 640.0 / 3.0 * PI ** 2 - 6848.0 / 21.0 * EGAMMA
    phase = phase + eta * (-15737765635.0 / 3048192.0 + 2255.0 / 12.0 * PI ** 2)
    phase = phase + eta * eta * 76055.0 / 1728.0 - eta * eta * eta * 127825.0 / 1296.0
    phase = phase + _LOG_6 * LN_4 + spin_spin_3pn(eta, x1, x2, chi1, chi2)
    for x, chi in ((x1, chi1), (x2, chi2)):
        phase = phase + PI * x * (1490.0 / 3.0 + x * 260.0) * chi
    return phase


def _phase_7(eta, x1, x2, chi1, chi2):
    phase = PI * (77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta - 74045.0 / 756.0 * eta * eta)
    for x, chi in ((x1, chi1), (x2, chi2)):
        phase = phase + chi * x * (-170978035.0 / 48384.0 + eta * 2876425.0 / 672.0 + eta * eta * 4735.0 / 144.0
                                   + x * (-7189233785.0 / 1524096.0 + eta * 458555.0 / 3024.0
                                          - eta * eta * 5345.0 / 72.0))
    return phase


def phasing(x1, x2, chi1, chi2):
    """``(v, l)``: the coefficients ``v_0 .. v_7`` of ``v^(k - 5)`` and
    ``l_5``, ``l_6`` of ``v^(k - 5) log v`` (lists, with the factor ``3 /
    (128 eta)``)."""
    eta = x1 * x2
    scale = 3.0 / (128.0 * eta)
    v5 = _phase_5(eta, x1, x2, chi1, chi2)
    v = [1.0, 0.0, 55.0 * eta / 9.0 + 3715.0 / 756.0, _phase_3(x1, x2, chi1, chi2),
         _phase_4(eta, x1, x2, chi1, chi2), v5, _phase_6(eta, x1, x2, chi1, chi2),
         _phase_7(eta, x1, x2, chi1, chi2)]
    return [scale * c for c in v], {5: scale * 3.0 * v5, 6: scale * _LOG_6}
