# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Quasi-normal-mode fits of the higher modes of IMRPhenomXHM
(``LALSimIMRPhenomXHM_qnm.c``, generated into :mod:`.._generated.hm_qnm`).

``fringLM(af)``, ``fdampLM(af)`` (``LM`` in 21, 33, 32, 44) are the
ringdown and damping frequencies of the mode for the dimensionless final
spin ``af``, in units of the remnant mass (divide by the final mass in units
of the total mass); ``re_l2m2lp2`` ... ``im_l3m2lp3`` are the real and
imaginary parts of the spheroidal-spherical mixing coefficients of the
(3,2) mode, and :func:`mixing_coefficients` returns the four coefficients as
LAL stores them (single precision, signs of the (3,2) ones flipped).
"""

from .._generated.hm_qnm import (  # noqa: F401
    fdamp21,
    fdamp32,
    fdamp33,
    fdamp44,
    fring21,
    fring32,
    fring33,
    fring44,
    im_l2m2lp2,
    im_l2m2lp3,
    im_l3m2lp2,
    im_l3m2lp3,
    re_l2m2lp2,
    re_l2m2lp3,
    re_l3m2lp2,
    re_l3m2lp3,
)
from .variables import mixing_coefficients, qnm_fits  # noqa: F401

__all__ = [
    "fring21", "fdamp21", "fring33", "fdamp33", "fring32", "fdamp32", "fring44", "fdamp44",
    "re_l2m2lp2", "re_l2m2lp3", "re_l3m2lp2", "re_l3m2lp3",
    "im_l2m2lp2", "im_l2m2lp3", "im_l3m2lp2", "im_l3m2lp3",
    "mixing_coefficients", "qnm_fits",
]
