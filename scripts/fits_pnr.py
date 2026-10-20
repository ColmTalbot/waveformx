# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``scripts/generate_fits.py`` spec of the parameter-space fits of the PNR
angles and of the PhenomXCP co-precessing deviations of IMRPhenomXPNR
(waveformrs ``imrphenomx/pnr_fits.rs``)."""

# module -> (rust source relative to src/, docstring, functions (None: all),
#            excluded functions, consts (None: all), extra prelude)
MODULES = {
    "pnr_fits": (
        "imrphenomx/pnr_fits.rs",
        "Parameter-space fits of IMRPhenomXPNR: the coefficients of the merger-ringdown\n"
        "``alpha`` and ``beta`` of the PNR angles (``LALSimIMRPhenomX_PNR_coefficients.c``,\n"
        "polynomials of degree (3, 3, 4) in ``eta``, ``chi``, ``cos(theta)`` whose coefficient\n"
        "tables ``COEFF_*[i][j][k]`` multiply ``eta^i chi^j cos(theta)^k``) and the PhenomXCP\n"
        "deviations of the (2,2) mode (``LALSimIMRPhenomX_PNR_deviations.c``,\n"
        "``XLALSimIMRPhenomXCP_*_l2m2(theta, eta, a1)``).",
        None,
        ["evaluate"],
        None,
        '''

def evaluate(coeff, eta, chi, costheta):
    """``IMRPhenomX_PNR_evaluate_coefficient_array``: the sum of ``c[i][j][k]
    eta^i chi^j costheta^k`` over the ``4 x 4 x 5`` table ``coeff``,
    accumulated in the order of LAL's loops (``i`` outer, ``k`` inner) as
    ``poly_sum += c * eta^i * chi^j * costheta^k`` (a fused multiply-add
    in LAL's arm64 build). LAL's powers come from ``pow``; here they are
    products (the same to an ulp). Vanishing coefficients are skipped:
    adding ``+-0`` leaves the sum unchanged."""
    poly_sum = 0.0
    for i, ci in enumerate(coeff):
        eta_i = eta ** i
        for j, cij in enumerate(ci):
            chi_j = chi ** j
            for k, c in enumerate(cij):
                if c == 0.0:
                    continue
                poly_sum = fma(c * eta_i * chi_j, costheta ** k, poly_sum)
    return poly_sum


evaluate_coefficient_array = evaluate
''',
    ),
}
