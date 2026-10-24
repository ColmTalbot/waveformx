# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""TaylorF2 as a table of coefficients per binary and a short formula per
frequency, shared by the JAX model (:mod:`.model`) and the MLX one
(:mod:`.mlx`), as IMRPhenomXAS (:mod:`..xas.core`).

LAL's TaylorF2 (``XLALSimInspiralTaylorF2Core`` with LAL's defaults: the
3.5PN phase with every spin term, the leading-order amplitude, no tidal
terms) is ``h = A v^(-7/2) exp(-i (Psi(v) - 2 phi_ref - Psi_ref - pi / 4))``,
``v = (pi M f)^(1/3)``, ``Psi = sum_k (v_k + l_k log v) v^(k - 5)``
(:mod:`.pn`). :func:`table` writes the phase in powers of the frequency
``f`` in Hz (and their products with ``log f``), the powers of the
binary's ``M`` folded into the coefficients, so that each frequency costs
a sum of eight terms, one power for the amplitude and a sine and cosine
(:func:`evaluate`). The leading term, ``T0 f^(-5/3)`` (about ``10^4``
radians for binary neutron stars from 20 Hz), and its value at the
reference frequency are in float64 whatever the precision, as in
:func:`..xas.core.phase`.

``f_ref = 0`` is the reference at coalescence (LAL's convention for
TaylorF2): no reference phase ``Psi_ref``. The masses are in solar masses
and the strain is multiplied by ``amp_scale`` (``1e21`` unless
``scaled_strain`` is false), so that the coefficients stay within the range
of float32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..xas.core import K0, frequency_powers, phase_basis, reduce
from ..xas.core import PHASE_BASIS as _XAS_BASIS
from ..xpnr._math import MPC_SI, MRSUN_SI, MTSUN_SI, PI, real
from . import pn

__all__ = ["BASIS", "LAYOUT", "SIZE", "OFFSETS", "table", "unpack", "frequency_powers", "evaluate", "phase",
           "amplitude"]

#: The powers ``(a, l)`` of the basis ``f^a (log f)^l`` of the phase but
#: for its leading term, ``f^(-5/3)`` (``v^(k - 5)`` for ``k = 2 .. 7``).
BASIS = ((-1, 0), (-2 / 3, 0), (-1 / 3, 0), (0, 1), (0, 0), (1 / 3, 0), (1 / 3, 1), (2 / 3, 0))
#: Their indices in :data:`..xas.core.PHASE_BASIS` (:func:`..xas.core.phase_basis`).
XAS_INDEX = tuple(_XAS_BASIS.index(key) for key in BASIS)
CONST = BASIS.index((0, 0))

#: The fields of a table: ``phase`` (the coefficients of :data:`BASIS`),
#: ``big`` (the coefficient ``T0`` of ``f^(-5/3)`` and its value ``A`` at
#: the reference frequency) and ``big_lo`` (what they lack of their float64
#: values), ``amp`` (the amplitude is ``amp f^(-7/6)``) and the factors
#: ``kp``, ``kc`` of the polarizations.
LAYOUT = (
    ("phase", (len(BASIS),)),
    ("big", (2,)),
    ("big_lo", (2,)),
    ("amp", ()),
    ("kp", ()),
    ("kc", ()),
)


def _offsets():
    out, start = {}, 0
    for name, shape in LAYOUT:
        size = int(np.prod(shape)) if shape else 1
        out[name] = (start, shape)
        start += size
    return out, start


OFFSETS, SIZE = _offsets()

#: The amplitude's coefficient is ``K_AMP (m1 m2)^(1/2) M^(-1/6) / D``
#: (masses in solar masses, ``D`` in Mpc): LAL's ``amp0 sqrt(5 / (32
#: eta)) (pi M)^(-7/6)``, ``amp0 = -4 m1 m2 / D sqrt(pi / 12)`` in
#: geometric units.
K_AMP = -4.0 * MRSUN_SI * MTSUN_SI / MPC_SI * float(np.sqrt(PI / 12.0) * np.sqrt(5.0 / 32.0)) \
    * float(PI * MTSUN_SI) ** (-7.0 / 6.0)


def unpack(flat):
    """The fields of a flat table (``(..., SIZE)``) as a dict of arrays."""
    out = {}
    for name, (start, shape) in OFFSETS.items():
        size = int(np.prod(shape)) if shape else 1
        out[name] = flat[..., start:start + size].reshape(flat.shape[:-1] + tuple(shape))
    return out


def table(params, amp_scale=1.0):
    """The table of a binary (a flat array of :data:`SIZE` values) from
    ``params``: ``mass_1``, ``mass_2`` (solar masses, either order),
    ``chi_1``, ``chi_2`` (aligned spins), ``luminosity_distance`` (Mpc),
    ``inclination``, ``phi_ref``, ``f_ref`` (Hz, zero for the reference at
    coalescence), as ``XLALSimInspiralChooseFDWaveformSequence``."""
    m1, m2, chi1, chi2, distance, inclination, phi_ref, f_ref = params
    total = m1 + m2
    x1, x2 = m1 / total, m2 / total
    v, logs = pn.phasing(x1, x2, chi1, chi2)
    m_sec = total * MTSUN_SI
    log_pim = jnp.log(PI * m_sec)
    # -Psi in powers of f: (v_k + l_k log v) v^(k - 5), log v = (log(pi M) + log f) / 3
    rows = [jnp.zeros((), real())] * len(BASIS)
    for k in range(2, 8):
        a = (k - 5) / 3.0
        s = jnp.exp(a * log_pim)
        coefficient = v[k]
        if k in logs:
            coefficient = coefficient + logs[k] * log_pim / 3.0
            rows[BASIS.index((a, 1))] = -logs[k] / 3.0 * s
        rows[BASIS.index((a, 0))] = -coefficient * s
    rows = jnp.stack([jnp.asarray(c, real()) for c in rows])

    # the reference: phi(f_ref) = 2 phi_ref + pi / 4 (none for f_ref = 0)
    referenced = f_ref > 0.0
    f_ref_safe = jnp.where(referenced, f_ref, 1.0)
    fields = dict(phase=rows)
    rest = _phase_rest(fields, frequency_powers(jnp.asarray(f_ref_safe, real())))
    constant = 2.0 * phi_ref + PI / 4.0 - jnp.where(referenced, rest[..., 0], 0.0)
    rows = rows + constant * jnp.asarray(np.eye(len(BASIS))[CONST], real())

    # the leading term in float64 (see ..xas.core.phase)
    m1d, m2d = (jnp.asarray(m, jnp.float64) for m in (m1, m2))
    t0 = K0 * jnp.cbrt(m1d + m2d) / (m1d * m2d)
    f_ref64 = jnp.asarray(f_ref_safe, jnp.float64)
    anchor = jnp.where(referenced, t0 * jnp.exp(-5.0 / 3.0 * jnp.log(f_ref64)), 0.0)
    big64 = jnp.stack([t0, anchor])
    big = big64.astype(real())
    big_lo = jax.lax.stop_gradient(big64 - big.astype(jnp.float64)).astype(real())

    amp = K_AMP * amp_scale * jnp.sqrt(m1 * m2) * jnp.exp(-1.0 / 6.0 * jnp.log(total)) / distance
    cos_i = jnp.cos(inclination)
    fields = dict(phase=rows, big=big, big_lo=big_lo, amp=amp, kp=0.5 * (1.0 + cos_i * cos_i), kc=-cos_i)
    return jnp.concatenate([jnp.asarray(fields[name], real()).reshape(-1) for name, _ in LAYOUT])


def _col(fields, name, *index):
    value = fields[name]
    return value[(Ellipsis,) + index + (None,)] if index else value[..., None]


def _phase_rest(fields, fp):
    """The phase but for the leading term (and ``-A``)."""
    basis = phase_basis(fp)
    total = _col(fields, "phase", CONST)
    for i, k in enumerate(XAS_INDEX):
        if i != CONST:
            total = total + _col(fields, "phase", i) * basis[k]
    return total


def phase(fields, fp):
    """The phase ``phi`` (``h = A exp(i phi)``) at the frequencies, reduced
    to ``[-pi, pi]``: the leading term, its value at the reference and the
    sum in float64, as :func:`..xas.core.phase`."""
    rest = _phase_rest(fields, fp)
    t0 = (_col(fields, "big", 0).astype(jnp.float64)
          + jax.lax.stop_gradient(_col(fields, "big_lo", 0)).astype(jnp.float64))
    anchor = (_col(fields, "big", 1).astype(jnp.float64)
              + jax.lax.stop_gradient(_col(fields, "big_lo", 1)).astype(jnp.float64))
    phi = t0 * fp[5] + rest.astype(jnp.float64) - anchor
    return reduce(phi).astype(rest.dtype)


def amplitude(fields, fp):
    """The amplitude ``A`` at the frequencies (negative: LAL's sign)."""
    return _col(fields, "amp") * fp[4]


def evaluate(fields, fp):
    """``(h_+, h_x)`` at the frequencies (powers ``fp`` from
    :func:`frequency_powers`) for a table (:func:`unpack`; one binary, or a
    batch with a trailing axis for the frequencies to broadcast against)."""
    amp = amplitude(fields, fp)
    phi = phase(fields, fp)
    c, s = jnp.cos(phi), jnp.sin(phi)
    ap = _col(fields, "kp") * amp
    ac = _col(fields, "kc") * amp
    return jax.lax.complex(ap * c, ap * s), jax.lax.complex(-ac * s, ac * c)
