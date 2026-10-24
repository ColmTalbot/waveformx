# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXHM as tables of coefficients per binary and a short formula
per frequency, shared by the JAX model (:mod:`.model`) and the MLX one
(:mod:`.mlx`); the (2,2) mode is IMRPhenomXAS's (:mod:`..xas.core`).

Each higher mode (:data:`.reference.HIGHER_MODES`) is written, region by
region, in powers of the frequency ``f`` in Hz (:data:`HM_BASIS`, the
powers of ``M`` folded into the coefficients as for the (2,2)), with the
amplitude scale ``amp0`` folded into the amplitude coefficients.

The phase of every mode is ``[inspiral] T0_lm f^(-5/3) - (m/2) A + rest(f)``
where ``T0_lm = (m/2)^(8/3) T0`` and ``A`` are the leading term and the
anchor of the (2,2) (:func:`..xas.core.phase`): the first two terms, of
``10^4`` radians for binary neutron stars, in extended precision, the rest
(at most ``~10^3`` radians) in the working precision. LAL's alignment of
the higher modes with the (2,2) (``deltaphi_lm``) and the
spheroidal-to-spherical rotation of the (3,2) ringdown contain ``A``
through the (2,2) phase; here it is taken out exactly (``(m/2) T0 (2f/m)^(-5/3)
= T0_lm f^(-5/3)``), so that the set-up computes no phase larger than
``rest`` in float32 either. The (3,2) ringdown is ``conj(a322) h22 +
conj(a323) h32_spheroidal`` with the (2,2) mode recycled, as in the
polarizations of LAL (``IMRPhenomXHM_MultiMode2``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..xas import core as xas_core
from ..xpnr import phase22 as phase22_module
from ..xpnr._generated import hm_ringdown
from ..xpnr._generated import ringdown as ringdown22
from ..xpnr._math import FRAC_PI_4, PI, Struct, fma, lean, powers, real
from ..xpnr.amp22 import amp_coefficients as amp22_coefficients
from ..xpnr.hm.inspiral import phase_ansatz_int as inspiral_phase_ansatz_int
from ..xpnr.hm.phase import ringdown_phase_ansatz, ringdown_phase_ansatz_int
from ..xpnr.internals import errors
from . import reference as ref

__all__ = ["HM_BASIS", "LAYOUT", "SIZE", "MODES", "table", "unpack", "evaluate", "modes_and_polarizations"]

MODES = ref.MODES
HIGHER_MODES = ref.HIGHER_MODES


def _warm_constants():
    """Compute the constants that the shared code computes lazily on first
    use (with ``jax.ensure_compile_time_eval``), which fails when the first
    use is inside the trace of a custom derivative rule (:mod:`.model`)."""
    from ..xpnr._math import POWERS_OF_LALPI
    from ..xpnr.hm import inspiral

    for name in powers(np.float64(np.pi)).keys():
        getattr(POWERS_OF_LALPI, name)
    for x in (2.0, 2.0 / 3.0, 0.5):
        inspiral._p2(x)


_warm_constants()

#: The powers ``(a, l)`` of the basis ``f^a (log f)^l`` of the phases of
#: the higher modes.
HM_BASIS = (
    (-3, 0), (-2, 0), (-4 / 3, 0), (-1, 0), (-2 / 3, 0), (-1 / 3, 0), (0, 0), (1 / 3, 0), (2 / 3, 0), (1, 0),
    (4 / 3, 0), (5 / 3, 0), (2, 0), (7 / 3, 0), (0, 1), (1 / 3, 1), (1, 1), (4 / 3, 1),
)
_INDEX = {key: i for i, key in enumerate(HM_BASIS)}
_CONSTANT = _INDEX[(0, 0)]
#: The terms of :data:`HM_BASIS` in each region (the constant apart): the
#: inspiral (the rescaled (2,2) inspiral, but its leading term), the
#: intermediate and the ringdown.
REGION_TERMS = (
    tuple(_INDEX[k] for k in HM_BASIS if k not in ((-3, 0), (-2, 0), (0, 0))),
    tuple(_INDEX[k] for k in ((1, 0), (0, 1), (-1, 0), (-2, 0), (-3, 0))),
    tuple(_INDEX[k] for k in ((1, 0), (-1, 0))),
)

#: The fields of the table of a higher mode: ``phase`` (the coefficients of
#: :data:`HM_BASIS` in the inspiral, intermediate and ringdown; for the
#: (3,2) the ringdown row is the linear term added to the phase of the
#: rotation), ``sph`` (the spheroidal phase of the (3,2): coefficients of
#: ``f``, 1, ``1/f``, ``1/f^3``), ``atan_a``, ``atan_w`` (``a atan(w (f -
#: shift))`` of the intermediate and ringdown regions), the boundaries of
#: the regions (Hz), the inspiral amplitude ``f^(-7/6) (|sum_k pn_k t^k| +
#: pseudo_0 t^7 + pseudo_1 t^8 + pseudo_2 t^9)`` (``t = f^(1/3)``), the
#: intermediate amplitude ``f^(-7/6) sum_j inter_j u^j`` (``u = (f -
#: inter_c) inter_h``), the ringdown
#: Lorentzian ``rd_0 / (exp(rd_1 u) (u^2 + rd_2))`` (``u = s (f - shift)``),
#: its exponential fall-off ``falloff_0 exp(-falloff_1 (f - falloff_2))``,
#: the cubic ``aux`` of the (3,2) below ``f_aux`` and the mixing
#: coefficients ``-a322``, ``-a323`` (``(re, im)``) of the (3,2).
MODE_LAYOUT = (
    ("phase", (3, len(HM_BASIS))),
    ("sph", (4,)),
    ("atan_a", (3,)),
    ("atan_w", (3,)),
    ("shift", ()),
    ("f_phase_in", ()),
    ("f_phase_im", ()),
    ("f_amp_in", ()),
    ("f_amp_im", ()),
    ("pn_re", (7,)),
    ("pn_im", (7,)),
    ("pseudo", (3,)),
    ("inter", (8,)),
    ("inter_c", ()),
    ("inter_h", ()),
    ("rd", (3,)),
    ("falloff", (3,)),
    ("aux", (4,)),
    ("f_aux", ()),
    ("mix", (4,)),
    ("on", ()),
)
#: The table of a binary: that of the (2,2) (:mod:`..xas.core`), those of
#: the higher modes, the factors ``(F+, Fx)`` (``(re, im)`` each) of each
#: mode of :data:`MODES` in the polarizations, ``amp0`` and ``s = M`` (s).
HEAD_LAYOUT = (
    ("xas", (xas_core.SIZE,)),
    ("factors", (len(MODES), 4)),
    ("amp0", ()),
    ("s", ()),
)
LAYOUT = HEAD_LAYOUT + tuple((f"hm{ell}{emm}", (sum(int(np.prod(shape)) if shape else 1 for _, shape in MODE_LAYOUT),))
                             for ell, emm in HIGHER_MODES)


def _offsets(layout):
    out, start = {}, 0
    for name, shape in layout:
        size = int(np.prod(shape)) if shape else 1
        out[name] = (start, shape)
        start += size
    return out, start


MODE_OFFSETS, MODE_SIZE = _offsets(MODE_LAYOUT)
HEAD_OFFSETS, HEAD_SIZE = _offsets(HEAD_LAYOUT)
OFFSETS, SIZE = _offsets(LAYOUT)


def _unpack(flat, offsets):
    out = {}
    for name, (start, shape) in offsets.items():
        size = int(np.prod(shape)) if shape else 1
        out[name] = flat[..., start:start + size].reshape(flat.shape[:-1] + tuple(shape))
    return out


def unpack_mode(flat):
    """The fields of the table of a higher mode."""
    return _unpack(flat, MODE_OFFSETS)


def unpack(flat):
    """The fields of a flat table: ``xas`` (those of :func:`..xas.core.unpack`),
    ``hm21`` ... (those of each higher mode), ``factors``, ``amp0``, ``s``."""
    out = _unpack(flat, OFFSETS)
    out["xas"] = xas_core.unpack(out["xas"])
    for ell, emm in HIGHER_MODES:
        out[f"hm{ell}{emm}"] = _unpack(out[f"hm{ell}{emm}"], MODE_OFFSETS)
    return out


def _flat(fields, layout):
    flat = []
    for name, shape in layout:
        value = fields[name]
        if isinstance(value, (list, tuple)):
            value = jnp.stack([jnp.asarray(v, real()) for v in value])
        flat.append(jnp.broadcast_to(jnp.asarray(value, real()), shape).reshape(-1))
    return jnp.concatenate(flat)


def _add_terms(row, terms, s, log_s):
    """Add ``c x^a (log x)^l`` (``x = s f``) to the coefficients ``row`` of
    :data:`HM_BASIS`."""
    for c, a, l in terms:
        scaled = c * s[a]
        row[_INDEX[(a, 0)]] = row[_INDEX[(a, 0)]] + (scaled * log_s if l else scaled)
        if l:
            row[_INDEX[(a, 1)]] = row[_INDEX[(a, 1)]] + scaled


# ---------------------------------------------------------------------------
# set-up
# ---------------------------------------------------------------------------


def _rest22(fields22, mf, s, leading=True):
    """The phase of the (2,2) at ``mf`` without its anchor, ``[inspiral] T0
    f^(-5/3) + rest(f)`` (without the leading term if not ``leading``), and
    whether ``mf`` is past its inspiral (in the working precision: the
    set-up evaluates it at ``Mf >~ 0.6 fMECO``)."""
    f = mf / s
    fp = xas_core.frequency_powers(f)
    rest, past = xas_core._phase_rest(fields22, fp)
    if leading:
        t0 = xas_core._col(fields22, "big", 0)
        rest = rest + jnp.where(past, 0.0, t0 * fp[2] * jnp.square(fp[1] * fp[1] * fp[2]))
    return rest[..., 0], past[..., 0]


def _spheroidal(wf, hm, phase22, fields22):
    """``GetSpheroidalCoefficients``: the spheroidal ringdown phase of the
    (3,2) aligned with the (2,2) ringdown, without the anchor of the (2,2)
    in ``phi0_s`` (after :func:`..xpnr.hm.phase.spheroidal_coefficients`)."""
    fringlm = hm.f_ring
    fdamplm = hm.f_damp
    freqs = [wf.f_ring, fma(-1.5, fdamplm, fringlm), fma(-0.5, fdamplm, fringlm), fma(0.5, fdamplm, fringlm)]
    rows, values = [], []
    for i, ff in enumerate(freqs):
        values.append(hm_ringdown.phase_32_fit(i, wf))
        ffm2 = 1.0 / (ff * ff)
        rows.append([1.0, fdamplm / (fdamplm * fdamplm + (ff - fringlm) * (ff - fringlm)), ffm2, ffm2 * ffm2])
    alpha0_s, alpha_l_s, alpha2_s, alpha4_s = ref._solve(rows, values)
    ph = Struct(alpha0_s=alpha0_s, alpha_l_s=alpha_l_s, alpha2_s=alpha2_s, alpha4_s=alpha4_s, phi0_s=0.0)
    view = Struct(f_ring=fringlm, f_damp=fdamplm, mixing_on=True)
    # time shift of the spheroidal ansatz
    fref_rd = wf.f_ring + wf.f_damp
    p_ref = powers(fref_rd)
    timeshift = phase22_module.time_shift_22(phase22, wf)
    dphi22_ref = fma(1.0 / wf.eta, phase22_module.dphase_22(fref_rd, p_ref, phase22), timeshift)
    ph.alpha0_s = ph.alpha0_s + dphi22_ref + hm_ringdown.phase_32_spheroidal_time_shift(wf) - \
        ringdown_phase_ansatz(fref_rd, p_ref, view, ph)
    # phase shift (the (2,2) phase at fRING without its anchor)
    fref_rd = wf.f_ring
    phi22_ref, _ = _rest22(fields22, fref_rd, wf.m_sec)
    ph.phi0_s = phi22_ref - ringdown_phase_ansatz_int(fref_rd, powers(fref_rd), view, ph) + \
        hm_ringdown.phase_32_spheroidal_phase_shift(wf)
    return ph


def _mode(wf, phase22, fields22, ell, emm):
    """The coefficients of a higher mode (:func:`.reference.mode_setup`)
    without the anchor of the (2,2) in its phase: the amplitude, the phase
    coefficients, ``D = deltaphi_lm + (m/2) A`` and, for the (3,2), the
    spheroidal phase and the data of the rotation."""
    hm = ref.mode_struct(wf, ell, emm)
    mix = Struct(alpha2_fit=hm_ringdown.phase_22_alpha2(wf), alpha_l_fit=hm_ringdown.phase_22_alpha_l(wf),
                 timeshift=0.0, phiref22=0.0)
    rotated = None
    if hm.mixing_on:
        spheroidal = _spheroidal(wf, hm, phase22, fields22)
        view = Struct(f_ring=hm.f_ring, f_damp=hm.f_damp, mixing_on=True, mode_tag=32, ell=3, emm=2,
                      f_amp_rd_falloff=hm.f_amp_rd_falloff, mixing_coeffs=hm.mixing_coeffs)
        mix = mix.replace(spheroidal=spheroidal, hm=view)
        amp22 = amp22_coefficients(wf)

        def rotation(mf, amp):
            # (the (2,2) ringdown ansatz rescaled by its leading order, with
            # the phase of the (2,2) without its anchor)
            p = powers(mf)
            a22 = ringdown22.amp_22_ansatz_exact(mf, wf.f_ring, wf.f_damp, amp22) * wf.amp_norm * p.m_seven_sixths
            phi22, _ = _rest22(fields22, mf, wf.m_sec)
            return ref.rotate(mf, p, mix, amp, (a22 * jnp.cos(phi22), a22 * jnp.sin(phi22)))
    else:
        def rotation(mf, amp):
            raise AssertionError("no rotation without mode mixing")
    amp = ref.amplitude_coefficients(wf, hm, rotation)
    if hm.mixing_on:
        def rotated(mf):
            return rotation(mf, amp)
    ph = ref.phase_coefficients(wf, hm, phase22, mix, amp, rotated)

    # D = deltaphi_lm + (m/2) A: (m/2) phi22(2 falign / m) - phi_lm(falign) -
    # 3 pi/4 (1 - m/2), with the leading terms, equal, taken out
    m_over_2 = emm * 0.5
    falign = jnp.where(wf.eta > hm.eta_emr, 0.6 * m_over_2 * wf.f_meco, m_over_2 * wf.f_meco)
    rest22, past22 = _rest22(fields22, falign / m_over_2, wf.m_sec, leading=False)
    leading = ph.phi[0] * jnp.power(falign, -5.0 / 3.0)
    phi_lm_rest = fma(ph.c1_insp, falign, inspiral_phase_ansatz_int(falign, powers(falign), ph.replace(
        phi=[0.0] + list(ph.phi[1:])))) + ph.c_insp
    d = fma(-(3.0 * FRAC_PI_4), 1.0 - m_over_2, m_over_2 * rest22) - phi_lm_rest - jnp.where(past22, leading, 0.0)
    if hm.mode_tag == 21:
        d = d + jnp.where(ref.pn21_amp_sign(0.008, wf), PI, 0.0)
    return hm, amp, ph, mix, d


def _mode_fields(wf, hm, amp, ph, mix, d):
    """The table of a higher mode (:data:`MODE_LAYOUT`)."""
    s = wf.m_sec
    p = powers(s)
    s_pow = {-3: p.m_three, -2: p.m_two, -4 / 3: p.m_four_thirds, -1: p.m_one, -2 / 3: p.m_two_thirds,
             -1 / 3: p.m_one_third, 0: 1.0, 1 / 3: p.one_third, 2 / 3: p.two_thirds, 1: s, 4 / 3: p.four_thirds,
             5 / 3: p.five_thirds, 2: p.two, 7 / 3: p.seven_thirds}
    zero = jnp.zeros((), real())
    tag = hm.mode_tag
    exponents = [(i - 5) / 3.0 for i in range(13)]
    exponents = [int(e) if float(e).is_integer() else e for e in exponents]
    ins = [(ph.phi[i], exponents[i], 0) for i in range(1, 13)]
    ins += [(ph.phi_l[i], exponents[i], 1) for i in (5, 6, 8, 9)]
    ins += [(ph.c1_insp, 1, 0), (ph.c_insp + d, 0, 0)]
    inter = [(ph.c0, 1, 0), (ph.c1, 0, 1), (-ph.c2, -1, 0), (-ph.c4 / 3.0, -3, 0), (d, 0, 0)]
    if tag == 32:
        inter.append((-0.5 * ph.c3, -2, 0))
        rd = [(ph.c1_rd, 1, 0), (ph.c_rd + d, 0, 0)]
    else:
        rd = [(ph.alpha0 + ph.c1_rd, 1, 0), (-hm.f_ring * hm.f_ring * ph.alpha2, -1, 0), (ph.c_rd + d, 0, 0)]
    rows = []
    for terms in (ins, inter, rd):
        row = [zero] * len(HM_BASIS)
        _add_terms(row, terms, s_pow, p.log)
        rows.append(jnp.stack([jnp.asarray(c, real()) for c in row]))
    w = s / hm.f_damp
    if tag == 32:
        sph = mix.spheroidal
        atan_a = [zero, ph.c_l, sph.alpha_l_s]
        sph_row = [sph.alpha0_s * s, sph.phi0_s, -sph.alpha2_s / s, -sph.alpha4_s / 3.0 * p.m_three]
        (m2r, m2i), (m3r, m3i) = hm.mixing_coeffs[2], hm.mixing_coeffs[3]
        mixing = [m2r, m2i, m3r, m3i]
    else:
        atan_a = [zero, ph.c_l, ph.alpha_l]
        sph_row = [zero] * 4
        mixing = [zero] * 4

    amp0 = wf.amp0
    pn_scale = amp.pn.pn_global_factor * amp.amp_norm * p.m_seven_sixths * amp0
    third = [1.0, p.one_third, p.two_thirds, s, p.four_thirds, p.five_thirds, p.two]
    pn_re = [amp.pn.pn_re[k] * third[k] * pn_scale for k in range(7)]
    pn_im = [amp.pn.pn_im[k] * third[k] * pn_scale for k in range(7)]
    fold = amp.pn_dominant * p.m_seven_sixths * amp0
    c = amp.inspiral_coefficients
    pseudo = [c[0] / amp.fcut_insp_seven_thirds * p.seven_thirds * fold,
              c[1] / amp.fcut_insp_eight_thirds * p.eight_thirds * fold,
              c[2] / amp.fcut_insp_three * p.three * fold]
    s_j = [1.0, s, p.two, p.three]
    inter_amp = [amp.inter[j] * p.m_seven_sixths * amp0 for j in range(len(amp.inter))]
    inter_amp += [zero] * (8 - len(inter_amp))
    rdc = amp.rd
    dfd = hm.f_damp * rdc[2]
    rd_amp = [rdc[0] * hm.f_damp * amp0, rdc[1] / dfd, dfd * dfd]
    falloff = [rdc[3] * amp0, rdc[4] * s, hm.f_amp_rd_falloff / s]
    if tag == 32:
        aux = [amp.rd_aux[j] * s_j[j] * amp0 for j in range(4)]
        f_aux = amp.f_rd_aux / s
    else:
        aux = [zero] * 4
        f_aux = zero
    fields = dict(
        phase=jnp.stack(rows), sph=sph_row, atan_a=atan_a, atan_w=[zero, w, w], shift=hm.f_ring / s,
        f_phase_in=ph.f_phase_match_in / s, f_phase_im=ph.f_phase_match_im / s,
        f_amp_in=amp.f_amp_match_in / s, f_amp_im=amp.f_amp_match_im / s,
        pn_re=pn_re, pn_im=pn_im, pseudo=pseudo, inter=inter_amp, inter_c=amp.inter_center / s,
        inter_h=s / amp.inter_half, rd=rd_amp, falloff=falloff, aux=aux,
        f_aux=f_aux, mix=mixing, on=jnp.where(hm.amp_zero, 0.0, 1.0),
    )
    return _flat(fields, MODE_LAYOUT)


def _setup(params, f_min, amp_scale):
    """The waveform variables and the table of the (2,2) (in lean mode)."""
    m1, m2, _, _, _, inclination, _, _ = params
    wf = xas_core.waveform_variables(params, f_min, amp_scale)
    wf.hm_release = 122022
    flat22, c22 = xas_core.table_of(wf, m1, m2, inclination)
    return wf, flat22, c22, xas_core.unpack(flat22)


def _head(params, wf, flat22):
    """The head of the table (:data:`HEAD_LAYOUT`)."""
    inclination = params[5]
    factors = []
    for ell, emm in MODES:
        fp, fc = ref.mode_factors(inclination, ell, emm)
        factors.append(jnp.stack([jnp.real(fp), jnp.imag(fp), jnp.real(fc), jnp.imag(fc)]).astype(real()))
    return _flat(dict(xas=flat22, factors=jnp.where(errors(wf), jnp.nan, jnp.stack(factors)), amp0=wf.amp0,
                      s=wf.m_sec), HEAD_LAYOUT)


def _mode_table(wf, c22, fields22, ell, emm):
    """The table of a higher mode (:data:`MODE_LAYOUT`). The odd modes
    vanish for equal masses and spins, where their set-up divides by zero
    (LAL skips them): there they are computed from detached inputs and
    replaced by zeros, so that neither their values nor their derivatives
    are NaN."""
    if (10 * ell + emm) not in (21, 33):
        return _mode_fields(wf, *_mode(wf, c22.phase, fields22, ell, emm))
    vanish = (wf.q == 1.0) & (wf.chi1l == wf.chi2l)
    args = jax.tree_util.tree_map(
        lambda x: jnp.where(vanish, jax.lax.stop_gradient(x), x) if isinstance(x, jax.Array) else x,
        (wf, c22.phase, fields22))
    hm, amp, ph, mix, d = _mode(*args, ell, emm)
    return jnp.where(vanish, 0.0, _mode_fields(args[0], hm, amp, ph, mix, d))


def table(params, f_min, amp_scale=1.0):
    """The table of a binary (a flat array of :data:`SIZE` values) from the
    parameters of :func:`..xas.core.table` and the first frequency."""
    with lean(True):
        wf, flat22, c22, fields22 = _setup(params, f_min, amp_scale)
        parts = [_head(params, wf, flat22)]
        parts += [_mode_table(wf, c22, fields22, ell, emm) for ell, emm in HIGHER_MODES]
        return jnp.concatenate(parts)


def head_table(params, f_min, amp_scale=1.0):
    """The head of :func:`table` alone (:data:`HEAD_LAYOUT`; for the MLX
    kernels, which set the modes up in parallel)."""
    with lean(True):
        wf, flat22, _, _ = _setup(params, f_min, amp_scale)
        return _head(params, wf, flat22)


def mode_table(params, f_min, amp_scale, ell, emm):
    """The table of the higher mode ``(ell, emm)`` alone (:data:`MODE_LAYOUT`)."""
    with lean(True):
        wf, _, c22, fields22 = _setup(params, f_min, amp_scale)
        return _mode_table(wf, c22, fields22, ell, emm)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def hm_basis(fp):
    """:data:`HM_BASIS` at the frequencies (from
    :func:`..xas.core.frequency_powers`): a list of arrays."""
    f, t, m1, log_f = fp[:4]
    t2 = t * t
    mt1 = t2 * m1  # f^(-1/3)
    m2 = m1 * m1
    return [m2 * m1, m2, m1 * mt1, m1, mt1 * mt1, mt1, None, t, t2, f, f * t, f * t2, f * f, f * f * t,
            log_f, t * log_f, f * log_f, f * t * log_f]


def _col(fields, name, *index):
    return xas_core._col(fields, name, *index)


def mode_rest(fields, basis, fp, s, amp0, tag, h22p=None):
    """The amplitude (``amp0`` included, ``>= 0``) and the phase but for
    ``[inspiral] T0_lm f^(-5/3) - (m/2) A`` of the higher mode ``tag``
    (``10 l + m``) of the table ``fields`` at the frequencies, and whether
    they are in its inspiral; ``s`` and ``amp0`` broadcast against the
    frequencies; ``h22p`` the (2,2) mode without its anchor, ``(amplitude,
    phase)``, recycled in the ringdown of the (3,2)."""
    f, t = fp[0], fp[1]
    m76 = fp[4]

    def col(name, *index):
        return _col(fields, name, *index)

    past_ins = f >= col("f_phase_in")
    rd_phase = f >= col("f_phase_im")
    sums = []
    for r, terms in enumerate(REGION_TERMS):
        if tag == 32 and r == 2:
            terms = (_INDEX[(1, 0)],)
        total = col("phase", r, _CONSTANT)
        for k in terms:
            total = total + col("phase", r, k) * basis[k]
        sums.append(total)
    y = f - col("shift")
    at_int = col("atan_a", 1) * jnp.arctan(col("atan_w", 1) * y)
    at_rd = col("atan_a", 2) * jnp.arctan(col("atan_w", 2) * y)

    # ringdown amplitude: the cubic of the (3,2) below f_aux, the fall-off,
    # the Lorentzian
    u = s * y
    # (the exponents bounded where the branches are not used: exp overflows
    # float32 beyond 88, and the derivative of the Lorentzian squares it; the
    # derivative of an unused branch would be NaN. In the ringdown |rd_1 u|
    # is a few)
    lorentzian = col("rd", 0) / (jnp.exp(jnp.clip(col("rd", 1) * u, -30.0, 30.0)) * (u * u + col("rd", 2)))
    falloff = col("falloff", 0) * jnp.exp(-col("falloff", 1) * jnp.maximum(f - col("falloff", 2), 0.0))
    amp_rd = jnp.where(f >= col("falloff", 2), falloff, lorentzian)
    if tag == 32:
        a = [col("aux", j) for j in range(4)]
        aux = fma(fma(fma(a[3], f, a[2]), f, a[1]), f, a[0])
        amp_rd = jnp.where(f < col("f_aux"), aux, amp_rd)
        # the rotation: conj(-a322) h22 + conj(-a323) A_s exp(i phi_s)
        phi_s = (col("sph", 0) * f + col("sph", 1) + col("sph", 2) * basis[_INDEX[(-1, 0)]]
                 + col("sph", 3) * basis[_INDEX[(-3, 0)]] + at_rd)
        a22, phi22 = h22p
        m2r, m2i, m3r, m3i = (col("mix", j) for j in range(4))
        c22, s22 = a22 * jnp.cos(phi22), a22 * jnp.sin(phi22)
        cs, ss = amp_rd * jnp.cos(phi_s), amp_rd * jnp.sin(phi_s)
        re = c22 * m2r + s22 * m2i + cs * m3r + ss * m3i
        im = s22 * m2r - c22 * m2i + ss * m3r - cs * m3i
        # The modulus and the argument of re + i im scaled by its larger part
        # (constant for the derivatives, which are then those of |z| and
        # arg z): above the cutoff of the (2,2) the rotation is the tail of
        # the fall-off, ~1e-20 of the peak, whose square, in the derivatives
        # of hypot and atan2, is below the normal numbers of float32.
        scale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(re), jnp.abs(im)))
        nonzero = scale > 1e-37
        scale = jnp.where(nonzero, scale, 1.0)
        re = jnp.where(nonzero, re / scale, 1.0)
        im = jnp.where(nonzero, im / scale, 0.0)
        amp_rd = jnp.where(nonzero, scale * jnp.sqrt(re * re + im * im), 0.0)
        rest_rd = sums[2] + jnp.arctan2(im, re)
    else:
        rest_rd = sums[2] + at_rd
    rest = jnp.where(rd_phase, rest_rd, jnp.where(past_ins, sums[1] + at_int, sums[0]))

    # amplitude
    pn_re = col("pn_re", 6)
    pn_im = col("pn_im", 6)
    for k in range(5, -1, -1):
        pn_re = pn_re * t + col("pn_re", k)
        pn_im = pn_im * t + col("pn_im", k)
    t3 = t * t * t
    t7 = t3 * t3 * t
    pseudo = t7 * (col("pseudo", 0) + t * (col("pseudo", 1) + t * col("pseudo", 2)))
    amp_ins = m76 * (jnp.hypot(pn_re, pn_im) + pseudo)
    v = (f - col("inter_c")) * col("inter_h")
    inter = col("inter", 7)
    for j in range(6, -1, -1):
        inter = inter * v + col("inter", j)
    amp_int = m76 * inter
    amp = jnp.where(f < col("f_amp_in"), amp_ins, jnp.where(f >= col("f_amp_im"), amp_rd, amp_int))
    # (LAL's FALSE_ZERO for negative amplitudes)
    amp = jnp.where(amp < 0.0, ref.FALSE_ZERO * amp0, amp)
    return amp, rest, ~past_ins


def _big(fields22, name, index):
    return (_col(fields22, name, index).astype(jnp.float64)
            + jax.lax.stop_gradient(_col(fields22, name + "_lo", index)).astype(jnp.float64))


def modes_and_polarizations(fields, fp):
    """The modes ``{(l, m): h_{l,-m}(f)}`` (:data:`MODES`) and ``(h_+,
    h_x)`` at the frequencies (powers ``fp`` from
    :func:`..xas.core.frequency_powers`) for a table (:func:`unpack`)."""
    f22 = fields["xas"]
    amp22 = xas_core.amplitude(f22, fp)
    phi22p = xas_core.phase(f22, fp, anchor=False)
    phi22 = xas_core.phase(f22, fp)
    modes = {(2, 2): jax.lax.complex(amp22 * jnp.cos(phi22), amp22 * jnp.sin(phi22))}
    basis = hm_basis(fp)
    t0 = _big(f22, "big", 0)
    anchor = _big(f22, "big", 1)
    s = fields["s"][..., None]
    amp0 = fields["amp0"][..., None]
    for ell, emm in HIGHER_MODES:
        tag = 10 * ell + emm
        m = fields[f"hm{ell}{emm}"]
        amp, rest, inspiral = mode_rest(m, basis, fp, s, amp0, tag, (amp22, phi22p))
        m_over_2 = emm * 0.5
        phi = jnp.where(inspiral, (m_over_2 ** (8.0 / 3.0)) * t0 * fp[5], 0.0) - m_over_2 * anchor
        phi = xas_core.reduce(phi + rest.astype(jnp.float64)).astype(rest.dtype)
        sign = -1.0 if ell % 2 else 1.0
        a = sign * _col(m, "on") * amp
        modes[(ell, emm)] = jax.lax.complex(a * jnp.cos(phi), a * jnp.sin(phi))
    factors = fields["factors"]
    hp = 0.0
    hc = 0.0
    for k, mode in enumerate(MODES):
        fpc = jax.lax.complex(factors[..., k, 0, None], factors[..., k, 1, None])
        fcc = jax.lax.complex(factors[..., k, 2, None], factors[..., k, 3, None])
        hp = hp + fpc * modes[mode]
        hc = hc + fcc * modes[mode]
    return modes, (hp, hc)


def mode_point(mode_flat, f, s, amp0, a22, phi22p, tag):
    """:func:`mode_rest` of the higher mode ``tag`` at one frequency ``f``
    (scalars; the table of the mode flat): ``(amplitude, rest, inspiral)``,
    the last as 0 or 1. The function the MLX kernels are generated from."""
    fields = jax.tree_util.tree_map(lambda x: x[None], unpack_mode(mode_flat))
    f = jnp.reshape(f, (1,))
    fp = xas_core.frequency_powers(f)
    amp, rest, inspiral = mode_rest(fields, hm_basis(fp), fp, s, amp0, tag, (a22, phi22p))
    return amp[0, 0], rest[0, 0], inspiral[0, 0].astype(amp.dtype)


def evaluate(fields, fp):
    """``(h_+, h_x)`` (see :func:`modes_and_polarizations`)."""
    return modes_and_polarizations(fields, fp)[1]
