Differences from LALSimulation
==============================

The models follow LALSimulation 6.2.0: its fits, ansatzes, parameters and
conventions. Where they differ, it is on purpose: so that the waveforms and
their derivatives are smooth in the binary's parameters, or so that they
are fast and stay accurate in single precision. This page lists the
differences a user can see.

All models
----------

* **Units and scale.** Masses are in solar masses and distances in Mpc,
  rather than SI units. The output is the strain multiplied by
  10\ :sup:`21` unless ``scaled_strain=False``, which gives LALSimulation's
  strain.
* **Single precision.** The pure functions run in float32 when the binary's
  parameters are float32, which LALSimulation does not offer.
* **Mass-scaled reference frequency.** ``reference_mf`` gives the binary at
  a reference frequency scaled by its total mass, an option LALSimulation
  does not have. With ``f_ref`` in Hz the models behave as LALSimulation's.
* **Options.** Only the ``LALDict`` settings in
  :class:`~waveformx.WaveformOptions` are supported, with LALSimulation's
  defaults for everything else.

IMRPhenomD
----------

The ringdown and damping frequencies of the remnant come from closed-form
fits in its spin rather than from LALSimulation's interpolation of a table.
The fits are as accurate as that interpolation, and smooth, so that the
derivatives are too.

IMRPhenomXHM
------------

* Only LALSimulation's current release of the model with its default
  settings is implemented, without multibanding.
* Where LALSimulation takes derivatives of the waveform by finite
  differences to fix its coefficients, waveformx uses exact derivatives,
  which changes the (3,2) mode slightly.

IMRPhenomXPNR
-------------

waveformx keeps LALSimulation's physics, but not its numerical procedure,
so its waveforms are close to LALSimulation's rather than equal to them.

* **The parity fix.** It includes the parity fix from the development
  version of LALSuite.
* **The precession angles.** It uses the SpinTaylor angles for every
  binary, also where LALSimulation switches to other angles (small in-plane
  spins, a failed integration of the precession equations).
  LALSimulation's discrete switches in the precession angles are replaced
  by continuous versions, so that the waveform and its derivatives are
  continuous in the binary's parameters. The waveform tends smoothly to
  that of aligned spins as the in-plane spins vanish, and is continuous at
  equal masses.
* **The precession equations** are solved once with an accurate
  integrator, and the angles are taken from that solution, rather than by
  repeating LALSimulation's integration step by step and interpolating its
  samples.
* **The interpolation grid.** The precession angles are interpolated with
  LALSimulation's splines, on a grid that is fine at low frequencies and
  coarser above, rather than on LALSimulation's uniform grid, so that long
  waveforms of light binaries are fast.
* **Higher modes.** Where LALSimulation takes derivatives by finite
  differences, waveformx uses exact derivatives. The discrete choices that
  the higher modes make in their amplitudes are kept, except that the test
  of whether the amplitude polynomial of the intermediate region crosses
  zero checks the polynomial itself, which changes the higher modes of a
  few binaries slightly.
* **The NNLO and MSA prescriptions** are the separate models
  IMRPhenomXPNR_NNLO and IMRPhenomXPNR_MSA rather than a setting
  (``precession_version`` is accepted and ignored). Each is used for every
  binary, without LALSimulation's switches between them; where the MSA
  angles cannot be computed, IMRPhenomXPNR_MSA raises an error rather than
  falling back to the NNLO angles. Both are continuous as the in-plane
  spins vanish, which for the MSA angles departs from LALSimulation at
  small in-plane spins.

The version of waveformx that reproduces LALSimulation's numerical
procedure step for step, with its switches to other precession angles, is
the git tag ``lal-faithful``.

NRSur7dq4
---------

The waveform is LALSimulation's, on its grid and with its length and start
time. The pure function evaluates it at any times and can be differentiated
with respect to every parameter.

TaylorF2
--------

Only LALSimulation's default settings are implemented: the phase to the
highest order with the spin terms, and the leading-order amplitude, without
tidal effects. As in LALSimulation's evaluation at given frequencies, the
waveform is not cut off at the last stable orbit.
