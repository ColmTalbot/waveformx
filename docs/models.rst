Models
======

Every model takes the binary at a reference frequency in Hz (``f_ref``) or
at one scaled by its total mass (``reference_mf``; see
:doc:`api/binaries`). Unless ``scaled_strain=False``, every model gives the
strain multiplied by 10\ :sup:`21`, so that it can be used in single
precision. Where a model departs from LALSimulation is described in
:doc:`lalsimulation`.

.. list-table::
   :header-rows: 1

   * - Model
     - Domain
     - Spins
     - Modes
     - Conditioned
     - PyTorch
     - MLX
   * - TaylorF2
     - frequency
     - aligned
     - (2,2)
     -
     - yes
     - yes
   * - IMRPhenomD
     - frequency
     - aligned
     - (2,2)
     -
     - yes
     - yes
   * - IMRPhenomXAS
     - frequency
     - aligned
     - (2,2)
     -
     - yes
     - yes
   * - IMRPhenomXHM
     - frequency
     - aligned
     - higher modes, returned
     -
     - yes
     - yes
   * - IMRPhenomXPNR, _NNLO, _MSA
     - frequency
     - precessing
     - higher modes, returned
     - time and frequency
     - yes
     - yes
   * - NRSur7dq4
     - time
     - precessing
     - higher modes, returned
     - time and frequency
     - yes
     - yes

TaylorF2
--------

The post-Newtonian inspiral of binaries with aligned spins, in the
stationary-phase approximation, with LALSimulation's default settings: the
phase to the highest order, with the spin terms, and the leading-order
amplitude. Tidal effects and the other orders of the phase and amplitude
are not implemented. A reference frequency of zero refers the phase to
coalescence.

IMRPhenomD
----------

The dominant mode of binaries with aligned spins, from inspiral to
ringdown. Its modes are not implemented.

IMRPhenomXAS
------------

The dominant mode of binaries with aligned spins, from inspiral to
ringdown, fast for few frequencies, for many binaries sharing their
frequencies and for long frequency grids. In single precision, the largest
part of the inspiral phase is computed with extra precision, so that it
stays accurate for long inspirals such as those of binary neutron stars;
TaylorF2, IMRPhenomD and IMRPhenomXHM do the same.

IMRPhenomXHM
------------

Binaries with aligned spins and higher modes: (2,±2), (2,±1), (3,±3),
(3,±2) and (4,±4). It returns the spherical-harmonic modes as well as the
polarizations (:func:`~waveformx.modes`, and
:func:`~waveformx.choose_fd_modes` on LALSimulation's uniform grid). Mode
arrays are not supported: select modes from the output instead.

IMRPhenomXPNR
-------------

Precessing spins with higher modes, tuned to numerical relativity. Its
derivatives are computed in forward mode, so ``jax.jvp`` and ``jax.jacfwd``
work as well as ``jax.grad``, and ``jax.vmap`` evaluates a batch of
binaries together. Batches and long frequency grids are computed a part at
a time, so that the memory a batch or a derivative needs does not grow with
the batch or with the number of frequencies.

It builds the precession angles on the SpinTaylor angles. Two other
precession prescriptions are models of their own: **IMRPhenomXPNR_NNLO**
builds them on the NNLO angles and **IMRPhenomXPNR_MSA** on the MSA angles.
Neither integrates the precession equations, so they compile and run
faster. They work everywhere IMRPhenomXPNR does: the functions of the
package, the conditioned waveforms, the reference converters, PyTorch and
MLX.

With ``reference_mf`` (recommended :data:`~waveformx.REFERENCE_MF`) the
spins and orientation are given late in the inspiral, near merger, for
every binary, and the waveform does not depend on the choice of reference:
a binary converted between two references gives the same waveform.

``extended_phase_grid=True`` (:class:`~waveformx.WaveformOptions`) selects
other interpolants of the precession angles, which follow the orbital
angular momentum where it passes close to the total angular momentum.

NRSur7dq4
---------

The numerical-relativity surrogate for precessing binaries with higher
modes. It gives LALSimulation's polarizations and modes, with the same
length and start time; its pure function
(:func:`~waveformx.td_polarizations_function`) evaluates the waveform at
any times and can be differentiated with respect to every parameter of the
binary. It needs the surrogate's data file (:doc:`installation`).

With ``reference_mf`` the spins and orientation are given at the same point
of the surrogate's inspiral for every mass, which lies inside the surrogate
for every binary it covers at the recommended
:data:`~waveformx.REFERENCE_MF`, and the waveform can be differentiated
with respect to every parameter without the reference moving with the mass.
