# waveformx

Gravitational-wave waveform models for compact binaries, implemented in
[JAX](https://github.com/jax-ml/jax). The models can be
compiled with `jax.jit`, evaluated for many binaries at once with
`jax.vmap`, and differentiated with respect to the binary's parameters with
`jax.grad`. They follow [LALSimulation](https://git.ligo.org/lscsoft/lalsuite)
and are checked against it.

The package has the same interface as the Python bindings of
[waveformrs](https://github.com/ColmTalbot/waveformrs).

## Models

- **IMRPhenomD** (frequency domain): the dominant mode of binaries whose
  spins are aligned with the orbital angular momentum, from inspiral to
  ringdown, written for speed in the same way as IMRPhenomXAS. It follows
  LALSimulation, except that the ringdown and damping frequencies of the
  remnant come from closed-form fits in its spin rather than from
  LALSimulation's interpolation of a table; the fits are as accurate as
  that interpolation, and smooth, so that the derivatives are too
  (`scripts/fit_phenomd_qnm.py` makes them).
  - It runs in single precision (float32) when the binary's parameters are
    float32, including its derivatives, with the extra precision of the
    inspiral phase of IMRPhenomXAS.
  - Its output is the strain multiplied by 10^21 unless
    `scaled_strain=False`.
  - It is also available in MLX, in single precision on Apple GPUs: the
    polarizations for one binary or a batch, with their Jacobian and
    derivatives for `mx.grad`.
  - Like every model, it can take the binary at a reference frequency
    scaled by its total mass (`reference_mf`) instead of one in Hz; for
    this model that only moves where the reference phase is given.
    `to_mass_scaled_reference` and `to_physical_reference` convert
    binaries between the two conventions (also in MLX), so that both give
    the same waveform.
  - Its modes and the conversions between the time and frequency domains
    are not implemented.
- **IMRPhenomXAS** (frequency domain): the dominant mode of binaries whose
  spins are aligned with the orbital angular momentum, written to be fast
  for few frequencies, for many binaries sharing their frequencies and for
  long frequency grids. It follows LALSimulation, with the same arithmetic
  in single and double precision (masses in solar masses, two fits
  rearranged so that they keep their precision near maximal spins).
  - It runs in single precision (float32) when the binary's parameters are
    float32, including its derivatives; the largest part of the inspiral
    phase is computed with extra precision, so that it stays accurate for
    long inspirals such as those of binary neutron stars.
  - Its output is the strain multiplied by 10^21, so that it can be used in
    single precision; the option `scaled_strain=False` gives the strain
    itself, as LALSimulation does.
  - It is also available in [MLX](https://github.com/ml-explore/mlx), in
    single precision on Apple GPUs (optional dependency `mlx`): the
    polarizations at given frequencies for one binary or a batch, with
    their Jacobian and derivatives for `mx.grad`.
  - Like every model, it can take the binary at a reference frequency
    scaled by its total mass (`reference_mf`) instead of one in Hz; for
    this model that only moves where the reference phase is given.
    `to_mass_scaled_reference` and `to_physical_reference` convert
    binaries between the two conventions (also in MLX), so that both give
    the same waveform.
  - Its modes and the conversions between the time and frequency domains
    are not implemented.
- **IMRPhenomXHM** (frequency domain): binaries with aligned spins and
  higher modes, (2,±2), (2,±1), (3,±3), (3,±2) and (4,±4), written for speed
  in the same way as IMRPhenomXAS, whose (2,2) mode it shares. It follows
  LALSimulation's current release of the model with its default settings
  only (one code path), without multibanding.
  - It returns the spherical-harmonic modes as well as the polarizations
    (`modes`, and `choose_fd_modes` on LALSimulation's uniform grid): the
    modes whose sum is the polarizations. Mode arrays are not supported;
    select modes from the output instead.
  - Where LALSimulation takes derivatives of the waveform by finite
    differences to fix its coefficients, it uses exact derivatives, which
    changes the (3,2) mode slightly.
  - It runs in single precision (float32) when the binary's parameters are
    float32, including its derivatives, with the same arithmetic as in
    double precision and the extra precision of the inspiral phase of
    IMRPhenomXAS, so that binary neutron stars stay accurate.
  - Its output is the strain multiplied by 10^21 unless
    `scaled_strain=False`.
  - It is also available in MLX, in single precision on Apple GPUs: the
    polarizations and the modes for one binary or a batch, with their
    Jacobian and derivatives for `mx.grad`.
  - Like every model, it can take the binary at a reference frequency
    scaled by its total mass (`reference_mf`); only the reference phase
    moves, and the converters of IMRPhenomXAS (also in MLX) give a binary
    whose every mode is the same in both conventions.
  - The conversions between the time and frequency domains are not
    implemented.
- **IMRPhenomXPNR** (frequency domain): precessing spins with higher modes,
  tuned to numerical relativity. It keeps LALSimulation's physics (its
  fits, ansatzes and construction of the precession angles) with the
  parity fix from the development version of LALSuite, but not its
  numerical procedure, so it is close to LALSimulation rather than equal
  to it. Its derivatives are computed in forward mode, so `jax.jvp` and
  `jax.jacfwd` work as well as `jax.grad`, and `jax.vmap` evaluates a
  batch of binaries together.
  - It uses the SpinTaylor precession angles for every binary, also where
    LALSimulation switches to other angles (small in-plane spins, a failed
    integration of the precession equations), and LALSimulation's discrete
    switches in the precession angles are replaced by continuous versions,
    so that the waveform and its derivatives are continuous in the
    binary's parameters.
  - It solves the precession equations once with an accurate integrator
    and takes the angles from that solution, rather than repeating
    LALSimulation's integration step by step and interpolating its
    samples, and it takes the derivatives in the higher modes exactly
    where LALSimulation uses finite differences.
  - The discrete choices that the higher modes make in their amplitudes
    are kept as in LALSimulation, except that the test of whether the
    amplitude polynomial of the intermediate region crosses zero checks
    the polynomial itself (LALSimulation reads the wrong roots), which
    changes the higher modes of a few binaries slightly.
  - A coarser grid for interpolating the precession angles
    (`extended_phase_grid`) makes light binaries faster.
  - It can take the binary at a reference frequency scaled by its total
    mass (`reference_mf`, recommended `REFERENCE_MF`) instead of a
    reference frequency in Hz: the spins and the orientation are then
    given late in the inspiral, near merger, for every binary, and the
    waveform works for every choice of modes, including those whose only
    modes are the (2,1) ones, and whether or not the reference lies in the
    requested band. With this option the waveform does not depend on the
    choice of reference: whatever `reference_mf` is, the model first
    carries the binary to `REFERENCE_MF` and takes everything that
    depends on the reference there, so a binary converted between two
    references gives the same waveform. `to_mass_scaled_reference` and
    `to_physical_reference` convert binaries between the two conventions,
    as pure functions that can be differentiated (also in MLX: the option
    is available there too).
  - It is also available in [MLX](https://github.com/ml-explore/mlx), in
    single precision on Apple GPUs (optional dependency `mlx`): the
    polarizations at given frequencies for one binary or a batch, with
    derivatives with respect to every parameter of the binary. It is
    produced from the JAX model itself rather than written again, so the
    two compute the same waveform.
  - Its output is the strain multiplied by 10^21, so that it can be used in
    single precision; the option `scaled_strain=False` gives the strain
    itself, as LALSimulation does.
  - Its pure function runs in single precision (float32) when the binary's
    parameters are float32, including its derivatives, and stays close to
    its double-precision waveform there.
  - The version that reproduces LALSimulation's numerics step for step
    (with its switches to other precession angles and the behaviour of the
    LALSimulation releases without the parity fix as options) is commit
    `0bbd3c5`, tag `lal-faithful`.
- **NRSur7dq4** (time domain): numerical-relativity surrogate for precessing
  binaries with higher modes. It gives LALSimulation's polarizations and
  modes, with the same length and start time; its pure function evaluates
  the waveform at any times and can be differentiated with respect to every
  parameter of the binary. It needs the surrogate data file from LALSuite's
  data repository.
  - Its output is the strain multiplied by 10^21, so that it can be used in
    single precision; the option `scaled_strain=False` gives the strain
    itself, as LALSimulation does.
  - Its pure function runs in single precision (float32) when the binary's
    parameters are float32, including its derivatives.
  - It is also available in [MLX](https://github.com/ml-explore/mlx), in
    single precision on Apple GPUs (optional dependency `mlx`): the pure
    function for one binary or a batch, with derivatives; LALSimulation's
    polarizations and modes on its time grid, for one binary or a batch;
    and the conditioned waveforms in the time and frequency domains.
  - Like every model, it can take the binary at a reference frequency
    scaled by its total mass (`reference_mf`): the spins and the
    orientation are then given at the same point of the surrogate's
    inspiral for every mass, which lies inside the surrogate for every
    binary it covers at the recommended `REFERENCE_MF`, and the waveform
    can be differentiated with respect to every parameter without the
    reference moving with the mass. `to_mass_scaled_reference` and
    `to_physical_reference` convert binaries between the two conventions
    with the surrogate's own dynamics, as pure functions that can be
    differentiated (also in MLX).
- **TaylorF2** (frequency domain): the post-Newtonian inspiral of binaries
  with aligned spins in the stationary-phase approximation, with
  LALSimulation's default settings (the phase to the highest order, with
  the spin terms, and the leading-order amplitude), written for speed in
  the same way as IMRPhenomXAS. Tidal effects and the other orders of the
  phase and amplitude are not implemented. As in LALSimulation's
  evaluation at given frequencies, the waveform is not cut off at the
  last stable orbit; a reference frequency of zero means the phase is
  referred to coalescence.
  - It runs in single precision (float32) when the binary's parameters are
    float32, including its derivatives, with the extra precision of the
    inspiral phase of IMRPhenomXAS.
  - Its output is the strain multiplied by 10^21 unless
    `scaled_strain=False`.
  - It is also available in MLX, in single precision on Apple GPUs: the
    polarizations for one binary or a batch, with their Jacobian and
    derivatives for `mx.grad`.
  - Like every model, it can take the binary at a reference frequency
    scaled by its total mass (`reference_mf`) instead of one in Hz; for
    this model that only moves where the reference phase is given.
    `to_mass_scaled_reference` and `to_physical_reference` convert
    binaries between the two conventions (also in MLX), so that both give
    the same waveform.
  - Its modes and the conversions between the time and frequency domains
    are not implemented.

## Features

- The waveformrs interface: polarizations, modes and batches of binaries,
  with LALSimulation's names, parameters and options.
- Pure JAX functions of a `Binary` for use inside your own `jit`, `vmap` and
  `grad` code.
- LALSimulation's conditioned waveforms and its conversions between the time
  and frequency domains (`SimInspiralTD` and `SimInspiralFD`):
  IMRPhenomXPNR in the time domain and NRSur7dq4 in the frequency domain,
  each also conditioned in its own domain, with LAL's tapers, high-pass
  filter, lengths and start times. They are also available as pure
  functions of the binary, which keep the length and start time of a
  reference binary and can be differentiated.
- Matches between waveforms (`waveformx.match`), as pure functions that
  can be compiled, vectorised and differentiated: maximised over time and
  phase, or over time and the detector's polarisation response, with the
  power spectral density given by the user. Waveforms of time-domain models
  are compared through their conditioned frequency-domain form. A helper
  optimises a match over the parameters that models define differently
  (the reference phase and the orientation of the in-plane spins) from
  several starting points, using the derivatives of the models.
- The mass-scaled reference frequency (`reference_mf`) for every model,
  with converters for each (`waveformx.to_mass_scaled_reference` and
  `waveformx.to_physical_reference` with `approximant=`). The models share
  the orientation conventions, but not the definition of the orbital
  phase at the reference: the same binary in IMRPhenomXPNR and NRSur7dq4
  still differs by a reference phase that depends on the binary.

## Report

`report/` holds a technical report on how the models were implemented and
how they compare with LALSimulation and waveformrs, with the scripts that
make its measurements (`report/measure_all.sh`).

## License

waveformx is free software, licensed under the GNU General Public License
version 3 or later ([LICENSE](LICENSE)). It follows LALSuite, the GNU
Scientific Library and waveformrs; [NOTICE.md](NOTICE.md) lists the original
code.
