# waveformjax

Gravitational-wave waveform models for compact binaries, implemented in
[JAX](https://github.com/jax-ml/jax). The models can be
compiled with `jax.jit`, evaluated for many binaries at once with
`jax.vmap`, and differentiated with respect to the binary's parameters with
`jax.grad`. They follow [LALSimulation](https://git.ligo.org/lscsoft/lalsuite)
and are checked against it.

The package has the same interface as the Python bindings of
[waveformrs](https://github.com/ColmTalbot/waveformrs).

## Models

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
  - The conversions between the time and frequency domains are not
    implemented.
- **IMRPhenomXPNR** (frequency domain): precessing spins with higher modes,
  tuned to numerical relativity. By default it includes the parity fix from
  the development version of LALSuite; the behaviour of the LALSimulation
  releases is available as an option. Its derivatives are computed in
  forward mode, so `jax.jvp` and `jax.jacfwd` work as well as `jax.grad`,
  and `jax.vmap` evaluates a batch of binaries together.
  - By default the model only uses LALSimulation's SpinTaylor precession
    angles, and reports an error (NaN from the pure functions) where
    LALSimulation would switch to other angles, such as for binaries with
    aligned spins. LALSimulation's switches to the other angles are
    available as an option (`version_fallback=True`); the model then
    compiles more slowly.
  - One of LALSimulation's checks of the higher modes, whether the
    amplitude polynomial of the intermediate region crosses zero, reads
    the wrong roots in LALSimulation; it is corrected here, which changes
    the higher modes of a few binaries slightly.
  - An extended variant (`extended_spintaylor=True`), which is not
    LALSimulation's waveform: it uses the SpinTaylor angles for every
    binary, including those LALSimulation cannot evaluate with them,
    replaces LALSimulation's discrete switches in the precession angles
    with continuous versions, and interpolates the angles in a form that
    stays smooth where the orbital angular momentum passes close to the
    total one, so that the waveform and its derivatives are continuous in
    the binary's parameters. The discrete choices that the higher modes
    make in their amplitudes are kept as in LALSimulation. By default it
    takes the precession angles directly from the solution of the
    precession equations rather than from LALSimulation's splines through
    samples of it (`extended_direct_angles`), which is much faster for
    light binaries; a coarser grid for interpolating the angles
    (`extended_phase_grid`) makes light binaries faster still.
  - A lean variant (`lean=True`): the extended variant with the parity fix,
    without the parts that only reproduce LALSimulation's numerical
    procedure. It solves the precession equations once with an accurate
    integrator instead of repeating LALSimulation's integration step by
    step, takes the derivatives in the higher modes exactly, and keeps
    masses and stored coefficients at full precision. LALSimulation's
    physics is unchanged, and the waveform is the extended variant's up
    to the accuracy of the integration. It compiles and runs faster than
    the other variants, and works in single precision.
  - The lean variant can take the binary at a reference frequency scaled
    by its total mass (`reference_mf`, recommended `REFERENCE_MF`) instead
    of a reference frequency in Hz: the spins and the orientation are then
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
  - The lean variant is also available in
    [MLX](https://github.com/ml-explore/mlx), in single precision on Apple
    GPUs (optional dependency `mlx`): the polarizations at given
    frequencies for one binary or a batch, with derivatives with respect
    to every parameter of the binary. It is produced from the JAX model
    itself rather than written again, so the two compute the same
    waveform.
  - Its output is the strain multiplied by 10^21, so that it can be used in
    single precision; the option `scaled_strain=False` gives the strain
    itself, as LALSimulation does.
  - Its pure function runs in single precision (float32) when the binary's
    parameters are float32, including its derivatives. The extended variant
    stays close to its double-precision waveform there; the default model,
    which repeats LALSimulation's integration of the precession equations
    step by step, is much less accurate in single precision.
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

## Report

`report/` holds a technical report on how the models were implemented and
how they compare with LALSimulation and waveformrs, with the scripts that
make its measurements (`report/measure_all.sh`).

## License

waveformjax is free software, licensed under the GNU General Public License
version 3 or later ([LICENSE](LICENSE)). It follows LALSuite, the GNU
Scientific Library and waveformrs; [NOTICE.md](NOTICE.md) lists the original
code.
