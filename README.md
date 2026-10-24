# waveformjax

Gravitational-wave waveform models for precessing compact binaries,
implemented in [JAX](https://github.com/jax-ml/jax). The models can be
compiled with `jax.jit`, evaluated for many binaries at once with
`jax.vmap`, and differentiated with respect to the binary's parameters with
`jax.grad`. They follow [LALSimulation](https://git.ligo.org/lscsoft/lalsuite)
and are checked against it.

The package has the same interface as the Python bindings of
[waveformrs](https://github.com/ColmTalbot/waveformrs).

## Models

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
