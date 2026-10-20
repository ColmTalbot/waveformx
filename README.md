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
  releases is available as an option.
- **NRSur7dq4** (time domain): numerical-relativity surrogate for precessing
  binaries. It needs the surrogate data file from LALSuite's data
  repository.

## Features

- The waveformrs interface: polarizations, modes and batches of binaries,
  with LALSimulation's names, parameters and options.
- Pure JAX functions of a `Binary` for use inside your own `jit`, `vmap` and
  `grad` code.

## License

waveformjax is free software, licensed under the GNU General Public License
version 3 or later ([LICENSE](LICENSE)). It follows LALSuite, the GNU
Scientific Library and waveformrs; [NOTICE.md](NOTICE.md) lists the original
code.
