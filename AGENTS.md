# Notes for agents working on this repository

- **Keep README.md up to date.** When you add a feature or a model, or remove
  one, update the lists in `README.md` in the same change. The README says
  what is implemented, in plain language: no technical details and no timing
  or accuracy numbers.
- **Record ported code in NOTICE.md.** The package is GPL-3.0-or-later
  because it follows LALSuite (GPL-2.0-or-later), GSL (GPL-3.0-or-later) and
  waveformrs (GPL-3.0-or-later). Before porting code from anywhere else, check
  that its license is compatible with GPL-3.0-or-later, and add what you port
  to `NOTICE.md`. Source files start with the same SPDX/copyright header as
  the existing ones.
- **The models must stay jit-compilable, vectorisable and differentiable.**
  Every model function takes a `Binary` pytree and returns arrays without
  Python control flow on traced values, so that `jax.jit`, `jax.vmap` and
  `jax.grad` (with respect to the binary's parameters) work. Run
  the model's `tests/test_transforms_<model>.py` after changing it.
- **Run LALSimulation single-threaded** (`OMP_NUM_THREADS=1`) when comparing:
  its PNR and MSA angles race under OpenMP.
- The reference implementation is the Rust crate waveformrs
  (`../waveformrs`), which reproduces LALSimulation 6.2.0 to rounding; its
  report (`report/waveformrs-report.html`) and comments describe LAL's
  quirks. Do not modify `../waveformrs` or `../lalsuite`.
