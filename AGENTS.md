# Notes for agents working on this repository

- **Plans stay out of git.** Write a new working plan to `<TOPIC>_PLAN.md`
  in the repository root, which `.gitignore` ignores (`/*_PLAN.md`). If a
  plan needs another name or place, add it to `.gitignore` in the same
  change. Do not commit plan files.
- **Keep README.md up to date.** When you add a feature or a model, or remove
  one, update the lists in `README.md` in the same change. The README says
  what is implemented, in plain language: no technical details and no timing
  or accuracy numbers. Those belong in the report (`report/`).
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
- **Tests**: `pytest` runs the quick tier (`tests/test_quick.py` and a few
  fast files, about a minute); `pytest tests` runs everything (an hour or
  more; `WAVEFORMX_SLOW=1` adds the random-binary suites). Compiled code is
  cached across runs in `~/.cache/waveformx/jax` (`WAVEFORMX_TEST_CACHE`
  moves it, empty turns it off): time compilation with the cache off.
- **MLX tests** (`tests/test_nrsur7dq4_mlx.py`, `tests/test_xpnr_mlx.py`,
  `tests/test_xas_mlx.py`, `tests/test_xhm_mlx.py`) are skipped without MLX: run them with a Python
  that has it (`MLX_PYTHON` in `report/measure_all.sh`). The MLX lean
  IMRPhenomXPNR (`models/xpnr/mlx`) is generated from the JAX lean model's
  jaxprs, so a change to the lean model changes it too: run
  `tests/test_xpnr_mlx.py` after changing the lean model or the code it
  calls. Likewise the set-up of the MLX IMRPhenomXAS is generated from
  `models/xas/core.table`, which calls IMRPhenomXPNR's (2,2) code
  (`internals.py`, `amp22.py`, `phase22.py`, the fits): run
  `tests/test_xas_mlx.py` after changing any of them. The MLX
  IMRPhenomXHM is generated from `models/xhm/core.py` (its set-up and
  `mode_point`), which also calls the higher-mode code of IMRPhenomXPNR
  (`hm/inspiral.py`, `hm/phase.py`, `hm/variables.py`): run
  `tests/test_xhm_mlx.py` after changing them.
- **Run LALSimulation single-threaded** (`OMP_NUM_THREADS=1`) when comparing:
  its PNR and MSA angles race under OpenMP.
- The reference implementation is the Rust crate waveformrs
  (`../waveformrs`), which reproduces LALSimulation 6.2.0 to rounding; its
  report (`report/waveformrs-report.html`) and comments describe LAL's
  quirks. Do not modify `../waveformrs` or `../lalsuite`.
- **The report** (`report/`) is built from measurements in `report/data`
  (`report/measure_all.sh` re-runs them all, on an idle machine). When a
  model, its accuracy or its cost changes, re-run the affected measurement
  and `python report/build_report.py`, and update the text of
  `report/template.html` where it describes what changed.
