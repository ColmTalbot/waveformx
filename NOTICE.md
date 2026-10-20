# Notices

waveformjax is Copyright (C) 2026 Colm Talbot.

waveformjax is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
The full text is in [LICENSE](LICENSE).

## Code this work is derived from

waveformjax implements IMRPhenomXPNR and NRSur7dq4 following
[LALSuite](https://git.ligo.org/lscsoft/lalsuite) and its Rust translation
[waveformrs](https://github.com/ColmTalbot/waveformrs) (GPL-3.0-or-later),
which also translates parts of the GNU Scientific Library. The copyright of
the original code stays with its authors. waveformrs's `NOTICE.md` lists the
LALSuite and GSL files of each part of these models, with their copyright
holders and licences: LALSuite is licensed under the GNU General Public
License version 2 or later, used here under version 3 or later, and GSL under
version 3 or later.

The table below lists the original files of each module of waveformjax. Keep
it up to date when code is ported.

| waveformjax | Original files |
|---|---|
| `src/waveformjax/models/xpnr/` | LALSuite: `LALSimIMRPhenomX*.c`, `LALSimIMRPhenomX_PNR*.c`, `LALSimIMRPhenomX_AntisymmetricWaveform.c`, `LALSimInspiralSpinTaylor.c`, `LALSimInspiralPNCoefficients.c`, lal `SphericalHarmonics.c`; GSL: `specfunc/elljac.c`, `specfunc/ellint.c`, `interpolation/cspline.c`, `poly/zsolve.c`, `ode-initval/rkf45.c` (as listed in waveformrs's `NOTICE.md`) |
| `src/waveformjax/models/nrsur7dq4/` | LALSuite: `LALSimIMRPrecessingNRSur.c`, `LALSimNRSurrogateUtilities.c`; GSL: `interpolation/cspline.c` |

## Data

The NRSur7dq4 data file (`NRSur7dq4_v1.0.h5`) is not part of this repository;
it is distributed with LALSuite's data.
