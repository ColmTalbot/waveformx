# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Check the built sdist and wheel (``python check_dist.py dist``).

* the wheel holds every module of ``src/waveformx`` (the generated ones and
  the MLX ports included), the version file written by setuptools-scm, and
  the licence files, with the licence in its metadata;
* the sdist holds the package and its tests, but not the report.
"""

import sys
import tarfile
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
dist = Path(sys.argv[1])
(wheel,) = dist.glob("waveformx-*.whl")
(sdist,) = dist.glob("waveformx-*.tar.gz")
errors = []

names = set(zipfile.ZipFile(wheel).namelist())
modules = {p.relative_to(root / "src").as_posix() for p in (root / "src" / "waveformx").rglob("*.py")}
modules.add("waveformx/_version.py")
errors += [f"{wheel.name}: missing {m}" for m in sorted(modules - names)]
errors += [f"{wheel.name}: unexpected {n}" for n in sorted(names) if not n.startswith(("waveformx/", "waveformx-"))]
(metadata,) = [n for n in names if n.endswith(".dist-info/METADATA")]
text = zipfile.ZipFile(wheel).read(metadata).decode()
for line in ("License-Expression: GPL-3.0-or-later", "License-File: LICENSE", "License-File: NOTICE.md"):
    if line not in text.splitlines():
        errors.append(f"{wheel.name}: no '{line}' in METADATA")
for licence in ("LICENSE", "NOTICE.md"):
    if not any(n.endswith(f".dist-info/licenses/{licence}") for n in names):
        errors.append(f"{wheel.name}: missing {licence}")

members = tarfile.open(sdist).getnames()
top = members[0].split("/")[0]
paths = {m.removeprefix(top + "/") for m in members}
for required in ("pyproject.toml", "LICENSE", "NOTICE.md", "README.md", "src/waveformx/__init__.py",
                 "src/waveformx/_version.py", "tests/conftest.py", "tests/test_quick.py"):
    if required not in paths:
        errors.append(f"{sdist.name}: missing {required}")
errors += [f"{sdist.name}: unexpected {p}" for p in sorted(paths) if p.startswith(("report/", "scripts/", ".github/"))]

print(f"{wheel.name}: {len(names)} files; {sdist.name}: {len(members)} members")
if errors:
    sys.exit("\n".join(errors))
