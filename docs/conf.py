# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The Sphinx configuration of the documentation (``pip install
".[docs]"``, then ``sphinx-build -W docs docs/_build/html``)."""

import waveformx

project = "waveformx"
author = "Colm Talbot"
copyright = "2026, Colm Talbot"
release = waveformx.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# the MLX ports import mlx, which is only for Apple silicon
autodoc_mock_imports = ["mlx"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"exclude-members": "__init__, __new__"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

exclude_patterns = ["_build"]
html_theme = "furo"
html_title = f"waveformx {release}"
