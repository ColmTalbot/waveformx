Development
===========

Tests
-----

``pytest`` runs the quick tier (``tests/test_quick.py`` and
``tests/test_api.py``); ``pytest tests`` runs everything, which takes an
hour or more. The tests compare with LALSimulation where it is installed
(``python-lalsimulation`` 6.2.0 from conda-forge) and use NRSur7dq4's data
file where it is found (:doc:`installation`); otherwise those tests are
skipped. ``AGENTS.md`` in the repository has the details.

Continuous integration
----------------------

GitHub Actions runs the quick tier on Linux and macOS for every pull
request and every push to ``main``, builds the sdist and wheel and checks
them (``.github/workflows/tests.yml``), and builds this documentation
(``.github/workflows/docs.yml``), which is published to GitHub Pages from
``main``.

Releases
--------

The version comes from the git tag (setuptools-scm); between releases it
is a development version such as ``0.2.dev3+g1234567``. To release:

1. tag the commit ``vX.Y.Z`` and push the tag;
2. publish a GitHub release from that tag.

``.github/workflows/release.yml`` then builds the sdist and wheel, checks
that their version is the tag's, and uploads them to PyPI with trusted
publishing. Running that workflow by hand (``workflow_dispatch``) uploads a
development version to TestPyPI instead.
