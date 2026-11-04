# Changelog

[v0.10.0] – 2026-11-04

Added

- Notebook support for downloading the archived regression input from Zenodo
  when it is missing, with checksum verification and reuse of the
  regression-owned configured artifact path.
- A config-driven in-memory analysis API for matching, disambiguation, and
  mobility workflows.
- Synthetic affiliation information to the synthetic-data example.

Changed

- Updated the analysis notebooks and synthetic fixtures for the
  configuration-driven pipeline architecture.
- Updated CI to use a supported `setup-uv` release and added Flake8
  configuration.
