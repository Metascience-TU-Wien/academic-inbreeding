# Changelog

## Unreleased

Added

- Synthetic mobility examples for university-affiliation inbreeding detection,
  with mapped TU Wien affiliations producing positive proxy cases for Ada and
  Jamie. The mobility notebook now displays only positive proxy profiles and
  omits requested columns that are absent from the result schema.

Changed

- Added an explanation to disambiguation reports when a hard publication-history
  safeguard overrides the raw matching score and keeps profiles distinct.

Fixed

- Prevented a candidate author's own name from being counted as shared-coauthor
  evidence when publication and profile names use different word orders.

[v0.10.1] – 2026-11-04

Fixed

- Included the missing name-matching and name-disambiguation modules required
  by workflow imports and test collection, fixing the GitHub Actions collection
  failure from v0.10.0.

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
