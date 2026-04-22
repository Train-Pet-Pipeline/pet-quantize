# Changelog

All notable changes to pet-quantize are documented here.
Format follows Keep a Changelog; versions follow SemVer.

## [2.0.1] - 2026-04-22

Phase 4 P5-A-6 final cut. Peer-dep pins forwarded to matrix 2026.09.

### Changed
- `pyproject.toml` peer-dep pins (P5-A-6):
  - `pet-schema @ git+...@v2.3.1` → `@v2.4.0`
  - `pet-infra @ git+...@v2.4.0` → `@v2.5.0`
- CI peer-dep pin `pet-infra @ git+...@v2.4.0` → `@v2.5.0` in both
  `ci.yml` and `peer-dep-smoke.yml`.

### Removed
- W&B residue (P2-C-2, shipped to dev 2026-04-22 ahead of this release):
  - `wandb` block deleted from `params.yaml`.
  - `wandb/` entry removed from `.gitignore`.

  ClearML (orchestrator P0-B/C) is the sole experiment tracker.

## [2.0.0] - 2026-04-21

Phase 3A — initial pet-infra v2 plugin port. See git history for details.
