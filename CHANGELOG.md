# Changelog

All notable changes to pet-quantize are documented here.
Format follows Keep a Changelog; versions follow SemVer.

## [2.1.0] - 2026-04-23

Phase 7 — ecosystem optimization pass for pet-quantize. Peer-dep
governance fix (spec §5.1 #1), W&B residue guard, version parity,
latent CI-workflow bug fix, pin refreshes to Phase 5/6 matrix row,
and `architecture.md`.

### Added
- `docs/architecture.md` (9-章 template per ecosystem-optimization spec §4.1).
- `.github/workflows/no-wandb-residue.yml` — positive-list CI guard
  scanning first-party code for `\bwandb\b` matches (Phase 7 7C).
- `tests/test_version.py` — `test_version_attribute_matches_metadata`
  parity between `pet_quantize.__version__` and `importlib.metadata`.
- README Prerequisites + quick-start + entry-point snippet + on-device
  smoke command.

### Changed
- **pet-infra migrated from hardpin to β peer-dep** (spec §5.1 #1):
  `pyproject.toml` drops `pet-infra @ git+...@v2.5.0` from
  `dependencies`. CI install-order already matched the peer-dep shape
  (Step 1 explicit install); only the pyproject line changed.
- `pyproject.toml` pin bumps:
  - `pet-schema @ v2.4.0` → `@v3.2.1` (7B, α hardpin style — tag only).
- `.github/workflows/{ci,peer-dep-smoke}.yml`:
  - Step 1 `pet-infra @v2.5.0` → `@v2.6.0` (matrix 2026.09).
  - Step 4 assertion tightened `startswith('2.')` → `startswith('2.6')`.
- `.github/workflows/quantize_validate.yml` — fixed broken invocation
  (`python -m pet_quantize.validate --model X --device rknn` would have
  raised `No module named pet_quantize.validate.__main__` on any
  self-hosted rk3576 trigger). Now uses
  `pytest src/pet_quantize/validate/ --device-id <serial>` matching the
  conftest.py CLI contract; inputs renamed `model_artifact` →
  `device_id` + `model_dir`.
- `plugins/_register.py` — "Install via matrix row 2026.08" →
  "Install via latest matrix row (...)" (stale string); header comment
  added explaining the delayed-guard pattern (option X adjudicated in
  Phase 7 plan).
- `plugins/converters/{audio_rknn_fp16,vision_rknn_fp16,vlm_rkllm_w4a16}.py`
  — 3-line comments explaining per-converter lazy-import rationale
  (top-level safe when the pointed-at convert module lazy-imports the
  SDK; VLM kept lazy because pet-eval's
  `test_module_load_does_not_import_rkllm_runner` covers that chain).
- `tests/conftest.py:sample_params` — added `gates.{vlm,audio}` and
  `audio.*` sections to match `params.yaml` structure.

### Fixed
- `pet_quantize.__version__` synced to pyproject `2.1.0` (was `2.0.0`
  when pyproject was `2.0.1` — drift since Phase 4 P5-A-6).

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
