# Phase 3B Audit: pet-quantize v1 Delete vs Preserve

Date: 2026-04-22
Branch: feature/phase-3b-audit-delete-v1
Base SHA: 184aeb2

## Summary

pet-quantize v1 surface is minimal. There is no `cli.py`, no `__main__.py`.
The only v1 components to purge are `config.py` (WandbConfig-bearing Pydantic
model), the `wandb` dependency, and their associated tests.

No `[project.scripts]` entries exist in `pyproject.toml`.

---

## Files to DELETE

### src/

| File | Reason |
|------|--------|
| `src/pet_quantize/config.py` | Houses `WandbConfig` + `QuantizeParams` root model tying vision/llm/audio triple into one rigid schema. Replaced by per-plugin config dicts read from params.yaml. Also contains `setup_logging()` wrapper — callers already import `pet_infra.logging.setup_logging` directly. |

Note: `cli.py` and `__main__.py` do not exist. Nothing to delete there.

### tests/

| File | Reason |
|------|--------|
| `tests/test_config.py` | Tests `pet_quantize.config.{QuantizeParams,load_params,setup_logging}` — all deleted. |

### Partial edits (wandb references in surviving files):

| File | Change |
|------|--------|
| `tests/conftest.py` | Remove `"wandb"` section from `sample_params` fixture dict (lines 85-88). The fixture is shared by other tests; we keep the file but strip the wandb key. |

### pyproject.toml

| Location | Change |
|----------|--------|
| `dependencies` list | Remove `"wandb"` dependency line. No `[project.scripts]` section exists. |

---

## Files to PRESERVE (SDK wrappers)

| Module | Path |
|--------|------|
| convert | `src/pet_quantize/convert/` — all 4 files |
| calibration | `src/pet_quantize/calibration/` — both files |
| inference | `src/pet_quantize/inference/` — all 3 files |
| packaging | `src/pet_quantize/packaging/` — all 3 files |
| validate | `src/pet_quantize/validate/` — all 4 test files + conftest |

**Dependency check:** None of the preserved modules import from
`pet_quantize.config`. They accept `dict` configs passed by callers, or
import `pet_infra.logging` directly. No inlining of constants required.

---

## Surprises / Notes

1. **No actual `import wandb` in source code.** The only wandb usage is:
   - `config.py`: `WandbConfig` Pydantic model (no runtime wandb calls)
   - `tests/conftest.py`: fixture dict has a `"wandb"` key (data, not import)
   - `tests/test_config.py`: asserts on `params.wandb.*` fields
   The `wandb` package is only a declared dependency in `pyproject.toml`.
   There are no `import wandb` / `wandb.*` runtime calls to remove.

2. **`setup_logging()` in config.py** is a thin wrapper around
   `pet_infra.logging.setup_logging`. It is not imported by any preserved
   module. Safe to delete with `config.py`.

3. **`pyproject.toml` has no `[project.scripts]`** — confirmed via grep.
   No CLI entry-point removal needed.

4. **`tests/conftest.py` `sample_params` fixture** includes a `"wandb"` key
   that feeds into `test_config.py`. After deleting `test_config.py`, this
   fixture is still used by `test_build_calib.py` and `test_validate_calib.py`
   (they use `sample_params_path` which writes the full dict to yaml). Those
   modules read from `params["calibration"]` — the extra `"wandb"` key in the
   yaml is harmless but should be removed for cleanliness.

---

## Post-delete expected state

- `src/pet_quantize/` contains: `__init__.py`, `calibration/`, `convert/`,
  `inference/`, `packaging/`, `validate/`
- `tests/` contains all test files except `test_config.py`
- `pyproject.toml` has no `wandb` dependency
- All preserved imports resolve successfully (no dependency on deleted config.py)
