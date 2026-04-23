# pet-quantize Architecture

## §1 Repository Responsibility

**pet-quantize** is the quantization / edge-conversion / packaging stage of the smart pet feeder pipeline.

It ships three registry-backed surfaces consumed by the `pet_infra` orchestrator:

1. **4 CONVERTERS plugins** (`pet_infra.registry.CONVERTERS`) —
   - `noop_converter` (zero-dep CI smoke),
   - `vlm_rkllm_w4a16` (VLM → RKLLM W4A16 for RK3576),
   - `audio_rknn_fp16` (audio CNN → RKNN FP16),
   - `vision_rknn_fp16` (ViT → RKNN FP16).
2. **3 DATASETS plugins** (`pet_infra.registry.DATASETS`) — content-addressable calibration subsets for VLM / vision / audio, writing tensor batches to `.cache/calibration` and exposing their URIs via `ModelCard.intermediate_artifacts["calibration_batch_uri"]`.
3. **2 inference runners** (`pet_quantize.inference.{rkllm_runner,rknn_runner}`) — PC-simulation and on-device (ADB) dual-mode; `RKLLMRunner` is imported at runtime by pet-eval's `QuantizedVlmEvaluator`.

Pipeline position:

```
pet-train → pet-eval → [pet-quantize] → pet-ota
                           │
                           └─ inference.rkllm_runner.RKLLMRunner
                              (cross-repo runtime consumer: pet-eval
                              QuantizedVlmEvaluator, lazy import)
```

**Does:**
- Accepts a trained `ModelCard` with `checkpoint_uri`; consumes an upstream calibration batch via `intermediate_artifacts`; produces an updated `ModelCard` with an `EdgeArtifact` (format `rkllm` / `rknn`) and a `QuantConfig`.
- Packages converted artifacts into signed release tarballs with manifest + SHA-256 checksums.
- Provides on-device smoke validation (`validate/`) runnable on a self-hosted rk3576 via the `quantize_validate` workflow.
- Reads every numeric threshold / sample rate / max_new_tokens / calibration-distribution from `params.yaml`.

**Does not:**
- Train checkpoints (pet-train) or evaluate metrics & gates (pet-eval).
- Publish OTA updates (pet-ota); packaging output is the handoff boundary.
- Maintain its own experiment tracker — `pet_infra.orchestrator + ClearMLLogger` is the sole logging path (W&B physically removed in Phase 4 P2-C-2; guarded by `no-wandb-residue.yml`).

---

## §2 I/O Contract

### Upstream dependencies

| Dependency | Mode | Locked version |
|---|---|---|
| pet-schema | α hardpin (in `pyproject.dependencies`) | v3.2.1 |
| pet-infra | β peer-dep (NOT in `pyproject.dependencies` as of v2.1.0) | v2.6.0 (compatibility_matrix 2026.09) |
| `rknn.api` / `rkllm.api` | optional SDK (vendor, not on PyPI) | set `PET_ALLOW_MISSING_SDK=1` in dev/CI |

CI install order (`.github/workflows/ci.yml` + `peer-dep-smoke.yml`) is 4-step: pet-infra peer-dep → editable `--no-deps` → re-resolve dev extras → version-prefix assertion on `pet_infra.__version__.startswith('2.6')`.

### Inputs

| Source | Consumer | Notes |
|---|---|---|
| `input_card: ModelCard` (pet-schema) | every CONVERTERS plugin | must carry `checkpoint_uri`; VLM cluster also needs `intermediate_artifacts["calibration_batch_uri"]` |
| Calibration source URIs | DATASETS plugins | modality-specific loaders (`calibration/{vlm,vision,audio}_loader.py`) |
| `params.yaml` | every plugin + packaging + validate | calibration / convert / inference / gates / validate / packaging / audio sections |

### Outputs

- `ModelCard` with appended `EdgeArtifact` + `QuantConfig`.
- Release tarball + manifest.json via `packaging/{build,sign,verify}_package.py`.
- On-device smoke reports from the `validate/` suite when executed against a real rk3576 (pytest artefacts).

### Downstream consumers

- **pet-eval:** `QuantizedVlmEvaluator` lazy-imports `pet_quantize.inference.rkllm_runner.RKLLMRunner` to run inference against the packaged `.rkllm` artifact — the only cross-repo runtime coupling out of pet-quantize.
- **pet-ota:** consumes the signed release tarball; manifest + SHA-256 checksums drive differential update packaging.

---

## §3 Architecture Overview

### Directory tree

```
src/pet_quantize/
├── __init__.py                            ← __version__ = "2.1.0"
├── calibration/                           ← per-modality data loaders + config validator
│   ├── build_calib_dataset.py             ← assemble balanced calibration set from SQLite frames
│   ├── validate_calib.py                  ← distribution / exclude-list admission rules
│   ├── audio_loader.py
│   ├── vision_loader.py
│   └── vlm_loader.py                      ← (image, prompt) pairs → (num_samples, 2048) int64
├── convert/                               ← SDK-bound wrapper functions (lazy-import rknn / rkllm / transformers)
│   ├── convert_audio.py                   ← PyTorch → ONNX → RKNN FP16
│   ├── convert_to_rknn.py                 ← vision ONNX → RKNN FP16
│   ├── convert_to_rkllm.py                ← HF weights + calib → RKLLM W4A16
│   └── export_vision_encoder.py           ← HF weights → ONNX
├── inference/                             ← dual-mode runners (PC simulation ↔ ADB on-device)
│   ├── rknn_runner.py                     ← RKNNRunner (top-level `from rknn.api import RKNN`)
│   ├── rkllm_runner.py                    ← RKLLMRunner (top-level `from rkllm.api import RKLLMRuntime`)
│   └── pipeline.py                        ← orchestrates ViT → LLM inference; FP16 vs quantized comparison
├── packaging/
│   ├── build_package.py                   ← tarball + manifest.json with SHA-256 + prompt files from pet-schema
│   ├── sign_package.py                    ← cryptographic signature
│   └── verify_package.py                  ← signature + manifest verification
├── plugins/
│   ├── _register.py                       ← entry-point target; delayed pet-infra guard; SDK-gated clusters
│   ├── converters/
│   │   ├── noop.py                        ← @CONVERTERS "noop_converter"  (zero-dep)
│   │   ├── vlm_rkllm_w4a16.py             ← @CONVERTERS "vlm_rkllm_w4a16"  (RKLLM)
│   │   ├── audio_rknn_fp16.py             ← @CONVERTERS "audio_rknn_fp16"  (RKNN)
│   │   └── vision_rknn_fp16.py            ← @CONVERTERS "vision_rknn_fp16" (RKNN)
│   └── datasets/
│       ├── vlm_calibration_subset.py      ← @DATASETS "vlm_calibration_subset"
│       ├── vision_calibration_subset.py   ← @DATASETS "vision_calibration_subset"
│       └── audio_calibration_subset.py    ← @DATASETS "audio_calibration_subset"
└── validate/                              ← on-device pytest smoke (NOT in default testpaths)
    ├── conftest.py                        ← adds --device-id CLI option
    ├── test_schema_compliance.py
    ├── test_kl_divergence.py
    ├── test_latency.py
    └── test_audio_accuracy.py

params.yaml                                ← calibration / convert / inference / gates / validate / packaging / audio
.github/workflows/
├── ci.yml                                 ← 4-step peer-dep install + ruff + mypy + pytest
├── peer-dep-smoke.yml                     ← isolated install-order contract test + register_all smoke
├── quantize_validate.yml                  ← self-hosted rk3576 smoke via pytest validate/ --device-id
└── no-wandb-residue.yml                   ← positive-list CI guard (Phase 7 7C)
```

### High-level dataflow

```
orchestrator                                            pet_quantize
─────────────                                            ────────────
recipe.yaml ──► compose_recipe ──► stage ──► CONVERTERS/DATASETS runner
                                                     │
                                                     │ _load_stage_kwargs(stage)
                                                     ▼
                                              plugin_cls(**kwargs)
                                                     │
                                                     ▼
                                            run(input_card, recipe)
                                                     │
                        (DATASETS)                  │                (CONVERTERS)
                  content-address cache ◄───────────┼────────────► lazy-import SDK wrapper
                    (.cache/calibration)            │                (convert_audio /
                                                     │                 convert_to_rknn /
                                                     │                 convert_to_rkllm)
                                                     ▼
                                          ModelCard with EdgeArtifact + QuantConfig
                                                     │
                                                     ▼
                                       packaging.build_package + sign_package
                                                     │
                                                     ▼
                                       tarball + manifest.json → pet-ota
```

---

## §4 Core Modules

### 4.1 `plugins/_register.py` — entry-point target

Declared in `pyproject.toml` under `[project.entry-points."pet_infra.plugins"] pet_quantize = …`. Layered guards so partial installs surface useful errors:

1. **pet-infra peer-dep guard** — inside `register_all()` (delayed variant, option X): raises `RuntimeError("pet-quantize requires pet-infra to be installed first. Install via latest matrix row (…)")` when `import pet_infra` fails. Kept inside the function so bare `import pet_quantize` stays cheap for IDE / static-analysis tooling.
2. **Always-available cluster** — `pet_quantize.plugins.converters.noop` (zero-dep).
3. **RKNN-gated cluster** — `try: import rknn.api` then import the 2 RKNN converter plugins + their 2 DATASETS feeders. On ImportError: re-raise unless `PET_ALLOW_MISSING_SDK=1`, in which case `logger.warning("rknn SDK missing; gated plugins skipped: …")`.
4. **RKLLM-gated cluster** — same pattern for `rkllm.api` + VLM converter + VLM dataset.

### 4.2 CONVERTERS plugins

All four implement the same contract: `run(input_card, recipe) -> ModelCard` appends an `EdgeArtifact` + sets `QuantConfig`.

| Plugin | SDK | Calibration needed | Output format |
|---|---|---|---|
| `noop_converter` | none | no | `onnx` placeholder (`noop:{card.id}` bytes) |
| `audio_rknn_fp16` | rknn | no (FP16) | `rknn` |
| `vision_rknn_fp16` | rknn | no (FP16, optimization level 3) | `rknn` + `vision_onnx_uri` in `intermediate_artifacts` |
| `vlm_rkllm_w4a16` | rkllm | **yes** (`calibration_batch_uri` required) | `rkllm` W4A16 |

All EdgeArtifacts carry `target_hardware=[target_platform]` (default `rk3576`) + SHA-256 over the final file bytes + `input_shape` spec.

### 4.3 DATASETS plugins

Content-addressable cache keyed on `sha256(modality|source_uri|num_samples)[:16]` — so orchestrator resume-from-cache sees a stable `card_id` across runs whenever the triple is unchanged. Output batches are cached under `cache_dir` (default `.cache/calibration`) as `.pt` tensors.

| Plugin | Output shape | Loader |
|---|---|---|
| `vlm_calibration_subset` | `(num_samples, 2048)` int64 | `calibration.vlm_loader.load_calibration_pairs` |
| `vision_calibration_subset` | per vision spec | `calibration.vision_loader` |
| `audio_calibration_subset` | per audio spec | `calibration.audio_loader` |

### 4.4 Inference runners (dual-mode)

Both `RKLLMRunner` and `RKNNRunner` expose the same lifecycle:

```
runner = RKLLMRunner(model_path, target=None, device_id=None)   # PC simulation
# or
runner = RKLLMRunner(model_path, target="rk3576", device_id="ADB_SERIAL")  # on-device

runner.init()                         # constructs RKLLMRuntime with/without target+device_id
text, latency_ms = runner.generate(prompt, visual_features, max_tokens=2048)
runner.release()                      # idempotent; the pet-eval consumer always wraps in try/finally
```

`FileNotFoundError` on missing `model_path` is raised eagerly from `__init__` — fail early before wasting an SDK init.

### 4.5 Packaging

- `build_package.py` collects converted `*.rknn` + `*.rkllm` via `_MODEL_FILE_MAP` glob patterns; adds prompt files read from the installed `pet_schema` package; emits `manifest.json` with per-file SHA-256 + timestamp + version metadata.
- `sign_package.py` signs the tarball with a pet-ota-trusted key.
- `verify_package.py` checks signature + manifest digests; used by pet-ota before publishing an update.

### 4.6 `validate/` on-device smoke

Pytest-native suite exposed under `src/pet_quantize/validate/` (not `tests/`) so it ships with the installed package but is excluded from default `pytest tests/` via `[tool.pytest.ini_options] testpaths = ["tests"]`. The `quantize_validate` workflow invokes it on `[self-hosted, rk3576]` runners via:

```bash
pytest src/pet_quantize/validate/ -v --device-id <adb-serial>
```

The conftest adds `--device-id` and loads `params.yaml` for thresholds; tests cover schema compliance, KL divergence (quantized vs FP16 reference), latency P95, and audio accuracy.

---

## §5 Extension Points

### Adding a converter

1. Drop `src/pet_quantize/plugins/converters/<name>.py` with a class decorated `@CONVERTERS.register_module(name="<name>")`.
2. Accept `**kwargs` in `__init__`; expose `run(input_card, recipe) -> ModelCard`.
3. If SDK-bound, import the SDK wrapper lazily inside `run()` (VLM pattern) *or* rely on the wrapper function itself lazy-importing the SDK (RKNN pattern). See §8.6.
4. Append the import to `_register.py` inside the matching SDK-gated cluster (or keep module-top if SDK-free).
5. Update `peer-dep-smoke.yml` expected-CONVERTERS set only if the plugin registers unconditionally.

### Adding a calibration dataset

1. `src/pet_quantize/plugins/datasets/<name>.py` decorated `@DATASETS.register_module`.
2. Implement content-addressable cache key (`sha256(modality|source_uri|num_samples)`) and write output tensor to `cache_dir`.
3. Return `input_card.model_copy(update={"intermediate_artifacts": {**..., "calibration_batch_uri": str(cache_path)}})` so a downstream CONVERTERS consumer can pick it up.

### Adding an inference runner mode

Both runners already cover PC-sim vs on-device via `target` / `device_id`. A third mode (e.g., remote inference server) would subclass the runner interface; the pet-eval consumer currently only calls `init / generate / release`, so that's the minimal contract to preserve.

---

## §6 Dependency Management

### Pin style

- **pet-schema** — α **hardpin** in `pyproject.dependencies` (`@v3.2.1`). Acceptable because pet-schema v3.x is additive only vs the ModelCard / EdgeArtifact / QuantConfig surface pet-quantize uses. Bumped in lockstep with matrix row.
- **pet-infra** — β **peer-dep**, NOT in `pyproject.dependencies`. Install order enforced by CI + `README.md` Prerequisites. Guard in `_register.py` surfaces a clear error if a consumer skips the prereq.
- **Vendor SDKs (rknn / rkllm)** — optional; `PET_ALLOW_MISSING_SDK=1` lets CI and dev skip without per-plugin cleanup.

### Install-order contract (DEV_GUIDE §11.4)

4-step in `.github/workflows/ci.yml` + `peer-dep-smoke.yml`:

1. `pip install 'pet-infra @ git+…@v2.6.0'`
2. `pip install -e . --no-deps`
3. `pip install -e ".[dev]"` (re-resolves dev extras; pet-infra stays at step-1 version because it's no longer a declared dep)
4. `python -c "import pet_infra; assert pet_infra.__version__.startswith('2.6')"`

### Version bump policy

- **patch** — docstring / comment-only changes; no plugin surface or runner lifecycle change.
- **minor** — new CONVERTERS / DATASETS plugin; `params.yaml` schema addition (not removal); peer-dep-surface tweak (e.g., pet-infra hardpin → peer-dep in 2.1.0).
- **major** — change to `ModelCard.edge_artifacts` contract, runner lifecycle (init/generate/release signature), or removal of a registered plugin name.

`test_version_attribute_matches_metadata` enforces parity between `pet_quantize.__version__` and `importlib.metadata.version("pet-quantize")`.

---

## §7 Local Dev and Test

```bash
# Prerequisites: shared pet-pipeline conda env + pet-infra pre-installed
conda activate pet-pipeline

# One-time: install pet-infra peer-dep (current matrix row)
pip install 'pet-infra @ git+https://github.com/Train-Pet-Pipeline/pet-infra@v2.6.0'

# From repo root:
make setup                          # pip install -e ".[dev]"
PET_ALLOW_MISSING_SDK=1 make test   # pytest tests/ -v        (64 tests)
make lint                           # ruff check src/ tests/ && mypy src/
make clean                          # drop .cache/ + .pytest_cache / .mypy_cache / .ruff_cache
```

Mini-E2E candidate (T6.3; zero SDK required):

```bash
PET_ALLOW_MISSING_SDK=1 pytest \
    tests/test_noop_converter.py \
    tests/test_plugin_register_noop.py \
    tests/test_plugin_register_missing_sdk.py -v
```

Covers noop CONVERTER contract + entry-point discovery + SDK-gate behavior (both `PET_ALLOW_MISSING_SDK=1` pass path and unset-gate raise path).

On-device smoke (requires real rk3576 + ADB):

```bash
pytest src/pet_quantize/validate/ -v --device-id <adb-serial>
```

---

## §8 Known Complex Points (Preserved for Good Reasons)

### 8.1 SDK-gated cluster pattern with `PET_ALLOW_MISSING_SDK` escape hatch

**Why preserved:** Rockchip's `rknn-toolkit2` and `rkllm` are vendor wheels not on PyPI, distributed per-target-silicon. Hard-requiring them would break every non-vendor dev environment (laptops / CI runners / IDE installs). Hard-excluding them would force CI to fake the runtime path with mocks everywhere. The cluster pattern — try the real SDK, skip-with-warning when the escape hatch is set — lets the same codebase serve both "fast PR smoke on ubuntu-latest" and "real conversion on a vendor workstation" without divergence.

**What would be lost by removing:** Either dev tooling friction (every workstation needs a full rknn install) or fake-mock sprawl (every SDK-bound plugin gets a stub). The cluster grouping also ensures we don't register a dataset plugin that has no matching converter available — orchestrator time-out instead of clean skip.

**Condition to revisit:** Rockchip publishes wheels to PyPI with `markers` supporting graceful absence, or conversion moves entirely to a remote service that pet-quantize just calls.

### 8.2 Dual-mode inference runners (PC simulation ↔ on-device ADB)

**Why preserved:** Pre-Phase 5 hardware bring-up, there is no real rk3576 in CI. PC simulation mode (`target=None`) lets pet-eval's `QuantizedVlmEvaluator` exercise the full `init / generate / release` path against the RKLLM runtime without any device. Once hardware arrives, flipping `target="rk3576"` + `device_id="<adb-serial>"` runs the exact same code on real silicon — no branch-per-mode inside callers.

**What would be lost by removing:** Either pet-eval's quantized evaluator becomes untestable without hardware (blocks every PR that touches the inference path), or duplicate "simulated" implementations proliferate across every caller.

**Condition to revisit:** Real rk3576 runners become part of the default PR CI matrix (Phase 5 hardware item).

### 8.3 Eager `FileNotFoundError` in runner `__init__`

**Why preserved:** RKLLM's `RKLLMRuntime(model_path=…)` will fail with an opaque SDK error deep inside the C binding if the model is missing. Checking at Python `__init__` surfaces the bad path with stack context before the runtime is even constructed, simplifying debugging for pet-eval (the main consumer) when a recipe's `artifact_uri` points at a stale path.

**What would be lost by removing:** Error messages move from "file not found: …" at plugin-construction time to an opaque SDK stack later in `init()`.

**Condition to revisit:** RKLLM SDK surfaces a typed exception for missing-model cases.

### 8.4 Content-addressable DATASETS cache key

**Why preserved:** Orchestrator resume-from-cache hinges on `card_id = hash(recipe_id, stage_name, stage_config_sha)`. A DATASETS plugin that writes a tensor file per run with a timestamped name would change `stage_config_sha` implicitly (via artefact path) and break cache hits. Keying on `sha256(modality|source_uri|num_samples)[:16]` locks the output path to the semantic input, so two runs with identical recipe + identical source → identical cache path → orchestrator cache hit.

**What would be lost by removing:** Every re-run that would have hit cache would re-walk the calibration loader and re-write the tensor — minutes per run turning into hours across a sweep.

**Condition to revisit:** Orchestrator adds a separate "artefact identity" field decoupled from filesystem path.

### 8.5 `validate/` subpackage under `src/` (not `tests/`)

**Why preserved:** The hardware smoke suite is *deliverable* runtime code, not test-only scaffolding — it ships with the installed package so a downstream consumer (pet-ota QA, vendor sanity checks, internal release gate) can `pytest src/pet_quantize/validate/ --device-id <adb>` against a real device without cloning the repo. `[tool.pytest.ini_options] testpaths = ["tests"]` excludes it from default `pytest tests/` runs so non-hardware CI skips it automatically. The `quantize_validate` workflow is the canonical invocation on `[self-hosted, rk3576]`.

**What would be lost by removing:** Either moving it to `tests/` breaks default `pytest tests/` on non-hardware runners (every CI run tries to ADB and fails), or removing it altogether loses the on-device gate.

**Condition to revisit:** The hardware smoke suite evolves into a standalone tool with its own entry point that lives outside pet-quantize.

### 8.6 VLM lazy import vs RKNN top-level import in converters

**Why preserved:** The two SDK clusters behave differently at the module level:

- `convert_audio.py` / `convert_to_rknn.py` / `export_vision_encoder.py` themselves lazy-import `rknn.api` / `transformers` inside their public functions — so the corresponding converter plugins (`audio_rknn_fp16.py` / `vision_rknn_fp16.py`) can import them at the module top without paying the SDK cost.
- `convert_to_rkllm.py` behaves the same (`rkllm.api` inside the function), **but** pet-eval has a `test_module_load_does_not_import_rkllm_runner` assertion that specifically covers the converter's module-load chain. Keeping the import inside `run()` in `vlm_rkllm_w4a16.py` makes that contract mechanical rather than "depends on a sibling module's hygiene".

**What would be lost by removing:** Either pet-eval's upstream test becomes flaky (any future refactor in `convert_to_rkllm.py` could accidentally trigger rkllm at module load), or audio/vision converters pay a grep cost every time a reader wonders "why is vlm different".

**Condition to revisit:** pet-eval drops the module-load assertion, or the SDK imports move into an explicit boundary package that all three converters share.

---

## §9 Phase 8+ Follow-ups

1. **Fresh-venv verification in CI** — Phase 7 plan 7A.5/7A.6 describe a `python -m venv /tmp/test-venv …` dry-run pair (one verifies the 4-step order succeeds; one verifies the guard trips without pet-infra). Add a nightly CI job that runs both — today the peer-dep contract is only validated on the PR-runner env state, which can mask path-dependency bugs.

2. **`_MODEL_FILE_MAP` params-driven** — `packaging/build_package.py:20-24` hardcodes the glob patterns (`vision_*.rknn`, `qwen2vl_*.rkllm`, `audio_*.rknn`) for tarball inclusion. When a new artifact type is added (e.g., a speech-to-text encoder), packaging silently drops it. Consider moving the map into `params.yaml:packaging.file_map` — but don't ship until there's a concrete second consumer (premature generalization risk noted in Phase 7 finding ⑪).

3. **`_register.py` converter-cluster → dataset-cluster coupling** — today the rknn cluster imports `(audio_rknn_fp16, vision_rknn_fp16)` converters AND `(audio_calibration_subset, vision_calibration_subset)` datasets inside the same try/except; likewise the rkllm cluster. This preserves "no converter, no dataset" pairing but is tightly coupled. When a calibration-only DATASETS plugin is needed without a matching converter, the coupling will surface as a refactor.

4. **RKLLMRunner / RKNNRunner lifecycle hooks** — `release()` is idempotent but neither runner implements `__enter__ / __exit__`. Adding the context-manager protocol would let pet-eval's `QuantizedVlmEvaluator` drop its manual try/finally block. Small cleanup; no active bug.

5. **`validate/` test data discovery** — `test_schema_compliance.py` uses `glob.glob(calib_dir + "*.jpg")` to find sample images; if the calibration output directory uses a deeper layout, the smoke silently `pytest.skip`s with "No calibration images available". Make the discovery rule explicit (params-driven glob pattern) so hardware smoke failures never look like "no data" when they actually mean "wrong path".
