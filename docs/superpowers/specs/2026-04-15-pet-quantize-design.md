# pet-quantize Design Spec

> **Status**: Approved  
> **Date**: 2026-04-15  
> **Scope**: Quantization, on-device conversion, inference interface, artifact packaging & signing

---

## 1. Purpose & Boundaries

pet-quantize receives merged HuggingFace FP16 weights from pet-train, converts them to
RK3576-compatible formats, validates conversion quality, exposes an inference interface for
pet-eval, and packages signed artifacts for pet-ota.

### Input

| Source | Format | Description |
|---|---|---|
| pet-train | HuggingFace weight directory | Merged LoRA + base (Qwen2-VL-2B), FP16 |
| pet-train | PyTorch checkpoint | Audio CNN model |
| pet-schema (tag) | Python package resource | prompt_system + prompt_user Jinja2 template |

### Output

| Target | Format | Description |
|---|---|---|
| pet-ota | Signed tarball + manifest.json | `artifacts/release/` |
| pet-eval | Python package interface | `pet_quantize.inference.run_quantized_pipeline()` |

### Relationship with pet-eval

- pet-quantize owns all RKNN/RKLLM SDK interactions and inference execution.
- pet-eval's `eval_quantized.py` has TODO stubs (`_run_on_device`, `_run_simulated`) that will
  be replaced with calls to `pet_quantize.inference` after pet-quantize merges.
- pet-quantize/validate/ provides fast smoke tests (dev-loop feedback); pet-eval/eval_quantized
  provides formal gate evaluation (CI/release decision).

### Relationship with pet-data

- `calibration/build_calib_dataset.py` queries pet-data's SQLite `frames` table to sample
  calibration frames, excluding training set and gold set frame IDs.

---

## 2. Module Structure

```
pet-quantize/
├── src/pet_quantize/
│   ├── __init__.py
│   ├── config.py                      # Pydantic params.yaml loader + JSON logging setup
│   ├── convert/
│   │   ├── __init__.py
│   │   ├── export_vision_encoder.py   # ViT -> ONNX (fp16)
│   │   ├── convert_to_rknn.py         # ONNX -> .rknn (vision encoder, FP16)
│   │   ├── convert_to_rkllm.py        # Merged LLM -> .rkllm (W8A8)
│   │   └── convert_audio.py           # Audio CNN -> INT8 .rknn
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── build_calib_dataset.py     # Distribution-aware sampling from pet-data SQLite
│   │   └── validate_calib.py          # Enforce distribution coverage constraints
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── rknn_runner.py             # ViT RKNN + Audio RKNN inference (device/simulated)
│   │   ├── rkllm_runner.py            # LLM RKLLM inference (device/simulated)
│   │   └── pipeline.py                # Combined VLM pipeline, returns outputs/timings/fp16_outputs
│   ├── validate/
│   │   ├── __init__.py
│   │   ├── test_schema_compliance.py  # Smoke: small sample schema compliance
│   │   ├── test_kl_divergence.py      # Smoke: small sample KL divergence
│   │   ├── test_latency.py            # On-device ADB P95 latency
│   │   ├── test_audio_accuracy.py     # INT8 audio basic accuracy
│   │   └── conftest.py                # ADB device fixture + simulated/device mode switch
│   └── packaging/
│       ├── __init__.py
│       ├── build_package.py           # Tarball + manifest.json generation
│       ├── sign_package.py            # RSA-2048 signing, skip when no private key
│       └── verify_package.py          # Signature + sha256 verification
├── tests/
│   ├── test_config.py
│   ├── test_export_vision_encoder.py
│   ├── test_convert_rknn.py
│   ├── test_convert_rkllm.py
│   ├── test_convert_audio.py
│   ├── test_build_calib.py
│   ├── test_validate_calib.py
│   ├── test_pipeline.py
│   ├── test_build_package.py
│   ├── test_sign_package.py
│   ├── test_verify_package.py
│   └── conftest.py
├── artifacts/
│   ├── converted/                     # Intermediate conversion products
│   └── release/                       # Final signed package
├── params.yaml
├── pyproject.toml
├── requirements.txt
└── Makefile
```

---

## 3. Key Flows

### 3.1 Full Pipeline (`make all`)

```
calibrate -> convert -> validate -> package
```

### 3.2 Calibration (`make calibrate`)

1. `build_calib_dataset.py` connects to pet-data SQLite, queries `frames` table.
2. Excludes training set frame IDs and gold set frame IDs.
3. Samples `calibration.frame_count` (200) frames respecting distribution constraints.
4. `validate_calib.py` enforces:
   - lighting: bright 40%, dim 20%, infrared_night 20%, unknown 20% (tolerance +-5%)
   - action_primary: eating 50%, sniffing_only 20%, leaving_bowl 15%, other 15% (tolerance +-5%)
   - min_breeds >= 5
   - Any violation rejects and halts the pipeline.

### 3.3 Conversion (`make convert`)

Execution order: 1->2 serial (ONNX before RKNN), 3 and 4 independent and parallel with 1->2.

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | export_vision_encoder.py | HF weight dir | vision_encoder.onnx |
| 2 | convert_to_rknn.py | vision_encoder.onnx + calibration data | vision_rk3576.rknn (FP16) |
| 3 | convert_to_rkllm.py | HF weight dir + calibration data | qwen2vl_2b_w8a8_rk3576.rkllm (W8A8) |
| 4 | convert_audio.py | PyTorch audio checkpoint + calibration data | audio_cnn_int8.rknn (INT8) |

Calibration data is required by steps 2, 3, 4 for quantization parameter determination
(scale/zero-point). Hence `calibrate` must complete before `convert`.

### 3.4 Validation (`make validate`)

Dual-mode pytest suite via `validate/conftest.py`:

**Without device** (default):
- RKNN-Toolkit2 PC simulator (`init_runtime(target=None)`)
- test_schema_compliance: ~30 frames smoke check
- test_kl_divergence: ~30 frames, FP16 vs W8A8
- test_latency: SKIP
- test_audio_accuracy: simulator INT8 inference

**With device** (`--device-id=xxx`):
- ADB push model to RK3576
- All above + test_latency: P95 measurement

Smoke thresholds are intentionally looser than pet-eval gate thresholds
(e.g., KL smoke: 0.05 vs gate: 0.02) to catch obvious regressions without
false-blocking development iteration.

### 3.5 Packaging (`make package`)

1. `build_package.py`:
   - Collects model files from `artifacts/converted/`
   - Reads prompt files from installed pet-schema package
   - Computes sha256 for each file
   - Reads schema_version and prompt_version from pet-schema package metadata
   - Generates `manifest.json` (version, schema_version, prompt_version, files, build_timestamp)
   - Creates tarball

2. `sign_package.py`:
   - Detects RSA private key (env var / CI secret)
   - With key: generates `.sig` file (RSA-2048)
   - Without key: skips, logs warning

3. `verify_package.py`:
   - Verifies sha256 for all files in manifest
   - With signature: verifies RSA signature
   - Without signature: warning, not failure

Output: `artifacts/release/`

---

## 4. Inference Interface

The `inference/` module is the bridge between pet-quantize and pet-eval.

### 4.1 VLM Pipeline

```python
# pet_quantize/inference/pipeline.py

def run_quantized_pipeline(
    model_dir: str,
    image_paths: list[str],
    device_id: str | None = None,
    params_path: str = "params.yaml",
) -> dict[str, Any]:
    """Unified VLM inference entry point.

    Returns:
        {
            "outputs": list[str],        # Quantized model JSON outputs
            "timings": list[float],      # Per-frame latency ms (empty in simulated mode)
            "fp16_outputs": list[str],   # FP16 reference outputs for KL comparison
        }
    """
```

### 4.2 Audio Inference

```python
# pet_quantize/inference/rknn_runner.py

def run_audio_inference(
    model_path: str,
    audio_paths: list[str],
    device_id: str | None = None,
) -> dict[str, Any]:
    """Audio CNN INT8 inference.

    Returns:
        {
            "predictions": list[str],
            "confidences": list[float],
            "timings": list[float],
        }
    """
```

### 4.3 Internal Components

| File | Responsibility |
|---|---|
| rknn_runner.py | Wraps RKNN SDK. Loads .rknn models. Supports `init_runtime(target=None)` for simulation and `init_runtime(target="rk3576", device_id=xxx)` for on-device. Also handles audio RKNN inference. |
| rkllm_runner.py | Wraps RKLLM SDK. Loads .rkllm models. Dual-mode support. |
| pipeline.py | Orchestrates ViT + LLM for full VLM inference. Also runs FP16 reference inference via transformers on the same inputs to produce fp16_outputs for KL comparison. |

### 4.4 Simulated Mode Performance

RKNN-Toolkit2 PC simulator is slow (~seconds per frame vs ~ms on device).
Mitigations:
- pet-quantize/validate/ smoke tests: ~30 frames (~1.5 min)
- pet-eval simulated mode: `simulated_sample_size: 50` in params.yaml (~2.5 min)
- On-device mode: no performance concern

---

## 5. Metric Evaluation Matrix

| Metric | pet-quantize/validate/ (smoke) | pet-eval/eval_quantized (gate) |
|---|---|---|
| **Without device** | | |
| Artifact loadable, format correct | Sole source | -- |
| schema_compliance >= 0.99 | ~30 frames quick check | Full benchmark |
| distribution_sum_error <= 0.01 | -- | Full benchmark |
| kl_divergence <= 0.02 | ~30 frames (loose: 0.05) | Full computation |
| latency_p95_ms <= 4000 | SKIP | SKIP |
| **With device** | | |
| latency_p95_ms <= 4000 | ADB measurement | ADB measurement (via pet_quantize.inference) |
| schema_compliance | Quick check | Full benchmark |
| kl_divergence | Quick check (loose) | Full computation (strict) |
| **Always skipped by eval_quantized** | | |
| anomaly_recall / false_positive | -- | SKIP (gold-set dependent, handled by eval_trained) |
| mood_spearman / narrative_bertscore | -- | SKIP (same reason) |
| **Audio (independent)** | | |
| audio_overall_accuracy >= 0.80 | INT8 basic check | -- (eval_audio.py handles) |
| audio_vomit_recall >= 0.70 | -- | -- (eval_audio.py handles) |

---

## 6. params.yaml

```yaml
# === Calibration ===
calibration:
  frame_count: 200
  tolerance: 0.05
  min_breeds: 5
  distribution:
    lighting:
      bright: 0.40
      dim: 0.20
      infrared_night: 0.20
      unknown: 0.20
    action_primary:
      eating: 0.50
      sniffing_only: 0.20
      leaving_bowl: 0.15
      other: 0.15
  exclude:
    train_ids_path: ""
    gold_set_path: ""
  data_db_path: ""

# === Conversion ===
convert:
  vision:
    input_size: [448, 448]
    onnx_opset: 17
    rknn_target: "rk3576"
    rknn_dtype: "fp16"
  llm:
    rkllm_target: "rk3576"
    quantization: "w8a8"
  audio:
    rknn_target: "rk3576"
    rknn_dtype: "int8"

# === Inference ===
inference:
  schema_version: "1.0"
  simulated_sample_size: 50
  device:
    adb_timeout: 30
    warmup_runs: 3
    latency_runs: 20

# === Validation (smoke) ===
validate:
  smoke_sample_size: 30
  kl_threshold: 0.05

# === Packaging ===
packaging:
  version: ""
  min_firmware: "2.0.0"

# === wandb ===
wandb:
  project: "pet-quantize"
  entity: ""
```

---

## 7. Dependencies

| Package | Purpose | Pinning |
|---|---|---|
| pet-schema | Prompt files, schema validation | `@v1.0.0` tag |
| rknn-toolkit2 | ONNX->RKNN conversion + PC simulation | `==2.x.y` |
| rknn-llm | LLM->RKLLM conversion + inference | `==1.x.y` |
| transformers | HF weight loading, ONNX export, FP16 ref inference | `>=4.44,<5.0` |
| onnx | ONNX intermediate format | `>=1.14,<2.0` |
| torch | PyTorch dependency | `==2.x.y` |
| pydantic | params.yaml validation | `>=2.0,<3.0` |
| pyyaml | YAML parsing | `>=6.0` |
| cryptography | RSA-2048 signing/verification | `>=41.0,<43.0` |

---

## 8. Test Strategy

### Unit Tests (tests/)

Mock RKNN/RKLLM SDK. Run on any environment via `make test`.

| Test File | Coverage |
|---|---|
| test_config.py | params.yaml loading, field validation |
| test_export_vision_encoder.py | Mock transformers, verify ONNX export call args |
| test_convert_rknn.py | Mock RKNN SDK, verify FP16 quantization config |
| test_convert_rkllm.py | Mock RKLLM SDK, verify W8A8 config |
| test_convert_audio.py | Mock RKNN SDK, verify INT8 config |
| test_build_calib.py | Mock SQLite, verify distribution sampling + exclusion logic |
| test_validate_calib.py | Boundary cases: exactly +-5% tolerance, breed < 5 rejection |
| test_pipeline.py | Mock runners, verify VLM pipeline composition |
| test_build_package.py | Manifest structure, sha256 computation, prompt file reading |
| test_sign_package.py | With-key signing, no-key skip paths |
| test_verify_package.py | Valid signature, tampered file detection, no-signature warning |

### Smoke Tests (validate/)

Real RKNN SDK calls with small sample. Run via `make validate`.

### Key Scenarios

- Calibration distribution at exactly +-5% boundary -> verify pass/reject
- Calibration frames insufficient (breed < 5) -> reject
- manifest.json sha256 tampered -> verify fails
- No private key -> sign skips, verify warns (not fails)
- Simulated mode -> timings empty, latency test skipped
- pet-schema package missing -> packaging fails with clear error

---

## 9. Development Order

### Phase 1: pet-quantize

1. Repository init (pyproject.toml, Makefile, params.yaml, config.py)
2. calibration/ -- calibration dataset build + distribution validation
3. convert/ -- four conversion scripts
4. inference/ -- inference interface (device/simulated dual-mode)
5. validate/ -- smoke tests (calls inference/)
6. packaging/ -- packaging and signing
7. Unit test completion
8. PR -> dev -> main, tag v1.0.0

### Phase 2: Cross-repo Integration (after pet-quantize merges)

| Step | Repo | Change |
|---|---|---|
| 1 | pet-eval | Small PR: add pet-quantize tag to requirements, replace `_run_on_device` and `_run_simulated` stubs with `pet_quantize.inference.run_quantized_pipeline` calls |
| 2 | pet-infra | Wire `quantize_validate.yml` CI workflow to pet-quantize `make all` |
| 3 | pet-infra | Add pet-quantize to `schema_guard.yml` repository_dispatch list |

---

## 10. Distillation vs Quantization KL -- Boundary Clarification

**Training-time distillation (pet-train)** -- training technique:
- Label smoothing (`label_smoothing_factor=0.1`): zero-cost softening, always on
- Full-vocab KL (`compute_kl_distillation_loss`): exact distillation with local 72B teacher
- Top-k approx KL (`compute_topk_kl_loss`): API-based approximate distillation

These are components of the training loss. Their mission ends when FP16 weights are produced.

**Quantization KL (pet-quantize -> pet-eval)** -- evaluation metric:
- Measures FP16 model vs W8A8 quantized model output distribution divergence
- Answers: "did quantization break the model?"
- Gate threshold: KL <= 0.02
- Completely independent of which distillation method was used during training

pet-quantize receives merged HuggingFace FP16 weights regardless of training process.
Quantization evaluation only measures the FP16 -> W8A8 conversion quality.
