# pet-quantize

Quantization, on-device conversion, artifact packaging and signing for the Train-Pet-Pipeline.

## Prerequisites

pet-quantize is a peer-dep consumer: **pet-infra must be installed first** from the current matrix row in `pet-infra/docs/compatibility_matrix.yaml`. `pet-schema` is a transitive dep of pet-quantize's own `pyproject.toml` (α hard-pin style). Both RK SDKs (`rknn.api`, `rkllm.api`) are optional in dev — set `PET_ALLOW_MISSING_SDK=1` to let the SDK-gated plugin clusters skip instead of hard-fail.

For local dev in the shared `pet-pipeline` conda env:

```bash
conda activate pet-pipeline

# 1. Install pet-infra peer-dep (current matrix row)
pip install 'pet-infra @ git+https://github.com/Train-Pet-Pipeline/pet-infra@v2.9.5'

# 2. Editable install
make setup   # → pip install -e ".[dev]"
make test    # → PET_ALLOW_MISSING_SDK=1 pytest tests/ -v
make lint    # → ruff check src/ tests/ && mypy src/
```

## Architecture

See `docs/architecture.md` for the full module map:
- CONVERTERS plugins (noop / vlm_rkllm_w4a16 / audio_rknn_fp16 / vision_rknn_fp16),
- DATASETS plugins (vlm / vision / audio calibration subsets),
- Inference runners (RKLLMRunner / RKNNRunner dual-mode: PC simulation vs on-device ADB),
- `validate/` hardware smoke suite run via the `quantize_validate` workflow.

## Plugin entry point

pet-quantize registers CONVERTERS and DATASETS plugins under the `pet_infra.plugins` entry point. SDK-gated clusters skip-with-warning when the corresponding SDK is absent and `PET_ALLOW_MISSING_SDK=1` is set:

```python
import os
os.environ["PET_ALLOW_MISSING_SDK"] = "1"  # CI default

from pet_quantize.plugins._register import register_all
from pet_infra.registry import CONVERTERS, DATASETS

register_all()
print(sorted(CONVERTERS.module_dict))  # at minimum: ['noop_converter']
print(sorted(DATASETS.module_dict))    # only populated if rknn/rkllm SDKs available
```

## On-device smoke validation

`src/pet_quantize/validate/` is a pytest-based smoke suite intentionally excluded from default `pytest tests/` (see `pyproject.toml:testpaths`) so non-hardware runners skip it. The `quantize_validate` workflow triggers it on self-hosted rk3576 runners:

```bash
pytest src/pet_quantize/validate/ -v --device-id <adb-serial>
```

## License

This project is licensed under the [Business Source License 1.1](LICENSE) (BSL 1.1).
On **2030-04-22** it converts automatically to the Apache License, Version 2.0.

> Note: BSL 1.1 is **source-available**, not OSI-approved open source.
> Production / commercial use requires a separate commercial license.

![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)
