.PHONY: setup test lint clean calibrate convert validate package all

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ && mypy src/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist/ *.egg-info \
		artifacts/converted/* artifacts/release/*

calibrate:
	python -m pet_quantize.calibration.build_calib_dataset
	python -m pet_quantize.calibration.validate_calib

convert:
	python -m pet_quantize.convert.export_vision_encoder
	python -m pet_quantize.convert.convert_to_rknn
	python -m pet_quantize.convert.convert_to_rkllm &
	python -m pet_quantize.convert.convert_audio &
	wait

validate:
	pytest src/pet_quantize/validate/ -v $(ARGS)

package:
	python -m pet_quantize.packaging.build_package
	python -m pet_quantize.packaging.sign_package
	python -m pet_quantize.packaging.verify_package

all: calibrate convert validate package
