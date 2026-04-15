-include ../pet-infra/shared/Makefile.include

.PHONY: setup test calibrate convert validate package all

setup:
	python -m pip install -e ".[dev]"

test:
	pytest tests/ -v

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
