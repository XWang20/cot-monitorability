UV ?= uv

.PHONY: check test test-datasets build

check:
	$(UV) run --locked ruff check src/cot_monitorability tests scripts
	$(UV) run --locked ruff format --check src/cot_monitorability tests scripts
	$(UV) run --locked mypy src/cot_monitorability/assets.py src/cot_monitorability/cot_control/cot_control.py src/cot_monitorability/openai_process/openai_process.py src/cot_monitorability/openai_intervention/openai_intervention.py src/cot_monitorability/immediacy/immediacy.py src/cot_monitorability/ncri/ncri.py src/cot_monitorability/monitorbench/monitorbench.py
	$(UV) run --locked inspect-evals-lint --all

test:
	COT_MONITORABILITY_OFFLINE=1 $(UV) run --locked pytest tests

test-datasets:
	COT_MONITORABILITY_OFFLINE=0 $(UV) run --locked python -m cot_monitorability.assets
	COT_MONITORABILITY_OFFLINE=1 RUN_DATASET_TESTS=1 $(UV) run --locked pytest tests

build:
	$(UV) build
