import hashlib
import importlib.util
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import write_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("status", ["success", "error"])
def test_indexer_reads_native_logs_without_publishing_endpoint(
    tmp_path, monkeypatch, status
):
    task = Task(
        version="1-A",
        dataset=[Sample(input="Reply OK.", target="OK")],
        solver=generate(),
        scorer=includes(),
    )
    model = get_model(
        "mockllm/index-test",
        custom_outputs=[ModelOutput.from_content("mockllm/index-test", "OK")],
        memoize=False,
    )
    log = eval(task, model=model, log_dir=str(tmp_path / "source"), display="none")[0]
    log.eval.model_base_url = "https://private.example.test/v1"
    log.status = status
    if status == "error":
        log.results = None
    path = tmp_path / f"{status}.eval"
    write_eval_log(log, path)

    spec = importlib.util.spec_from_file_location(
        "index_eval_logs", ROOT / "scripts/index_eval_logs.py"
    )
    indexer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(indexer)
    monkeypatch.setattr(indexer, "ROOT", tmp_path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    result = indexer.read_run(path)

    assert result["status"] == status
    assert result["task_version"] == "1-A"
    assert result["completed_samples"] == (1 if status == "success" else None)
    assert "model_base_url" not in result
    assert result["sha256"] == checksum
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum


def test_latest_results_keep_successful_run_after_failed_retry():
    spec = importlib.util.spec_from_file_location(
        "index_eval_logs", ROOT / "scripts/index_eval_logs.py"
    )
    indexer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(indexer)
    base = {"task": "cot_monitorability/ncri_bench", "model": "openai/example"}
    runs = [
        {**base, "created": "2026-01-01", "status": "success", "file": "old.eval"},
        {
            **base,
            "model": "openai-api/openai/example",
            "created": "2026-01-02",
            "status": "success",
            "file": "new.eval",
        },
        {**base, "created": "2026-01-03", "status": "error", "file": "failed.eval"},
        {
            **base,
            "task": "inspect_evals/ncri_bench",
            "created": "2026-01-04",
            "status": "success",
            "file": "legacy.eval",
        },
    ]
    assert [run["file"] for run in indexer.latest_completed(runs)] == ["new.eval"]


def test_latest_results_preserve_separate_effort_settings():
    spec = importlib.util.spec_from_file_location(
        "index_eval_logs", ROOT / "scripts/index_eval_logs.py"
    )
    indexer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(indexer)
    base = {
        "task": "cot_monitorability/cot_controllability_qa",
        "model": "openai/example",
        "status": "success",
    }
    runs = [
        {
            **base,
            "created": "2026-01-01",
            "reasoning_effort": "low",
            "file": "low.eval",
        },
        {
            **base,
            "created": "2026-01-02",
            "reasoning_effort": "high",
            "file": "high.eval",
        },
        {
            **base,
            "created": "2026-01-03",
            "reasoning_effort": "low",
            "file": "low-new.eval",
        },
    ]
    assert {r["file"] for r in indexer.latest_completed(runs)} == {
        "low-new.eval",
        "high.eval",
    }
