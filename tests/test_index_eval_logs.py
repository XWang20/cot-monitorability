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
