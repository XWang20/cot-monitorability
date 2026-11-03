import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.log import ModelEvent, read_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.dataset_download
@pytest.mark.skipif(
    os.environ.get("RUN_DATASET_TESTS") != "1", reason="prepare pinned assets first"
)
@pytest.mark.parametrize("setting", ["default", "cli", "task", "cli_over_task"])
def test_native_inspect_actor_effort_precedence(task_entry, setting, tmp_path):
    if task_entry["name"] == "ncri_bench":
        pytest.skip("NCRI uses a separate no-CoT protocol")
    factory = getattr(importlib.import_module(task_entry["module"]), task_entry["name"])
    kwargs = (
        {"target_reasoning_effort": "low" if setting == "task" else "high"}
        if setting in ("task", "cli_over_task")
        else {}
    )
    task = factory(**kwargs)
    default = (
        "medium" if task_entry["name"].startswith("openai_monitorability") else "high"
    )
    task.scorer = [includes()]
    actor = get_model(
        "mockllm/actor",
        custom_outputs=[ModelOutput.from_content("mockllm/actor", "ANSWER: A")] * 4,
        memoize=False,
    )
    config = {"reasoning_effort": "low"} if setting in ("cli", "cli_over_task") else {}
    log = eval(
        task, model=actor, limit=1, display="none", log_dir=str(tmp_path), **config
    )[0]
    assert log.status == "success", log.error
    events = [event for event in log.samples[0].events if isinstance(event, ModelEvent)]
    expected = default if setting == "default" else "low"
    assert events
    assert all(event.config.reasoning_effort == expected for event in events)
    expected_calls = (
        2
        if task_entry["name"] in ("monitorbench", "openai_monitorability_intervention")
        else 1
    )
    assert len(events) == expected_calls


@pytest.mark.dataset_download
@pytest.mark.skipif(
    os.environ.get("RUN_DATASET_TESTS") != "1", reason="prepare pinned assets first"
)
def test_native_cli_effort_override_reaches_model_request(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "cot_monitorability/cot_controllability_qa",
            "--model",
            "mockllm/actor",
            "--reasoning-effort",
            "low",
            "--limit",
            "1",
            "--display",
            "none",
            "--log-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    log = read_eval_log(next(tmp_path.glob("*.eval")))
    assert log.status == "success", log.error
    events = [event for event in log.samples[0].events if isinstance(event, ModelEvent)]
    assert len(events) == 1
    assert events[0].config.reasoning_effort == "low"


@pytest.mark.dataset_download
@pytest.mark.skipif(
    os.environ.get("RUN_DATASET_TESTS") != "1", reason="prepare pinned assets first"
)
def test_monitorbench_single_generation_respects_native_effort(tmp_path):
    from cot_monitorability.monitorbench.monitorbench import monitorbench

    task = monitorbench(subset="process.multistep")
    task.scorer = [includes()]
    log = eval(
        task,
        model="mockllm/actor",
        reasoning_effort="low",
        limit=1,
        display="none",
        log_dir=str(tmp_path),
    )[0]
    assert log.status == "success", log.error
    events = [event for event in log.samples[0].events if isinstance(event, ModelEvent)]
    assert len(events) == 1
    assert events[0].config.reasoning_effort == "low"
