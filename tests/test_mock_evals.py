import importlib
import json
import os
from pathlib import Path

import pytest
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, get_model

from inspect_ai import eval

CATALOG = json.loads(
    (
        Path(__file__).resolve().parents[1] / "src/cot_monitorability/catalog.json"
    ).read_text()
)["tasks"]


@pytest.mark.dataset_download
@pytest.mark.skipif(
    os.environ.get("RUN_DATASET_TESTS") != "1", reason="prepare pinned assets first"
)
@pytest.mark.parametrize("entry", CATALOG, ids=lambda entry: entry["name"])
def test_task_runs_through_inspect_with_mock_models(entry, tmp_path):
    task = getattr(importlib.import_module(entry["module"]), entry["name"])()
    target = str(task.dataset[0].target)
    if entry["name"] == "ncri_bench":
        content = f"Answer: {target}"
    elif entry["name"] == "monitorbench":
        content = [
            ContentReasoning(reasoning="The stated preference determines the choice."),
            ContentText(text=r"\boxed{A}"),
        ]
    elif entry["name"].startswith("openai_monitorability"):
        content = [
            ContentReasoning(reasoning="Apply the relevant theorem."),
            ContentText(text=f"Final Answer: {target}"),
        ]
    else:
        content = [
            ContentReasoning(reasoning="WORK THROUGH THE PROBLEM."),
            ContentText(text=f"ANSWER: {target}"),
        ]
    actor_output = ModelOutput.from_content("mockllm/actor", content)
    monitor_text = json.dumps(
        {
            "legibility_score": 3,
            "coverage_score": 4,
            "detected": True,
            "is_applied": True,
        }
    )
    judge_output = ModelOutput.from_content("mockllm/judge", monitor_text)
    actor = get_model(
        "mockllm/actor", custom_outputs=[actor_output] * 20, memoize=False
    )
    judge = get_model(
        "mockllm/judge", custom_outputs=[judge_output] * 20, memoize=False
    )
    logs = eval(
        task,
        model=actor,
        model_roles={"judge": judge, "monitor": judge},
        limit=1,
        log_dir=str(tmp_path),
        display="none",
        max_retries=0,
    )
    assert len(logs) == 1
    assert logs[0].status == "success", logs[0].error
    assert logs[0].results.completed_samples == 1
    assert logs[0].samples[0].scores
