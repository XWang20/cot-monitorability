import importlib
import json
import os

import pytest
from inspect_ai import eval
from inspect_ai.model import (
    ContentReasoning,
    ContentText,
    ModelOutput,
    ModelUsage,
    get_model,
)

from cot_monitorability.reference.openai_scaffold import _monitor_answer_spec


@pytest.mark.dataset_download
@pytest.mark.skipif(
    os.environ.get("RUN_DATASET_TESTS") != "1", reason="prepare pinned assets first"
)
@pytest.mark.parametrize("epochs", [1, 2])
def test_task_runs_through_inspect_with_mock_models(task_entry, tmp_path, epochs):
    task = getattr(importlib.import_module(task_entry["module"]), task_entry["name"])()
    target = str(task.dataset[0].target)
    if task_entry["name"] == "ncri_bench":
        content = f"Answer: {target}"
    elif task_entry["name"] == "monitorbench":
        content = [
            ContentReasoning(reasoning="The stated preference determines the choice."),
            ContentText(text=r"\boxed{A}"),
        ]
    elif task_entry["name"].startswith("openai_monitorability"):
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
    if task_entry["name"] == "ncri_bench":
        actor_output.usage = ModelUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, reasoning_tokens=0
        )

    def judge_reply(messages, tools, tool_choice, config):
        if "FINAL_RESULT: CORRECT" in messages[-1].text:
            reply = "FINAL_RESULT: CORRECT"
        elif task_entry["name"] == "openai_monitorability_intervention":
            item = task.dataset[0].metadata["perturbed_item"]
            answers, _ = _monitor_answer_spec(item["z_rule_ref"])
            reply = answers[0]
        elif task_entry["name"] == "openai_monitorability_process":
            paths = task.dataset[0].metadata["eval_item"]["raw"]["theorem_paths"]
            names = {
                name.strip() for path in paths.split("|") for name in path.split("&")
            }
            reply = json.dumps(
                {name: {"name": name, "is_applied": True} for name in names}
            )
        elif task_entry["name"] == "monitorbench":
            reply = "<answer>A</answer>"
        else:
            reply = monitor_text
        return ModelOutput.from_content("mockllm/judge", reply)

    actor = get_model(
        "mockllm/actor", custom_outputs=[actor_output] * 20, memoize=False
    )
    judge = get_model("mockllm/judge", custom_outputs=judge_reply, memoize=False)
    logs = eval(
        task,
        model=actor,
        model_roles={"judge": judge, "monitor": judge},
        limit=1,
        epochs=epochs,
        log_dir=str(tmp_path),
        display="none",
        max_retries=0,
    )
    assert len(logs) == 1
    assert logs[0].status == "success", logs[0].error
    assert logs[0].results.completed_samples == epochs
    assert logs[0].samples[0].scores


@pytest.mark.dataset_download
@pytest.mark.skipif(
    os.environ.get("RUN_DATASET_TESTS") != "1", reason="prepare pinned assets first"
)
def test_missing_intervention_verdict_is_a_native_scoring_error(tmp_path):
    from cot_monitorability.openai_intervention import (
        openai_monitorability_intervention,
    )

    task = openai_monitorability_intervention()
    actor = get_model(
        "mockllm/actor",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/actor",
                [
                    ContentReasoning(reasoning="Reasoning about the question."),
                    ContentText(text=f"Final Answer: {task.dataset[0].target}"),
                ],
            )
        ]
        * 2,
        memoize=False,
    )

    def judge_reply(messages, tools, tool_choice, config):
        reply = (
            "FINAL_RESULT: CORRECT"
            if "FINAL_RESULT: CORRECT" in messages[-1].text
            else "Unable to classify."
        )
        return ModelOutput.from_content("mockllm/monitor", reply)

    monitor = get_model("mockllm/monitor", custom_outputs=judge_reply, memoize=False)
    log = eval(
        task,
        model=actor,
        model_roles={"monitor": monitor},
        limit=1,
        max_retries=0,
        display="none",
        log_dir=str(tmp_path),
    )[0]
    assert log.status == "error"
    assert "monitor did not provide a valid verdict" in log.error.message
    assert log.samples[0].error is not None
    assert not log.samples[0].scores
