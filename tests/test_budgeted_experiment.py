import asyncio
import importlib.util
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

from cot_monitorability.experiment_costs import BudgetExceeded, RequestBudget

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "budgeted_experiment", ROOT / "scripts/run_monitorability.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

PRICING = {
    "timezone": "Asia/Shanghai",
    "weekday_peak_windows": [],
    "peak_per_million": {"input": 0, "output": 1_000_000, "cached_input": 0},
    "offpeak_per_million": {"input": 0, "output": 1_000_000, "cached_input": 0},
}


@pytest.mark.asyncio
async def test_budget_refuses_dispatch_before_exceeding_the_total(tmp_path):
    budget = RequestBudget(tmp_path, limit=3, baseline=2)
    with pytest.raises(BudgetExceeded):
        await budget.reserve({"messages": [], "max_tokens": 2}, PRICING, {})
    assert budget.snapshot()["known_total"] == 2
    assert budget.snapshot()["unresolved"] == []


@pytest.mark.asyncio
async def test_workers_wait_for_reserved_budget_and_only_charge_actual_usage(tmp_path):
    first, second = RequestBudget(tmp_path, 3), RequestBudget(tmp_path, 3)
    reservation = await first.reserve(
        {"messages": [], "max_tokens": 2}, PRICING, {"model": "ds"}
    )
    waiting = asyncio.create_task(
        second.reserve({"messages": [], "max_tokens": 2}, PRICING, {"model": "kimi"})
    )
    await asyncio.sleep(0.02)
    assert not waiting.done()
    first.settle(
        reservation, {"prompt_tokens": 0, "completion_tokens": 1}, PRICING, 200
    )
    next_reservation = await asyncio.wait_for(waiting, 2)
    status = first.snapshot()
    assert status["known_total"] == 1
    assert status["outstanding_reservations"] == 2
    assert status["available"] == 0
    assert next_reservation["model"] == "kimi"


@pytest.mark.asyncio
async def test_missing_usage_keeps_the_cost_reservation(tmp_path):
    budget = RequestBudget(tmp_path, 3)
    reservation = await budget.reserve({"messages": [], "max_tokens": 2}, PRICING, {})
    budget.settle(reservation, None, PRICING, 503)
    assert budget.snapshot()["outstanding_reservations"] == 2
    assert budget.snapshot()["new_known_cost"] == 0


def test_frozen_selection_catches_changed_prefill_and_uses_the_same_order():
    manifest = {}
    task = Task(
        dataset=[
            Sample(
                id="a", input="Q", target="A", metadata={"ncri_messages": ["prefill"]}
            )
        ]
    )
    runner.freeze_dataset(task, "bank", manifest, 50)
    assert manifest["bank"]["selected"] == 1
    changed = Task(
        dataset=[
            Sample(
                id="a", input="Q", target="A", metadata={"ncri_messages": ["different"]}
            )
        ]
    )
    with pytest.raises(ValueError, match="Frozen"):
        runner.freeze_dataset(changed, "bank", manifest, 50)


def test_native_phase_preserves_completed_log_paths_and_cost_metadata(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    state.mkdir()
    (state / "selection.json").write_text("{}")
    experiment = runner.Experiment(
        {"preflight_samples": 5, "max_tasks": 1, "max_samples": 1},
        state,
        tmp_path / "logs",
        RequestBudget(state, 500),
    )
    actor = get_model(
        "mockllm/source",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/source",
                [ContentReasoning(reasoning="A full trace."), ContentText(text="A")],
            )
        ],
        memoize=False,
    )
    monkeypatch.setattr(experiment, "model", lambda *args: actor)
    tasks = [
        Task(
            dataset=[
                Sample(
                    id="q1",
                    input="Question",
                    target="A",
                    metadata={"experiment_condition": "condition"},
                )
            ],
            solver=generate(),
            scorer=includes(),
        )
    ]
    paths = experiment.phase(tasks, {"id": "ds"}, "test", "openai", False)
    assert len(paths) == 1
    assert read_eval_log(paths[0]).results.metadata["run_cost"]["amount"] == 0
    assert experiment.history[0]["audit"]["passed"]


def test_resume_cannot_silently_change_the_shared_budget(tmp_path):
    RequestBudget(tmp_path, 500, baseline=88, uncertainty=15)
    with pytest.raises(ValueError, match="Budget settings"):
        RequestBudget(tmp_path, 500, baseline=0, uncertainty=15)


@pytest.mark.asyncio
async def test_all_zero_usage_is_unmetered_and_keeps_the_reservation(tmp_path):
    budget = RequestBudget(tmp_path, 3)
    reservation = await budget.reserve({"messages": [], "max_tokens": 2}, PRICING, {})
    budget.settle(
        reservation, {"prompt_tokens": 0, "completion_tokens": 0}, PRICING, 200
    )
    assert budget.snapshot()["responses"][0]["cost"] is None
    assert budget.snapshot()["outstanding_reservations"] == 2
