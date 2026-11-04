import asyncio
import importlib.util
from pathlib import Path

import pytest
from inspect_ai import Task, eval
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
    assert paths[0].parent == tmp_path / "logs" / "task" / "ds"
    assert read_eval_log(paths[0]).results.metadata["run_cost"]["amount"] == 0
    assert experiment.history[0]["audit"]["passed"]


def test_stage_sample_guard_and_ncri_summary_location(tmp_path):
    tasks = [
        Task(
            dataset=[Sample(id=str(i), input="q", target="A") for i in range(7)],
            solver=generate(),
            scorer=includes(),
        ),
        Task(
            dataset=[Sample(id=f"b{i}", input="q", target="A") for i in range(3)],
            solver=generate(),
            scorer=includes(),
        ),
    ]
    assert runner.expected_samples(tasks) == 10
    assert runner.expected_samples(tasks, 5) == 8
    summary = runner.ncri_summary_path(tmp_path / "logs", "deepseek-nt")
    assert summary.parent == tmp_path / "logs" / "ncri" / "deepseek"
    plain = Task(
        dataset=[Sample(id="q", input="q", target="A")],
        solver=generate(),
        scorer=includes(),
    )
    assert runner.task_dir_name(plain) == "task"
    assert runner.dataset_dir_name(plain) == "default"
    intervention = Task(
        dataset=[
            Sample(
                id="q",
                input="q",
                target="A",
                metadata={"dataset": "intervention/daft.jsonl"},
            )
        ],
        solver=generate(),
        scorer=includes(),
    )
    assert runner.dataset_dir_name(intervention) == "daft"
    process = Task(
        dataset=[
            Sample(
                id="q",
                input="q",
                target="A",
                metadata={"dataset": "process/process_gpqa_physics.jsonl"},
            )
        ],
        solver=generate(),
        scorer=includes(),
    )
    assert runner.dataset_dir_name(process) == "gpqa_physics"


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


def test_frozen_messages_ignore_generated_message_ids():
    from inspect_ai.model import ChatMessageUser

    manifest = {}
    first = Task(
        dataset=[
            Sample(
                id="q", input=[ChatMessageUser(content="Q")], target="A", metadata={}
            )
        ]
    )
    second = Task(
        dataset=[
            Sample(
                id="q", input=[ChatMessageUser(content="Q")], target="A", metadata={}
            )
        ]
    )
    runner.freeze_dataset(first, "messages", manifest, 50)
    runner.freeze_dataset(second, "messages", manifest, 50)
    assert manifest["messages"]["selected"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("judge", [False, True])
async def test_bounded_retry_keeps_prompt_and_actor_settings(judge):
    from inspect_ai.model import GenerateConfig

    from cot_monitorability.openai_common import _complete_response

    class Fake:
        config = GenerateConfig(
            max_tokens=8192 if judge else 65536, reasoning_effort="high"
        )

        def __init__(self):
            self.calls = []

        async def generate(self, messages, **options):
            self.calls.append((messages, options))
            output = ModelOutput.from_content("fake", "A")
            if len(self.calls) == 1:
                output.choices[0].stop_reason = "max_tokens"
            return output

    model = Fake()
    result = await _complete_response(model, ["same prompt"], judge=judge)
    assert result.stop_reason == "stop"
    assert model.calls[0][0] == model.calls[1][0]
    if judge:
        assert model.calls[1][1]["config"].max_tokens == 65536
    else:
        assert model.calls[1][1] == {"cache": False}


def test_recorded_pair_reuses_both_actor_outputs_and_restores_tokens(
    tmp_path, monkeypatch
):
    from inspect_ai.model import ModelUsage
    from inspect_ai.solver import chain

    from cot_monitorability.generation_reuse import (
        ACTIVE_REUSE,
        GenerationReuse,
        reuse_completed_generation,
    )
    from cot_monitorability.openai_common import paired_intervention_solver

    outputs = []
    for text in ["PERTURBED", "CONTROL"]:
        out = ModelOutput.from_content(
            "mockllm/pair", [ContentReasoning(reasoning=text), ContentText(text="A")]
        )
        out.usage = ModelUsage(
            input_tokens=3, output_tokens=5, total_tokens=8, reasoning_tokens=4
        )
        outputs.append(out)
    actor = get_model("mockllm/pair", custom_outputs=outputs, memoize=False)

    def sample():
        return Sample(
            id="pair",
            input="PERTURBED QUESTION",
            target="A",
            metadata={"unperturbed_prompt": "CONTROL QUESTION"},
        )

    eval(
        Task(
            dataset=[sample()], solver=paired_intervention_solver(), scorer=includes()
        ),
        model=actor,
        display="none",
        log_dir=str(tmp_path / "source"),
    )
    cache = GenerationReuse()
    cache.load([tmp_path / "source"])
    token = ACTIVE_REUSE.set(cache)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Complete recorded pairs must not call either actor arm")

    monkeypatch.setattr(actor, "generate", forbidden)
    try:
        eval(
            Task(
                dataset=[sample()],
                solver=chain(reuse_completed_generation(), generate()),
                scorer=includes(),
            ),
            model=actor,
            display="none",
            log_dir=str(tmp_path / "reused"),
        )
    finally:
        ACTIVE_REUSE.reset(token)
    path = next((tmp_path / "reused").glob("*.eval"))
    cache.restore(path)
    result = read_eval_log(path)
    assert result.samples[0].output.message.content[0].reasoning == "PERTURBED"
    assert (
        result.samples[0].metadata["unperturbed_message_dump"]["content"][0][
            "reasoning"
        ]
        == "CONTROL"
    )
    assert result.samples[0].model_usage[result.eval.model].reasoning_tokens == 8
    before = result.stats.model_usage[result.eval.model].total_tokens
    cache.restore(path)
    assert (
        read_eval_log(path).stats.model_usage[result.eval.model].total_tokens == before
    )


def test_local_profile_rejects_an_external_endpoint_before_dispatch(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    state.mkdir()
    experiment = runner.Experiment(
        {"connection_profile": "local-proxy"},
        state,
        tmp_path / "logs",
        RequestBudget(state, 500),
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="loopback"):
        experiment.model(
            {"id": "ds", "model": "openai-api/openai/test"}, "actor", "high"
        )
    assert experiment.budget.snapshot()["unresolved"] == []


def test_known_incomplete_sample_is_explicit_and_does_not_generate(
    tmp_path, monkeypatch
):
    from inspect_ai.solver import chain

    state = tmp_path / "state"
    state.mkdir()
    (state / "selection.json").write_text("{}")
    rule = {
        "model": "mockllm/source",
        "dataset": "daft",
        "sample_id": "q1",
        "reason": "64K exhausted",
    }
    config = {
        "preflight_samples": 5,
        "max_tasks": 1,
        "max_samples": 1,
        "known_incomplete": [rule],
    }
    experiment = runner.Experiment(
        config, state, tmp_path / "logs", RequestBudget(state, 500)
    )
    actor = get_model("mockllm/source", memoize=False)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Known incomplete sample must not generate again")

    monkeypatch.setattr(actor, "generate", forbidden)
    monkeypatch.setattr(experiment, "model", lambda *args: actor)
    task = Task(
        dataset=[Sample(id="q1", input="Q", target="A", metadata={"dataset": "daft"})],
        solver=chain(runner.identify_condition("daft"), generate()),
        scorer=includes(),
    )
    paths = experiment.phase([task], {"id": "mockllm/source"}, "test", "openai", False)
    loaded = read_eval_log(paths[0])
    assert loaded.samples[0].error is not None
    assert (
        loaded.samples[0].metadata["expected_incomplete"]["reason"] == "64K exhausted"
    )
    assert experiment.history[0]["audit"]["passed"]
    assert experiment.history[0]["audit"]["totals"]["expected_incomplete"] == 1


@pytest.mark.asyncio
async def test_provider_native_no_cot_request_is_reserved_without_flat_effort(tmp_path):
    import httpx

    budget = RequestBudget(tmp_path, 3)
    body = {
        "model": "deepseek-v4.1-flash-1017-nt",
        "messages": [],
        "max_tokens": 2,
        "extra_body": {"thinking": {"type": "disabled"}, "reasoning_effort": None},
    }
    request = httpx.Request(
        "POST", "http://example.test/v1/chat/completions", json=body
    )
    await budget.hooks(PRICING, lambda: {"role": "actor"}, "none")["request"][0](
        request
    )
    assert budget.snapshot()["outstanding_reservations"] == 2


def test_no_cot_reuse_matches_base_model_and_nonthinking_alias():
    from cot_monitorability.generation_reuse import generation_key

    common = ("Q", "A", {"ncri_messages": [{"role": "user", "content": "Q"}]})
    normal = generation_key(
        "openai-api/openai/deepseek-v4.1-flash-1017", "none", *common
    )
    native = generation_key(
        "openai-api/openai/deepseek-v4.1-flash-1017-nt", None, *common
    )
    assert normal == native
