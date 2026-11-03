import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.scorer import includes
from inspect_ai.solver import chain, generate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cot_control_experiment", ROOT / "scripts/run_cot_control.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
PRICING = json.loads((ROOT / "configs/cot_control_ds50.json").read_text())["pricing"]


@pytest.mark.parametrize(
    "time,factor",
    [
        ("2026-11-03T01:00:00+00:00", 1),
        ("2026-11-03T04:00:00+00:00", 0.5),
        ("2026-11-03T06:00:00+00:00", 1),
        ("2026-11-03T10:00:00+00:00", 0.5),
        ("2026-11-08T01:00:00+00:00", 0.5),
    ],
)
def test_fee_uses_local_time_and_does_not_double_charge_reasoning(time, factor):
    usage = {
        "prompt_tokens": 200,
        "completion_tokens": 1000,
        "prompt_tokens_details": {"cached_tokens": 100},
        "completion_tokens_details": {"reasoning_tokens": 500},
    }
    result = experiment.estimate_cost(usage, time, PRICING)
    assert result["cost"] == pytest.approx(0.0064812 * factor)


def test_fee_without_usage_remains_unknown():
    assert (
        experiment.estimate_cost({}, "2026-11-03T06:00:00+00:00", PRICING)["cost"]
        is None
    )


def test_reused_generation_skips_the_actor_and_keeps_the_original_output(
    tmp_path, monkeypatch
):
    sample = Sample(
        id="q1",
        input="Choose A.",
        target="A",
        metadata={
            "dataset": "gpqa",
            "mode": "uppercase_thinking",
            "raw_id": 0,
        },
    )
    output = ModelOutput.from_content(
        "mockllm/actor",
        [
            ContentReasoning(reasoning="THINK THROUGH THE QUESTION."),
            ContentText(text="ANSWER: A"),
        ],
    )
    output.usage = ModelUsage(
        input_tokens=10, output_tokens=5, total_tokens=15, reasoning_tokens=4
    )
    key = experiment.fingerprint(sample.input, sample.target, sample.metadata)
    monkeypatch.setattr(
        experiment,
        "CACHE",
        {
            key: {
                "output": output.model_dump(mode="json"),
                "messages": [],
                "source": {
                    "log": "original.eval",
                    "sample_id": "q1",
                    "task_version": "4-A",
                },
            }
        },
    )
    monkeypatch.setattr(experiment, "REUSED", [])

    def forbidden(*args):
        pytest.fail("Reused valid generations must not make an actor API call")

    actor = get_model("mockllm/actor", custom_outputs=forbidden, memoize=False)
    log = eval(
        Task(
            dataset=[sample],
            scorer=includes(),
            solver=chain(
                experiment.prepare_sample(),
                generate(),
                experiment.remember_generation(),
            ),
        ),
        model=actor,
        display="none",
        log_dir=str(tmp_path),
    )[0]
    assert log.status == "success", log.error
    assert log.samples[0].output.model_dump(mode="json") == output.model_dump(
        mode="json"
    )
    assert log.samples[0].metadata["reused_generation"]["log"] == "original.eval"
    assert not log.samples[0].model_usage


@pytest.mark.asyncio
async def test_fee_hooks_survive_the_compatible_provider_and_write_a_ledger(tmp_path):
    def respond(request):
        return httpx.Response(
            200,
            json={
                "id": "completion-test",
                "object": "chat.completion",
                "created": 0,
                "model": "model-test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        event_hooks=experiment.ledger_hooks("actor", tmp_path, PRICING),
    ) as client:
        model = get_model(
            "openai-api/openai/model-test",
            base_url="https://example.test/v1",
            api_key="test-key",
            http_client=client,
            memoize=False,
            config=GenerateConfig(
                reasoning_effort="high",
                extra_body={"allowed_openai_params": ["reasoning_effort"]},
            ),
        )
        experiment.SAMPLE.set(
            {"dataset": "gpqa", "mode": "uppercase_thinking", "sample_id": "q1"}
        )
        await model.generate("Q")
        await model.__aexit__(None, None, None)
    records = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["sample_id"] == "q1"
    assert records[0]["cost"] > 0
    assert records[0]["currency"] == "CNY"


def test_native_eval_set_retains_cost_hooks_across_conditions(tmp_path):
    from inspect_ai import eval_set, task
    from inspect_ai.solver import generate

    def respond(request):
        return httpx.Response(
            200,
            json={
                "id": "completion-test",
                "object": "chat.completion",
                "created": 0,
                "model": "model-test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    @task
    def condition(number):
        return Task(
            dataset=[Sample(id=number, input="Reply OK.", target="OK")],
            solver=generate(),
            scorer=includes(),
        )

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        event_hooks=experiment.ledger_hooks("actor", state_dir, PRICING),
    )
    model = get_model(
        "openai-api/openai/model-test",
        base_url="https://example.test/v1",
        api_key="test-key",
        http_client=client,
        memoize=False,
        config=GenerateConfig(
            reasoning_effort="high",
            extra_body={"allowed_openai_params": ["reasoning_effort"]},
        ),
    )
    success, logs = eval_set(
        [condition(1), condition(2)],
        model=model,
        log_dir=str(tmp_path / "logs"),
        max_tasks=1,
        max_samples=1,
        retry_attempts=0,
        display="none",
    )
    assert success
    assert len(logs) == 2
    assert len((state_dir / "requests.jsonl").read_text().splitlines()) == 2
