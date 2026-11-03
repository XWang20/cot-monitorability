import os
from types import SimpleNamespace

import pytest
from inspect_ai import eval
from inspect_ai.log import ModelEvent
from inspect_ai.model import (
    ContentReasoning,
    ContentText,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.scorer import Target

from cot_monitorability.immediacy import components as immediacy
from cot_monitorability.immediacy.immediacy import cot_immediacy_2510
from cot_monitorability.ncri.components import REPO_ROOT, G, R
from cot_monitorability.ncri.ncri import ncri_bench

pytestmark = [
    pytest.mark.dataset_download,
    pytest.mark.skipif(
        os.environ.get("RUN_DATASET_TESTS") != "1", reason="prepare pinned assets first"
    ),
]


@pytest.mark.parametrize("no_prefill", [False, True])
def test_ncri_frozen_conversation_runs_unchanged_through_inspect(tmp_path, no_prefill):
    task = ncri_bench(banks="o_gsm1k", limit_per_bank=1, no_prefill=no_prefill)
    sample = task.dataset[0]
    shots, items = G.load_bank("o_gsm1k")
    canonical = R.build_messages(items[0], shots[:10], use_prefill=not no_prefill)
    assert "problem" not in items[0]
    assert [(message.role, message.text) for message in sample.input] == [
        (message["role"], message["content"]) for message in canonical
    ]
    output = ModelOutput.from_content("mockllm/actor", f"Answer: {sample.target}")
    output.usage = ModelUsage(
        input_tokens=10, output_tokens=5, total_tokens=15, reasoning_tokens=0
    )
    requests = []

    def respond(messages, tools, tool_choice, config):
        requests.append([(message.role, message.text) for message in messages])
        return output

    actor = get_model("mockllm/actor", custom_outputs=respond, memoize=False)
    log = eval(task, model=actor, display="none", log_dir=str(tmp_path))[0]
    assert log.status == "success", log.error
    score = log.samples[0].scores["ncri_scorer"]
    assert score.value["primary_correct"] == 1
    assert score.value["valid_no_cot"] == 1
    events = [event for event in log.samples[0].events if isinstance(event, ModelEvent)]
    assert len(events) == 1
    assert requests == [
        [(message["role"], message["content"]) for message in canonical]
    ]


def test_ncri_all_available_banks_construct_without_dropping_frozen_items():
    task = ncri_bench(banks="all_ncri")
    banks = {
        bank
        for bank, entry in G.bank_manifest()["ncri"].items()
        if (REPO_ROOT / "data" / entry["file"]).exists()
    }
    assert {sample.metadata["bank"] for sample in task.dataset} == banks
    assert len({sample.id for sample in task.dataset}) == len(task.dataset)
    frozen = [sample for sample in task.dataset if sample.metadata["bank"] == "o_gsm1k"]
    assert frozen
    assert all(isinstance(sample.input, list) for sample in frozen)


@pytest.mark.asyncio
async def test_immediacy_judges_correct_mmlu_pro_choices_beyond_d(monkeypatch):
    task = cot_immediacy_2510(dataset="mmlu_pro", correct_only=True)
    calls = []

    class Judge:
        async def generate(self, messages):
            calls.append(messages)
            return ModelOutput.from_content(
                "mockllm/judge", '{"legibility_score":4,"coverage_score":4}'
            )

    monkeypatch.setattr(immediacy, "get_model", lambda **kwargs: Judge())
    letters = {sample.target for sample in task.dataset} - set("ABCD")
    assert letters == set("EFGHIJ")
    for letter in sorted(letters):
        sample = next(sample for sample in task.dataset if sample.target == letter)
        state = SimpleNamespace(
            input_text=sample.input,
            output=ModelOutput.from_content(
                "mockllm/actor",
                [
                    ContentReasoning(reasoning="Reasoning for the correct option."),
                    ContentText(text=f"ANSWER: {letter}"),
                ],
            ),
        )
        score = await immediacy.cot_immediacy_judge_scorer(correct_only=True)(
            state, Target(sample.target)
        )
        assert score.value["correct"] == 1
        assert score.value["legibility_pct"] == 100
    assert len(calls) == 6
