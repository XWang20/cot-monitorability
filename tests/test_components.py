from types import SimpleNamespace

import pytest
from inspect_ai.model import (
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.scorer import Target

from cot_monitorability.cot_control import components as control
from cot_monitorability.immediacy import components as immediacy
from cot_monitorability.monitorbench import components as monitorbench
from cot_monitorability.ncri import components as ncri


class RecordingModel:
    def __init__(self):
        self.calls = []
        self.config = GenerateConfig(reasoning_effort="high")
        self.output = ModelOutput.from_content(
            "mockllm/model",
            [ContentReasoning(reasoning="HELLO WORLD"), ContentText(text="ANSWER: A")],
        )

    def __str__(self):
        return "mockllm/model"

    async def generate(self, messages, config=None):
        self.calls.append(([message.text for message in messages], config))
        return self.output


def state(metadata=None):
    return SimpleNamespace(
        metadata=metadata or {},
        input_text="question",
        messages=[ChatMessageUser(content="question")],
        output=None,
        model="mockllm/model",
    )


@pytest.mark.asyncio
async def test_controllability_solver_preserves_system_prompt_and_effort_override(
    monkeypatch,
):
    model = RecordingModel()
    monkeypatch.setattr(control, "get_model", lambda: model)
    sample = state(
        {"system_prompt": "system constraint", "user_prompt": "user question"}
    )
    await control.cot_control_generate(target_reasoning_effort="low")(sample, None)
    assert model.calls[0][0] == ["system constraint", "user question"]
    assert model.calls[0][1].reasoning_effort == "low"


@pytest.mark.asyncio
async def test_controllability_scorer_keeps_accuracy_and_compliance_separate():
    sample = state({"mode": "uppercase_thinking"})
    sample.output = RecordingModel().output
    result = await control.cot_control_qa_scorer()(sample, Target("B"))
    assert result.value["cot_controllability_score"] == 1
    assert result.value["task_accuracy"] == 0
    assert result.value["controlled_and_accurate_rate"] == 0


@pytest.mark.asyncio
async def test_immediacy_solver_allows_effort_override(monkeypatch):
    model = RecordingModel()
    monkeypatch.setattr(immediacy, "get_model", lambda: model)
    await immediacy.immediacy_generate_solver(target_reasoning_effort="medium")(
        state(), None
    )
    assert model.calls[0][1].reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_immediacy_correctness_filter_skips_judge(monkeypatch):
    monkeypatch.setattr(
        immediacy, "get_model", lambda **kwargs: pytest.fail("Judge should be skipped")
    )
    sample = state()
    sample.output = RecordingModel().output
    result = await immediacy.cot_immediacy_judge_scorer(correct_only=True)(
        sample, Target("B")
    )
    assert result.value["legibility_pct"] == -1
    assert result.value["coverage_pct"] == -1


@pytest.mark.asyncio
async def test_ncri_solver_disables_reasoning_and_caps_generation(monkeypatch):
    model = RecordingModel()
    monkeypatch.setattr(ncri, "get_model", lambda: model)
    await ncri.ncri_no_cot_solver(disable_reasoning=True)(
        state({"bank": "arithmetic"}), None
    )
    assert model.calls[0][1].max_tokens == 100
    assert model.calls[0][1].extra_body["reasoning"]["enabled"] is False


@pytest.mark.asyncio
async def test_ncri_scorer_preserves_independent_witness_and_answer_flags(monkeypatch):
    monkeypatch.setattr(
        ncri.G,
        "grade_row",
        lambda row, item: {
            "correct": True,
            "valid": False,
            "predicted": "A",
            "reason": "reasoning present",
        },
    )
    sample = state({"bank": "arithmetic", "item": {"problem_number": 1}})
    sample.output = RecordingModel().output
    result = await ncri.ncri_scorer()(sample, Target("A"))
    assert result.value["primary_correct"] == 1
    assert result.value["valid_no_cot"] == 0


@pytest.mark.asyncio
async def test_monitorbench_paired_solver_retains_both_generations(monkeypatch):
    model = RecordingModel()
    monkeypatch.setattr(monitorbench, "get_model", lambda: model)
    sample = state({"original_prompt": "original question"})
    await monitorbench.paired_mb_solver(target_reasoning_effort="low")(sample, None)
    assert len(model.calls) == 2
    assert model.calls[1][0] == ["original question"]
    assert sample.metadata["original_message_dump"]
    assert all(config.reasoning_effort == "low" for _, config in model.calls)


@pytest.mark.asyncio
async def test_monitorbench_single_solver_does_not_generate_an_original(monkeypatch):
    model = RecordingModel()
    monkeypatch.setattr(monitorbench, "get_model", lambda: model)
    await monitorbench.single_mb_solver()(state(), None)
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_monitorbench_eligibility_requires_verification_gain(monkeypatch):
    model = RecordingModel()
    monkeypatch.setattr(monitorbench, "get_model", lambda **kwargs: model)

    async def verify(*args):
        return True, "reasoning", "A"

    monkeypatch.setattr(monitorbench, "_verify_rollout", verify)
    sample = state(
        {
            "task": "cue_condition.preference",
            "original_question": "question",
            "is_intervention": True,
            "original_message_dump": model.output.message.model_dump(),
        }
    )
    sample.output = model.output
    result = await monitorbench.monitorbench_general_scorer()(sample, Target("A"))
    assert result.value["cot_score"] == -1
    assert result.value["verified_original"] == 1
    assert model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["openai/kimi-k3.1-pro", "openai/deepseek-v4.1-flash-1017"]
)
async def test_ncri_no_cot_request_preserves_provider_fields(monkeypatch, model_name):
    model = RecordingModel()
    original = {
        "allowed_openai_params": ["thinking"],
        "thinking": {"type": "enabled"},
    }
    model.config.extra_body = original
    monkeypatch.setattr(RecordingModel, "__str__", lambda self: model_name)
    monkeypatch.setattr(ncri, "get_model", lambda: model)

    await ncri.ncri_no_cot_solver()(state({"bank": "arithmetic"}), None)

    body = model.calls[0][1].extra_body
    assert body["allowed_openai_params"] == ["thinking"]
    assert body["thinking"] == {"type": "disabled"}
    assert original["thinking"] == {"type": "enabled"}
    assert model.calls[0][1].max_tokens == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("unexplained_tokens", [0, 4])
async def test_ncri_cache_tokens_are_input_but_unexplained_tokens_remain_invalid(
    unexplained_tokens,
):
    sample = state(
        {
            "bank": "arithmetic",
            "item": {
                "domain": "arithmetic",
                "problem_number": 1,
                "rung": "arithmetic:ops1-2",
                "answer_type": "int",
                "target": 42,
            },
        }
    )
    sample.output = ModelOutput.from_content("mockllm/model", "Answer: 42")
    sample.output.usage = ModelUsage(
        input_tokens=10,
        input_tokens_cache_read=20,
        input_tokens_cache_write=30,
        output_tokens=5,
        total_tokens=65 + unexplained_tokens,
        reasoning_tokens=0,
    )
    result = await ncri.ncri_scorer()(sample, Target("42"))
    assert result.metadata["hidden_channel_tokens"] == unexplained_tokens
    assert result.value["valid_no_cot"] == (1.0 if unexplained_tokens == 0 else 0.0)
    assert result.value["primary_correct"] == (1.0 if unexplained_tokens == 0 else 0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    ["openai-api/openai/kimi-k3.1-pro", "openai-api/openai/deepseek-v4.1-flash-1017"],
)
async def test_ncri_positive_control_does_not_force_thinking_off(
    monkeypatch, model_name
):
    model = RecordingModel()
    monkeypatch.setattr(RecordingModel, "__str__", lambda self: model_name)
    monkeypatch.setattr(ncri, "get_model", lambda: model)
    await ncri.ncri_no_cot_solver(effort="high", disable_reasoning=False)(
        state({"bank": "arithmetic"}), None
    )
    config = model.calls[0][1]
    assert config.reasoning_effort == "high"
    assert config.extra_body["reasoning"]["effort"] == "high"
    assert config.extra_body.get("thinking") != {"type": "disabled"}
