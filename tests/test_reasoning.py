from types import SimpleNamespace

import pytest
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, ModelUsage
from inspect_ai.scorer import Target

from cot_monitorability.cot_control.components import extract_reasoning_and_text
from cot_monitorability.immediacy.components import _extract_thought_and_final
from cot_monitorability.monitorbench.components import _verify_rollout
from cot_monitorability.ncri.components import ncri_scorer
from cot_monitorability.openai_common import _extract_cot_and_final_answer
from cot_monitorability.reasoning import full_reasoning


@pytest.mark.parametrize("redacted", [False, True])
def test_full_trace_extractors_reject_summary_or_encrypted_only_content(redacted):
    output = ModelOutput.from_content(
        "mockllm/actor",
        [
            ContentReasoning(
                reasoning="encrypted-blob" if redacted else "",
                summary="A condensed explanation.",
                redacted=redacted,
            ),
            ContentText(text="ANSWER: A"),
        ],
    )
    state = SimpleNamespace(output=output)
    for extract in [
        lambda: full_reasoning(output.message),
        lambda: extract_reasoning_and_text(state),
        lambda: _extract_thought_and_final(state),
        lambda: _extract_cot_and_final_answer(output.message),
    ]:
        with pytest.raises(ValueError, match="Full reasoning is unavailable"):
            extract()


def test_full_reasoning_is_used_when_both_full_text_and_summary_exist():
    output = ModelOutput.from_content(
        "mockllm/actor",
        [
            ContentReasoning(reasoning="The complete trace.", summary="Short summary."),
            ContentText(text="ANSWER: A"),
        ],
    )
    assert full_reasoning(output.message) == "The complete trace."


@pytest.mark.asyncio
async def test_monitorbench_rejects_summary_only_actor_before_verification():
    message = ModelOutput.from_content(
        "mockllm/actor",
        [
            ContentReasoning(reasoning="", summary="Only a summary."),
            ContentText(text="ANSWER: A"),
        ],
    ).message
    with pytest.raises(ValueError, match="Full reasoning is unavailable"):
        await _verify_rollout("cue_condition.preference", message, "Q", "A", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("redacted", [False, True])
async def test_ncri_treats_reasoning_summary_as_cot_even_with_zero_counter(redacted):
    output = ModelOutput.from_content(
        "mockllm/actor",
        [
            ContentReasoning(
                reasoning="", summary="Work through the problem.", redacted=redacted
            ),
            ContentText(text="Answer: 42"),
        ],
    )
    output.usage = ModelUsage(
        input_tokens=10, output_tokens=5, total_tokens=15, reasoning_tokens=0
    )
    state = SimpleNamespace(
        model="mockllm/actor",
        output=output,
        metadata={
            "bank": "arithmetic",
            "item": {
                "domain": "arithmetic",
                "problem_number": 1,
                "rung": "arithmetic:ops1-2",
                "answer_type": "int",
                "target": 42,
            },
        },
    )
    score = await ncri_scorer()(state, Target("42"))
    assert score.value["valid_no_cot"] == 0
    assert score.value["primary_correct"] == 0
    assert score.metadata["reason"] == "reasoned"


def test_partial_readable_trace_does_not_hide_an_unreadable_reasoning_block():
    message = ModelOutput.from_content(
        "mockllm/actor",
        [
            ContentReasoning(reasoning="A readable part."),
            ContentReasoning(reasoning="encrypted-blob", redacted=True),
        ],
    ).message
    with pytest.raises(ValueError, match="Full reasoning is unavailable"):
        full_reasoning(message)
