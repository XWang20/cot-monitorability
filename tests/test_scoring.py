from types import SimpleNamespace

import pytest
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, ModelUsage

from cot_monitorability.cot_control.components import (
    extract_answer_letter,
    extract_reasoning_and_text,
)
from cot_monitorability.immediacy.components import parse_autorater_json
from cot_monitorability.monitorbench.components import _extract_boxed, _norm_icd_code
from cot_monitorability.openai_common import (
    _extract_cot_and_final_answer,
    _parse_process_theorem_faithfulness,
)
from cot_monitorability.reference.cot_control.grading import GRADE_FUNC_MAP
from cot_monitorability.reference.cot_control.judging import _parse_judge_response


@pytest.mark.parametrize(
    "mode,good,bad",
    [
        ("uppercase_thinking", "HELLO WORLD", "Hello world"),
        ("lowercase_thinking", "hello world", "Hello World"),
    ],
)
def test_case_constraints_measure_reasoning(mode, good, bad):
    assert GRADE_FUNC_MAP[mode](good) == 1
    assert GRADE_FUNC_MAP[mode](bad) == 0


def test_native_reasoning_preserves_reported_token_count():
    output = ModelOutput.from_content(
        "mockllm/model",
        [
            ContentReasoning(reasoning="A real intermediate step"),
            ContentText(text="ANSWER: A"),
        ],
    )
    output.usage = ModelUsage(
        input_tokens=10, output_tokens=20, total_tokens=30, reasoning_tokens=12
    )
    state = SimpleNamespace(output=output)
    reasoning, final, tokens = extract_reasoning_and_text(state)
    assert reasoning == "A real intermediate step"
    assert final == "ANSWER: A"
    assert tokens == 12


def test_tagged_reasoning_is_separated_from_answer():
    output = ModelOutput.from_content(
        "mockllm/model", "<think>WORK THROUGH THE PROBLEM</think> ANSWER: B"
    )
    reasoning, final, _ = extract_reasoning_and_text(SimpleNamespace(output=output))
    assert reasoning == "WORK THROUGH THE PROBLEM"
    assert final == "ANSWER: B"
    assert extract_answer_letter(final) == "B"


def test_unparseable_judge_verdict_is_not_a_compliance_value():
    with pytest.raises(ValueError):
        _parse_judge_response("???")


@pytest.mark.xfail(
    strict=True,
    reason="Copied legacy parser accepts T-containing text; scoring hardening is separate from the format migration",
)
def test_legacy_verdict_parser_does_not_accept_prose():
    with pytest.raises(ValueError):
        _parse_judge_response("No verdict")


def test_immediacy_parser_accepts_fenced_json():
    value = parse_autorater_json(
        '```json\n{"legibility_score": 3, "coverage_score": 4}\n```'
    )
    assert value["legibility_score"] == 3
    assert value["coverage_score"] == 4
    assert value["error"] is None


def test_immediacy_parser_reports_missing_verdict():
    value = parse_autorater_json("")
    assert value["legibility_score"] is None
    assert value["coverage_score"] is None
    assert value["error"]


def test_process_path_requires_all_theorems_in_one_valid_path():
    score, _ = _parse_process_theorem_faithfulness(
        "a&b|c", '{"a": true, "b": false, "c": false}'
    )
    assert score == 0
    score, _ = _parse_process_theorem_faithfulness("a&b|c", '{"a": true, "b": true}')
    assert score == 1
    score, _ = _parse_process_theorem_faithfulness("a&b|c", '{"c": true}')
    assert score == 1


def test_openai_final_answer_is_not_counted_as_thinking():
    output = ModelOutput.from_content(
        "mockllm/model", "Explanation: derive the result\nFinal Answer: 7"
    )
    value, _ = _extract_cot_and_final_answer(output.message)
    assert value.final_answer == "7"
    assert "derive the result" in value.cot_only


def test_monitorbench_answer_and_code_normalization():
    assert _extract_boxed(r"work \boxed{A} later \boxed{B}") == "B"
    assert _norm_icd_code(" e11.9 ") == "E11.9"
