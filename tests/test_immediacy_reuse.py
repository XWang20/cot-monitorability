import hashlib

import pytest
from inspect_ai import Task, eval, score
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

from cot_monitorability.immediacy.components import cot_immediacy_judge_scorer


def test_immediacy_scores_recorded_output_without_actor_generation(
    tmp_path, monkeypatch
):
    original_output = ModelOutput.from_content(
        "mockllm/source",
        [
            ContentReasoning(reasoning="Work through the question."),
            ContentText(text="ANSWER: A"),
        ],
    )
    actor = get_model("mockllm/source", custom_outputs=[original_output], memoize=False)
    source = eval(
        Task(
            dataset=[Sample(id="q1", input="Choose A or B.", target="A")],
            solver=generate(),
            scorer=includes(),
        ),
        model=actor,
        log_dir=str(tmp_path / "original"),
        display="none",
    )[0]
    original_path = next((tmp_path / "original").glob("*.eval"))
    checksum = hashlib.sha256(original_path.read_bytes()).hexdigest()
    recorded_output = source.samples[0].output.model_dump(mode="json")

    async def forbidden(*args, **kwargs):
        raise AssertionError("Rescoring must not generate a new actor response.")

    monkeypatch.setattr(actor, "generate", forbidden)
    judge = get_model(
        "mockllm/judge",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/judge", '{"legibility_score":3,"coverage_score":4}'
            )
        ],
        memoize=False,
    )
    rescored = score(
        source,
        cot_immediacy_judge_scorer(),
        model=actor,
        model_roles={"judge": judge},
        action="append",
        display="none",
    )
    destination = tmp_path / "rescored.eval"
    write_eval_log(rescored, destination)
    loaded = read_eval_log(destination)

    assert loaded.samples[0].output.model_dump(mode="json") == recorded_output
    assert loaded.eval.task == source.eval.task
    assert "cot_immediacy_judge_scorer" not in source.samples[0].scores
    result = loaded.samples[0].scores["cot_immediacy_judge_scorer"]
    assert result.value["legibility_pct"] == 75
    assert result.value["coverage_pct"] == 100
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == checksum


@pytest.mark.parametrize(
    "reply,is_error",
    [
        ('{"legibility_score":0,"coverage_score":0}', False),
        ("not JSON", True),
        ('{"legibility_score":8,"coverage_score":4}', True),
    ],
)
def test_immediacy_invalid_judge_is_a_scoring_error_not_zero(tmp_path, reply, is_error):
    actor = get_model(
        "mockllm/actor",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/actor",
                [
                    ContentReasoning(reasoning="A readable reasoning step."),
                    ContentText(text="ANSWER: A"),
                ],
            )
        ],
        memoize=False,
    )
    judge = get_model(
        "mockllm/judge",
        custom_outputs=[ModelOutput.from_content("mockllm/judge", reply)],
        memoize=False,
    )
    log = eval(
        Task(
            dataset=[Sample(input="Choose A.", target="A")],
            solver=generate(),
            scorer=cot_immediacy_judge_scorer(),
        ),
        model=actor,
        model_roles={"judge": judge},
        log_dir=str(tmp_path),
        fail_on_error=False,
        display="none",
    )[0]
    sample = log.samples[0]
    if is_error:
        assert sample.error is not None
        assert "Invalid Immediacy judge response" in sample.error.message
        assert not sample.scores
    else:
        assert sample.error is None
        assert sample.scores["cot_immediacy_judge_scorer"].value["legibility_pct"] == 0
        assert sample.scores["cot_immediacy_judge_scorer"].value["coverage_pct"] == 0
