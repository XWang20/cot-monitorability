import hashlib

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ContentText, ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import SampleScore, Score, scorer
from inspect_ai.solver import generate

from cot_monitorability.ncri.components import (
    P,
    ncri_placement,
    ncri_placement_score,
    ncri_primary_accuracy,
    ncri_rung_counts,
    ncri_valid_rate,
)
from cot_monitorability.ncri.logs import aggregate_ncri_logs, refresh_ncri_metrics


def observation(rung, correct, number=0, bank=None):
    return SampleScore(
        score=Score(
            value={"primary_correct": int(correct), "valid_no_cot": 1},
            metadata={
                "domain": bank or rung.split(":")[0],
                "problem_number": number,
                "rung": rung,
                "scorable": True,
                "correct": bool(correct),
            },
        )
    )


@pytest.mark.parametrize("model", list(P.DEMO_COUNTS))
def test_index_reproduces_the_frozen_published_reference_models(model):
    rows = []
    for rung, (correct, total) in P.DEMO_COUNTS[model].items():
        rows.extend(observation(rung, i < correct, number=i) for i in range(total))
    assert ncri_placement_score()(rows) == pytest.approx(
        P.DEMO_PUBLISHED[model], abs=5e-4
    )


def test_index_applies_annex_hard_rung_mapping_and_primary_failures(monkeypatch):
    rung = next(iter(P.RUNGS))
    monkeypatch.setattr(
        P, "ncri_hard_rung_of", lambda **kwargs: {("hard-bank", "1"): rung}
    )
    hard = observation("local-label", False, number=1, bank="hard-bank")
    hard.score.metadata["correct"] = True
    excluded = observation(
        rung, True, number=P.ANNEXED_ITEMS[0][1], bank=P.ANNEXED_ITEMS[0][0]
    )
    assert ncri_rung_counts([hard, excluded]) == {rung: (0, 1)}


def test_missing_index_is_refused_and_partial_coverage_is_explicit():
    with pytest.raises(ValueError, match="No scorable"):
        ncri_placement([])
    result = ncri_placement([observation(next(iter(P.RUNGS)), True)])
    assert result["coverage_domains"] == 1
    assert not result["ranked"]


@scorer(metrics=[ncri_primary_accuracy(), ncri_valid_rate()])
def stored_ncri(correct, bank, rung):
    async def score(state, target):
        return Score(
            value={"primary_correct": int(correct), "valid_no_cot": 1},
            metadata={
                "domain": bank,
                "problem_number": 999,
                "rung": rung,
                "scorable": True,
                "correct": bool(correct),
                "reasoning_tokens": 0,
                "hidden_channel_tokens": 0,
            },
        )

    return score


def test_log_refresh_and_model_pooling_preserve_outputs_without_actor_calls(
    tmp_path, monkeypatch
):
    actor = get_model(
        "mockllm/ncri",
        custom_outputs=[ModelOutput.from_content("mockllm/ncri", "42")],
        memoize=False,
    )
    paths = []
    before = []
    rungs = list(P.RUNGS)[:2]
    for bank, correct, rung in zip(["arithmetic", "brew"], [True, False], rungs):
        output = ModelOutput.from_content("mockllm/ncri", [ContentText(text="42")])
        output.usage = ModelUsage(
            input_tokens=4, output_tokens=1, total_tokens=5, reasoning_tokens=0
        )
        model = get_model("mockllm/ncri", custom_outputs=[output], memoize=False)
        log = eval(
            Task(
                dataset=[Sample(id=bank + "__999", input="Q", target="42")],
                solver=generate(),
                scorer=stored_ncri(correct, bank, rung),
            ),
            model=model,
            display="none",
            log_dir=str(tmp_path / bank),
        )[0]
        path = next((tmp_path / bank).glob("*.eval"))
        log.eval.task = "cot_monitorability/ncri_bench"
        log.eval.task_args = {"banks": bank, "k_shot": 10, "disable_reasoning": True}
        log.samples[0].scores = {
            "ncri_scorer": next(iter(log.samples[0].scores.values()))
        }
        log.results.scores[0].name = log.results.scores[0].scorer = "ncri_scorer"
        log.eval.scorers[0].name = "ncri_scorer"
        write_eval_log(log, path)
        paths.append(path)
        before.append(hashlib.sha256(path.read_bytes()).hexdigest())

    async def forbidden(*args, **kwargs):
        raise AssertionError("Index recomputation must not generate any response")

    monkeypatch.setattr(actor, "generate", forbidden)
    loaded = read_eval_log(paths[0])
    original = loaded.samples[0].model_dump(mode="json")
    refresh_ncri_metrics(loaded)
    assert loaded.samples[0].model_dump(mode="json") == original
    assert list(loaded.results.scores[0].metrics) == [
        "ncri_placement_score",
        "ncri_valid_rate",
    ]
    combined = tmp_path / "model-index.eval"
    placement = aggregate_ncri_logs(paths, combined)
    expected = P.place({rungs[0]: (1, 1), rungs[1]: (0, 1)})
    assert placement["display"] == pytest.approx(expected["display"])
    merged = read_eval_log(combined)
    assert merged.results.completed_samples == 2
    assert merged.eval.metadata["aggregation"]["new_model_requests"] == 0
    assert merged.results.metadata["run_cost"]["incremental_cost"] == 0
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before
