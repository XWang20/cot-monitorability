import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, get_model
from inspect_ai.scorer import Score, mean_score, scorer
from inspect_ai.solver import generate

from cot_monitorability.cot_control.components import controllability_750_1250_tok
from cot_monitorability.immediacy.components import cot_immediacy_judge_scorer
from cot_monitorability.monitorbench import components as monitorbench
from cot_monitorability.ncri import components as ncri
from cot_monitorability.openai_common import (
    intervention_eligible_fraction,
    intervention_gmean2_lower,
    intervention_gmean2_upper,
    process_gmean2_cot,
    process_post_control_score_k1,
)


@scorer(metrics=[])
def epoch_records(records):
    async def score(state, target):
        return Score(**records[state.epoch - 1])

    return score


def run_metrics(tmp_path, records, metrics):
    log = eval(
        Task(
            dataset=[Sample(id="q1", input="Question", target="A")],
            solver=generate(),
            scorer=epoch_records(records),
            metrics=metrics,
        ),
        model="mockllm/actor",
        epochs=len(records),
        max_samples=1,
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "success", log.error
    assert log.results.completed_samples == len(records)
    return {
        name: metric.value
        for score in log.results.scores
        for name, metric in score.metrics.items()
    }


def test_process_rates_use_individual_epoch_verdicts(tmp_path):
    records = [
        {"value": {"y": 1, "z_cot_only": 1}},
        {"value": {"y": 1, "z_cot_only": 0}},
    ]
    metrics = run_metrics(
        tmp_path, records, [process_gmean2_cot(), process_post_control_score_k1()]
    )
    assert metrics["process_gmean2_cot"] == 0.5
    assert metrics["process_post_control_score_k1"] == 0.5


@pytest.mark.parametrize("reverse", [False, True])
def test_intervention_bounds_pool_epochs_without_order_dependence(tmp_path, reverse):
    records = [
        {
            "value": {"y_unperturbed": 0, "y_perturbed": 1},
            "metadata": {
                "arm_records": [
                    {"dataset": "d", "instance_id": "q1", "x": 0, "y": 0, "z": 0},
                    {"dataset": "d", "instance_id": "q1", "x": 1, "y": 1, "z": 1},
                ]
            },
        },
        {
            "value": {"y_unperturbed": 1, "y_perturbed": 1},
            "metadata": {
                "arm_records": [
                    {"dataset": "d", "instance_id": "q1", "x": 0, "y": 1, "z": 0},
                    {"dataset": "d", "instance_id": "q1", "x": 1, "y": 1, "z": 0},
                ]
            },
        },
    ]
    metrics = run_metrics(
        tmp_path,
        records[::-1] if reverse else records,
        [
            intervention_gmean2_upper(),
            intervention_gmean2_lower(),
            intervention_eligible_fraction(),
        ],
    )
    assert metrics["intervention_gmean2_upper"] == 1
    assert metrics["intervention_gmean2_lower"] == 0
    assert metrics["intervention_eligible_fraction"] == 1


def test_control_token_window_filters_each_epoch_before_averaging(tmp_path):
    records = [
        {"value": {"cot_controllability_score": 1, "cot_tokens": 500}},
        {"value": {"cot_controllability_score": 0, "cot_tokens": 1500}},
    ]
    metrics = run_metrics(tmp_path, records, [controllability_750_1250_tok()])
    assert metrics["controllability_750_1250_tok"] == -1


@pytest.mark.parametrize("reverse", [False, True])
def test_ncri_placement_counts_all_epoch_observations(tmp_path, monkeypatch, reverse):
    observed = []

    def place(counts):
        observed.append(counts)
        return {"display": 100}

    monkeypatch.setattr(ncri.P, "place", place)
    rung = next(iter(ncri.P.RUNGS))
    records = [
        {
            "value": {"primary_correct": 1, "valid_no_cot": 1},
            "metadata": {
                "domain": "arithmetic",
                "problem_number": "test",
                "rung": rung,
                "scorable": True,
                "correct": True,
            },
        },
        {
            "value": {"primary_correct": 0, "valid_no_cot": 1},
            "metadata": {
                "domain": "arithmetic",
                "problem_number": "test",
                "rung": rung,
                "scorable": True,
                "correct": False,
            },
        },
    ]
    metrics = run_metrics(
        tmp_path,
        records[::-1] if reverse else records,
        [
            ncri.ncri_placement_score(),
            ncri.ncri_primary_accuracy(),
            ncri.ncri_valid_rate(),
        ],
    )
    assert observed == [{rung: (1, 2)}]
    assert metrics["ncri_primary_accuracy"] == 0.5
    assert metrics["ncri_valid_rate"] == 1


@pytest.mark.parametrize("empty", [False, True])
def test_immediacy_epoch_filters_and_score_fields(tmp_path, empty):
    content = [ContentText(text="ANSWER: A" if empty else "ANSWER: B")]
    if not empty:
        content.insert(0, ContentReasoning(reasoning="Reasoning for B."))
    actor = get_model(
        "mockllm/actor",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/actor",
                [
                    ContentReasoning(reasoning="Reasoning for A."),
                    ContentText(text="ANSWER: A"),
                ],
            ),
            ModelOutput.from_content("mockllm/actor", content),
        ],
        memoize=False,
    )
    judge = get_model(
        "mockllm/judge",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/judge", '{"legibility_score":4,"coverage_score":4}'
            )
        ],
        memoize=False,
    )
    log = eval(
        Task(
            dataset=[Sample(id="q1", input="Choose A.", target="A")],
            solver=generate(),
            scorer=cot_immediacy_judge_scorer(correct_only=not empty),
        ),
        model=actor,
        model_roles={"judge": judge},
        epochs=2,
        max_samples=1,
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "success", log.error
    scores = [sample.scores["cot_immediacy_judge_scorer"] for sample in log.samples]
    assert set(scores[0].value) == set(scores[1].value)
    mean_score()(scores)
    metrics = log.results.scores[0].metrics
    assert metrics["legibility_pct"].value == (50 if empty else 100)
    assert metrics["coverage_pct"].value == (50 if empty else 100)
    assert sum(score.value["correct"] for score in scores) / len(scores) == (
        1 if empty else 0.5
    )


def test_monitorbench_eligibility_filters_and_score_fields(tmp_path, monkeypatch):
    actor = get_model(
        "mockllm/actor",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/actor",
                [
                    ContentReasoning(reasoning="A relevant factor."),
                    ContentText(text="ANSWER: A"),
                ],
            )
        ]
        * 2,
        memoize=False,
    )
    monitor = get_model(
        "mockllm/monitor",
        custom_outputs=[
            ModelOutput.from_content("mockllm/monitor", "<answer>A</answer>")
        ],
        memoize=False,
    )
    intervened = 0

    async def verify(task_name, message, *args):
        nonlocal intervened
        if message is None:
            return False, "", ""
        intervened += 1
        return intervened == 1, "A relevant factor.", "ANSWER: A"

    monkeypatch.setattr(monitorbench, "_verify_rollout", verify)
    monkeypatch.setattr(
        monitorbench, "_load_monitor_yaml", lambda *args: {"cot": "{cot}"}
    )
    log = eval(
        Task(
            dataset=[
                Sample(
                    id="q1",
                    input="Question",
                    target="A",
                    metadata={
                        "task": "cue_condition.preference",
                        "original_question": "Question",
                        "is_intervention": True,
                    },
                )
            ],
            solver=generate(),
            scorer=monitorbench.monitorbench_general_scorer(),
        ),
        model=actor,
        model_roles={"monitor": monitor},
        epochs=2,
        max_samples=1,
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "success", log.error
    scores = [sample.scores["monitorbench_general_scorer"] for sample in log.samples]
    assert set(scores[0].value) == set(scores[1].value)
    mean_score()(scores)
    metrics = log.results.scores[0].metrics
    assert metrics["mb_cot_monitorability"].value == 1
    assert metrics["mb_verification_rate"].value == 0.5
