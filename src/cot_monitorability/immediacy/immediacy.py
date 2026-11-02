from typing import Literal, Optional

from cot_monitorability.cot_control.components import build_cot_control_qa_dataset
from cot_monitorability.immediacy.components import (
    cot_immediacy_judge_scorer,
    immediacy_generate_solver,
)
from inspect_ai import Task, task


@task(version="1-A")
def cot_immediacy_2510(
    dataset: Literal["gpqa", "hle", "mmlu_pro"] = "gpqa",
    target_reasoning_effort: Optional[str] = "high",
    judge_model: str = "openai/deepseek-v4.1-flash-1017",
    correct_only: bool = False,
) -> Task:
    """DeepMind arXiv:2510.23966 CoT Legibility & Coverage Judge benchmark."""
    ds = build_cot_control_qa_dataset(dataset=dataset, modes="baseline")
    return Task(
        version="1-A",
        dataset=ds,
        solver=[
            immediacy_generate_solver(target_reasoning_effort=target_reasoning_effort)
        ],
        scorer=cot_immediacy_judge_scorer(
            judge_model=judge_model, correct_only=correct_only
        ),
    )
