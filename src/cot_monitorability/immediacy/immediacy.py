from typing import Literal, Optional

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from cot_monitorability.cot_control.components import build_cot_control_qa_dataset
from cot_monitorability.immediacy.components import (
    cot_immediacy_judge_scorer,
    immediacy_generate_solver,
)


@task(version="4-A")
def cot_immediacy_2510(
    dataset: Literal["gpqa", "hle", "mmlu_pro"] = "gpqa",
    target_reasoning_effort: Optional[str] = "high",
    judge_model: str = "openai-api/openai/deepseek-v4.1-flash-1017",
    correct_only: bool = False,
) -> Task:
    """DeepMind arXiv:2510.23966 CoT Legibility & Coverage Judge benchmark."""
    ds = build_cot_control_qa_dataset(dataset=dataset, modes="baseline")
    return Task(
        version="4-A",
        config=GenerateConfig.model_validate(
            {
                "reasoning_effort": None
                if target_reasoning_effort == "default"
                else target_reasoning_effort
            }
        ),
        dataset=ds,
        solver=[immediacy_generate_solver()],
        scorer=cot_immediacy_judge_scorer(
            judge_model=judge_model, correct_only=correct_only
        ),
    )
