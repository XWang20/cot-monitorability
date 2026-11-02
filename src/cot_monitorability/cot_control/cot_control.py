from typing import Literal, Optional

from inspect_ai import Task, task

from cot_monitorability.cot_control.components import (
    build_cot_control_qa_dataset,
    cot_control_generate,
    cot_control_qa_scorer,
)


@task(version="1-A")
def cot_controllability_qa(
    dataset: Literal["gpqa", "hle", "mmlu_pro", "all"] = "gpqa",
    modes: str = "word_suppression",
    target_reasoning_effort: Optional[str] = "high",
    adversarial_pressure: str = "na",
    non_reasoning: bool = False,
    monitor_aware: bool = False,
    eval_aware: bool = False,
    cot_sys_prompt: bool = False,
    elicit_brevity: bool = False,
    sample_limit_per_mode: Optional[int] = None,
    judge_model: str = "openai/deepseek-v4.1-flash-1017",
    check_meta_discussion: bool = False,
) -> Task:
    """CoT-Control-QA (arXiv:2603.05706) main-paper benchmark in Inspect AI."""
    return Task(
        version="1-A",
        dataset=build_cot_control_qa_dataset(
            dataset=dataset,
            modes=modes,
            adversarial_pressure=adversarial_pressure,
            non_reasoning=non_reasoning,
            monitor_aware=monitor_aware,
            eval_aware=eval_aware,
            cot_sys_prompt=cot_sys_prompt,
            elicit_brevity=elicit_brevity,
            sample_limit_per_mode=sample_limit_per_mode,
        ),
        solver=[cot_control_generate(target_reasoning_effort=target_reasoning_effort)],
        scorer=cot_control_qa_scorer(
            judge_model=judge_model,
            check_meta_discussion=check_meta_discussion,
        ),
    )
