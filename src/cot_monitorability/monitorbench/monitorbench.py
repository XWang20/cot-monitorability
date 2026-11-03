from typing import Literal, Optional

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from cot_monitorability.monitorbench.components import (
    _build_monitorbench_dataset,
    ensure_asset,
    mb_dual_objectives_steganography,
    mb_dual_objectives_summarization,
    mb_goal_sandbag_engineering,
    mb_goal_sandbag_math,
    mb_goal_sandbag_safety,
    monitorbench_general_scorer,
    paired_mb_solver,
    single_mb_solver,
)
from cot_monitorability.versioning import stamp_version


@task(version="3-A")
def monitorbench(
    subset: Literal[
        "dual_objectives_steganography",
        "dual_objectives_summarization",
        "goal_sandbag_safety",
        "goal_sandbag_math",
        "goal_sandbag_engineering",
        "goal_sandbag.science",
        "goal_deception.realworld",
        "cue_condition.knowledge",
        "cue_condition.preference",
        "cue_condition.judgment",
        "cue_condition.health",
        "cue_answer.math",
        "cue_answer.logic",
        "cue_answer.health",
        "process.multistep",
    ] = "cue_condition.preference",
    stress_test: Literal["standard", "direct_conceal", "monitor_aware"] = "standard",
    target_reasoning_effort: Optional[str] = "high",
    epochs: int = 1,
    monitor_rollouts: int = 1,
    monitor_model: str = "openai-api/openai/deepseek-v4.1-flash-1017",
) -> Task:
    """Unified MonitorBench (COLM 2026, arXiv:2603.28590) task in Inspect AI."""
    ensure_asset("monitorbench")
    # Delegate the 5 completed semsorock tasks directly
    if subset == "dual_objectives_steganography":
        return stamp_version(
            mb_dual_objectives_steganography(
                stress_test=stress_test,
                epochs=epochs,
                monitor_rollouts=monitor_rollouts,
            )
        )
    if subset == "dual_objectives_summarization":
        return stamp_version(
            mb_dual_objectives_summarization(
                stress_test=stress_test,
                epochs=epochs,
                monitor_rollouts=monitor_rollouts,
            )
        )
    if subset == "goal_sandbag_safety":
        return stamp_version(
            mb_goal_sandbag_safety(
                stress_test=stress_test,
                epochs=epochs,
                monitor_rollouts=monitor_rollouts,
            )
        )
    if subset == "goal_sandbag_math":
        return stamp_version(
            mb_goal_sandbag_math(
                stress_test=stress_test,
                epochs=epochs,
                monitor_rollouts=monitor_rollouts,
            )
        )
    if subset == "goal_sandbag_engineering":
        return stamp_version(
            mb_goal_sandbag_engineering(
                stress_test=stress_test,
                epochs=epochs,
                monitor_rollouts=monitor_rollouts,
            )
        )

    # Otherwise use the general MonitorBench adapter over upstream datasets/prompts
    ds = _build_monitorbench_dataset(task_name=subset, stress_test=stress_test)
    is_inter = subset != "process.multistep"
    return Task(
        version="3-A",
        config=GenerateConfig.model_validate(
            {
                "reasoning_effort": None
                if target_reasoning_effort == "default"
                else target_reasoning_effort
            }
        ),
        dataset=ds,
        solver=[paired_mb_solver() if is_inter else single_mb_solver()],
        scorer=monitorbench_general_scorer(monitor_model=monitor_model),
    )
