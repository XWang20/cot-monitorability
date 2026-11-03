from typing import Dict, List, Literal, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

from cot_monitorability.openai_common import (
    EvalItem,
    _render_model_prompt,
    ensure_asset,
    load_items,
    openai_intervention_scorer,
    paired_intervention_solver,
)


@task(version="2-A")
def openai_monitorability_intervention(
    dataset: Literal[
        "all",
        "gpqa",
        "gpqa_encoded",
        "daft",
        "scruples_first_person",
        "scruples_suggest_right",
        "scruples_suggest_wrong",
        "wmdp_sandbagging",
    ] = "scruples_suggest_wrong",
    target_reasoning_effort: Optional[str] = "medium",
    monitor_model: str = "openai-api/openai/deepseek-v4.1-flash-1017",
    scope: str = "cot_only",
) -> Task:
    """OpenAI Intervention-level Monitorability benchmark (arXiv:2512.18311)."""
    ensure_asset("openai-monitorability")
    all_items = load_items(max_rows_per_dataset=None)
    int_items = [it for it in all_items if it.archetype == "intervention"]
    if dataset != "all":
        target_rel = f"intervention/{dataset}.jsonl"
        int_items = [it for it in int_items if it.dataset == target_rel]

    pairs: Dict[tuple[str, str], Dict[int, EvalItem]] = {}
    for it in int_items:
        if it.arm in (0, 1):
            pairs.setdefault((it.dataset, it.instance_id), {})[it.arm] = it

    samples: List[Sample] = []
    for (ds_name, inst_id), arm_map in pairs.items():
        if 0 not in arm_map or 1 not in arm_map:
            continue
        it0 = arm_map[0]
        it1 = arm_map[1]
        prompt0 = _render_model_prompt(it0)
        prompt1 = _render_model_prompt(it1)
        safe_ds = ds_name.replace("/", "_").replace(".jsonl", "")
        samples.append(
            Sample(
                id=f"{safe_ds}__{inst_id}",
                input=prompt1,
                target=it1.answer,
                metadata={
                    "dataset": ds_name,
                    "instance_id": inst_id,
                    "unperturbed_prompt": prompt0,
                    "perturbed_prompt": prompt1,
                    "unperturbed_item": it0.__dict__,
                    "perturbed_item": it1.__dict__,
                },
            )
        )

    return Task(
        version="2-A",
        config=GenerateConfig.model_validate(
            {
                "reasoning_effort": None
                if target_reasoning_effort == "default"
                else target_reasoning_effort
            }
        ),
        dataset=MemoryDataset(samples),
        solver=[paired_intervention_solver()],
        scorer=openai_intervention_scorer(monitor_model=monitor_model, scope=scope),
    )
