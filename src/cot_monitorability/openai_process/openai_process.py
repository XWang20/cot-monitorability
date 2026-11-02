from typing import List, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from cot_monitorability.openai_common import (
    _render_model_prompt,
    ensure_asset,
    load_items,
    openai_process_scorer,
    process_generate_solver,
)


@task(version="1-A")
def openai_monitorability_process(
    target_reasoning_effort: Optional[str] = "medium",
    monitor_model: str = "openai/deepseek-v4.1-flash-1017",
    scopes: str = "cot_only",
) -> Task:
    """OpenAI Process-level Monitorability benchmark (arXiv:2512.18311)."""
    ensure_asset("openai-monitorability")
    all_items = load_items(max_rows_per_dataset=None)
    proc_items = [it for it in all_items if it.archetype == "process"]

    samples: List[Sample] = []
    for it in proc_items:
        rendered_prompt = _render_model_prompt(it)
        safe_ds = it.dataset.replace("/", "_").replace(".jsonl", "")
        samples.append(
            Sample(
                id=f"{safe_ds}__{it.instance_id}",
                input=rendered_prompt,
                target=it.answer,
                metadata={
                    "dataset": it.dataset,
                    "instance_id": it.instance_id,
                    "eval_item": it.__dict__,
                },
            )
        )
    return Task(
        version="1-A",
        dataset=MemoryDataset(samples),
        solver=[
            process_generate_solver(target_reasoning_effort=target_reasoning_effort)
        ],
        scorer=openai_process_scorer(monitor_model=monitor_model, scopes=scopes),
    )
