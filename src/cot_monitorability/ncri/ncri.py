from typing import Optional

from cot_monitorability.ncri.components import (
    R,
    build_ncri_dataset,
    ncri_no_cot_solver,
    ncri_scorer,
)
from inspect_ai import Task, task


@task(version="1-A")
def ncri_bench(
    banks: str = "arithmetic",
    k_shot: int = R.DEFAULT_K_SHOT,
    no_prefill: bool = False,
    no_delib_system_v2: bool = False,
    effort: Optional[str] = None,
    disable_reasoning: bool = True,
    limit_per_bank: Optional[int] = None,
) -> Task:
    """nocot-bench NCRI (No-Chain-of-thought Reasoning Index) in Inspect AI."""
    return Task(
        version="1-A",
        dataset=build_ncri_dataset(
            banks=banks,
            k_shot=k_shot,
            no_prefill=no_prefill,
            no_delib_system_v2=no_delib_system_v2,
            limit_per_bank=limit_per_bank,
        ),
        solver=[
            ncri_no_cot_solver(
                no_prefill=no_prefill,
                effort=effort,
                disable_reasoning=disable_reasoning,
            )
        ],
        scorer=ncri_scorer(),
    )
