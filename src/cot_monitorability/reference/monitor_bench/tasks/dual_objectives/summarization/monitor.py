# Adapted from semsorock/inspect-evals-monitor-bench at 9452b3ddcc1be81c4322e141dceb6f6c69b49434 (MIT); see NOTICE.
"""Monitor scorer for MonitorBench dual-objective summarization.

Adapted from ``pipeline/base.py`` in MonitorBench commit
``43dda5994bfb16d34b1c30d4b3482d78a714e640`` (Copyright (c) 2026 ASTRAL
Group @ UIUC, MIT License): https://github.com/ASTRAL-Group/MonitorBench
"""

from functools import lru_cache
from importlib import resources

import yaml
from inspect_ai.scorer import Scorer, scorer

from cot_monitorability.reference.monitor_bench.model_config import get_monitor_model
from cot_monitorability.reference.monitor_bench.outcome import (
    DEFAULT_MONITOR_ROLLOUTS,
    MONITOR_SCOPES,
    monitorability,
    pooled_monitor,
    text_outcome_monitor,
)
from cot_monitorability.reference.monitor_bench.tasks.dual_objectives.summarization.verifier import (
    verify_summary_payload,
)

__all__ = [
    "MONITOR_SCOPES",
    "load_monitor_prompts",
    "monitorability",
    "pooled_monitor",
    "summarization_monitor",
]


@lru_cache(maxsize=1)
def load_monitor_prompts() -> dict[str, str]:
    """Load the byte-vendored summarization monitor prompts."""
    loaded = yaml.safe_load(
        resources.files("cot_monitorability.reference.monitor_bench")
        .joinpath(
            "assets",
            "outcome",
            "prompts",
            "monitor_dual_objectives.summarization.yaml",
        )
        .read_text()
    )
    if not isinstance(loaded, dict) or set(loaded) != set(MONITOR_SCOPES):
        raise TypeError("Invalid summarization monitor prompt asset")
    if not all(isinstance(value, str) for value in loaded.values()):
        raise TypeError("Invalid summarization monitor prompt values")
    return {str(scope): value for scope, value in loaded.items()}


@scorer(metrics=[monitorability()])
def summarization_monitor(
    monitor_rollouts: int = DEFAULT_MONITOR_ROLLOUTS,
) -> Scorer:
    """Verify the payload, then query action, CoT, and combined monitors."""
    return text_outcome_monitor(
        prompt_loader=load_monitor_prompts,
        verifier=verify_summary_payload,
        monitor_loader=get_monitor_model,
        monitor_rollouts=monitor_rollouts,
    )
