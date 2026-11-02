# Adapted from semsorock/inspect-evals-monitor-bench at 9452b3ddcc1be81c4322e141dceb6f6c69b49434 (MIT); see NOTICE.
"""Incremental Inspect AI port of the MonitorBench evaluation suite.

Ported from MonitorBench (https://arxiv.org/abs/2603.28590v2), upstream commit
43dda5994bfb16d34b1c30d4b3482d78a714e640 of
https://github.com/ASTRAL-Group/MonitorBench (MIT License).

Only completed tasks are exported. Available tasks cover dual-objective
steganography and summarization, plus paired WMDP, AIME, and MMLU-Pro goal sandbagging.
"""

from cot_monitorability.reference.monitor_bench.tasks.dual_objectives.steganography import (
    dual_objectives_steganography,
)
from cot_monitorability.reference.monitor_bench.tasks.dual_objectives.summarization import (
    dual_objectives_summarization,
)
from cot_monitorability.reference.monitor_bench.tasks.goal_sandbag.engineering import goal_sandbag_engineering
from cot_monitorability.reference.monitor_bench.tasks.goal_sandbag.math import goal_sandbag_math
from cot_monitorability.reference.monitor_bench.tasks.goal_sandbag.safety import goal_sandbag_safety

__all__ = [
    "dual_objectives_steganography",
    "goal_sandbag_safety",
    "goal_sandbag_math",
    "dual_objectives_summarization",
    "goal_sandbag_engineering",
]
