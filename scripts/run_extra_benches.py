#!/usr/bin/env python3
"""Run MonitorBench and Immediacy at the shared 50-sample specification.

Uses the monitorability experiment's budget ledger and model factory, but
writes resumable logs under <log-dir>/<bench>/<model>/<dataset>/. One model at
a time: the caller runs DeepSeek first, audits, then Kimi.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from inspect_ai import eval_set
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.solver import chain, solver

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "monitorability_runner", ROOT / "scripts/run_monitorability.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from cot_monitorability.experiment_costs import RequestBudget  # noqa: E402
from cot_monitorability.generation_reuse import ACTIVE_REUSE  # noqa: E402
from cot_monitorability.immediacy.immediacy import cot_immediacy_2510  # noqa: E402
from cot_monitorability.monitorbench.monitorbench import monitorbench  # noqa: E402

SUBSETS = [
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
]
IMMEDIACY = ["gpqa", "hle", "mmlu_pro"]


@solver
def declared_incomplete(sample_ids, reason):
    async def solve(state, generate):
        if state.sample_id in sample_ids:
            state.metadata["expected_incomplete"] = {
                "sample_id": state.sample_id,
                "reason": reason,
            }
            raise ValueError(reason)
        return state

    return solve


def build(bench, item):
    if bench == "monitorbench":
        return monitorbench(subset=item, stress_test="standard")
    return cot_immediacy_2510(dataset=item, correct_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/monitorability_50.json"
    )
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--baseline-cost", type=float)
    parser.add_argument("--uncertainty-reserve", type=float)
    parser.add_argument("--model", action="append")
    parser.add_argument(
        "--bench", action="append", choices=["monitorbench", "immediacy"]
    )
    parser.add_argument("--item", nargs="+")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--skip-sample", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    state = args.state_dir.resolve()
    logs = args.log_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    previous = state / "extra-protocol.json"
    prior = json.loads(previous.read_text()) if previous.exists() else {}
    baseline = (
        args.baseline_cost
        if args.baseline_cost is not None
        else prior.get("baseline_cost", 0)
    )
    reserve = (
        args.uncertainty_reserve
        if args.uncertainty_reserve is not None
        else prior.get("uncertainty_reserve", 0)
    )
    previous.write_text(
        json.dumps(
            {
                "config": config,
                "baseline_cost": baseline,
                "uncertainty_reserve": reserve,
            },
            indent=2,
        )
        + "\n"
    )
    budget = RequestBudget(state, config["budget_cny"], baseline, reserve)
    experiment = runner.Experiment(config, state, logs, budget)
    manifest_path = state / "extra-selection.json"
    experiment.manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    experiment.reuse.load([state, logs])
    ACTIVE_REUSE.set(experiment.reuse)

    benches = args.bench or ["monitorbench", "immediacy"]
    actors = config["actors"]
    ids = args.model or [a["id"] for a in actors]
    history = []
    for specification in [a for a in actors if a["id"] in ids]:
        for bench in benches:
            items = SUBSETS if bench == "monitorbench" else IMMEDIACY
            if args.item:
                items = [item for item in items if item in args.item]
            for item in items:
                condition = f"{bench}/{item}"
                task = build(bench, item)
                available = len(task.dataset)
                expected = min(args.limit, available)
                if not args.dry_run:
                    runner.freeze_dataset(
                        task, condition, experiment.manifest, args.limit
                    )
                    manifest_path.write_text(
                        json.dumps(experiment.manifest, indent=2) + "\n"
                    )
                    expected = len(task.dataset)
                print(
                    json.dumps(
                        {
                            "stage": condition,
                            "model": specification["id"],
                            "available": available,
                            "selected": expected,
                        }
                    ),
                    flush=True,
                )
                if args.dry_run:
                    continue
                task.solver = chain(
                    declared_incomplete(
                        set(args.skip_sample or []),
                        "Declared incomplete by the operator after repeated 1800s timeouts.",
                    ),
                    runner.identify_condition(condition),
                    task.solver,
                    runner.require_complete_actor(),
                )
                experiment.current = {
                    "model": specification["id"],
                    "phase": bench,
                    "preflight": False,
                    "dataset": item,
                }
                actor = experiment.model(specification, "actor", "high")
                roles = {
                    "monitor": experiment.model(specification, "monitor", "high"),
                    "judge": experiment.model(specification, "judge", "high"),
                }
                destination = logs / bench / specification["id"] / item
                destination.mkdir(parents=True, exist_ok=True)
                success, headers = eval_set(
                    [task],
                    model=actor,
                    model_roles=roles,
                    log_dir=str(destination),
                    log_dir_allow_dirty=True,
                    max_tasks=1,
                    max_samples=config["max_samples"],
                    max_retries=0,
                    retry_attempts=0,
                    retry_cleanup=False,
                    fail_on_error=True,
                    continue_on_fail=True,
                    display="none",
                    tags=[state.name, bench, item],
                    metadata={
                        "budget_currency": "CNY",
                        "budget_limit": budget.limit,
                        "sample_manifest_sha256": runner.hashlib.sha256(
                            manifest_path.read_bytes()
                        ).hexdigest(),
                        "experiment_stage": dict(experiment.current),
                    },
                )
                by_id = {
                    read_eval_log(p, header_only=True).eval.eval_id: p
                    for p in destination.glob("*.eval")
                }
                paths = [by_id[h.eval.eval_id] for h in headers]
                for path in paths:
                    log = read_eval_log(path, resolve_attachments=True)
                    unexpected = [
                        sample for sample in (log.samples or []) if sample.error
                    ]
                    errors = [sample for sample in (log.samples or []) if sample.error]
                    if unexpected and log.status == "success":
                        log.status = "error"
                        log.error = unexpected[0].error
                        write_eval_log(log, path)
                    elif (
                        errors
                        and all(
                            sample.metadata.get("expected_incomplete")
                            for sample in errors
                        )
                        and log.status != "success"
                    ):
                        log.status = "success"
                        log.error = None
                        write_eval_log(log, path)
                audit = runner.audit_logs(paths, "openai")
                if audit["totals"]["samples"] != expected:
                    audit["issues"].append(
                        {
                            "error": f"audited {audit['totals']['samples']} samples, expected {expected}"
                        }
                    )
                    audit["passed"] = False
                history.append(
                    {
                        **experiment.current,
                        "success": success,
                        "audit": audit,
                        "files": [str(p) for p in paths],
                    }
                )
                (state / "extra-status.json").write_text(
                    json.dumps({"history": history}, indent=2) + "\n"
                )
                experiment.embed_costs(paths)
                snapshot = budget.snapshot()
                print(
                    json.dumps(
                        {
                            "stage": condition,
                            "model": specification["id"],
                            "success": success,
                            "passed": audit["passed"],
                            "completed": audit["totals"]["completed"],
                            "issues": len(audit["issues"]),
                            "known_total": round(snapshot["known_total"], 3),
                            "available": round(snapshot["available"], 2),
                        }
                    ),
                    flush=True,
                )
                if not audit["passed"]:
                    raise SystemExit(f"stage {condition} did not pass")
    print(json.dumps({"completed": True}), flush=True)


if __name__ == "__main__":
    main()
