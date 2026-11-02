#!/usr/bin/env python3
"""Create a deterministic results index from Inspect log headers."""

import hashlib
import json
import pathlib
from datetime import datetime

from inspect_ai.log import read_eval_log

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "eval_logs"


def read_run(path):
    header = read_eval_log(str(path), header_only=True).model_dump(
        mode="json", exclude_none=True
    )
    evaluation = header["eval"]
    results = header.get("results") or {}
    stats = header.get("stats") or {}
    started = stats.get("started_at")
    finished = stats.get("completed_at")
    elapsed = None
    if started and finished:
        elapsed = (
            datetime.fromisoformat(finished) - datetime.fromisoformat(started)
        ).total_seconds()
    return {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "task": evaluation["task"],
        "task_version": evaluation.get("task_version"),
        "tags": evaluation.get("tags") or [],
        "model": evaluation["model"],
        "status": header["status"],
        "created": evaluation.get("created"),
        "started_at": started,
        "completed_at": finished,
        "elapsed_seconds": elapsed,
        "reasoning_effort": evaluation.get("model_generate_config", {}).get(
            "reasoning_effort"
        ),
        "task_args": evaluation.get("task_args", {}),
        "eval_config": {
            key: evaluation.get("config", {}).get(key) for key in ("limit", "epochs")
        },
        "completed_samples": results.get("completed_samples"),
        "total_samples": results.get("total_samples"),
        "scores": [
            {
                "scorer": score["name"],
                "scored_samples": score.get("scored_samples"),
                "metrics": {
                    key: item["value"] for key, item in score.get("metrics", {}).items()
                },
            }
            for score in results.get("scores", [])
        ],
        "model_usage": stats.get("model_usage", {}),
        "revision": evaluation.get("revision"),
        "packages": evaluation.get("packages", {}),
    }


def primary_result(run):
    metrics = {
        key: value for score in run["scores"] for key, value in score["metrics"].items()
    }
    if metrics.get("ncri_valid_rate") == 0:
        return "Not reportable (no valid no-CoT samples)"
    if metrics.get("intervention_reportable") == 0:
        return "Not reportable (eligible fraction below 10%)"
    names = (
        "cot_controllability_score",
        "process_gmean2_cot",
        "intervention_gmean2_lower",
        "legibility_pct",
        "coverage_pct",
        "ncri_primary_accuracy",
        "ncri_valid_rate",
        "mb_cot_monitorability",
    )
    return (
        "; ".join(f"{name}={metrics[name]:.4g}" for name in names if name in metrics)
        or "Not recorded"
    )


def display_model(run):
    return run["model"].removeprefix("openai/").removeprefix("openai-api/openai/")


def latest_completed(runs):
    latest = {}
    for run in sorted(runs, key=lambda run: run["created"] or ""):
        if run["status"] == "success" and run["task"].startswith("cot_monitorability/"):
            latest[(run["task"], display_model(run))] = run
    return [latest[key] for key in sorted(latest)]


def markdown_table(runs):
    lines = [
        "| Task / subset | Version | Model | Effort | Samples | Status | Main metrics |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        args = run["task_args"]
        subset = (
            args.get("dataset")
            or args.get("banks")
            or args.get("subset")
            or args.get("scopes")
            or ""
        )
        if args.get("modes"):
            subset += "/" + args["modes"]
        effort = run["reasoning_effort"] or (
            "disabled" if args.get("disable_reasoning") else "Not recorded"
        )
        model = display_model(run)
        task = run["task"]
        version = (
            run["task_version"] if run["task_version"] is not None else "Not recorded"
        )
        samples = (
            f"{run['completed_samples']}/{run['total_samples']}"
            if run["completed_samples"] is not None and run["total_samples"] is not None
            else "Not recorded"
        )
        lines.append(
            f"| {task} / {subset} | {version} | {model} | {effort} | "
            f"{samples} | "
            f"{run['status']} | {primary_result(run)} |"
        )
    return lines


def render_markdown(runs):
    latest = latest_completed(runs)
    selected = {run["file"] for run in latest}
    other = [run for run in runs if run["file"] not in selected]
    lines = [
        "# Evaluation results",
        "",
        f"Index of {len(runs)} stored runs. Raw "
        + chr(96)
        + ".eval"
        + chr(96)
        + " files are kept by the "
        "operator; this directory versions their metadata and checksums.",
        "",
        "Interpret each run using its recorded task namespace, version, configuration, "
        "and package revision. Limited runs do not establish full-dataset performance.",
        "",
        "## Latest completed runs",
        "",
        "One latest completed run per package task and model. The listed configuration "
        "and sample count belong to that run.",
        "",
    ]
    lines += markdown_table(latest)
    if other:
        lines += [
            "",
            "## Other recorded runs",
            "",
            "Earlier configurations, diagnostic attempts, and legacy runs.",
            "",
        ]
        lines += markdown_table(other)
    lines += [
        "",
        "## Interpretation",
        "",
        "- Counts refer to Inspect samples. Intervention samples contain paired generations.",
        "- Runs, subsets, and repetitions are listed separately; their samples are not pooled.",
        "- Effort comes from stored generation configuration or explicit no-CoT task settings.",
        "- Metrics measure different quantities across tasks. Do not combine them into a ranking.",
        "- A -1 value can denote an empty eligible window rather than a negative score.",
        "- Some historical aggregate metrics were reconstructed from stored sample scores "
        "on 2026-10-30. The indexer reads headers and does not modify logs.",
        "",
        "[index.json](index.json) contains file paths, SHA-256, task and package "
        "versions, timing, model usage, task arguments, and all recorded metrics.",
        "",
        "## Inspecting and indexing logs",
        "",
        "Run from the repository root:",
        "",
        "```bash",
        "uv run inspect view --recursive --log-dir eval_logs",
        "uv run python scripts/index_eval_logs.py",
        "```",
        "",
        "The indexer refuses to replace the index when no local logs are present.",
        "",
    ]
    return "\n".join(lines)


def main():
    runs = [read_run(path) for path in sorted(LOG_DIR.rglob("*.eval"))]
    if not runs:
        raise SystemExit(
            "No local .eval files found; refusing to replace the existing index."
        )
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / "index.json").write_text(
        json.dumps(
            {"schema_version": 1, "runs": runs},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (LOG_DIR / "README.md").write_text(render_markdown(runs), encoding="utf-8")
    print(f"Indexed {len(runs)} Inspect logs without changing the original files.")


if __name__ == "__main__":
    main()
