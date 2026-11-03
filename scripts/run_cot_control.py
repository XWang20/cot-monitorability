"""Run a resumable CoT-Control matrix with validated generation reuse and a cost ledger."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from dotenv import dotenv_values
from inspect_ai import eval_set
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput, ModelUsage, get_model
from inspect_ai.solver import chain, solver

from cot_monitorability.cot_control.components import MAIN_PAPER_MODES
from cot_monitorability.cot_control.cot_control import cot_controllability_qa
from cot_monitorability.reasoning import full_reasoning

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ContextVar("cot_control_sample", default={})
CACHE = {}
REUSED = []
CACHE_LOCK = threading.Lock()
LEDGER_LOCK = threading.Lock()


def fingerprint(input_value, target, metadata):
    payload = {
        "input": input_value,
        "target": target,
        "dataset": metadata.get("dataset"),
        "mode": metadata.get("mode"),
        "raw_id": metadata.get("raw_id"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def valid_generation(output):
    if output is None or output.stop_reason != "stop":
        return False
    try:
        text = full_reasoning(output.message)
    except ValueError:
        return False
    usage = output.usage
    return bool(text) or bool(
        usage and usage.reasoning_tokens == 0 and output.completion.strip()
    )


def estimate_cost(usage, started_at, pricing):
    zone = ZoneInfo(pricing["timezone"])
    local = datetime.fromisoformat(started_at).astimezone(zone)
    minute = local.strftime("%H:%M")
    peak = local.weekday() < 5 and any(
        start <= minute < end for start, end in pricing["weekday_peak_windows"]
    )
    rates = pricing["peak_per_million"] if peak else pricing["offpeak_per_million"]
    if (
        not usage
        or usage.get("prompt_tokens") is None
        or usage.get("completion_tokens") is None
    ):
        return {"cost": None, "rates": rates, "period": "peak" if peak else "offpeak"}
    prompt = int(usage["prompt_tokens"])
    cached = int(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or 0
    )
    output = int(usage["completion_tokens"])
    uncached = max(0, prompt - cached)
    cost = (
        uncached * rates["input"]
        + cached * rates["cached_input"]
        + output * rates["output"]
    ) / 1_000_000
    return {
        "cost": cost,
        "rates": rates,
        "period": "peak" if peak else "offpeak",
        "uncached_input_tokens": uncached,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        ),
    }


def load_reuse(paths, model, effort, wanted):
    accepted = {}
    for directory in paths:
        for path in sorted(directory.rglob("*.eval")):
            try:
                log = read_eval_log(path, resolve_attachments=True)
            except Exception:
                continue
            if not log.eval.task.endswith("cot_controllability_qa"):
                continue
            canonical = log.eval.model.removeprefix("openai/").removeprefix(
                "openai-api/openai/"
            )
            expected = model.removeprefix("openai/").removeprefix("openai-api/openai/")
            if (
                canonical != expected
                or log.eval.model_generate_config.reasoning_effort != effort
            ):
                continue
            for sample in log.samples or []:
                key = fingerprint(sample.input, sample.target, sample.metadata)
                if key not in wanted or not valid_generation(sample.output):
                    continue
                actor_events = [
                    event
                    for event in sample.events
                    if getattr(event, "event", None) == "model"
                    and event.role is None
                    and event.call
                ]
                # Older SDKs could put Responses summaries in the reasoning field.
                summary_only_response = any(
                    item.get("type") == "reasoning"
                    and item.get("summary")
                    and not item.get("content")
                    for event in actor_events
                    if isinstance(event.call.response, dict)
                    for item in event.call.response.get("output", [])
                    if isinstance(item, dict)
                )
                if summary_only_response:
                    continue
                accepted[key] = {
                    "output": sample.output.model_dump(mode="json"),
                    "messages": [
                        message.model_dump(mode="json") for message in sample.messages
                    ],
                    "source": {
                        "log": str(path.relative_to(ROOT)),
                        "sample_id": sample.id,
                        "task_version": log.eval.task_version,
                    },
                }
    return accepted


@solver
def prepare_sample():
    async def solve(state, generate):
        SAMPLE.set(
            {
                "dataset": state.metadata["dataset"],
                "mode": state.metadata["mode"],
                "sample_id": state.sample_id,
            }
        )
        key = fingerprint(state.input, state.target.text, state.metadata)
        with CACHE_LOCK:
            record = copy.deepcopy(CACHE.get(key))
        if record is not None:
            from inspect_ai.model import (
                ChatMessageAssistant,
                ChatMessageSystem,
                ChatMessageUser,
            )

            classes = {
                "assistant": ChatMessageAssistant,
                "system": ChatMessageSystem,
                "user": ChatMessageUser,
            }
            state.output = ModelOutput.model_validate(record["output"])
            state.messages = [
                classes[message["role"]](**message) for message in record["messages"]
            ]
            state.metadata["reused_generation"] = record["source"]
            state.completed = True
            with CACHE_LOCK:
                REUSED.append({**SAMPLE.get(), **record["source"]})
        return state

    return solve


@solver
def remember_generation():
    async def solve(state, generate):
        if not valid_generation(state.output):
            raise ValueError("Generation is truncated or full reasoning is unavailable")
        key = fingerprint(state.input, state.target.text, state.metadata)
        with CACHE_LOCK:
            CACHE[key] = {
                "output": state.output.model_dump(mode="json"),
                "messages": [
                    message.model_dump(mode="json") for message in state.messages
                ],
                "source": {
                    "log": "this_experiment",
                    "sample_id": state.sample_id,
                    "task_version": state.metadata.get("experiment_task_version"),
                },
            }
        return state

    return solve


def ledger_hooks(role, state_dir, pricing):
    async def request_hook(request):
        request.extensions["cot_started_at"] = datetime.now(timezone.utc).isoformat()
        request.extensions["cot_sample"] = dict(SAMPLE.get())
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            if body.get(
                "reasoning_effort"
            ) != "high" or "reasoning_effort" not in body.get(
                "allowed_openai_params", []
            ):
                raise ValueError(
                    "Experiment requests must carry high reasoning effort with proxy passthrough"
                )

    async def response_hook(response):
        if not response.request.url.path.endswith("/chat/completions"):
            return
        await response.aread()
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        started = response.request.extensions.get(
            "cot_started_at", datetime.now(timezone.utc).isoformat()
        )
        record = {
            **response.request.extensions.get("cot_sample", {}),
            "role": role,
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": response.status_code,
            "currency": pricing["currency"],
            **estimate_cost(payload.get("usage"), started, pricing),
            "response_id": payload.get("id"),
            "proxy_reported_cost": response.headers.get("x-litellm-response-cost"),
        }
        with LEDGER_LOCK:
            with (state_dir / "requests.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    return {"request": [request_hook], "response": [response_hook]}


def restore_reused_history(log_dir):
    """Restore recorded actor events and tokens without making or billing new requests."""
    restored = 0
    for path in sorted(log_dir.rglob("*.eval")):
        header = read_eval_log(path, header_only=True)
        if header.status == "started":
            continue
        if header.results is None:
            continue
        if (header.results.metadata or {}).get("reuse_transcripts_restored"):
            continue
        log = read_eval_log(path, resolve_attachments=True)
        changed = False
        added_usage = ModelUsage()
        for sample in log.samples or []:
            source = sample.metadata.get("reused_generation")
            if not source or sample.metadata.get("reused_history_restored"):
                continue
            if source["log"] == "this_experiment":
                candidates = [
                    other for other in log_dir.rglob("*.eval") if other != path
                ]
            else:
                candidates = [ROOT / source["log"]]
            actor_event = None
            for candidate in candidates:
                original = read_eval_log(candidate, resolve_attachments=True)
                old_sample = next(
                    (
                        item
                        for item in original.samples or []
                        if item.id == source["sample_id"]
                    ),
                    None,
                )
                if old_sample is None:
                    continue
                actor_event = next(
                    (
                        event
                        for event in reversed(old_sample.events)
                        if isinstance(event, ModelEvent)
                        and event.role is None
                        and event.output
                        and event.error is None
                        and event.output.message.model_dump(mode="json")
                        == sample.output.message.model_dump(mode="json")
                    ),
                    None,
                )
                if actor_event is not None:
                    break
            if actor_event is None:
                continue
            event = actor_event.model_copy(
                deep=True,
                update={
                    "uuid": str(uuid.uuid4()),
                    "span_id": None,
                    "input_refs": None,
                    "model": log.eval.model,
                    "cache": "read",
                    "metadata": {
                        **(actor_event.metadata or {}),
                        "reused_from": source,
                        "original_model": actor_event.model,
                        "original_event_uuid": actor_event.uuid,
                        "new_actor_request": False,
                    },
                },
            )
            sample.events.insert(0, event)
            usage = (
                sample.output.usage.model_copy(deep=True)
                if sample.output.usage
                else ModelUsage()
            )
            usage.total_cost = None
            sample.model_usage[log.eval.model] = (
                sample.model_usage.get(log.eval.model, ModelUsage()) + usage
            )
            sample.metadata["reused_history_restored"] = True
            sample.metadata["new_actor_cost_cny"] = 0
            added_usage += usage
            restored += 1
            changed = True
        if changed:
            metadata = dict(log.results.metadata or {})
            metadata["new_call_model_usage"] = {
                name: usage.model_dump(mode="json")
                for name, usage in log.stats.model_usage.items()
            }
            metadata["reuse_transcripts_restored"] = all(
                not item.metadata.get("reused_generation")
                or item.metadata.get("reused_history_restored")
                for item in log.samples
            )
            metadata["token_usage_includes_reused_generations"] = True
            metadata["billing_source"] = (
                "per-request ledger; restoring history does not charge again"
            )
            log.results.metadata = metadata
            log.stats.model_usage[log.eval.model] = (
                log.stats.model_usage.get(log.eval.model, ModelUsage()) + added_usage
            )
            write_eval_log(log, path)
    return restored


def write_report(log_dir, state_dir, pricing, final=None):
    with LEDGER_LOCK:
        lines = (
            (state_dir / "requests.jsonl").read_text().splitlines()
            if (state_dir / "requests.jsonl").exists()
            else []
        )
    records = [json.loads(line) for line in lines if line.strip()]
    costs = {}
    for role in ("actor", "judge"):
        selected = [row for row in records if row["role"] == role]
        costs[role] = {
            "requests": len(selected),
            "cost": sum(row["cost"] for row in selected if row["cost"] is not None),
            "unmetered_requests": sum(row["cost"] is None for row in selected),
            "output_tokens": sum(row.get("output_tokens", 0) for row in selected),
            "uncached_input_tokens": sum(
                row.get("uncached_input_tokens", 0) for row in selected
            ),
            "cached_input_tokens": sum(
                row.get("cached_input_tokens", 0) for row in selected
            ),
        }
    latest = {}
    for path in sorted(log_dir.rglob("*.eval")):
        try:
            log = read_eval_log(path, header_only=True)
        except Exception:
            continue
        latest[(log.eval.task_args.get("dataset"), log.eval.task_args.get("modes"))] = {
            "file": str(path.relative_to(ROOT)),
            "status": log.status,
            "dataset": log.eval.task_args.get("dataset"),
            "mode": log.eval.task_args.get("modes"),
            "completed_samples": log.results.completed_samples if log.results else 0,
            "total_samples": log.results.total_samples if log.results else 50,
            "metrics": {
                name: metric.value
                for score in (log.results.scores if log.results else [])
                for name, metric in score.metrics.items()
            },
        }
    logs = list(latest.values())
    with CACHE_LOCK:
        reused = list(REUSED)
    report = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "currency": pricing["currency"],
        "cost_basis": "provided discounted rates per million tokens at request start; reasoning is included in output",
        "costs": costs,
        "total_new_cost": sum(row["cost"] for row in costs.values()),
        "reused_generation_records": reused,
        "logs": logs,
        "final": final,
    }
    target = state_dir / "report.json"
    temporary = state_dir / "report.json.tmp"
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--reuse-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.log_dir = args.log_dir.resolve()
    args.reuse_dir = [path.resolve() for path in args.reuse_dir]
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    pricing = config["pricing"]
    tasks = []
    wanted = set()
    plan = []
    for dataset in config["datasets"]:
        for mode in MAIN_PAPER_MODES[1:]:
            task = cot_controllability_qa(
                dataset=dataset,
                modes=mode,
                target_reasoning_effort=config["reasoning_effort"],
            )
            for sample in task.dataset:
                sample.metadata["experiment_task_version"] = task.version
            selected = list(task.dataset)[: config["samples_per_condition"]]
            assert len(selected) == config["samples_per_condition"]
            wanted.update(
                fingerprint(sample.input, sample.target, sample.metadata)
                for sample in selected
            )
            plan.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "sample_ids": [sample.id for sample in selected],
                    "raw_ids": [sample.metadata["raw_id"] for sample in selected],
                }
            )
            task.solver = chain(prepare_sample(), task.solver, remember_generation())
            tasks.append(task)
    with CACHE_LOCK:
        CACHE.update(
            load_reuse(
                args.reuse_dir + [args.log_dir],
                config["model"],
                config["reasoning_effort"],
                wanted,
            )
        )
        reused_count = len(CACHE)
    (args.state_dir / "protocol.json").write_text(
        json.dumps(
            {
                "config": config,
                "plan": plan,
                "initial_reusable_generations": reused_count,
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "selection": "first eligible rows after the upstream loader's mode-specific filters",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "conditions": len(tasks),
                "target_samples": len(wanted),
                "reuse_candidates": reused_count,
            }
        ),
        flush=True,
    )
    values = dotenv_values(ROOT / ".env")
    base_url = os.environ.get("OPENAI_BASE_URL") or values.get("OPENAI_BASE_URL")
    key = (
        os.environ.get("OPENAI_API_KEY")
        or values.get("OPENAI_API_KEY")
        or os.environ.get("PROXY_API_KEY")
        or values.get("PROXY_API_KEY")
    )
    if not base_url or not key:
        raise ValueError(
            "Configure OPENAI_BASE_URL and OPENAI_API_KEY in the operator environment"
        )
    body = {
        "allowed_openai_params": ["reasoning_effort"],
        "reasoning": {"effort": "high"},
    }
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: reporter(stop, args.log_dir, args.state_dir, pricing),
        daemon=True,
    )
    thread.start()
    success = False
    # Native eval-set owns the asynchronous model lifecycle.
    actor_client = httpx.AsyncClient(
        event_hooks=ledger_hooks("actor", args.state_dir, pricing), timeout=300
    )
    judge_client = httpx.AsyncClient(
        event_hooks=ledger_hooks("judge", args.state_dir, pricing), timeout=300
    )
    actor = get_model(
        config["model"],
        base_url=base_url,
        api_key=key,
        memoize=False,
        http_client=actor_client,
        responses_api=False,
        stream=False,
        config=GenerateConfig(
            reasoning_effort="high", extra_body=body, max_retries=0, timeout=300
        ),
    )
    judge = get_model(
        config["model"],
        base_url=base_url,
        api_key=key,
        memoize=False,
        http_client=judge_client,
        responses_api=False,
        stream=False,
        config=GenerateConfig(
            reasoning_effort="high",
            extra_body=body,
            temperature=0,
            max_tokens=8192,
            max_retries=0,
            timeout=300,
        ),
    )
    try:
        success, _ = eval_set(
            tasks,
            model=actor,
            model_roles={"judge": judge},
            log_dir=str(args.log_dir),
            limit=config["samples_per_condition"],
            max_samples=config["max_samples"],
            max_tasks=config.get("max_tasks", 4),
            max_retries=0,
            retry_attempts=config["retry_attempts"],
            retry_cleanup=False,
            continue_on_fail=True,
            fail_on_error=True,
            display="none",
            tags=["cot-control-ds50"],
            metadata={"pricing_currency": pricing["currency"]},
        )
    finally:
        stop.set()
        thread.join(timeout=30)
        restore_reused_history(args.log_dir)
        report = write_report(args.log_dir, args.state_dir, pricing, final=success)
        with LEDGER_LOCK:
            records = (
                [
                    json.loads(line)
                    for line in (args.state_dir / "requests.jsonl")
                    .read_text()
                    .splitlines()
                    if line.strip()
                ]
                if (args.state_dir / "requests.jsonl").exists()
                else []
            )
        for entry in report["logs"]:
            selected = [
                row
                for row in records
                if row.get("dataset") == entry["dataset"]
                and row.get("mode") == entry["mode"]
            ]
            log_path = ROOT / entry["file"]
            log = read_eval_log(log_path, resolve_attachments=True)
            if log.results:
                log.results.metadata = {
                    **(log.results.metadata or {}),
                    "run_cost": {
                        "currency": pricing["currency"],
                        "amount": sum(
                            row["cost"] for row in selected if row["cost"] is not None
                        ),
                        "request_count": len(selected),
                        "unmetered_requests": sum(
                            row["cost"] is None for row in selected
                        ),
                        "pricing": pricing,
                        "basis": "provided rates and metered tokens; includes this condition's retries",
                    },
                }
                write_eval_log(log, log_path)
        print(
            json.dumps(
                {
                    "final": success,
                    "cost": report["total_new_cost"],
                    "currency": pricing["currency"],
                }
            ),
            flush=True,
        )


def reporter(stop, log_dir, state_dir, pricing):
    while not stop.wait(30):
        try:
            restore_reused_history(log_dir)
            report = write_report(log_dir, state_dir, pricing)
            print(
                json.dumps(
                    {
                        "progress": sum(
                            log["completed_samples"] for log in report["logs"]
                        ),
                        "cost": report["total_new_cost"],
                        "currency": pricing["currency"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            print(json.dumps({"report_error": type(error).__name__}), flush=True)


if __name__ == "__main__":
    main()
