"""Run a budgeted, resumable multi-model Inspect experiment with shared generation caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values
from inspect_ai import eval_set
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import CachePolicy, GenerateConfig, get_model
from inspect_ai.solver import chain, solver
from openai import DefaultAsyncHttpxClient

from cot_monitorability.experiment_costs import RequestBudget
from cot_monitorability.generation_reuse import (
    ACTIVE_REUSE,
    GenerationReuse,
    reuse_completed_generation,
)
from cot_monitorability.ncri.logs import aggregate_ncri_logs, refresh_ncri_metrics
from cot_monitorability.ncri.ncri import ncri_bench
from cot_monitorability.openai_intervention.openai_intervention import (
    openai_monitorability_intervention,
)
from cot_monitorability.openai_process.openai_process import (
    openai_monitorability_process,
)
from cot_monitorability.reasoning import full_reasoning

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ContextVar("monitorability_sample", default={})
KNOWN_INCOMPLETE = ContextVar("known_incomplete_samples", default=[])


def canonical(model):
    return model.removeprefix("openai-api/openai/").removeprefix("openai/")


def sample_hash(sample):
    payload = sample.model_dump(mode="json", exclude={"metadata"})
    if isinstance(payload.get("input"), list):
        payload["input"] = [
            {k: v for k, v in message.items() if k != "id"}
            for message in payload["input"]
        ]
    payload["protocol"] = {
        key: value
        for key, value in sample.metadata.items()
        if not key.startswith("experiment_")
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def freeze_dataset(task, key, manifest, count):
    samples = list(task.dataset)
    if key not in manifest:
        samples = samples[: min(count, len(samples))]
        manifest[key] = {
            "ids": [s.id for s in samples],
            "hashes": [sample_hash(s) for s in samples],
            "available": len(task.dataset),
            "selected": len(samples),
        }
    else:
        by_id = {s.id: s for s in samples}
        samples = [by_id[identifier] for identifier in manifest[key]["ids"]]
        if [sample_hash(s) for s in samples] != manifest[key]["hashes"]:
            raise ValueError(f"Frozen inputs, targets or protocol changed: {key}")
    for index, sample in enumerate(samples):
        sample.metadata["experiment_selection_index"] = index
    task.dataset = MemoryDataset(samples)


@solver
def identify_condition(condition: str):
    async def solve(state, generate):
        state.metadata["experiment_condition"] = condition
        from inspect_ai.model import get_model

        for record in KNOWN_INCOMPLETE.get():
            if (
                canonical(str(get_model())) == record["model"]
                and state.sample_id == record["sample_id"]
                and state.metadata.get("dataset") == record["dataset"]
            ):
                state.metadata["expected_incomplete"] = record
                raise ValueError(record["reason"])
        CONTEXT.set({"condition": condition, "sample_id": state.sample_id})
        return state

    return solve


@solver
def require_complete_actor():
    async def solve(state, generate):
        if (
            state.output is None
            or not state.output.choices
            or state.output.stop_reason != "stop"
        ):
            raise ValueError("Actor generation did not finish normally")
        thought = full_reasoning(state.output.message)
        if state.output.usage and state.output.usage.reasoning_tokens and not thought:
            raise ValueError(
                "Actor reported reasoning tokens without readable reasoning"
            )
        if "unperturbed_message_dump" in state.metadata:
            from inspect_ai.model import ChatMessageAssistant

            thought = full_reasoning(
                ChatMessageAssistant.model_validate(
                    state.metadata["unperturbed_message_dump"]
                )
            )
            if state.metadata.get("unperturbed_rtok") and not thought:
                raise ValueError(
                    "Unperturbed actor reported reasoning tokens without readable reasoning"
                )
            if state.metadata.get("unperturbed_stop_reason") != "stop":
                raise ValueError("Unperturbed actor generation did not finish normally")
        return state

    return solve


def build_tasks(config, manifest):
    tasks = {"cot": [], "scopes": [], "ncri": []}
    for scope in config["scopes"]:
        task = openai_monitorability_process(
            target_reasoning_effort="high", scopes=scope
        )
        key = "process/physics/" + scope
        freeze_dataset(task, key, manifest, config["samples_per_condition"])
        task.solver = chain(
            identify_condition(key),
            reuse_completed_generation(),
            task.solver,
            require_complete_actor(),
        )
        tasks["cot" if scope == "cot_only" else "scopes"].append(task)
        for dataset in config["intervention_datasets"]:
            task = openai_monitorability_intervention(
                dataset=dataset, target_reasoning_effort="high", scope=scope
            )
            key = "intervention/" + dataset + "/" + scope
            freeze_dataset(task, key, manifest, config["samples_per_condition"])
            task.solver = chain(
                identify_condition(key),
                reuse_completed_generation(),
                task.solver,
                require_complete_actor(),
            )
            tasks["cot" if scope == "cot_only" else "scopes"].append(task)
    for bank in config["ncri_banks"]:
        task = ncri_bench(banks=bank, k_shot=10, disable_reasoning=True)
        key = "ncri/" + bank + "/no_cot_10shot"
        freeze_dataset(task, key, manifest, config["samples_per_condition"])
        task.solver = chain(
            identify_condition(key), reuse_completed_generation(), task.solver
        )
        tasks["ncri"].append(task)
    return tasks


def expected_samples(tasks, preflight_samples=None):
    """Samples a stage must yield (one epoch), including the preflight limit."""
    if preflight_samples is None:
        return sum(len(task.dataset) for task in tasks)
    return sum(min(preflight_samples, len(task.dataset)) for task in tasks)


TASK_DIRS = {
    "openai_monitorability_process": "process",
    "openai_monitorability_intervention": "intervention",
    "ncri_bench": "ncri",
}


def path_slug(value):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", value).strip("-") or "unknown"


def task_dir_name(task):
    """One directory per task family, so each eval-set log dir stays resumable."""
    name = task.name.rsplit("/", 1)[-1]
    return TASK_DIRS.get(name, path_slug(name))


def ncri_summary_path(log_dir, model_id):
    """Model-level NCRI summary, next to that model's bank runs."""
    name = "ncri-index_" + hashlib.sha256(model_id.encode()).hexdigest()[:12] + ".eval"
    return Path(log_dir) / path_slug(model_id) / "ncri-index" / name


def audit_logs(paths, kind):
    issues, totals = (
        [],
        {"samples": 0, "completed": 0, "no_cot_invalid": 0, "expected_incomplete": 0},
    )
    for path in paths:
        log = read_eval_log(path, resolve_attachments=True)
        if log.status != "success":
            issues.append({"file": str(path), "error": "evaluation did not complete"})
        for sample in log.samples or []:
            totals["samples"] += 1
            if sample.error:
                if sample.metadata.get("expected_incomplete"):
                    totals["expected_incomplete"] += 1
                    continue
                issues.append(
                    {
                        "file": str(path),
                        "sample": sample.id,
                        "error": sample.error.message[-1200:],
                    }
                )
                continue
            totals["completed"] += 1
            if kind == "ncri":
                score = next(iter(sample.scores.values()))
                meta = score.metadata
                if (
                    meta.get("reasoning_tokens") != 0
                    or meta.get("hidden_channel_tokens") != 0
                ):
                    issues.append(
                        {
                            "file": str(path),
                            "sample": sample.id,
                            "error": "NCRI reasoning counters are nonzero or absent",
                        }
                    )
                if not score.value.get("valid_no_cot"):
                    totals["no_cot_invalid"] += 1
            else:
                if log.eval.task.endswith("openai_monitorability_process"):
                    result = next(iter(sample.scores.values()))
                    answers = result.metadata.get("monitor_answers", {})
                    if not answers or any(
                        not json.loads(value) for value in answers.values()
                    ):
                        issues.append(
                            {
                                "file": str(path),
                                "sample": sample.id,
                                "error": "Process monitor returned no parseable theorem verdicts",
                            }
                        )
                if sample.output.stop_reason != "stop":
                    issues.append(
                        {
                            "file": str(path),
                            "sample": sample.id,
                            "error": "truncated actor output",
                        }
                    )
                try:
                    full_reasoning(sample.output.message)
                    if "unperturbed_message_dump" in sample.metadata:
                        from inspect_ai.model import ChatMessageAssistant

                        full_reasoning(
                            ChatMessageAssistant.model_validate(
                                sample.metadata["unperturbed_message_dump"]
                            )
                        )
                except ValueError as error:
                    issues.append(
                        {"file": str(path), "sample": sample.id, "error": str(error)}
                    )
    return {"passed": not issues, "totals": totals, "issues": issues}


class Experiment:
    def __init__(self, config, state, logs, budget):
        self.config, self.state, self.logs, self.budget = config, state, logs, budget
        self.manifest = (
            json.loads((state / "selection.json").read_text())
            if (state / "selection.json").exists()
            else {}
        )
        self.history = []
        self.reuse = GenerationReuse()
        self.current = {}
        self.stop = threading.Event()
        self.report_lock = threading.Lock()

    def report(self):
        with self.report_lock:
            return self._report()

    def _report(self):
        costs = self.budget.snapshot()
        rows = costs.pop("responses")
        by_role = {}
        for row in rows:
            key = row["actor_model"] + "/" + row["role"]
            item = by_role.setdefault(key, {"requests": 0, "cost": 0})
            item["requests"] += 1
            item["cost"] += row.get("cost") or 0
        live = []
        try:
            result = subprocess.run(
                [
                    str(
                        Path(sys.executable).with_name(
                            "inspect.exe" if os.name == "nt" else "inspect"
                        )
                    ),
                    "ctl",
                    "task",
                    "list",
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                live = [
                    {"task": t["task"], "status": t["status"], "samples": t["samples"]}
                    for t in json.loads(result.stdout)["tasks"]
                    if t["pid"] == os.getpid() and t["status"] == "running"
                ]
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
        result = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "current": self.current,
            "history": self.history,
            "live_tasks": live,
            "costs": costs,
            "by_model_role": by_role,
        }
        temporary = self.state / "status.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(self.state / "status.json")
        return result

    def reporter(self):
        while not self.stop.wait(30):
            try:
                result = self.report()
                print(
                    json.dumps(
                        {
                            "stage": self.current,
                            "known_total_cny": result["costs"]["known_total"],
                            "in_flight": sum(
                                x["samples"]["in_flight"] for x in result["live_tasks"]
                            ),
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                print(json.dumps({"report_error": str(error)}), flush=True)

    def model(self, specification, role, effort):
        values = dotenv_values(ROOT / ".env")
        base_url = os.environ.get("OPENAI_BASE_URL") or values.get("OPENAI_BASE_URL")
        api_key = (
            os.environ.get("OPENAI_API_KEY")
            or values.get("OPENAI_API_KEY")
            or values.get("PROXY_API_KEY")
        )
        if self.config.get("connection_profile") == "local-proxy":
            if urlparse(base_url or "").hostname not in (
                "localhost",
                "127.0.0.1",
                "::1",
            ):
                raise ValueError(
                    "Local proxy profile requires a loopback OPENAI_BASE_URL"
                )
        if not base_url or not api_key:
            raise ValueError(
                "Configure OPENAI_BASE_URL and OPENAI_API_KEY in the operator environment"
            )
        chosen = specification if role == "actor" else self.config["monitor"]
        body = {"allowed_openai_params": ["reasoning_effort"]}
        model_name = chosen["model"]
        generation_effort = effort
        if role == "actor" and effort == "none" and specification.get("ncri_model"):
            model_name = specification["ncri_model"]
            generation_effort = None
            body = {"extra_body": specification["ncri_disable_body"]}
        elif effort != "none":
            body["reasoning"] = {"effort": effort}

        def context():
            return {
                **CONTEXT.get(),
                **self.current,
                "actor_model": specification["id"],
                "role": role,
            }

        client = DefaultAsyncHttpxClient(
            timeout=600,
            event_hooks=self.budget.hooks(chosen["pricing"], context, effort),
        )
        return get_model(
            model_name,
            base_url=base_url,
            api_key=api_key,
            memoize=False,
            http_client=client,
            client_timeout=600,
            max_retries=0,
            responses_api=False,
            stream=False,
            config=GenerateConfig(
                reasoning_effort=generation_effort,
                extra_body=body,
                temperature=0 if role != "actor" else None,
                max_tokens=8192
                if role != "actor"
                else self.config.get("actor_max_tokens", 65536),
                adaptive_connections=64,
                max_retries=0,
                timeout=600,
                cache=CachePolicy(expiry="30D", scopes={"experiment": self.state.name}),
            ),
        )

    def phase(self, tasks, specification, name, kind, preflight):
        effort = "none" if kind == "ncri" else "high"
        self.current = {
            "model": specification["id"],
            "phase": name,
            "preflight": preflight,
        }
        self.reuse.load([self.state / "preflight", self.logs])
        ACTIVE_REUSE.set(self.reuse)
        known = self.config.get("known_incomplete", [])
        KNOWN_INCOMPLETE.set(known)
        allow_incomplete = any(
            record["model"] == specification["id"]
            and any(
                sample.id == record["sample_id"]
                and sample.metadata.get("dataset") == record["dataset"]
                for task in tasks
                for sample in task.dataset
            )
            for record in known
        )
        actor = self.model(specification, "actor", effort)
        roles = (
            {}
            if kind == "ncri"
            else {
                "monitor": self.model(specification, "monitor", "high"),
                "judge": self.model(specification, "judge", "high"),
            }
        )
        base = (
            self.state / "preflight" / path_slug(specification["id"])
            if preflight
            else self.logs / path_slug(specification["id"])
        )
        groups = {}
        for task in tasks:
            groups.setdefault(task_dir_name(task), []).append(task)
        success = True
        paths = []
        for group_name, group_tasks in groups.items():
            destination = base / group_name
            destination.mkdir(parents=True, exist_ok=True)
            group_success, headers = eval_set(
                group_tasks,
                model=actor,
                model_roles=roles,
                log_dir=str(destination),
                log_dir_allow_dirty=True,
                limit=self.config["preflight_samples"] if preflight else None,
                max_tasks=self.config["max_tasks"],
                max_samples=self.config["max_samples"],
                max_retries=0,
                retry_attempts=0,
                retry_cleanup=False,
                fail_on_error=not allow_incomplete,
                continue_on_fail=True,
                display="none",
                tags=[self.state.name, name],
                metadata={
                    "budget_currency": "CNY",
                    "budget_limit": self.budget.limit,
                    "sample_manifest_sha256": hashlib.sha256(
                        (self.state / "selection.json").read_bytes()
                    ).hexdigest(),
                    "experiment_stage": dict(self.current),
                },
            )
            success = success and group_success
            by_id = {
                read_eval_log(p, header_only=True).eval.eval_id: p
                for p in destination.glob("*.eval")
            }
            paths.extend(by_id[header.eval.eval_id] for header in headers)
        for path in paths:
            self.reuse.restore(path)
            log = read_eval_log(path, resolve_attachments=True)
            unexpected = [
                sample
                for sample in (log.samples or [])
                if sample.error and not sample.metadata.get("expected_incomplete")
            ]
            if unexpected and log.status == "success":
                log.status = "error"
                log.error = unexpected[0].error
                write_eval_log(log, path)
        audit = audit_logs(paths, kind)
        expected = expected_samples(
            tasks, self.config["preflight_samples"] if preflight else None
        )
        if audit["totals"]["samples"] != expected:
            audit["issues"].append(
                {
                    "error": (
                        f"Audited {audit['totals']['samples']} samples but the "
                        f"task matrix defines {expected}"
                    )
                }
            )
            audit["passed"] = False
        self.history.append(
            {
                **self.current,
                "success": success,
                "audit": audit,
                "files": [str(p) for p in paths],
            }
        )
        self.report()
        if not success or not audit["passed"]:
            raise RuntimeError(
                f"Stage {name} did not pass its validity checks; see status.json"
            )
        self.embed_costs(paths)
        if kind == "ncri":
            for path in paths:
                log = read_eval_log(path, resolve_attachments=True)
                refresh_ncri_metrics(log)
                write_eval_log(log, path)
            if not preflight:
                summary = ncri_summary_path(self.logs, specification["id"])
                summary.parent.mkdir(parents=True, exist_ok=True)
                aggregate_ncri_logs(paths, summary)
        return paths

    def embed_costs(self, paths):
        rows = self.budget.snapshot()["responses"]
        for path in paths:
            log = read_eval_log(path, resolve_attachments=True)
            conditions = {s.metadata.get("experiment_condition") for s in log.samples}
            selected = [
                r
                for r in rows
                if r["actor_model"] == self.current["model"]
                and r.get("condition") in conditions
                and r.get("preflight") == self.current["preflight"]
                and r.get("phase") == self.current["phase"]
            ]
            run_cost = {
                "currency": "CNY",
                "amount": sum(r.get("cost") or 0 for r in selected),
                "request_count": len(selected),
                "unmetered_requests": sum(r.get("cost") is None for r in selected),
                "basis": "metered requests at recorded request-time rates; cache hits and reused actor outputs incur no new cost",
            }
            if log.results is not None:
                log.results.metadata = {
                    **(log.results.metadata or {}),
                    "run_cost": run_cost,
                }
            else:
                log.eval.metadata = {**(log.eval.metadata or {}), "run_cost": run_cost}
            write_eval_log(log, path)

    def run(self):
        thread = threading.Thread(target=self.reporter, daemon=True)
        thread.start()
        try:
            for specification in self.config["actors"]:
                for preflight in [True, False]:
                    tasks = build_tasks(self.config, self.manifest)
                    (self.state / "selection.json").write_text(
                        json.dumps(self.manifest, indent=2) + "\n"
                    )
                    self.phase(
                        tasks["cot"], specification, "cot_only", "openai", preflight
                    )
                    self.phase(
                        tasks["scopes"],
                        specification,
                        "other_scopes",
                        "openai",
                        preflight,
                    )
                    self.phase(
                        tasks["ncri"], specification, "no_cot", "ncri", preflight
                    )
                print(
                    json.dumps({"model": specification["id"], "completed": True}),
                    flush=True,
                )
            self.current = {"completed": True}
        finally:
            self.stop.set()
            thread.join(timeout=15)
            self.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--baseline-cost", type=float)
    parser.add_argument("--uncertainty-reserve", type=float)
    args = parser.parse_args()
    previous_path = args.state_dir / "protocol.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if args.baseline_cost is None:
        args.baseline_cost = previous.get("baseline_cost", 0)
    if args.uncertainty_reserve is None:
        args.uncertainty_reserve = previous.get("uncertainty_reserve", 0)
    config = json.loads(args.config.read_text())
    args.state_dir.mkdir(parents=True, exist_ok=True)
    (args.state_dir / "protocol.json").write_text(
        json.dumps(
            {
                "config": config,
                "baseline_cost": args.baseline_cost,
                "uncertainty_reserve": args.uncertainty_reserve,
            },
            indent=2,
        )
        + "\n"
    )
    budget = RequestBudget(
        args.state_dir,
        config["budget_cny"],
        args.baseline_cost,
        args.uncertainty_reserve,
    )
    Experiment(config, args.state_dir.resolve(), args.log_dir.resolve(), budget).run()


if __name__ == "__main__":
    main()
