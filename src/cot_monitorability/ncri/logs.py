"""Recompute NCRI headline metrics and pool recorded banks without model calls."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from inspect_ai.log import EvalMetric, read_eval_log, write_eval_log
from inspect_ai.model import ModelUsage
from inspect_ai.scorer import SampleScore

from cot_monitorability.ncri.components import (
    ncri_placement,
    ncri_primary_accuracy,
    ncri_valid_rate,
)


def observations(log):
    return [
        SampleScore(
            score=sample.scores.get("ncri_scorer", next(iter(sample.scores.values()))),
            sample_id=sample.id,
            sample_metadata=sample.metadata,
        )
        for sample in (log.samples or [])
        if not sample.error and sample.scores
    ]


def refresh_ncri_metrics(log):
    scores = observations(log)
    if not scores or log.results is None:
        return None
    placement = ncri_placement(scores)
    current = next(s for s in log.results.scores if s.name == "ncri_scorer")
    supplemental = dict((current.metadata or {}).get("supplemental_metrics", {}))
    supplemental["ncri_primary_accuracy"] = EvalMetric(
        name="cot_monitorability/ncri_primary_accuracy",
        value=ncri_primary_accuracy()(scores),
    ).model_dump(mode="json")
    supplemental.pop("ncri_placement_score", None)
    current.metrics = {
        "ncri_placement_score": EvalMetric(
            name="cot_monitorability/ncri_placement_score", value=placement["display"]
        ),
        "ncri_valid_rate": EvalMetric(
            name="cot_monitorability/ncri_valid_rate", value=ncri_valid_rate()(scores)
        ),
    }
    current.metadata = {
        **(current.metadata or {}),
        "supplemental_metrics": supplemental,
        "ncri_placement": placement,
        "metric_schema": "8-A",
    }
    log.results.metadata = {**(log.results.metadata or {}), "ncri_placement": placement}
    log.results.headline = None
    for definition in log.eval.scorers or []:
        if definition.name == "ncri_scorer" and definition.metrics:
            definition.metrics[0] = definition.metrics[0].model_copy(
                update={"name": "cot_monitorability/ncri_placement_score"}
            )
    return placement


def aggregate_ncri_logs(paths, destination):
    """Pool one completed bank run per model/protocol; preserve source observations."""
    sources = [
        (Path(path), read_eval_log(path, resolve_attachments=True)) for path in paths
    ]
    if not sources:
        raise ValueError("NCRI aggregation requires source logs")
    model = sources[0][1].eval.model
    protocol_keys = (
        "k_shot",
        "no_prefill",
        "no_delib_system_v2",
        "disable_reasoning",
        "effort",
    )
    defaults = {
        "k_shot": 10,
        "no_prefill": False,
        "no_delib_system_v2": False,
        "disable_reasoning": True,
        "effort": None,
    }
    protocol = {
        key: sources[0][1].eval.task_args.get(key, defaults[key])
        for key in protocol_keys
    }
    samples, banks, seen = [], [], set()
    for path, log in sources:
        if log.status != "success" or log.eval.model != model:
            raise ValueError("NCRI aggregation requires completed runs of one model")
        if {
            key: log.eval.task_args.get(key, defaults[key]) for key in protocol_keys
        } != protocol:
            raise ValueError("NCRI source protocols differ")
        bank = log.eval.task_args.get("banks")
        if not bank or "," in bank or bank == "all_ncri" or bank in banks:
            raise ValueError("Select exactly one source run per NCRI bank")
        banks.append(bank)
        for sample in log.samples or []:
            score = next(iter(sample.scores.values())) if sample.scores else None
            if (
                sample.error
                or score is None
                or score.metadata.get("reasoning_tokens") != 0
                or score.metadata.get("hidden_channel_tokens") != 0
            ):
                raise ValueError(
                    "Cannot publish NCRI placement with missing or nonzero reasoning witnesses"
                )
            identity = (sample.id, sample.epoch)
            if identity in seen:
                raise ValueError("NCRI source observations overlap")
            seen.add(identity)
            samples.append(sample.model_copy(deep=True))
    merged = sources[0][1].model_copy(deep=True)
    merged.samples = sorted(samples, key=lambda sample: (str(sample.id), sample.epoch))
    merged.eval.task_id = uuid.uuid4().hex
    merged.eval.eval_id = uuid.uuid4().hex
    merged.eval.task_version = "8-A"
    merged.eval.task_args = {
        **protocol,
        "banks": "all_ncri",
        "limit_per_bank": max(
            len({sample.id for sample in log.samples}) for _, log in sources
        ),
    }
    merged.eval.dataset.name = "recorded-ncri-banks"
    merged.eval.dataset.samples = len({s.id for s in samples})
    merged.eval.dataset.sample_ids = list(dict.fromkeys(s.id for s in merged.samples))
    merged.eval.config.limit = None
    merged.eval.tags = list(
        dict.fromkeys([*(merged.eval.tags or []), "ncri-model-index"])
    )
    merged.eval.metadata = {
        **(merged.eval.metadata or {}),
        "aggregation": {
            "kind": "pooled_recorded_ncri_banks",
            "banks": banks,
            "new_model_requests": 0,
            "sources": [
                {
                    "file": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "task_version": log.eval.task_version,
                }
                for path, log in sources
            ],
        },
    }
    merged.results.total_samples = merged.results.completed_samples = (
        merged.results.logged_samples
    ) = len(samples)
    merged.results.scores[0].scored_samples = len(samples)
    merged.stats.started_at = min(log.stats.started_at for _, log in sources)
    merged.stats.completed_at = max(log.stats.completed_at for _, log in sources)
    merged.stats.model_usage = {}
    merged.stats.role_usage = {}
    for _, log in sources:
        for name, usage in log.stats.model_usage.items():
            merged.stats.model_usage[name] = (
                merged.stats.model_usage.get(name, ModelUsage()) + usage
            )
    merged.results.metadata = {
        "run_cost": {
            "currency": "CNY",
            "amount": sum(
                (log.results.metadata or {}).get("run_cost", {}).get("amount", 0)
                for _, log in sources
            ),
            "incremental_cost": 0,
            "basis": "costs of the source runs; pooling makes no model requests",
        }
    }
    placement = refresh_ncri_metrics(merged)
    write_eval_log(merged, destination)
    return placement
