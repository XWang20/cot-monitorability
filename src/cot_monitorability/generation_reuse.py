"""Replay complete generations from public Inspect logs before generating again."""

from __future__ import annotations

import hashlib
import json
import uuid
from contextvars import ContextVar

from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelUsage,
)
from inspect_ai.solver import solver

from cot_monitorability.reasoning import full_reasoning

ACTIVE_REUSE = ContextVar("experiment_generation_reuse", default=None)


def generation_key(model, effort, input_value, target, metadata):
    if isinstance(input_value, list):
        input_value = [
            m.model_dump(mode="json", exclude={"id"})
            if hasattr(m, "model_dump")
            else {k: v for k, v in m.items() if k != "id"}
            for m in input_value
        ]
    model = model.removeprefix("openai-api/openai/").removeprefix("openai/")
    if model.endswith("-nt") and effort in (None, "none"):
        model = model[:-3]
        effort = "none"
    payload = {
        "model": model,
        "effort": effort,
        "input": input_value,
        "target": target,
        "unperturbed_prompt": metadata.get("unperturbed_prompt"),
        "ncri_messages": metadata.get("ncri_messages"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class GenerationReuse:
    def __init__(self):
        self.records = {}

    def load(self, directories):
        for directory in directories:
            for path in sorted(directory.rglob("*.eval")):
                log = read_eval_log(path, resolve_attachments=True)
                for sample in log.samples or []:
                    output = sample.output
                    no_cot = log.eval.task.endswith("ncri_bench")
                    if (
                        not output
                        or not output.choices
                        or (output.stop_reason != "stop" and not no_cot)
                    ):
                        continue
                    if no_cot:
                        verdict = next(iter(sample.scores.values()), None)
                        if (
                            not output.usage
                            or output.usage.reasoning_tokens != 0
                            or verdict is None
                            or verdict.metadata.get("hidden_channel_tokens") != 0
                        ):
                            continue
                    try:
                        thought = full_reasoning(output.message)
                        if (
                            output.usage
                            and output.usage.reasoning_tokens
                            and not thought
                        ):
                            continue
                        wanted = [output.message.model_dump(mode="json")]
                        if "unperturbed_message_dump" in sample.metadata:
                            unperturbed = ChatMessageAssistant.model_validate(
                                sample.metadata["unperturbed_message_dump"]
                            )
                            unperturbed_thought = full_reasoning(unperturbed)
                            if (
                                sample.metadata.get("unperturbed_rtok")
                                and not unperturbed_thought
                            ):
                                continue
                            if sample.metadata.get("unperturbed_stop_reason") != "stop":
                                continue
                            wanted.append(unperturbed.model_dump(mode="json"))
                    except ValueError:
                        continue
                    events = [
                        e
                        for e in sample.events
                        if e.event == "model"
                        and e.role is None
                        and e.output
                        and e.output.choices
                        and (e.output.stop_reason == "stop" or no_cot)
                        and e.output.message.model_dump(mode="json") in wanted
                    ]
                    if len(events) < len(wanted):
                        continue
                    key = generation_key(
                        log.eval.model,
                        "none"
                        if no_cot
                        else log.eval.model_generate_config.reasoning_effort,
                        sample.input,
                        sample.target,
                        sample.metadata,
                    )
                    self.records[key] = {
                        "output": output,
                        "messages": sample.messages,
                        "metadata": sample.metadata,
                        "events": events,
                        "source_log": str(path),
                        "source_sample": sample.id,
                    }

    def restore(self, path):
        log = read_eval_log(path, resolve_attachments=True)
        changed = False
        for sample in log.samples or []:
            meta = sample.metadata
            if not meta.get("recorded_generation_key") or meta.get(
                "recorded_generation_events_restored"
            ):
                continue
            record = self.records[meta["recorded_generation_key"]]
            usage = ModelUsage()
            for original in record["events"]:
                sample.events.insert(
                    0,
                    original.model_copy(
                        deep=True,
                        update={
                            "uuid": str(uuid.uuid4()),
                            "span_id": None,
                            "input_refs": None,
                            "cache": "read",
                            "metadata": {
                                "source_log": record["source_log"],
                                "source_event_uuid": original.uuid,
                                "new_actor_request": False,
                            },
                        },
                    ),
                )
                if original.output.usage:
                    usage += original.output.usage
            usage.total_cost = None
            sample.model_usage[log.eval.model] = (
                sample.model_usage.get(log.eval.model, ModelUsage()) + usage
            )
            log.stats.model_usage[log.eval.model] = (
                log.stats.model_usage.get(log.eval.model, ModelUsage()) + usage
            )
            meta["recorded_generation_events_restored"] = True
            changed = True
        if changed:
            write_eval_log(log, path)


@solver
def reuse_completed_generation():
    async def solve(state, generate):
        from inspect_ai.model import get_model

        cache = ACTIVE_REUSE.get()
        if cache is None:
            return state
        model = get_model()
        key = generation_key(
            str(model),
            model.config.reasoning_effort,
            state.input,
            state.target.text,
            state.metadata,
        )
        record = cache.records.get(key)
        if record:
            state.output = record["output"].model_copy(deep=True)
            classes = {
                "user": ChatMessageUser,
                "assistant": ChatMessageAssistant,
                "system": ChatMessageSystem,
            }
            state.messages = [
                classes[m.role].model_validate(m.model_dump(mode="json"))
                for m in record["messages"]
            ]
            for name in [
                "perturbed_message_dump",
                "unperturbed_message_dump",
                "perturbed_rtok",
                "unperturbed_rtok",
                "unperturbed_text",
                "unperturbed_stop_reason",
            ]:
                if name in record["metadata"]:
                    state.metadata[name] = record["metadata"][name]
            state.metadata["recorded_generation_key"] = key
            state.metadata["recorded_generation_source"] = {
                "log": record["source_log"],
                "sample_id": record["source_sample"],
            }
            state.completed = True
        return state

    return solve
