#!/usr/bin/env python3
"""Inspect AI adaptation for nocot-bench (NCRI — No-Chain-of-thought Reasoning Index).

Upstream repository: https://github.com/neelnanda-io/nocot-bench
Reuses `nocot.run`, `nocot.grade`, `nocot.witnesses`, and `nocot.place` directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentReasoning,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from cot_monitorability.assets import asset_path, ensure_asset
from cot_monitorability.reference.nocot import grade as G
from cot_monitorability.reference.nocot import place as P
from cot_monitorability.reference.nocot import run as R

REPO_ROOT = asset_path("ncri")


def _to_inspect_messages(raw_msgs: list[dict[str, str]]) -> list[Any]:
    out = []
    for m in raw_msgs:
        role = m.get("role", "user")
        content = str(m.get("content", ""))
        if role == "system":
            out.append(ChatMessageSystem(content=content))
        elif role == "assistant":
            out.append(ChatMessageAssistant(content=content))
        else:
            out.append(ChatMessageUser(content=content))
    return out


@solver
def ncri_no_cot_solver(
    no_prefill: bool = False,
    effort: Optional[str] = None,
    disable_reasoning: bool = True,
) -> Solver:
    """Execute the canonical NCRI no-CoT request (K-shot + assistant prefill + max_tokens=100)."""

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        meta = state.metadata or {}
        bank = str(meta.get("bank", "arithmetic"))
        raw_msgs = meta.get("ncri_messages") or []
        if raw_msgs:
            state.messages = _to_inspect_messages(raw_msgs)

        model = get_model()
        m_name = str(model)
        # Route deepseek-v4.1-flash to its -nt alias; for deepseek-v4.1-flash-1017 and kimi-k3.1-pro,
        # use thinking: {"type": "disabled"} with temperature=None to cleanly disable reasoning.
        if disable_reasoning and not effort and m_name.endswith("deepseek-v4.1-flash"):
            model = get_model(f"{m_name}-nt")

        max_tok = R.OFFICIAL_MAX_TOKENS.get(bank, R.MAX_TOKENS)
        extra_body: Dict[str, Any] = dict(model.config.extra_body or {})
        temp: Optional[float] = 0.0
        if effort:
            extra_body["reasoning"] = {"effort": effort}
        elif disable_reasoning:
            if (
                "kimi-k3.1-pro" in m_name.lower()
                or "deepseek-v4.1-flash-1017" in m_name.lower()
            ):
                extra_body["thinking"] = {"type": "disabled"}
                temp = None
            else:
                extra_body["reasoning"] = {"enabled": False}

        out = await model.generate(
            state.messages,
            config=GenerateConfig(
                temperature=temp,
                max_tokens=max_tok,
                reasoning_effort=effort,
                extra_body=extra_body if extra_body else None,
            ),
        )
        state.output = out
        state.messages.append(out.message)
        return state

    return solve


@metric
def ncri_placement_score() -> Metric:
    """Compute sealed NCRI 15.2 placement display score (`100 + 14.42695 * theta`) via `nocot.place`."""

    def compute(scores: list[SampleScore]) -> float:
        counts: Dict[str, List[int]] = {}
        annex = {(b, str(p)) for b, p in P.ANNEXED_ITEMS}
        for s in scores:
            meta = s.score.metadata or {}
            if not meta.get("scorable", False):
                continue
            bank = str(meta.get("domain", ""))
            pn = str(meta.get("problem_number", ""))
            if (bank, pn) in annex:
                continue
            rung = meta.get("rung")
            if not rung or rung not in P.RUNGS:
                continue
            k_n = counts.setdefault(str(rung), [0, 0])
            k_n[0] += 1 if meta.get("correct", False) else 0
            k_n[1] += 1
        if not counts:
            return 0.0
        try:
            res = P.place({r: (v[0], v[1]) for r, v in counts.items()})
            return float(res["display"])
        except BaseException:
            return 0.0

    return compute


@metric
def ncri_primary_accuracy() -> Metric:
    """Primary no-CoT accuracy (floor): reasoned rows are scored wrong."""

    def compute(scores: list[SampleScore]) -> float:
        vals = [
            float(s.score.value.get("primary_correct", 0.0))
            for s in scores
            if isinstance(s.score.value, dict)
            and s.score.metadata.get("scorable", False)
        ]
        return sum(vals) / len(vals) if vals else 0.0

    return compute


@metric
def ncri_valid_rate() -> Metric:
    """Fraction of scorable rows certified as clean no-CoT by all 3 witnesses (W1, W2, W3)."""

    def compute(scores: list[SampleScore]) -> float:
        vals = [
            float(s.score.value.get("valid_no_cot", 0.0))
            for s in scores
            if isinstance(s.score.value, dict)
            and s.score.metadata.get("scorable", False)
        ]
        return sum(vals) / len(vals) if vals else 0.0

    return compute


@scorer(metrics=[ncri_placement_score(), ncri_primary_accuracy(), ncri_valid_rate()])
def ncri_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        item = dict(meta.get("item") or {})
        bank = str(meta.get("bank", item.get("domain", "arithmetic")))

        msg = state.output.message if state.output else None
        usage_obj = state.output.usage if state.output else None

        reasoning_parts: List[str] = []
        if msg and isinstance(msg.content, list):
            for b in msg.content:
                if isinstance(b, ContentReasoning) and b.reasoning:
                    reasoning_parts.append(b.reasoning)
        reasoning_text = "\n".join(reasoning_parts).strip()
        raw_text = (msg.text or "") if msg else ""

        # Inspect reports cached input separately from uncached input.
        prompt_tok = (
            sum(
                int(getattr(usage_obj, field, 0) or 0)
                for field in (
                    "input_tokens",
                    "input_tokens_cache_read",
                    "input_tokens_cache_write",
                )
            )
            if usage_obj
            else 0
        )
        comp_tok = int(getattr(usage_obj, "output_tokens", 0) or 0) if usage_obj else 0
        tot_tok = (
            int(getattr(usage_obj, "total_tokens", 0) or (prompt_tok + comp_tok))
            if usage_obj
            else 0
        )
        rtok_raw = getattr(usage_obj, "reasoning_tokens", None) if usage_obj else None
        rtok_val = int(rtok_raw) if rtok_raw is not None else 0

        usage_dict = {
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "total_tokens": tot_tok,
            "completion_tokens_details": {"reasoning_tokens": rtok_val},
        }

        raw_row = {
            "model": str(state.model),
            "domain": bank,
            "problem_number": item.get("problem_number"),
            "rung": item.get("rung"),
            "arm": "inspect",
            "provider": "inspect_proxy",
            "status": "ok" if msg else "api_error",
            "finish_reason": getattr(state.output, "stop_reason", "stop")
            if state.output
            else "error",
            "raw_text": raw_text,
            "reasoning_text": reasoning_text,
            "usage": usage_dict,
            "rtok_field_present": True,
            "reasoning_tokens": rtok_val,
            "hidden_channel_tokens": tot_tok - (prompt_tok + comp_tok),
        }

        graded = G.grade_row(raw_row, item)
        primary_correct = 1.0 if graded.get("correct") else 0.0
        valid_no_cot = 1.0 if graded.get("valid") else 0.0

        return Score(
            value={
                "primary_correct": primary_correct,
                "valid_no_cot": valid_no_cot,
                "reasoning_tokens": float(rtok_val),
            },
            answer=str(
                graded.get("predicted")
                if graded.get("predicted") is not None
                else raw_text[:80]
            ),
            explanation=(
                f"[{bank} #{item.get('problem_number')} rung={item.get('rung')}] "
                f"correct={graded.get('correct')} | valid_no_cot={graded.get('valid')} "
                f"(reason={graded.get('reason')}, rtok={rtok_val}) | "
                f"pred={graded.get('predicted')!r} vs gold={graded.get('correct_answer')!r}"
            ),
            metadata=graded,
        )

    return score


def build_ncri_dataset(
    banks: str = "arithmetic",
    k_shot: int = R.DEFAULT_K_SHOT,
    no_prefill: bool = False,
    no_delib_system_v2: bool = False,
    limit_per_bank: Optional[int] = None,
) -> MemoryDataset:
    ensure_asset("ncri")
    manifest = G.bank_manifest().get("ncri", {})
    if banks == "all_ncri":
        # Exclude gpqa if not fetched locally
        bank_list = [
            b
            for b in sorted(manifest.keys())
            if (REPO_ROOT / "data" / manifest[b]["file"]).exists()
        ]
    else:
        bank_list = [b.strip() for b in banks.split(",") if b.strip()]

    no_delib_text = R.NO_DELIB_SYSTEM_TEXT_V2 if no_delib_system_v2 else None
    samples: List[Sample] = []

    for b_name in bank_list:
        shots, eval_items = G.load_bank(b_name)
        selected_shots = shots[:k_shot]
        if limit_per_bank is not None:
            eval_items = eval_items[:limit_per_bank]
        for it in eval_items:
            msgs = R.build_messages(
                it,
                selected_shots,
                use_prefill=not no_prefill,
                no_delib=no_delib_text,
            )
            gold = (
                it.get("target") if it.get("target") is not None else it.get("answer")
            )
            samples.append(
                Sample(
                    id=f"{b_name}__{it.get('problem_number')}",
                    input=R.user_message(it),
                    target=str(gold),
                    metadata={
                        "bank": b_name,
                        "rung": it.get("rung"),
                        "problem_number": it.get("problem_number"),
                        "item": it,
                        "ncri_messages": msgs,
                    },
                )
            )
    return MemoryDataset(samples)


__all__ = ["R", "build_ncri_dataset", "ncri_no_cot_solver", "ncri_scorer"]
