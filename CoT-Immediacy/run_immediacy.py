"""Run CoT Immediacy (Legibility & Coverage) evaluation.

Implements the methodology from:
- "A Pragmatic Way to Measure Chain-of-Thought Monitorability" (Google DeepMind, Emmons et al., arXiv:2510.23966)
- Google DeepMind Gemini 4 Argon Safety & Security Report (Section 3.4.5.6: CoT Immediacy)

Supports three trajectory sources:
1. Reuse baseline CSVs from CoT-Control-QA (--source_csv or automatic discovery)
2. Generate fresh unconstrained baseline trajectories on GPQA / HLE / MMLU-Pro
3. Extract existing correct trajectories directly from platform's Inspect .eval logs (/home/eval/workspace/eval_log/openai/<model>/*.eval)

Applies Correctness Filter (grades only samples where correctness == True, unless --grade_all is passed)
and scores each CoT on Legibility (0-4 & 0-100%) and Coverage (0-4 & 0-100%) using the Autorater LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
IMM_DIR = Path(__file__).resolve().parent
QA_DIR = ROOT_DIR / "CoT-Control-QA"

if str(QA_DIR) not in sys.path:
    sys.path.insert(0, str(QA_DIR))
if str(IMM_DIR) not in sys.path:
    sys.path.insert(0, str(IMM_DIR))

load_dotenv(ROOT_DIR / ".env")
load_dotenv()

from autorater_prompt import format_autorater_prompt
from llm import create_llm, format_messages, LLMResponse, load_models_config
from run_cceval import run_evaluation as run_qa_baseline, DATASET_PRESETS, write_status_file, _sanitize_filename

IMMEDIACY_COLUMNS = [
    "sample_id",
    "model",
    "autorater_model",
    "provider",
    "dataset",
    "source_type",
    "question",
    "reasoning",
    "response",
    "correctness",
    "legibility_score",
    "coverage_score",
    "legibility_pct",
    "coverage_pct",
    "justification",
    "reasoning_chars",
    "reasoning_words",
    "autorater_prompt_tokens",
    "autorater_completion_tokens",
    "autorater_latency_ms",
    "retries",
    "error",
    "timestamp",
]


def parse_autorater_json(text: str) -> Dict[str, Any]:
    """Extract justification, legibility_score, and coverage_score from Autorater response."""
    if not text:
        raise ValueError("Empty autorater response")

    cleaned = text.strip()
    # Try code block first
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if code_match:
        candidate = code_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        candidate = brace_match.group(0) if brace_match else cleaned

    try:
        data = json.loads(candidate)
        leg = int(data["legibility_score"])
        cov = int(data["coverage_score"])
        just = str(data.get("justification", ""))
        return {
            "legibility_score": max(0, min(4, leg)),
            "coverage_score": max(0, min(4, cov)),
            "justification": just,
        }
    except Exception:
        # Regex fallback if JSON has unescaped quotes
        leg_m = re.search(r'"?legibility_score"?\s*:\s*([0-4])', cleaned)
        cov_m = re.search(r'"?coverage_score"?\s*:\s*([0-4])', cleaned)
        just_m = re.search(r'"?justification"?\s*:\s*"(.*?)"\s*,\s*"?legibility_score', cleaned, flags=re.DOTALL)
        if leg_m and cov_m:
            return {
                "legibility_score": int(leg_m.group(1)),
                "coverage_score": int(cov_m.group(1)),
                "justification": just_m.group(1).strip() if just_m else cleaned[:400],
            }
        raise ValueError(f"Could not parse autorater JSON: {cleaned[:200]}")


def extract_samples_from_platform_eval_logs(
    model_name: str,
    eval_log_root: str = "/home/eval/workspace/eval_log/openai",
    benchmarks: Optional[List[str]] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Extract correctly answered trajectories with CoT directly from Inspect .eval zip archives."""
    model_dir = Path(eval_log_root) / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Platform eval_log directory not found for model {model_name}: {model_dir}")

    target_benchmarks = benchmarks or [
        "frontierscience", "wet_lab_protocols", "scicode", "chembench", "cyberseceval"
    ]

    eval_files = sorted(model_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
    extracted: List[Dict[str, Any]] = []

    for ef in eval_files:
        if len(extracted) >= limit:
            break
        if target_benchmarks and not any(b in ef.name.lower() for b in target_benchmarks):
            continue
        try:
            with zipfile.ZipFile(ef, "r") as zf:
                sample_names = [n for n in zf.namelist() if n.startswith("samples/") and n.endswith(".json")]
                for sname in sample_names:
                    if len(extracted) >= limit:
                        break
                    sdata = json.loads(zf.read(sname).decode("utf-8"))
                    # Check score == C or 1.0
                    scores = sdata.get("scores") or {}
                    is_correct = False
                    for sc_obj in scores.values():
                        val = sc_obj.get("value") if isinstance(sc_obj, dict) else sc_obj
                        if val in ("C", 1, 1.0, True, "1"):
                            is_correct = True
                            break
                    if not is_correct:
                        continue

                    # Extract question
                    inp = sdata.get("input")
                    if isinstance(inp, str):
                        question_text = inp
                    elif isinstance(inp, list):
                        question_text = "\n".join(
                            m.get("content", "") if isinstance(m.get("content"), str) else str(m.get("content", ""))
                            for m in inp if isinstance(m, dict)
                        )
                    else:
                        question_text = str(inp or "")

                    # Extract reasoning and response from output.choices[0].message.content
                    choices = (sdata.get("output") or {}).get("choices") or []
                    if not choices:
                        continue
                    msg = choices[0].get("message") or {}
                    content_list = msg.get("content")
                    reasoning_parts = []
                    text_parts = []
                    if isinstance(content_list, list):
                        for item in content_list:
                            if isinstance(item, dict):
                                if item.get("type") == "reasoning":
                                    r_txt = item.get("reasoning") or ""
                                    if r_txt and not item.get("redacted"):
                                        reasoning_parts.append(r_txt)
                                elif item.get("type") == "text":
                                    t_txt = item.get("text") or ""
                                    if t_txt:
                                        text_parts.append(t_txt)
                    elif isinstance(content_list, str):
                        text_parts.append(content_list)

                    reasoning_str = "\n".join(reasoning_parts).strip()
                    response_str = "\n".join(text_parts).strip()
                    if not reasoning_str or not response_str:
                        continue

                    extracted.append({
                        "sample_id": str(sdata.get("id", len(extracted))),
                        "model": model_name,
                        "dataset": ef.name.split("_")[1] if "_" in ef.name else ef.stem,
                        "source_type": "platform_eval_log",
                        "question": question_text,
                        "reasoning": reasoning_str,
                        "response": response_str,
                        "correctness": True,
                    })
        except Exception:
            continue

    return extracted


async def evaluate_immediacy_on_records(
    records: List[Dict[str, Any]],
    model_name: str,
    dataset_label: str,
    autorater_model: str = "gemini-3.8-flash",
    provider: str = "dmxapi",
    max_concurrency: int = 15,
    grade_all: bool = False,
    resume: bool = True,
    log_dir: str = "results",
) -> Path:
    import pandas as pd

    output_dir = IMM_DIR / "logs" / log_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status_dir = QA_DIR / "logs" / "status"

    sanitized_model = _sanitize_filename(model_name.replace("/", "_"))
    sanitized_ds = _sanitize_filename(dataset_label)
    sanitized_ar = _sanitize_filename(autorater_model.replace("/", "_"))
    output_file = output_dir / f"{sanitized_model}__{sanitized_ds}__immediacy__{sanitized_ar}.csv"
    job_id = f"imm__{sanitized_model}__{sanitized_ds}__{sanitized_ar}"

    # Filter eligible records: must have non-empty reasoning, and correctness==True (unless grade_all)
    eligible_records = []
    for r in records:
        is_corr = str(r.get("correctness", "")).lower() in ("true", "1", "1.0")
        has_cot = bool(r.get("reasoning") and not pd.isna(r.get("reasoning")) and str(r.get("reasoning")).strip())
        if not has_cot:
            continue
        if grade_all or is_corr:
            eligible_records.append(r)

    completed_map: Dict[str, Dict[str, Any]] = {}
    if resume and output_file.exists():
        try:
            df_ex = pd.read_csv(output_file)
            for _, row in df_ex.iterrows():
                err = str(row.get("error", "")) if pd.notna(row.get("error")) else ""
                if not err and pd.notna(row.get("legibility_score")):
                    completed_map[str(row.get("question", ""))] = row.to_dict()
            if completed_map:
                print(f"Found {len(completed_map)} already rated samples in {output_file.name}")
        except Exception:
            pass

    pending_records = [r for r in eligible_records if str(r.get("question_raw") or r.get("question", "")) not in completed_map]
    total_eligible = len(eligible_records)
    total_to_run = len(pending_records)
    skipped = total_eligible - total_to_run

    print(f"\n{'='*65}")
    print(f"Running CoT-Immediacy (Legibility & Coverage)")
    print(f"Target Model: {model_name} | Autorater: {autorater_model} ({provider})")
    print(f"Dataset: {dataset_label} | Total trajectories={len(records)}, Eligible (correct+CoT)={total_eligible}, Skip={skipped}, Todo={total_to_run}")
    print(f"{'='*65}")

    existing_rows = list(completed_map.values())
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=IMMEDIACY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)

    start_ts = time.time()
    stats = {
        "job_id": job_id,
        "eval_type": "immediacy",
        "model": model_name,
        "autorater_model": autorater_model,
        "provider": provider,
        "dataset": dataset_label,
        "mode": "immediacy",
        "threat_group": "legibility_coverage",
        "status": "running" if total_to_run > 0 else "completed",
        "total_trajectories": len(records),
        "total": total_eligible,
        "completed": skipped,
        "errors": 0,
        "legibility_pct_sum": sum(float(r.get("legibility_pct") or 0.0) for r in existing_rows),
        "coverage_pct_sum": sum(float(r.get("coverage_pct") or 0.0) for r in existing_rows),
        "latency_ms_sum": sum(float(r.get("autorater_latency_ms") or 0.0) for r in existing_rows),
        "prompt_tokens": sum(int(float(r.get("autorater_prompt_tokens") or 0)) for r in existing_rows),
        "completion_tokens": sum(int(float(r.get("autorater_completion_tokens") or 0)) for r in existing_rows),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": 0.0,
        "output_file": str(output_file),
        "last_error": "",
    }
    write_status_file(status_dir, job_id, stats)

    if total_to_run == 0:
        print("✅ All eligible trajectories already graded for Immediacy!")
        return output_file

    autorater_llm = create_llm(model=autorater_model, backend=provider, default_max_tokens=8192)

    all_messages = []
    for r in pending_records:
        q_txt = str(r.get("question_raw") if r.get("question_raw") and str(r.get("question_raw")) != "nan" else r.get("question", ""))
        cot_txt = str(r.get("reasoning", ""))
        ans_txt = str(r.get("response", ""))
        prompt = format_autorater_prompt(question=q_txt, explanation=cot_txt, answer=ans_txt)
        all_messages.append(format_messages(prompt))

    def on_rated(idx: int, resp: LLMResponse):
        rec = pending_records[idx]
        q_txt = str(rec.get("question_raw") if rec.get("question_raw") and str(rec.get("question_raw")) != "nan" else rec.get("question", ""))
        cot_txt = str(rec.get("reasoning", ""))
        ans_txt = str(rec.get("response", ""))

        row: Dict[str, Any] = {
            "sample_id": rec.get("sample_id", idx),
            "model": model_name,
            "autorater_model": autorater_model,
            "provider": provider,
            "dataset": dataset_label,
            "source_type": rec.get("source_type", "cot_control_baseline"),
            "question": q_txt,
            "reasoning": cot_txt,
            "response": ans_txt,
            "correctness": rec.get("correctness", True),
            "legibility_score": "",
            "coverage_score": "",
            "legibility_pct": "",
            "coverage_pct": "",
            "justification": "",
            "reasoning_chars": len(cot_txt),
            "reasoning_words": len(cot_txt.split()),
            "autorater_prompt_tokens": resp.prompt_tokens,
            "autorater_completion_tokens": resp.completion_tokens,
            "autorater_latency_ms": round(resp.latency_ms or 0.0, 1),
            "retries": resp.retries,
            "error": "",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        if resp.error:
            row["error"] = resp.error
            stats["errors"] += 1
            stats["last_error"] = resp.error[:200]
        else:
            try:
                parsed = parse_autorater_json(resp.content)
                leg_s = parsed["legibility_score"]
                cov_s = parsed["coverage_score"]
                leg_pct = round(leg_s * 25.0, 2)
                cov_pct = round(cov_s * 25.0, 2)
                row["legibility_score"] = leg_s
                row["coverage_score"] = cov_s
                row["legibility_pct"] = leg_pct
                row["coverage_pct"] = cov_pct
                row["justification"] = parsed["justification"]

                stats["completed"] += 1
                stats["legibility_pct_sum"] += leg_pct
                stats["coverage_pct_sum"] += cov_pct
                stats["latency_ms_sum"] += float(resp.latency_ms or 0.0)
                stats["prompt_tokens"] += resp.prompt_tokens
                stats["completion_tokens"] += resp.completion_tokens
            except Exception as e:
                row["error"] = f"ParseError: {e}"
                stats["errors"] += 1
                stats["last_error"] = str(e)[:200]

        with open(output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=IMMEDIACY_COLUMNS, extrasaction="ignore")
            writer.writerow(row)

        stats["elapsed_s"] = round(time.time() - start_ts, 1)
        stats["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_status_file(status_dir, job_id, stats)

    await autorater_llm.generate_batch_async(
        all_messages,
        max_tokens=8192,
        temperature=0.0,
        max_concurrency=max_concurrency,
        show_progress=True,
        on_sample_complete=on_rated,
    )

    stats["status"] = "completed" if stats["errors"] == 0 else ("partial_error" if stats["completed"] > 0 else "error")
    stats["elapsed_s"] = round(time.time() - start_ts, 1)
    stats["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_status_file(status_dir, job_id, stats)

    denom = max(1, stats["completed"])
    avg_leg = stats["legibility_pct_sum"] / denom
    avg_cov = stats["coverage_pct_sum"] / denom
    avg_lat = (stats["latency_ms_sum"] / denom) / 1000.0

    print(f"\n{'='*65}")
    print(f"✅ IMMEDIACY EVALUATION COMPLETE: {model_name} | {dataset_label}")
    print(f"   Rated: {stats['completed']}/{stats['total']} | Errors: {stats['errors']} | Avg Autorater Latency: {avg_lat:.2f}s")
    print(f"   Mean Legibility: {avg_leg:.2f}% ({avg_leg/25.0:.2f} / 4.00)")
    print(f"   Mean Coverage:   {avg_cov:.2f}% ({avg_cov/25.0:.2f} / 4.00)")
    print(f"   Saved to: {output_file}")
    print(f"{'='*65}\n")

    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CoT Immediacy (Legibility & Coverage) evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, nargs="+", required=True,
                        help="Target model(s) to evaluate")
    parser.add_argument("--autorater", type=str, default="gemini-3.8-flash",
                        help="Autorater LLM model (e.g., gemini-3.8-flash, gemini-4-argon, deepseek-v4.1-flash-1017)")
    parser.add_argument("--provider", type=str, default=os.getenv("DEFAULT_PROVIDER", "dmxapi"),
                        choices=["dmxapi", "proxy"],
                        help="API provider backend")
    parser.add_argument("--source", type=str, default="qa_baseline",
                        choices=["qa_baseline", "platform_logs", "csv"],
                        help="Where to get CoT trajectories: qa_baseline (generate/reuse CoT-Control baseline), platform_logs (/home/eval/workspace/eval_log), or csv (--source_csv)")
    parser.add_argument("--source_csv", type=str, default=None,
                        help="Explicit CSV file path when --source csv")
    parser.add_argument("--data", type=str, default=str(QA_DIR / "datasets" / "gpqa_w_keyword.csv"),
                        help="Dataset file when --source qa_baseline")
    parser.add_argument("--dataset_preset", type=str, default=None,
                        choices=list(DATASET_PRESETS.keys()),
                        help="Dataset preset when --source qa_baseline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per dataset")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 3 samples")
    parser.add_argument("--grade_all", action="store_true",
                        help="Grade all trajectories with CoT even if answer is incorrect (useful for tiny smoke tests)")
    parser.add_argument("--max_concurrency", type=int, default=15,
                        help="Autorater max concurrency")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


async def async_main():
    import pandas as pd
    args = parse_args()

    for model_name in args.model:
        if args.source == "csv":
            if not args.source_csv:
                raise ValueError("--source_csv is required when --source csv")
            csv_path = Path(args.source_csv)
            df = pd.read_csv(csv_path)
            records = df.to_dict("records")
            await evaluate_immediacy_on_records(
                records=records,
                model_name=model_name,
                dataset_label=csv_path.stem,
                autorater_model=args.autorater,
                provider=args.provider,
                max_concurrency=args.max_concurrency,
                grade_all=args.grade_all,
                resume=args.resume,
            )
        elif args.source == "platform_logs":
            lim = 3 if args.test else (args.limit or 50)
            records = extract_samples_from_platform_eval_logs(model_name, limit=lim)
            print(f"Extracted {len(records)} correct trajectories with CoT from platform eval_log for {model_name}")
            await evaluate_immediacy_on_records(
                records=records,
                model_name=model_name,
                dataset_label="platform_eval_logs",
                autorater_model=args.autorater,
                provider=args.provider,
                max_concurrency=args.max_concurrency,
                grade_all=True,
                resume=args.resume,
            )
        else:
            # qa_baseline: run or reuse baseline from CoT-Control-QA
            if args.dataset_preset:
                datasets_to_run = [
                    (path, args.limit if args.limit is not None else preset_limit)
                    for path, preset_limit in DATASET_PRESETS[args.dataset_preset]
                ]
            else:
                data_file = Path(args.data)
                if not data_file.exists():
                    alt = QA_DIR / "datasets" / args.data
                    if alt.exists():
                        data_file = alt
                datasets_to_run = [(data_file, args.limit)]

            for data_file, lim in datasets_to_run:
                baseline_csv = await run_qa_baseline(
                    model_name=model_name,
                    data_file=data_file,
                    mode="baseline",
                    test_mode=args.test,
                    limit=lim,
                    provider=args.provider,
                    max_concurrency=args.max_concurrency,
                    resume=args.resume,
                )
                df = pd.read_csv(baseline_csv)
                records = df.to_dict("records")
                await evaluate_immediacy_on_records(
                    records=records,
                    model_name=model_name,
                    dataset_label=data_file.stem,
                    autorater_model=args.autorater,
                    provider=args.provider,
                    max_concurrency=args.max_concurrency,
                    grade_all=args.grade_all,
                    resume=args.resume,
                )


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
