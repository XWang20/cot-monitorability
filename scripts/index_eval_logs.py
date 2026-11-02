#!/usr/bin/env python3
"""Create a deterministic results index from Inspect log headers."""

import hashlib
import json
import pathlib
import zipfile
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "eval_logs"


def read_run(path):
    with zipfile.ZipFile(path) as archive:
        header = json.loads(archive.read("header.json"))
    evaluation = header["eval"]
    results = header.get("results", {})
    stats = header.get("stats", {})
    started = stats.get("started_at")
    finished = stats.get("completed_at")
    elapsed = None
    if started and finished:
        elapsed = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    return {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "task": evaluation["task"],
        "model": evaluation["model"],
        "model_base_url": evaluation.get("model_base_url"),
        "status": header["status"],
        "created": evaluation.get("created"),
        "started_at": started,
        "completed_at": finished,
        "elapsed_seconds": elapsed,
        "reasoning_effort": evaluation.get("model_generate_config", {}).get("reasoning_effort"),
        "task_args": evaluation.get("task_args", {}),
        "eval_config": {key: evaluation.get("config", {}).get(key) for key in ("limit", "epochs")},
        "completed_samples": results.get("completed_samples"),
        "total_samples": results.get("total_samples"),
        "scores": [
            {
                "scorer": score["name"],
                "scored_samples": score.get("scored_samples"),
                "metrics": {key: item["value"] for key, item in score.get("metrics", {}).items()},
            }
            for score in results.get("scores", [])
        ],
        "model_usage": stats.get("model_usage", {}),
        "revision": evaluation.get("revision"),
        "packages": evaluation.get("packages", {}),
    }


def primary_result(run):
    metrics = {key: value for score in run["scores"] for key, value in score["metrics"].items()}
    if run["task"].endswith("openai_monitorability_intervention") and metrics.get("intervention_reportable") == 0:
        return "不可报告（eligible fraction 未达 10%）"
    names = (
        "cot_controllability_score", "process_gmean2_cot", "intervention_gmean2_lower",
        "legibility_pct", "coverage_pct", "ncri_placement_score", "mb_cot_monitorability",
    )
    return "; ".join(f"{name}={metrics[name]:.4g}" for name in names if name in metrics) or "未记录"


def render_markdown(runs):
    lines = [
        "# Inspect 评测结果索引", "",
        f"当前索引包含 {len(runs)} 份历史评测日志，覆盖 {len({run['task'] for run in runs})} 类基准。",
        "原始 `.eval` 保存在服务器项目的 `eval_logs/` 下；GitHub 保存本索引和 `index.json`。", "",
        "```bash", "./run_inspect_view.sh", "```", "",
        "## 已记录的运行", "",
        "| 任务 / 子集 | 模型 | 实际 effort | 完成样本 | 状态 | 主要指标 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        args = run["task_args"]
        subset = args.get("dataset") or args.get("banks") or args.get("subset") or args.get("scopes") or ""
        if args.get("modes"):
            subset += "/" + args["modes"]
        effort = run["reasoning_effort"] or ("disabled" if args.get("disable_reasoning") else "未记录")
        model = run["model"].removeprefix("openai/")
        task = run["task"].removeprefix("inspect_evals/")
        lines.append(f"| {task} / {subset} | {model} | {effort} | {run['completed_samples']}/{run['total_samples']} | {run['status']} | {primary_result(run)} |")
    lines += [
        "", "## 数据说明", "",
        "- 创建日期（UTC）：" + ", ".join(sorted({run["created"][:10] for run in runs if run["created"]})) + "。完成样本数与任务配置逐份记录在表格和 JSON 中。干预基准的一个样本包含一对生成。",
        "- 部分数据集存在两组运行。索引逐份保留，不将重复或不同子集的样本合并。",
        "- 当前启动脚本为 Process / Intervention 默认选择 medium。实际 effort 以每份日志的生成配置为准，不根据当前默认值推断。",
        "- NCRI 的任务参数指定 disable_reasoning；各任务的 mean CoT token 指标、模型用量和配置保存在 index.json。",
        "- 指标定义因基准而异，结果不合并成统一排行榜。干预指标未达可报告阈值时标为不可报告；原始数值保留在 JSON。",
        "- 固定长度窗口的 -1 等无有效样本标记保留原值，不解释为负分。",
        "- 历史日志的部分汇总指标由原会话在 2026-10-30 根据已有样本补齐；本脚本只读取 header，不修改日志。",
        "- file、SHA-256、时间、子集、完整指标和来源信息见 [index.json](index.json)。原始日志路径以服务器项目目录为基准。", "",
        "```bash", "python3 scripts/index_eval_logs.py", "```", "",
    ]
    return "\n".join(lines)


def main():
    runs = [read_run(path) for path in sorted(LOG_DIR.rglob("*.eval"))]
    if not runs:
        raise SystemExit("No local .eval files found; refusing to replace the published index with an empty index.")
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / "index.json").write_text(json.dumps({"schema_version": 1, "runs": runs}, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (LOG_DIR / "README.md").write_text(render_markdown(runs))
    print(f"Indexed {len(runs)} Inspect logs without changing the original files.")


if __name__ == "__main__":
    main()
