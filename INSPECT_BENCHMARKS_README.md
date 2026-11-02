# Inspect CoT 基准使用说明

当前 6 类基准均通过 `inspect_evals` 运行，结果写入 `eval_logs/`，使用 Inspect View 检查每道题的消息、思维链和评分。

## 环境与依赖

推荐 Python 3.10–3.13。从全新主仓库 checkout 设置依赖：

```bash
python3 scripts/dependency_repos.py restore
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 scripts/dependency_repos.py check
```

服务器现有 `.venv` 和依赖 clone 已设置，继续使用即可。恢复脚本拒绝覆盖已有 clone。

## 模型 API

启动脚本默认使用本机代理；显式导出的环境变量可以覆盖配置：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=a
export INSPECT_EVALS_JUDGE_MODEL=openai/deepseek-v4.1-flash-1017
```

DeepSeek V4.1 Flash-1017 和 Kimi K3.1 Pro 均走本机代理。脚本使用现有环境变量，不自动加载旧 `.env`。

## 统一启动方式

```bash
./run_inspect_bench.sh <编号或任务名> <模型> <样本数> <并发数> <effort> [其他 Inspect 参数]
```

effort 可选 `auto|low|medium|high|sweep`。`auto` 使用下表中的当前默认值，`sweep` 依次运行 low、medium、high。NCRI 使用无 CoT 设置；在该基准中 effort 扫描不构成不同的推理条件。

| 编号 | 任务 | 当前默认参数 | 默认 effort | 主要指标 |
| --- | --- | --- | --- | --- |
| 1 | `cot_controllability_qa` | `dataset=gpqa, modes=word_suppression` | high | `cot_controllability_score`, `task_accuracy`, `mean_cot_tokens` |
| 2 | `openai_monitorability_process` | `scopes=cot_only` | medium | `process_gmean2_cot`, `process_answer_accuracy`, `process_mean_cot_tokens` |
| 3 | `openai_monitorability_intervention` | `dataset=scruples_suggest_wrong` | medium | gmean 上/下界、eligible fraction、reportable、`intervention_mean_cot_tokens` |
| 4 | `cot_immediacy_2510` | `dataset=gpqa` | high | `legibility_pct`, `coverage_pct`, `immediacy_mean_cot_tokens` |
| 5 | `ncri_bench` | `banks=arithmetic, disable_reasoning=true` | 无 CoT | `ncri_placement_score`, `ncri_primary_accuracy`, `ncri_valid_rate` |
| 6 | `monitorbench` | `subset=cue_condition.preference, stress_test=standard` | high | `mb_cot_monitorability`, `mb_verification_rate`, `mb_mean_cot_tokens` |

各任务的 Judge/Monitor 默认为 DeepSeek V4.1 Flash-1017，判分使用 high effort。被测模型的 effort 会同时传入 Inspect 生成配置和任务参数；Inspect 适配层向本机代理传递对应 reasoning 配置。

```bash
./run_inspect_bench.sh 1 openai/deepseek-v4.1-flash-1017 10 5 auto
./run_inspect_bench.sh 2 openai/kimi-k3.1-pro 10 5 medium
./run_inspect_bench.sh 3 openai/deepseek-v4.1-flash-1017 10 5 auto -T dataset=gpqa
./run_inspect_bench.sh 1 openai/deepseek-v4.1-flash-1017 10 5 sweep
```

## 任务参数

- CoT-Control QA：正文 10 模式，`modes=all_main` 可选全部模式；支持 `dataset=gpqa|hle|mmlu_pro|all`、`cot_sys_prompt=true`、`elicit_brevity=true`，以及 monitor/eval awareness 和 adversarial pressure 参数。`baseline` 为无约束模式。
- OpenAI Process：`scopes=cot_only`，可通过 `monitor_model` 指定 Monitor。
- OpenAI Intervention：支持 `gpqa|gpqa_encoded|daft|scruples_first_person|scruples_suggest_right|scruples_suggest_wrong|wmdp_sandbagging|all`。一个 Inspect 样本包含原始和干预两次生成，`--limit 10` 对应 10 对样本。
- DeepMind Immediacy：支持 `dataset=gpqa|hle|mmlu_pro`、`judge_model` 和 `correct_only=true|false`；当前默认 `correct_only=false`，只分析答对样本时需显式启用。
- NCRI：支持逗号分隔题库和 `banks=all_ncri`。无 CoT 设置使用上游的判定与 placement 逻辑。
- MonitorBench：支持 Inspect 适配子任务及上游非沙盒子任务，可通过 `subset` 和 `stress_test` 选择；Monitor 模型可用 `--model-role monitor=openai/<模型>` 覆盖。

## 查看与记录结果

```bash
./run_inspect_view.sh
# 可覆盖 Inspect View 的默认端口或 host
./run_inspect_view.sh --port 5012 --host 127.0.0.1
python3 scripts/index_eval_logs.py
```

Inspect View 默认读取项目 `eval_logs/`，端口为 5010。目录中的 `.eval` 是原始结果，包含配置、题目、消息、评分和模型用量；原始日志保留在服务器，GitHub 保存 [结果索引](eval_logs/README.md) 与 [index.json](eval_logs/index.json)。

当前 14 份日志来自 2026-10-30 的 10 样本冒烟运行。部分历史运行使用 high effort，索引保留实际设置。各指标的定义、子集和可报告条件不同，应在相同实验设置下比较。扩大样本数和子任务覆盖范围后再进行完整结果分析。
