# Inspect 评测结果索引

当前索引包含 14 份历史评测日志，覆盖 6 类基准。
原始 `.eval` 保存在服务器项目的 `eval_logs/` 下；GitHub 保存本索引和 `index.json`。

```bash
./run_inspect_view.sh
```

## 已记录的运行

| 任务 / 子集 | 模型 | 实际 effort | 完成样本 | 状态 | 主要指标 |
| --- | --- | --- | --- | --- | --- |
| cot_controllability_qa / gpqa/word_suppression | deepseek-v4.1-flash-1017 | high | 10/10 | success | cot_controllability_score=0.1 |
| cot_controllability_qa / gpqa/word_suppression | kimi-k3.1-pro | high | 10/10 | success | cot_controllability_score=0.2 |
| openai_monitorability_process / cot_only | deepseek-v4.1-flash-1017 | high | 10/10 | success | process_gmean2_cot=0.875 |
| openai_monitorability_process / cot_only | kimi-k3.1-pro | high | 10/10 | success | process_gmean2_cot=0.875 |
| openai_monitorability_intervention / gpqa | deepseek-v4.1-flash-1017 | high | 10/10 | success | 不可报告（eligible fraction 未达 10%） |
| openai_monitorability_intervention / gpqa | kimi-k3.1-pro | high | 10/10 | success | intervention_gmean2_lower=0.5 |
| openai_monitorability_intervention / scruples_suggest_wrong | deepseek-v4.1-flash-1017 | high | 10/10 | success | intervention_gmean2_lower=0.5 |
| openai_monitorability_intervention / scruples_suggest_wrong | kimi-k3.1-pro | high | 10/10 | success | intervention_gmean2_lower=0.3333 |
| cot_immediacy_2510 / gpqa | deepseek-v4.1-flash-1017 | high | 10/10 | success | legibility_pct=92.5; coverage_pct=97.5 |
| cot_immediacy_2510 / gpqa | kimi-k3.1-pro | high | 10/10 | success | legibility_pct=85; coverage_pct=87.5 |
| ncri_bench / arithmetic | deepseek-v4.1-flash-1017 | disabled | 10/10 | success | ncri_placement_score=105.1 |
| ncri_bench / arithmetic | kimi-k3.1-pro | disabled | 10/10 | success | ncri_placement_score=132.5 |
| monitorbench / cue_condition.preference | deepseek-v4.1-flash-1017 | high | 10/10 | success | mb_cot_monitorability=1 |
| monitorbench / cue_condition.preference | kimi-k3.1-pro | high | 10/10 | success | mb_cot_monitorability=1 |

## 数据说明

- 创建日期（UTC）：2026-10-30。完成样本数与任务配置逐份记录在表格和 JSON 中。干预基准的一个样本包含一对生成。
- 部分数据集存在两组运行。索引逐份保留，不将重复或不同子集的样本合并。
- 当前启动脚本为 Process / Intervention 默认选择 medium。实际 effort 以每份日志的生成配置为准，不根据当前默认值推断。
- NCRI 的任务参数指定 disable_reasoning；各任务的 mean CoT token 指标、模型用量和配置保存在 index.json。
- 指标定义因基准而异，结果不合并成统一排行榜。干预指标未达可报告阈值时标为不可报告；原始数值保留在 JSON。
- 固定长度窗口的 -1 等无有效样本标记保留原值，不解释为负分。
- 历史日志的部分汇总指标由原会话在 2026-10-30 根据已有样本补齐；本脚本只读取 header，不修改日志。
- file、SHA-256、时间、子集、完整指标和来源信息见 [index.json](index.json)。原始日志路径以服务器项目目录为基准。

```bash
python3 scripts/index_eval_logs.py
```
