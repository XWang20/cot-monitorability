# CoT Monitorability — Inspect Benchmarks

本仓库将 6 类思维链可监控性基准统一接入 `inspect_evals`。当前工作重点是评测适配、推理 effort 配置和 `.eval` 结果检查。

**结果入口：[eval_logs/](eval_logs/README.md)**。原始日志位于服务器的同名目录，GitHub 保存结果索引、完整汇总指标和日志 SHA-256。现有结果为 DeepSeek V4.1 Flash-1017 与 Kimi K3.1 Pro 的 14 次冒烟测试，每次 10 个样本。

| 编号 | Inspect 任务 | 当前启动脚本默认 effort |
| --- | --- | --- |
| 1 | `inspect_evals/cot_controllability_qa` | high |
| 2 | `inspect_evals/openai_monitorability_process` | medium |
| 3 | `inspect_evals/openai_monitorability_intervention` | medium |
| 4 | `inspect_evals/cot_immediacy_2510` | high |
| 5 | `inspect_evals/ncri_bench` | disable_reasoning |
| 6 | `inspect_evals/monitorbench` | high |

## 运行与查看

被测模型和 Judge/Monitor 统一使用本机 OpenAI 兼容代理，默认 `http://127.0.0.1:8000/v1`。

```bash
# 单个基准：编号、模型、样本数、并发数、effort
./run_inspect_bench.sh 1 openai/deepseek-v4.1-flash-1017 10 5 auto
./run_inspect_bench.sh 1 openai/kimi-k3.1-pro 10 5 high

# 同一基准扫描 low / medium / high
./run_inspect_bench.sh 1 openai/deepseek-v4.1-flash-1017 10 5 sweep

# Inspect View 浏览原始日志（默认端口 5010）
./run_inspect_view.sh

# 根据本机原始日志更新 GitHub 结果索引
python3 scripts/index_eval_logs.py
```

首次设置环境请参阅 [基准使用说明](INSPECT_BENCHMARKS_README.md)。
历史日志按实际设置记录，部分运行的 effort 与当前默认值不同；具体配置见结果索引。

## 仓库内容

- `eval_logs/`：服务器原始 `.eval` 与仓库结果索引。
- `run_inspect_bench.sh`、`run_inspect_view.sh`：评测与日志浏览入口。
- `dependencies.lock.json`、`patches/`：7 个独立依赖的固定版本及本地改动。
- `inspect_ai/`、`inspect_evals/`、`repos/`：由依赖恢复脚本管理的独立源码 clone。
- [仓库管理](REPOSITORY_MANAGEMENT.md)、[来源与许可](docs/SOURCES.md)。

原始日志和运行环境保存在服务器。主仓库从自己的初始提交开始，基准源码在依赖 clone 中保留上游历史。

## Inspect Evals 登记准备

本仓库当前为本地评测集成。正式发布将按官方 Register 流程准备独立任务源码、可安装包、测试和两模型全样本日志。当前差距和任务规范见 [上游提交准备](docs/UPSTREAM_CONTRIBUTION.md)。
