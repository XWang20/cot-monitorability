# CoT Monitorability — Inspect Tasks

本仓库提供可安装的独立 Inspect 评测包，包含 6 类思维链可监控性任务。任务源码直接位于 `src/cot_monitorability/`，每个任务有 `eval.yaml`、README 和 `1-A` 版本。

**历史结果入口：[eval_logs/](eval_logs/README.md)**。14 份原始 `.eval` 保留在服务器，GitHub 保存索引和 SHA-256。这些日志对应原本地集成，不作为新包 `1-A` 的全量模型报告。

## 安装与运行

```bash
uv sync --locked --group dev --group test

# 包入口
uv run inspect eval cot_monitorability/cot_controllability_qa --model openai/<模型> --limit 10

# 与 Register Source URL 对应的源码入口
uv run inspect eval src/cot_monitorability/cot_control/cot_control.py@cot_controllability_qa --model openai/<模型> --limit 10
```

第一次构造任务时，数据资产按固定 commit 和 SHA-256 下载到缓存。可在导入任务前设置绝对路径 `COT_MONITORABILITY_CACHE_DIR`；预取后设置 `COT_MONITORABILITY_OFFLINE=1` 可禁止下载。运行不要求存在服务器原来的 `repos/` 或修改版 `inspect_evals` clone。

| 任务 | 源码目录 | 默认 effort | 默认样本数 |
| --- | --- | --- | --- |
| `cot_controllability_qa` | [cot_control](src/cot_monitorability/cot_control) | high | 427 |
| `openai_monitorability_process` | [openai_process](src/cot_monitorability/openai_process) | medium | 52 |
| `openai_monitorability_intervention` | [openai_intervention](src/cot_monitorability/openai_intervention) | medium | 150 |
| `cot_immediacy_2510` | [immediacy](src/cot_monitorability/immediacy) | high | 445 |
| `ncri_bench` | [ncri](src/cot_monitorability/ncri) | 无 CoT | 83 |
| `monitorbench` | [monitorbench](src/cot_monitorability/monitorbench) | high | 108 |

样本数对应任务默认参数；切换模式、数据集或子任务会改变数量。Judge/Monitor 使用 Inspect Model Roles，可通过 `--model-role judge=...` 或 `--model-role monitor=...` 覆盖。

## 本机代理与结果

服务器启动脚本默认使用 `http://127.0.0.1:8000/v1`，将代理兼容参数留在运行配置中。

```bash
./run_inspect_bench.sh 1 openai/deepseek-v4.1-flash-1017 10 5 auto
./run_inspect_bench.sh 1 openai/kimi-k3.1-pro 10 5 high
./run_inspect_bench.sh 1 openai/deepseek-v4.1-flash-1017 10 5 sweep
./run_inspect_view.sh
python3 scripts/index_eval_logs.py
```

## 检查与登记格式

```bash
uv run ruff check src/cot_monitorability tests scripts/prepare_registration.py
uv run ruff format --check src/cot_monitorability tests scripts/prepare_registration.py
uv run pytest tests
uv build

# 从已提交的源码生成本地登记格式，不发送 issue、不调用 Register
uv run python scripts/prepare_registration.py
```

生成的 `register/<id>/eval.yaml` 指向本仓库固定 40 字符 commit 的任务源码。格式说明与后续发布条件见 [上游提交准备](docs/UPSTREAM_CONTRIBUTION.md)。NCRI 已记录用户提供的 [LessWrong 原文](https://www.lesswrong.com/posts/eRmzz8J8Qkzqvzrgg/astra-can-do-a-concerning-amount-with-no-chain-of-thought)，arXiv 项保持待填。

`dependencies.lock.json`、`patches/` 和依赖 clone 仍保留为旧服务器环境的复现记录；它们不再是安装新评测包的前提。来源和许可证见 [docs/SOURCES.md](docs/SOURCES.md)、`NOTICE` 与 `licenses/`。

旧 patch 的迁移去向、保留原因及尚未确认的兼容性见 [迁移对应表](docs/MIGRATION_MAP.md)。
