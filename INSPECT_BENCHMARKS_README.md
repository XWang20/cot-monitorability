# 运行独立 Inspect 评测包

## 环境

使用 Python 3.11–3.13 和 uv：

```bash
uv sync --locked --group dev --group test
```

服务器为了保留原来的 `.venv` 和运行中的 Inspect View，将新包安装到 `.venv-package`。服务器上的 uv 命令应增加 `UV_PROJECT_ENVIRONMENT=.venv-package`；启动脚本会优先使用这个环境。新 checkout 默认使用 uv 创建的 `.venv`。

任务源码、参考函数与元数据随包安装。数据通过 `asset_specs.json` 中固定的源码归档加载，不依赖原来的 clone 路径。

```bash
# 预取所有固定资产
uv run python -m cot_monitorability.assets

# 已预取时可以离线构造和测试任务
export COT_MONITORABILITY_OFFLINE=1
```

缓存默认使用平台用户缓存目录。`COT_MONITORABILITY_CACHE_DIR` 必须是绝对路径，并在导入任务前设置。

## 模型与角色

标准包入口与源码入口都可以运行：

```bash
uv run inspect eval cot_monitorability/openai_monitorability_process --model openai/<被测模型> --model-role monitor=openai/<Monitor模型> --limit 10
uv run inspect eval src/cot_monitorability/immediacy/immediacy.py@cot_immediacy_2510 --model openai/<被测模型> --model-role judge=openai/<Judge模型> --limit 10
```

根据所选 provider 配置凭据与 URL。任务代码不指定 API 服务地址。本机部署使用：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=a
export INSPECT_EVALS_JUDGE_MODEL=openai/deepseek-v4.1-flash-1017
```

## 统一启动脚本

```bash
./run_inspect_bench.sh <编号或任务名> <模型> <样本数> <并发数> <auto|low|medium|high|sweep> [其他 Inspect 参数]
```

| 编号 | 任务 | 当前默认参数 | 被测模型默认 effort |
| --- | --- | --- | --- |
| 1 | `cot_controllability_qa` | GPQA / word_suppression | high |
| 2 | `openai_monitorability_process` | cot_only | medium |
| 3 | `openai_monitorability_intervention` | scruples_suggest_wrong | medium |
| 4 | `cot_immediacy_2510` | GPQA / correct_only=false | high |
| 5 | `ncri_bench` | arithmetic / disable_reasoning=true | 无 CoT |
| 6 | `monitorbench` | cue_condition.preference / standard | high |

Judge/Monitor 默认 DeepSeek V4.1 Flash-1017，使用 high effort。脚本通过公共 Inspect CLI 的 `--extra-body` 和角色配置适配本机代理；外部服务可设置 `COT_USE_LITELLM_COMPAT=0` 关闭这些部署参数。

```bash
./run_inspect_bench.sh 1 openai/deepseek-v4.1-flash-1017 10 5 auto
./run_inspect_bench.sh 2 openai/kimi-k3.1-pro 10 5 medium
./run_inspect_bench.sh 3 openai/deepseek-v4.1-flash-1017 10 5 auto -T dataset=gpqa
./run_inspect_bench.sh 1 openai/deepseek-v4.1-flash-1017 10 5 sweep
```

NCRI 保持无 CoT 设置，全局 effort 参数不会改变这个任务的约束。其他任务参数和实际默认样本数见各任务目录 README 与 eval.yaml。

## 原始日志与索引

```bash
./run_inspect_view.sh
./run_inspect_view.sh --port 5012 --host 127.0.0.1
python3 scripts/index_eval_logs.py
```

原始 `.eval` 保存在服务器 `eval_logs/`，GitHub 只保存索引。现有 14 份日志来自原集成的 10 样本冒烟测试，按原始配置和 SHA-256 保留；没有在迁移中改写成新包版本的结果。

本次 mock 运行检查任务构造、模型角色、solver/scorer 调用和日志生成。模型性能、论文方法一致性、两模型全量日志和人工轨迹审阅属于后续正式发布检查。
