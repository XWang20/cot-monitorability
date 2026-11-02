# Inspect Evals 登记格式

本仓库已整理为独立 Inspect Python 包。此文件区分已完成的格式准备与实际登记前的工作；当前仓库继续保持私有，没有提交 Register。

## 已准备的格式

| 项目 | 当前内容 |
| --- | --- |
| 安装包 | `pyproject.toml`、PEP 517 构建、uv.lock、Inspect 插件入口 |
| 任务源码 | `src/cot_monitorability/` 下 6 个直接定义 `@task` 的源码入口 |
| 任务版本 | 每个默认任务为 `1-A` |
| 元数据与文档 | 每个任务有 `eval.yaml`、README、默认样本数与配置说明 |
| 外部资产 | 固定源码 commit、归档 SHA-256 和独立缓存，复用前验证文件 |
| 模型配置 | Judge/Monitor 使用公共 Model Roles；本机 API 参数由部署脚本管理 |
| 来源许可 | `NOTICE`、`licenses/` 与参考文件的来源声明 |
| 格式草稿 | `scripts/prepare_registration.py` 生成标准 `register/<id>/eval.yaml`，源码 URL 固定到已提交 SHA |
| 校验 | 按官方元数据模型核对任务元数据，并用固定 Register JSON schema 验证草稿 |

数据和参考函数已经迁移到包资源及固定资产缓存。新包不需要修改版 `inspect_evals` 或服务器上的原始 `repos/` 目录。

```bash
uv sync --locked --group dev --group test
uv run pytest tests
uv build
uv run python scripts/prepare_registration.py
```

代码应先提交，再生成登记草稿。生成器读取指定 commit 中的任务与 catalog，确保 Source URL 对应已提交的直接任务定义。它只写本地文件。

## 实际登记前仍需完成

- 选择公开的源码仓库或发布位置；本次不会修改私有仓库的可见性。
- 逐项审阅方法、数据过滤、Judge Prompt、判分和指标聚合，并与论文及固定参考实现比较。
- 对拟登记的任务配置运行两模型全部样本，人工核对轨迹，准备可公开的运行日志；已有 14 份旧集成冒烟日志不作为本包 `1-A` 全量运行证据。
- 确认版本化论文 URL。NCRI 原始说明已引用用户提供的 [LessWrong 文章](https://www.lesswrong.com/posts/eRmzz8J8Qkzqvzrgg/astra-can-do-a-concerning-amount-with-no-chain-of-thought)，arXiv 项按用户要求留待补充。
- 当前沿用的 CoT-Control 宽松 T/F 解析存在已标记测试：无结论但包含字母 T 的文本可能被判为 T。格式迁移保留原语义，正式方法审阅需处理或说明该问题。其他 Judge 无效结论、空样本和多 epoch 聚合也需完成评分质量检查。
- MonitorBench 已有 4 个社区子任务登记。当前包默认 preference 子集；登记其他变体时需检查重复条目和任务名称。

## 官方流程与参考

当前官方 [CONTRIBUTING.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/CONTRIBUTING.md) 不再接收新的 eval 实现代码 PR。新实现通过 [Inspect Evals Register](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/register/README.md) 指向作者维护的任务源码。

每个任务对应一个 Register Eval Submission issue，提供论文 URL、固定 40 字符 commit 的源码 URL和必要行号；后续机器人和维护者检查代码、元数据和日志。本次只生成可检查的源码与元数据格式。

质量检查参考 [BEST_PRACTICES.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/BEST_PRACTICES.md)、[EVALUATION_CHECKLIST.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/EVALUATION_CHECKLIST.md)、[TASK_VERSIONING.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/TASK_VERSIONING.md) 和 [实现模板](https://github.com/Generality-Labs/inspect-evals-template)。格式通过不代表方法、评分和公开发布检查已经完成。
