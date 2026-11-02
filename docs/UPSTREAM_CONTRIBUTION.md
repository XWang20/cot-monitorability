# Inspect Evals 提交准备

核对日期：2026-11-02。当前状态：**本地集成可运行，登记准备尚未完成**。

## 官方贡献流程

当前官方 [CONTRIBUTING.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/CONTRIBUTING.md) 写明不再接收新的 eval 实现代码提交，新实现通过 [Inspect Evals Register](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/register/README.md) 登记。已有实现的代码改动 PR 只接受预批准贡献者。

新任务实现保留在作者自己的源码仓库。通过 Register Eval Submission issue 提供版本化论文 URL、固定 40 字符 commit 的任务源码 URL，并在多任务文件中提供行号。机器人生成登记 PR，维护者核对代码与运行日志。

`patches/` 的用途是复现本地依赖改动。登记时的 Source URL 应指向我们发布包中直接定义 `@task` 的 Python 文件。

## 登记的必需条件与当前差距

| 官方条件 | 当前状态 | 准备工作 |
| --- | --- | --- |
| 公开 GitHub 源码仓库，固定可获取 commit | 主仓库为私有仓库 | 完成代码审阅后，由用户决定公开发布的位置；当前仓库继续保持私有 |
| `pyproject.toml` 含 `[project]`，支持 `uv sync`，声明 `inspect_ai` | 当前为本地运行环境，使用 requirements 和源码恢复脚本 | 将任务迁移为独立、可安装的 Python 包，增加项目元数据和依赖锁 |
| Source URL 对应直接定义 `@task` 的源码 | 任务目前分散在未纳入主仓库 Git 的依赖 clone 中 | 将 6 个任务的实现整理到主仓库的源码包；提供固定 SHA 与任务行号 |
| 外部资产稳定托管并固定版本 | 依赖源码和 NCRI 归档已固定及校验；当前加载还依赖 clone 的目录结构 | 以包资源和缓存加载数据、Prompt、参考实现；在元数据中声明固定来源和校验 |
| 任务命名不与现有任务冲突 | 4 个 MonitorBench 子任务已有登记 | 复用既有登记，对新增子任务提供独立入口和准确描述 |
| 两模型全样本运行及日志核对 | 已有 14 份 10 样本冒烟日志 | 任务质量检查完成后再运行完整样本；人工核对轨迹与论文方法，按登记流程提供日志 |

Register 将部分 lint 检查作为信息性结果，不要求全部 lint 通过才能登记。我们仍采用官方实现规范作为代码质量要求。

## 发布包的任务规范

后续迁移按 [BEST_PRACTICES.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/BEST_PRACTICES.md)、[EVALUATION_CHECKLIST.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/EVALUATION_CHECKLIST.md) 和 [实现模板](https://github.com/Generality-Labs/inspect-evals-template) 执行：

- 每个评测模块放入 Python 包，并包含 `__init__.py`。对外使用 `@task`，将 dataset、solver、scorer 和 metric 分开组织，提供文档和版本。
- 任务代码通过包导入或固定资产加载运行，移除依赖服务器 checkout 位置的 `sys.path` 修改与跨仓库动态加载。
- 使用 Inspect Model Roles 管理 Judge/Monitor，允许覆盖模型与配置。仅在 solver/scorer 内解析模型。8000 代理、凭据和 provider 兼容配置由运行配置管理。
- 模型失败、评测实现错误和 Judge 无有效结论分别处理。模型失败计相应分数；确定性的实现错误抛异常；无法判分的 Judge 结果按框架支持的 unscored 机制记录原因。避免将解析失败或缺少有效样本统一记为零。
- 可归约的评分量放在 `Score.value`，避免使用仅存于 metadata 的量产生错误的多 epoch 聚合；提供与指标类型匹配的标准误或其他不确定性估计。
- Dataset ID 稳定且区分完整记录。资产固定到 commit/revision，并记录过滤规则、实际样本数、许可和 canary。
- 从上游复制或改写的文件增加来源声明；根 `NOTICE` 记录对应来源、版权与文件。与原论文或参考判分代码的偏差逐项写入任务 README。
- 自定义 dataset、solver、scorer、parser 和 metric 需要确定性测试。模型输出使用 `mockllm/model`，外部 API 使用 mock，临时文件使用 `tmp_path`。分别验证合法、错误、空输出、无有效样本、边界值和多 epoch 情况。
- 改变结果或任务接口时按 [TASK_VERSIONING.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/TASK_VERSIONING.md) 更新版本、任务 changelog 和必要的发布说明。

参考检查命令（在迁移后的发布包中运行）：

```bash
uv sync --group dev --group test
uv run ruff check
uv run ruff format --check
uv run mypy src tests
uv run pytest tests
uvx inspect-evals-lint --preset single-eval --all
```

这些检查尚未在当前本地适配上全部执行，不能据此宣称符合发布规范。14 份日志及任务构造检查确认了当前本地运行状态，完整方法审阅和质量检查仍需完成。

## 各任务需要核对的内容

| 任务 | 发布前核对重点 |
| --- | --- |
| CoT-Control QA | 正文模式、约束解析和合规判分与固定参考实现一致；明确 baseline、空 CoT 和固定长度窗口无样本的处理；验证不同数据集过滤和 ID |
| OpenAI Process | Monitor 判分解析、公式和聚合与参考实现一致；角色覆盖有效；Judge 错误与被测模型错误区分 |
| OpenAI Intervention | 成对样本对应关系、干预方向、gmean 上/下界与 eligible fraction 判定；无可报告样本的返回与解释 |
| DeepMind Immediacy | 与版本化论文 Appendix C 对照 Judge Prompt、评分量表和正确性过滤；当前 `correct_only=false` 需与论文对应设置区分 |
| NCRI | K-shot、prefill、无 CoT 检验和冻结 placement 保持一致；当前 arithmetic 冒烟子集不能描述为全 NCRI 结果 |
| MonitorBench | 区分现有登记与新增子任务；核对验证器和 Monitor 的职责；当前只对 preference 子集有冒烟记录 |

目前上游已有以下 MonitorBench 登记，应在后续工作中优先复用：

- `monitor_bench-goal-sandbag-safety`
- `monitorbench-dual-objectives-summarization`
- `monitorbench-goal-sandbag-math`
- `monitorbench-steganography`

原始运行日志继续保存在服务器。官方登记流程要求的日志公开性与去标识处理在正式发布前单独确认。官方贡献指南要求人类在提交前审阅和测试 AI 生成的生产代码。
