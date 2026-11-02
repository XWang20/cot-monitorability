# 迁移记录与原始 patch

7 份原始 patch 和 `dependencies.lock.json` 均保留在主仓库。它们记录原始依赖版本、本地改动和可恢复的旧运行环境。独立包成为当前运行入口后，这些迁移记录仍需保留。

## 对应关系

| 原始记录 | 新包中的对应位置 | 确认范围 |
| --- | --- | --- |
| `repos-CoTControl.patch` | `cot_control/` 与 `reference/cot_control/` | 任务、生成器、判分和指标代码迁移；默认样本指纹一致，mock 与组件测试通过 |
| `repos-monitorability-evals.patch` | `openai_process/`、`openai_intervention/`、`openai_common.py` 与对应参考模块 | Process / Intervention 适配迁移；默认样本指纹一致，mock 运行通过 |
| `repos-nocot-bench.patch` | `ncri/` 与 `reference/nocot/` | 无 CoT 请求、判定与 placement 的适配和参考代码保留；arithmetic 默认样本一致 |
| `repos-MonitorBench.patch` | `monitorbench/` 与 `reference/monitor_bench/` | 通用非沙盒适配与社区子任务代码迁移；preference 默认样本一致，单次/成对生成和 eligibility 测试通过 |
| `repos-inspect-evals-monitor-bench.patch` | `reference/monitor_bench/model_config.py` | 旧的 model_roles 导入兼容改为公共 Model Roles API；保留 Monitor 默认采样参数 |
| `inspect_evals.patch` 中 CoT 相关部分 | `_registry.py`、6 个任务入口、`immediacy/` | 原注册 wrapper 改为独立插件；Immediacy 实现迁入包。其余 fork 扩展继续保存在旧 patch 中 |
| `inspect_ai.patch` | 公开 Inspect SDK、启动脚本的 `--extra-body` 与角色配置 | 新包不再应用整个旧框架 patch。已验证包构造、mock 运行和 OpenAI provider 初始化；尚未逐项完成旧框架功能及真实 API 的等价回归 |

新源码的实际路径均位于 `src/cot_monitorability/`。格式迁移提交为 `b07b231705416dc10108caf7cab634e0056ba377`，不是通过删掉旧 patch 来完成的。

## 保留的边界

`inspect_ai.patch` 含 56 个文件的差异，`inspect_evals.patch` 含 589 个文件的差异。它们包含从原私有 fork 导入的框架或其他基准扩展，不能将整个文件都解释为本项目新写的 CoT 适配。

6 类任务适配已进入新包。使用公开 SDK 接管旧框架功能，需要按功能核对，不能仅凭版本更新就认定旧改动全部得到覆盖。以下项目在进行真实模型迁移回归时应检查：

- 本机代理的 reasoning_effort、nested reasoning 和额外参数透传。
- DeepSeek / Kimi 的原生 reasoning 字段、token 用量和无 CoT 设置。
- Judge/Monitor 的角色覆盖、请求配置和并发/重试行为。
- 旧 fork 的日志查看、断点续跑和其他共享框架修复是否仍被项目依赖。

当前 mock 和样本指纹检查不代表这些真实 API 或框架功能已逐项等价。

## 维护方式

日常任务开发修改新包源码，更新对应文档、版本和测试。原 patch 作为迁移来源与旧环境基准保留，不因新包暂时不调用它们而删除。

`scripts/dependency_repos.py check` 可以核对旧 clone 与原锁文件和 patch；该机制与新包的资产缓存分开维护。若继续修改旧环境，先保存当前锁文件和 patch 快照，再更新新的快照，以便区分迁移前后的状态。

移除旧 patch 或旧环境前，需要另行完成上述回归，并保留可恢复的来源记录。当前未进行这种移除。
