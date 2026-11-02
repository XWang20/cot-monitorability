# 来源与许可

主仓库维护 6 类 Inspect 评测集成、启动脚本和结果索引。各依赖 clone 保留上游 Git 历史，本地适配由主仓库的固定版本和补丁记录。

| 用途 | 上游源码 |
| --- | --- |
| Inspect AI 框架与 View | https://github.com/UKGovernmentBEIS/inspect_ai |
| Inspect Evals 注册与 Immediacy Judge | https://github.com/UKGovernmentBEIS/inspect_evals |
| CoT-Control QA | https://github.com/YuehHanChen/CoTControl |
| OpenAI Process / Intervention | https://github.com/openai/monitorability-evals |
| NCRI | https://github.com/neelnanda-io/nocot-bench |
| MonitorBench | https://github.com/ASTRAL-Group/MonitorBench |
| MonitorBench Inspect 适配 | https://github.com/semsorock/inspect-evals-monitor-bench |

准确的上游 commit、补丁和校验和见 [dependencies.lock.json](../dependencies.lock.json)。

相关论文与原始说明：

- CoT-Control：[Reasoning Models Struggle to Control their Chains of Thought](https://arxiv.org/abs/2603.05706)。
- [OpenAI monitorability-evals 对应论文](https://arxiv.org/abs/2512.18311)。
- DeepMind Immediacy：[A Pragmatic Way to Measure Chain-of-Thought Monitorability](https://arxiv.org/abs/2510.23966)，使用 Appendix C 的 Judge 设置。
- NCRI 与 MonitorBench 的数据来源、许可和任务说明以对应上游仓库为准。

根目录保留原 CoT-Control 的 MIT LICENSE。各源码、题库、压缩包和数据集的许可、引用要求与 canary 信息保留在对应依赖 clone 中。

新评测包使用 uv.lock 中固定的公开 Inspect SDK。任务源码和引用的参考函数位于 src/cot_monitorability，来源与许可证见根 NOTICE 和 licenses。任务数据归档及 SHA-256 见包内 asset_specs.json；旧 dependencies.lock.json 保留旧服务器环境来源。
