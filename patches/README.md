# 旧环境迁移 patch

本目录保留 7 份迁移及 fork 差异记录。它们与根 `dependencies.lock.json` 一起固定旧环境的来源版本和改动，可用于复现、差异审阅和回归比较。

当前独立评测包的任务源码已移入 `src/cot_monitorability/`。本目录仍需保留；对应关系和确认范围见 [docs/MIGRATION_MAP.md](../docs/MIGRATION_MAP.md)。

`inspect_ai.patch` 和 `inspect_evals.patch` 含导入 fork 的整体差异，范围超过 6 个 CoT 任务。逐项完成真实 API 和框架功能回归前，不将其视为已完全被新 SDK 覆盖。
