# 仓库与依赖管理

主仓库维护 6 类 Inspect 基准集成的启动入口、结果索引和依赖快照。主仓库从自己的初始提交开始，作者关联 `XWang20`。

## 评测结果

原始 `.eval` 位于服务器 `eval_logs/` 下。GitHub 只保存 `eval_logs/README.md` 和 `eval_logs/index.json`。
索引记录运行配置、完整汇总指标和原始日志 SHA-256；脚本只读取日志 header，不修改原始文件。

```bash
python3 scripts/index_eval_logs.py
```

在没有本机原始 `.eval` 的新 checkout 中，索引脚本会拒绝生成空索引。已有 GitHub 索引仍可直接查看。
旧平台源码和 CSV 输出已从活跃目录移除，服务器的 `tmp/` 下保留本次清理备份。

## 固定依赖源码

`dependencies.lock.json` 为 7 个独立 clone 记录公开 HTTPS 来源、上游 commit、补丁及 SHA-256。
`checkout_head` 记录原本地来源；恢复使用固定 `base_commit` 加补丁。依赖 clone 保留各自上游历史。
Inspect 补丁包含现有 fork 的兼容性和基准扩展，以保留当前运行代码。

```bash
python3 scripts/dependency_repos.py check
python3 scripts/dependency_repos.py snapshot
```

修改依赖后，先在对应 clone 中暂存或提交新增源码文件，再运行 `snapshot`。该命令保存已跟踪的暂存及未暂存改动，保留已有固定来源和基准 commit，并拒绝遗漏未跟踪的源码。
评审锁文件与补丁后，将它们和对应文档、启动脚本一起提交到主仓库。

## 新 checkout 恢复

```bash
python3 scripts/dependency_repos.py restore
python3 scripts/dependency_repos.py check
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`restore` 拒绝覆盖已有依赖目录。它 clone 固定来源、checkout 固定 commit，并将补丁应用到 index 和工作区。
NCRI 题库由上游发布压缩包恢复，压缩包和解压文件均校验 SHA-256，并保留 canary。解压题库仍由依赖自己的 Git 忽略规则管理。
已有 clone 缺少题库时，可运行 `python3 scripts/dependency_repos.py prepare-data`，该命令拒绝覆盖已变更数据。

更新上游版本时，明确选择新的公开来源与基准 commit，重新应用本地改动、验证，再运行 `snapshot`。主仓库依赖目录位置固定为 `inspect_ai/`、`inspect_evals/` 与 `repos/`。

## Git remotes

主仓库的 `origin` 为 `https://github.com/XWang20/cot-monitorability.git`。`upstream` 保留原 CoT-Control 地址供源码对照。
各基准的原始来源与许可见 [docs/SOURCES.md](docs/SOURCES.md)。
