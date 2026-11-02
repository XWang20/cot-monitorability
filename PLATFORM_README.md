# CoT Monitorability 评测与监控平台 (`cot-monitorability`)

本仓库是基于原始开源评测库 [`YuehHanChen/CoTControl`](https://github.com/YuehHanChen/CoTControl)（ICML 2026 *Reasoning Models Struggle to Control their Chains of Thought*）与 Google DeepMind 思维链默认可监控性评估方法（*A Pragmatic Way to Measure Chain-of-Thought Monitorability*, `arXiv:2510.23966` & *Gemini 4 Argon Safety Report* Section 3.4.5.6 / 3.4.5.8）构建的 **CoT 可监控性双支柱评测与监控平台**。

---

## 一、 平台核心能力

1. **双支柱评测体系（Controllability + Immediacy）**
   * **支柱一：`CoT Controllability`（对抗可控性，`CoT-Control-QA/`）**
     * 支持原论文 **9 种基础约束** + Google DeepMind《Gemini 4 Argon 安全报告》扩充的 **5 种纯正则约束**（共 **14 种约束**，以及 `baseline` 无约束基线）。
     * 内置三大机构官方 System Card 预设模式包：`--mode_preset openai8`（8项）、`--mode_preset anthropic9`（9项）、`--mode_preset deepmind14`（14项）。
     * 内置数据集预设：`--dataset_preset dm300`（GPQA/HLE/MMLU-Pro 各前100题=300题，对齐 DeepMind 报告）与 `--dataset_preset full1214`（1,214道 QA 全集，对齐 OpenAI/Anthropic 系统卡）。
     * 同时计算 **Strict Controllability（全文 100% 严格合规率）** 与 **Continuous Controllability（合规词/句占比软指标）**，并按 **Information Suppression（信息抑制组）** 与 **Stylistic Modification（风格修改组）** 自动聚合。
   * **支柱二：`CoT Immediacy`（可读性 Legibility & 覆盖度 Coverage，`CoT-Immediacy/`）**
     * 完整实现 DeepMind `2510.23966` Appendix C 的 Autorater Prompt 与 **Correctness Filter（仅评答对样本）**。
     * 支持 **3 种轨迹来源**：
       1. `--source qa_baseline`：自动生成或零成本复用 `CoT-Control-QA` 的无约束 `baseline` 轨迹；
       2. `--source platform_logs`：直接从本机监测平台 `/home/eval/workspace/eval_log/openai/<model>/*.eval` 中提取高难基准（如 `FrontierScience`、`SciCode`、`BioLP`）已答对的原生 `<think>` 轨迹（**被测模型 0 生成成本**）；
       3. `--source csv --source_csv <path>`：对任意已有 CSV 结果文件进行评分。

2. **多模型与双通道适配（`models.yaml` + `CoT-Control-QA/llm.py`）**
   * **双通道无缝切换**：
     * `--provider dmxapi`：开发环境默认走 `https://www.dmxapi.cn`；
     * `--provider proxy`：正式测评走本机平台统一代理 `http://127.0.0.1:8000/v1`（LiteLLM Proxy）。
   * **跨架构思维链自动提取**：
     * 原生返回 `reasoning_content` 的模型（DeepSeek、MiMo、MiniMax、GLM、Qwen、Doubao、Hunyuan、ERNIE、Kimi、Grok）；
     * Anthropic 模型（自动路由 `/v1/messages` 并开启 `thinking: {type: adaptive, display: summarized}` + `effort: max`）；
     * Google Gemini 模型（自动注入 `extra_body.google.thinking_config.include_thoughts = True` 提取思维链摘要）；
     * OpenAI 隐藏思维链模型（如 `gpt-6.1-sol`，自动切换 `--non_reasoning` 使用 `<output_reasoning>` 标签评测外部思维链可控性）。

3. **高可靠并发、断点续跑与错误处理**
   * 基于单实例复用的 `aiohttp.ClientSession` + `asyncio.Semaphore` 控制并发；
   * 针对 HTTP `429 Rate Limit`、`500/502/503/504/524` 及网络超时内置**指数退避 + 随机抖动重试（Exponential Backoff with Jitter）**；
   * **逐条实时落盘（Incremental Checkpointing）**：每完成 1 道题立即追加写入 CSV 并原子更新 `CoT-Control-QA/logs/status/<job_id>.json`；意外中断后重跑自动跳过已完成样本，仅补跑失败或未跑样本。

4. **人类可视化监控看板（`monitor_platform.py` + `run_dashboard.sh`）— ⚠️ 已退役**
   * **当前主界面已切换为 Inspect View（端口 `5010`）**：直接浏览 `eval_logs/` 下的 `.eval` 日志，可查看每道题的完整消息、`<think>` 思维链与 Scorer 判分详情。
   * 下述 Flask 看板为历史实现，代码保留但**默认不再占用 5010**（已改为 `5011`），仅作为离线回看 CSV 结果的备用界面：
   * 提供 **4 大视图**：
     1. **模型总榜与双支柱对比**：同屏展示 Baseline/Controlled 准确率、Strict/Continuous 可控率、信息抑制/风格修改分组可控率、平均思考词数、平均耗时、Legibility & Coverage 得分；
     2. **Controllability 14 项约束细分矩阵**：展示每个模型在每种约束下的合规率与准确率；
     3. **任务进度、耗时、Token 与错误实时监控**：每 3 秒自动刷新进度条、单题耗时、累计 Token、预估人民币花费与最近报错信息；
     4. **样本级轨迹穿透抽屉 & 在线任务启动器**：点击任意指标或单元格即可弹出抽屉查看原始 Prompt、`<think>` 思维链、最终输出及 Autorater 评分理由，或直接在网页端启动新评测。

---

## 二、 快速使用指南

### 1. 查看评测结果

**主界面：Inspect View（端口 5010）**
```bash
.venv/bin/inspect view --log-dir eval_logs --port 5010 --host 0.0.0.0
```

**遗留 Flask 看板（可选，端口 5011）** —— 已退役，仅用于离线回看 CSV 结果，不再占用 5010：
```bash
./run_dashboard.sh status   # 查看运行状态
./run_dashboard.sh start    # 启动后台看板 (默认端口 5011)
./run_dashboard.sh restart  # 重启看板
```

### 2. 运行多模型冒烟测试（Controllability + Immediacy 一键联测）
```bash
.venv/bin/python monitor_platform.py smoke \
  --models deepseek-v4.1-flash-1017 mimo-v2.6-pro minimax-m3 glm-5.3 qwen3.8-max-0902 claude-sonnet-5-5 gemini-3.8-flash gpt-6.1-sol \
  --modes baseline word_suppression uppercase_thinking no_lettere \
  --limit 2 \
  --autorater gemini-3.8-flash \
  --provider dmxapi
```

### 3. 运行 CoT Controllability（可控性）评测
```bash
# 使用 DMXAPI 跑单个或多个约束模式
.venv/bin/python CoT-Control-QA/run_cceval.py \
  --model deepseek-v4.1-flash-1017 mimo-v2.6-pro \
  --mode word_suppression uppercase_thinking ignore_question \
  --limit 10 \
  --provider dmxapi

# 使用 DeepMind 官方 300 题子集 + DeepMind 14 项完整约束（切换到正式代理 proxy）
.venv/bin/python CoT-Control-QA/run_cceval.py \
  --model deepseek-v4-pro-0813 \
  --dataset_preset dm300 \
  --mode_preset deepmind14 \
  --provider proxy
```

### 4. 运行 CoT Immediacy（可读性 Legibility & 覆盖度 Coverage）评测
```bash
# 方式 A：直接复用 CoT-Control-QA 的 baseline 轨迹进行打分
.venv/bin/python CoT-Immediacy/run_immediacy.py \
  --model deepseek-v4.1-flash-1017 mimo-v2.6-pro \
  --source qa_baseline \
  --limit 10 \
  --autorater gemini-3.8-flash \
  --provider dmxapi

# 方式 B：0 生成成本！直接读取监测平台 /home/eval/workspace/eval_log 下已有的高难基准答对轨迹打分
.venv/bin/python CoT-Immediacy/run_immediacy.py \
  --model deepseek-v4-pro-0813 \
  --source platform_logs \
  --limit 50 \
  --autorater gemini-3.8-flash \
  --provider dmxapi
```
