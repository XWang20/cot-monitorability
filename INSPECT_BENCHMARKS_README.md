# CoT Monitorability — Inspect Evals 统一评测说明

本目录（`/home/wangxing/cot-monitorability`）已将全部 6 个思维链可监控性（CoT Monitorability）基准统一接入 **`inspect_evals`** 框架，默认连接本机统一代理端口 `http://127.0.0.1:8000/v1`（密码 `a`），并保留各上游 Git 历史，以固定版本和补丁记录当前适配源码。

---

## 一、上游仓库与版本管理

`inspect_ai/`、`inspect_evals/` 和 `repos/` 下的 5 个基准仓库仍保留各自 Git 历史。
主仓库通过 `dependencies.lock.json` 固定上游 commit，通过 `patches/` 保存本地源码改动，
支持恢复到新的项目目录；各 clone 和评测日志仍由 `.gitignore` 忽略。

```bash
python3 scripts/dependency_repos.py check     # 核对当前源码与快照
python3 scripts/dependency_repos.py snapshot  # 修改依赖后更新补丁
python3 scripts/dependency_repos.py restore   # 在全新项目 checkout 中恢复依赖
```

锁定版本、补丁校验和及当前本地 HEAD 以 `dependencies.lock.json` 为准。
详见 [仓库管理说明](REPOSITORY_MANAGEMENT.md)。恢复脚本拒绝覆盖已有依赖目录。

---

## 二、8000 端口代理与模型配置说明

- **环境变量**：
  ```bash
  export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
  export OPENAI_API_KEY=a
  export INSPECT_EVALS_JUDGE_MODEL=openai/deepseek-v4.1-flash-1017
  ```
- **被测模型与统一 Judge 模型**：
  - **DeepSeek V4.1 Flash-1017（被测模型 & 全部 6 个基准的默认 Judge/Monitor 模型）**：`openai/deepseek-v4.1-flash-1017`（直接走本机 `:8000` 代理，已将 Judge 最大生成长度设为 `8192` 以容纳其自身思考 Token）。
  - **Kimi K3.1 Pro（被测模型）**：`openai/kimi-k3.1-pro`（已由本机 `:8000` 代理提供，使用标准 OpenAI 兼容接口；不再自动切换到 DMXAPI。保留模型参数兼容处理）。
  - **启动配置**：`run_inspect_bench.sh` 默认使用本机代理，不再加载旧 `.env`；可显式导出 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 覆盖默认值，也支持 `PROXY_API_KEY` 作为凭据备用变量。

---

## 三、6 个 Benchmark 的 `inspect eval` 调用方式与参数

您可以直接使用一键脚本 `./run_inspect_bench.sh`，或直接运行原生 `inspect eval` 命令：

### 1. CoT-Control-QA（正文 10 种模式，`inspect_evals/cot_controllability_qa`）
- **支持参数**：
  - `-T dataset=gpqa|hle|mmlu_pro|all`（默认 `gpqa`）
  - `-T modes=word_suppression|multiple_word_suppression|uppercase_thinking|lowercase_thinking|alternating_case|repeat_sentences|end_of_sentence|meow_between_words|ignore_question|baseline|all_main`
  - `-T adversarial_pressure=na|none|shutdown|money|reward_model|caps|profanity|repetition|replaced|policy|supervisor`
  - `-T monitor_aware=true|false`、`-T eval_aware=true|false`、`-T check_meta_discussion=true|false`
- **示例命令**：
  ```bash
  .venv/bin/inspect eval inspect_evals/cot_controllability_qa \
    --model openai/deepseek-v4.1-flash --reasoning-effort high \
    --limit 10 --max-connections 5 \
    -T dataset=gpqa -T modes=word_suppression
  ```

### 2. OpenAI Process-level Monitorability（`inspect_evals/openai_monitorability_process`）
- **复刻内容**：完整复用 `openai/monitorability-evals` 的 `process/process_gpqa_physics.jsonl`（52 题）、`model_prompts/process_labeling_model_prompt.txt` 与 `monitor_prompts/templates/process.txt`，评估模型思维链是否显式/隐式覆盖解题必经定理路径（Theorem Path Faithfulness `gmean` & `TPR`）。
- **示例命令**：
  ```bash
  .venv/bin/inspect eval inspect_evals/openai_monitorability_process \
    --model openai/deepseek-v4.1-flash --reasoning-effort high \
    --limit 10 --max-connections 5 \
    -T monitor_model=openai/judge -T scopes=cot_only
  ```

### 3. OpenAI Intervention-level Monitorability（`inspect_evals/openai_monitorability_intervention`）
- **复刻内容**：完整复用 `openai/monitorability-evals` 的成对因果干预（Unperturbed `arm=0` vs. Perturbed `arm=1`）与官方 `metric/intervention_gmean_metric.py`（`bootstrapped_gmean_metric`）。每个 Inspect Sample 内置成对的 `unperturbed` 和 `perturbed` 两次生成，因此 `--limit 10` 会完整评测 10 对（20 条）因果样本。
- **支持参数**：
  - `-T dataset=gpqa|gpqa_encoded|daft|scruples_first_person|scruples_suggest_right|scruples_suggest_wrong|wmdp_sandbagging|all`
- **示例命令**：
  ```bash
  .venv/bin/inspect eval inspect_evals/openai_monitorability_intervention \
    --model openai/deepseek-v4.1-flash --reasoning-effort high \
    --limit 10 --max-connections 5 \
    -T dataset=gpqa
  ```

### 4. DeepMind `arXiv:2510.23966` Immediacy Judge（`inspect_evals/cot_immediacy_2510`）
- **复刻内容**：复刻 DeepMind *A Pragmatic Way to Measure Chain-of-Thought Monitorability* Appendix C 的 Autorater Judge（评估 `legibility_pct` 0–100% 与 `coverage_pct` 0–100%）。
- **支持参数**：
  - `-T dataset=gpqa|hle|mmlu_pro`
  - `-T judge_model=openai/judge`
  - `-T correct_only=true|false`
- **示例命令**：
  ```bash
  .venv/bin/inspect eval inspect_evals/cot_immediacy_2510 \
    --model openai/deepseek-v4.1-flash --reasoning-effort high \
    --limit 10 --max-connections 5 \
    -T dataset=gpqa
  ```

### 5. `nocot-bench` NCRI（`inspect_evals/ncri_bench`）
- **复刻内容**：完整复用 `neelnanda-io/nocot-bench` 的 `nocot.run`（K-shot turns + `Answer:` prefill + `max_tokens=100`）、`nocot.witnesses`（W1/W2/W3 三重无思维链鉴证）、`nocot.grade` 与 `nocot.place`（直接解算 NCRI 15.2 冻结难度阶梯上的 Rasch 能力值 `100 + 14.42695 * theta`）。对 DeepSeek 模型会自动路由至 8000 代理的 `-nt` 通道以确保 `reasoning_tokens=0`。
- **支持参数**：
  - `-T banks=arithmetic|sudoku|shortpath|cfg|chain|all_ncri`（支持逗号分隔的任意 NCRI 题库或 `all_ncri` 全部 19 个本地题库）
- **示例命令**：
  ```bash
  .venv/bin/inspect eval inspect_evals/ncri_bench \
    --model openai/deepseek-v4.1-flash \
    --limit 10 --max-connections 5 \
    -T banks=arithmetic
  ```

### 6. MonitorBench（`inspect_evals/monitorbench`）
- **复刻内容**：整合了 `semsorock/inspect-evals-monitor-bench` 已完成的 5 个子任务（`dual_objectives_steganography`、`dual_objectives_summarization`、`goal_sandbag_safety`、`goal_sandbag_math`、`goal_sandbag_engineering`）以及 `ASTRAL-Group/MonitorBench` 其余非沙盒任务（`goal_sandbag.science`、`goal_deception.realworld`、`cue_condition.knowledge/preference/judgment/health`、`cue_answer.math/logic/health`、`process.multistep`）。
- **示例命令**：
  ```bash
  .venv/bin/inspect eval inspect_evals/monitorbench \
    --model openai/deepseek-v4.1-flash \
    --model-role monitor=openai/judge \
    --reasoning-effort high \
    --limit 10 --max-connections 5 \
    -T subset=cue_condition.preference -T stress_test=standard
  ```

---

## 四、可视化查看 `.eval` 日志

所有跑出的 `.eval` 日志均保存在 `/home/wangxing/cot-monitorability/eval_logs/` 下，可随时使用 Inspect View 查看每道题的完整消息、`<think>` 思维链及 Scorer 判分详情：
```bash
/home/wangxing/cot-monitorability/.venv/bin/inspect view --log-dir /home/wangxing/cot-monitorability/eval_logs --port 5010 --host 0.0.0.0
```
