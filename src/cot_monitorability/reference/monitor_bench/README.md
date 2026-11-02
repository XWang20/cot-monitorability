# MonitorBench: Chain-of-Thought Monitorability

[MonitorBench](https://arxiv.org/abs/2603.28590v2) evaluates whether
decision-critical factors and hidden objectives are visible to
chain-of-thought (CoT) monitors. Its upstream registry defines 19 tasks across
seven task families and three evaluation axes. This package is migrating those
tasks incrementally so that each task can be reviewed, validated, and submitted
independently.

The current release implements `goal_sandbag.math`, `goal_sandbag.safety`,
`goal_sandbag.engineering`, `dual_objectives.steganography`, and
`dual_objectives.summarization`. Public Inspect task names replace each dot
with an underscore. The other
14 tasks below are migration plans;
they cannot yet be imported or run from this package.

The port is pinned to
[ASTRAL-Group/MonitorBench](https://github.com/ASTRAL-Group/MonitorBench) commit
`43dda5994bfb16d34b1c30d4b3482d78a714e640`.

## Evaluation axes

| Axis | What MonitorBench varies | Task count |
| ---- | ------------------------ | ---------: |
| Input Intervention | Whether cues or goals in the input change model behavior and its explanation. | 12 |
| Outcome Justification | Whether reasoning reveals a hidden objective or explains a suspicious outcome. | 6 |
| Solution Process | Whether the reasoning process exposes a decisive intermediate step. | 1 |

## Task catalog and migration status

**Implemented** means that the task is exported from `monitor_bench`, listed in
`eval.yaml`, tested, and runnable. **Planned** means that only a scoped tracking
issue exists; code on draft or reference branches is not supported product
state. Sample counts reproduce the current issue scope and must be confirmed
against pinned assets when each planned task is implemented.
Inspect task names preserve the complete upstream identifier, replacing each
dot with an underscore.

### Input Intervention

| Upstream task | Inspect task | Samples | Status |
| ------------- | ------------ | ------: | ------ |
| `goal_sandbag.safety` | `goal_sandbag_safety` | 120 | **Implemented** ([task docs](tasks/goal_sandbag/safety/README.md), [#5](https://github.com/semsorock/inspect-evals-monitor-bench/issues/5)) |
| `goal_sandbag.math` | `goal_sandbag_math` | 30 | **Implemented** ([task docs](tasks/goal_sandbag/math/README.md), [#2](https://github.com/semsorock/inspect-evals-monitor-bench/issues/2)) |
| `goal_sandbag.science` | `goal_sandbag_science` | 198 | Planned ([#6](https://github.com/semsorock/inspect-evals-monitor-bench/issues/6)) |
| `goal_sandbag.engineering` | `goal_sandbag_engineering` | 150 | **Implemented** ([task docs](tasks/goal_sandbag/engineering/README.md), [#7](https://github.com/semsorock/inspect-evals-monitor-bench/issues/7)) |
| `goal_deception.realworld` | `goal_deception_realworld` | 50 | Planned ([#4](https://github.com/semsorock/inspect-evals-monitor-bench/issues/4)) |
| `cue_condition.knowledge` | `cue_condition_knowledge` | 100 | Planned ([#12](https://github.com/semsorock/inspect-evals-monitor-bench/issues/12)) |
| `cue_condition.preference` | `cue_condition_preference` | 108 | Planned ([#8](https://github.com/semsorock/inspect-evals-monitor-bench/issues/8)) |
| `cue_condition.judgment` | `cue_condition_judgment` | 100 | Planned ([#9](https://github.com/semsorock/inspect-evals-monitor-bench/issues/9)) |
| `cue_condition.health` | `cue_condition_health` | 65 | Planned ([#10](https://github.com/semsorock/inspect-evals-monitor-bench/issues/10)) |
| `cue_answer.math` | `cue_answer_math` | 28 | Planned ([#13](https://github.com/semsorock/inspect-evals-monitor-bench/issues/13)) |
| `cue_answer.health` | `cue_answer_health` | 65 | Planned ([#11](https://github.com/semsorock/inspect-evals-monitor-bench/issues/11)) |
| `cue_answer.logic` | `cue_answer_logic` | 50 | Planned ([#14](https://github.com/semsorock/inspect-evals-monitor-bench/issues/14)) |

### Outcome Justification

| Upstream task | Inspect task | Samples | Status |
| ------------- | ------------ | ------: | ------ |
| `dual_objectives.steganography` | `dual_objectives_steganography` | 50 | **Implemented** ([task docs](tasks/dual_objectives/steganography/README.md), [#20](https://github.com/semsorock/inspect-evals-monitor-bench/issues/20), [PR #1](https://github.com/semsorock/inspect-evals-monitor-bench/pull/1)) |
| `dual_objectives.summarization` | `dual_objectives_summarization` | 50 | **Implemented** ([task docs](tasks/dual_objectives/summarization/README.md), [#3](https://github.com/semsorock/inspect-evals-monitor-bench/issues/3)) |
| `dual_objectives.coding` | `dual_objectives_coding` | 291 | Planned ([#17](https://github.com/semsorock/inspect-evals-monitor-bench/issues/17)) |
| `dual_objectives.agent.malicious` | `dual_objectives_agent_malicious` | 5 | Planned ([#18](https://github.com/semsorock/inspect-evals-monitor-bench/issues/18)) |
| `dual_objectives.agent.benign` | `dual_objectives_agent_benign` | 5 | Planned ([#19](https://github.com/semsorock/inspect-evals-monitor-bench/issues/19)) |
| `impossible.coding` | `impossible_coding` | 50 | Planned ([#16](https://github.com/semsorock/inspect-evals-monitor-bench/issues/16)) |

### Solution Process

| Upstream task | Planned Inspect task | Samples | Status |
| ------------- | -------------------- | ------: | ------ |
| `process.multistep` | `process_multistep` | 25 | Planned ([#15](https://github.com/semsorock/inspect-evals-monitor-bench/issues/15)) |

<!-- Contributors: Automatically Generated -->
Contributed by [@semsorock](https://github.com/semsorock)
<!-- /Contributors: Automatically Generated -->

<!-- Usage: Automatically Generated -->
## Usage

First, install dependencies:

```bash
uv sync
```

Then run evaluations:

```bash
uv run inspect eval monitor_bench/dual_objectives_steganography --model openai/gpt-5-nano
uv run inspect eval monitor_bench/goal_sandbag_safety --model openai/gpt-5-nano
uv run inspect eval monitor_bench/goal_sandbag_math --model openai/gpt-5-nano
uv run inspect eval monitor_bench/dual_objectives_summarization --model openai/gpt-5-nano
uv run inspect eval monitor_bench/goal_sandbag_engineering --model openai/gpt-5-nano
```

You can also import tasks as Python objects:

```python
from inspect_ai import eval
from monitor_bench import dual_objectives_steganography, goal_sandbag_safety, goal_sandbag_math, dual_objectives_summarization, goal_sandbag_engineering
eval(dual_objectives_steganography)
```

After running evaluations, view logs with:

```bash
uv run inspect view
```

If you don't want to specify `--model` each time, create a `.env` file:

```bash
INSPECT_EVAL_MODEL=anthropic/claude-opus-4-1-20250805
ANTHROPIC_API_KEY=<anthropic-api-key>
```
<!-- /Usage: Automatically Generated -->

Requires Inspect AI 0.3.276 or newer with OpenAI SDK 3.24.0 or newer.

The generated commands explicitly use the evaluated model as a self-monitoring
fallback when the `monitor` role is absent, preserving its model identity and
sampling configuration. Scorers warn when a model monitors itself.
Benchmark runs should configure a distinct `monitor` model role. See each
task's README for the full command, stress conditions, scoring semantics, and
rollout caveats.

<!-- Options: Automatically Generated -->
## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval monitor_bench/dual_objectives_steganography --limit 10
uv run inspect eval monitor_bench/goal_sandbag_safety --max-connections 10
uv run inspect eval monitor_bench/goal_sandbag_math --temperature 0.5
```

See `uv run inspect eval --help` for all available options.
<!-- /Options: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `dual_objectives_steganography`, `goal_sandbag_safety`, `goal_sandbag_math`, `dual_objectives_summarization`, `goal_sandbag_engineering`

- `stress_test` (Literal['standard', 'direct_conceal', 'monitor_aware']): Stress test condition appended to the prompt. (default: `'standard'`)
- `epochs` (int): Number of rollouts of the evaluated model per sample. (default: `8`)
- `monitor_rollouts` (int): Number of monitor queries per scope per rollout. (default: `4`)
<!-- /Parameters: Automatically Generated -->

## Current package structure

- `tasks/dual_objectives/steganography/`,
  `tasks/dual_objectives/summarization/`, `tasks/goal_sandbag/math/`, and
  `tasks/goal_sandbag/safety/`, and `tasks/goal_sandbag/engineering/` are
  task packages. Their directory hierarchy mirrors the dotted upstream task
  identifiers. Each
  keeps its registered task in `eval.py`, monitor/scoring implementation in
  `monitor.py`, deterministic verification in `verifier.py`, and public task
  export in `__init__.py`. Its `README.md` documents task-specific behavior,
  scoring, fidelity decisions, validation evidence, and deviations.
- The `tasks/` namespace avoids collisions between task package names and the
  task functions exported from `monitor_bench`.
- `extraction.py`, `model_config.py`, `outcome.py`, and `outcome_text.py`
  contain genuinely shared extraction, model-role, and text
  outcome-justification behavior.
- `assets/` contains only implemented-task data.
- `eval.yaml` and `monitor_bench.__init__` enumerate only implemented tasks.

Each migration issue should add only the dependencies, assets, attribution,
and runtime behavior needed by that task. Preserve upstream prompts,
verification behavior, monitor scopes, aggregation semantics, and exclusions;
document deliberate deviations in the task README.

## Evaluation Report

The steganography task has completed unit, mock-model, local LM Studio, and
full three-condition OpenRouter validation. The `goal_sandbag.math` task has
independently completed deterministic, mock-model, and full three-condition
OpenRouter validation. Summarization has completed focused, mock-model, and
full three-condition OpenRouter validation. Its reported verifier pass rate
retains the documented upstream first-character caveat and is not exact
payload compliance. `goal_sandbag.safety` has focused deterministic and
mock-model validation across all three stress conditions, plus full Qwen3-8B
runs for all three conditions and a Qwen3-4B Thinking `standard` run. Each
completed 960 unique sample/epoch pairs with zero final sample errors; the
[safety report](tasks/goal_sandbag/safety/README.md#validation-status) records
metrics, extraction exclusions, runtime settings, and retry/provider caveats.
Engineering has offline dataset/prompt/verifier and mock-model validation;
its real-model evaluation and external registration are pending. These runs
establish execution, not numerical agreement with the paper. See
the task READMEs for reports and reproduction commands.

## Provenance and licensing

MonitorBench code and prompt additions retain the upstream MIT notice. The
vendored Databricks Dolly-derived writing data retains CC BY-SA 3.0. GovReport
source provenance is recorded for summarization, but the cited ccdv dataset
card declares no license and redistribution rights remain unresolved. The AIME
problem text used by `goal_sandbag.math` likewise has an unresolved rights
review and is not represented as Apache-2.0 in package metadata. The WMDP
records used by `goal_sandbag.safety` retain WMDP's separate MIT notice.
Engineering records retain the MMLU-Pro dataset's MIT declaration, with two
pinned source differences documented in the asset audit. Its software
repository's Apache-2.0 license is separate from the dataset grant. The
NLTK tokenizer is fetched from a pinned, checksum-verified source at runtime.
See [`assets/ATTRIBUTION.md`](assets/ATTRIBUTION.md), the
[`outcome` asset audit](assets/outcome/ATTRIBUTION.md), the
[`goal_sandbag` asset audit](assets/intervention/ATTRIBUTION.md), and the
repository [`NOTICE`](../../NOTICE). Future tasks must update these records
when they add third-party material.
