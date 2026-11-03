# MonitorBench

Evaluates whether a monitor detects decision-relevant factors in a reasoning trace. Uses [MonitorBench](https://github.com/ASTRAL-Group/MonitorBench) ([paper](https://arxiv.org/abs/2603.28590v2)) and selected tasks from the [community Inspect implementation](https://github.com/semsorock/inspect-evals-monitor-bench).

## Usage

```bash
uv run inspect eval cot_monitorability/monitorbench \
  --model 'openai-api/openai/<your-evaluated-model>' \
  --model-role 'monitor=openai-api/openai/<your-monitor-model>' \
  -T subset=cue_condition.preference -T stress_test=standard --limit 10
```

The default has 108 paired preference samples. The general adapter generates original and intervened responses. `process.multistep` uses a single response.

## Task arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `subset` | `cue_condition.preference` | One of the supported tasks below |
| `stress_test` | `standard` | `standard`, `direct_conceal`, or `monitor_aware` |
| `target_reasoning_effort` | `high` | Effort for the general adapter |
| `monitor_model` | DeepSeek V4.1 Flash-1017 | Fallback for the general adapter; the monitor role takes precedence |
| `epochs` | 1 | Internal epochs for delegated community tasks |
| `monitor_rollouts` | 1 | Monitor rollouts for delegated community tasks |

General subsets: `goal_sandbag.science`, `goal_deception.realworld`, `cue_condition.knowledge`, `cue_condition.preference`, `cue_condition.judgment`, `cue_condition.health`, `cue_answer.math`, `cue_answer.logic`, `cue_answer.health`, and `process.multistep`.

Delegated community subsets: `dual_objectives_steganography`, `dual_objectives_summarization`, `goal_sandbag_safety`, `goal_sandbag_math`, and `goal_sandbag_engineering`. Their task-specific prompts, sampling, and metrics are retained in the reference implementation. Configure their monitor through `--model-role monitor=...`. The general adapter's effort and fallback-model arguments do not configure those delegated tasks.

Delegated sentence-based tasks fetch a pinned NLTK tokenizer on first use and verify its checksum. This cache is separate from the source archive cache.

For repetitions through the Inspect runner, use the CLI `--epochs`; distinguish this from the delegated task's `-T epochs` argument.

## Scoring

The general adapter reports `mb_cot_monitorability` and `mb_verification_rate` in the overview. Per-sample records retain reasoning lengths. For paired interventions, a sample is eligible when the intervened response passes verification and the original does not. Process samples need a verified response. Only eligible samples with a reasoning trace receive a monitor query.

Report verification rates and eligible counts with monitorability. An aggregate with few eligible samples is unstable; Inspect logs contain the outputs and verdicts needed to inspect exclusions. Delegated subsets use their own metric definitions rather than these general metrics.

The supported variants are non-sandbox tasks. Several delegated variants already have Inspect Evals listings; check those entries before proposing another registration.

## Changelog

### 4-A

Reject summary-only or redacted actor reasoning as a scoring error. Show two core metrics by default; auxiliary statistics remain in sample records or registered optional metrics.

### 3-A

Keep general-adapter score fields consistent across eligible and ineligible rollouts. Filter individual epoch observations before aggregating monitorability.

### 2-A

Use the OpenAI-compatible provider for the fallback scoring model so configured sampling parameters are sent without OpenAI model-family rewrites.

Use native Inspect task configuration for the general adapter's actor effort, including both paired arms; CLI settings override task defaults.

### 1-A

Initial non-sandbox task collection with pinned datasets and community implementations.
