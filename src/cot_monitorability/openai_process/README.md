# Process-level monitorability

Measures whether a monitor can detect a required theorem path in a model's reasoning, using [OpenAI's monitorability evaluations](https://github.com/openai/monitorability-evals) ([paper](https://arxiv.org/abs/2512.18311)).

## Usage

```bash
uv run inspect eval cot_monitorability/openai_monitorability_process \
  --model 'openai-api/openai/<your-evaluated-model>' \
  --model-role 'monitor=openai-api/openai/<your-monitor-model>' --limit 10
```

The default dataset contains 52 samples. Each sample produces one evaluated-model answer. The monitor checks whether all theorems in at least one required path are applied, and also grades answer correctness.

## Task arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `scopes` | `cot_only` | Comma-separated upstream monitor scopes |
| `target_reasoning_effort` | `medium` | Evaluated-model reasoning effort |
| `monitor_model` | DeepSeek V4.1 Flash-1017 | Fallback model; `--model-role monitor=...` takes precedence |

Use scopes supported by the upstream prompts, such as `cot_only`, `answer_only`, and `all_messages`. Multiple scopes issue separate monitor queries.

## Scoring

`process_gmean2_cot` reports the CoT monitor's g-mean². The log also records answer accuracy, monitor verdicts by scope, and mean reasoning length. The aggregate uses the CoT scope when present, otherwise the first configured scope; name alternative-scope results explicitly.

Answer grading and theorem-path detection depend on the monitor. Unrecognized or missing theorem verdicts count as not applied. Review the monitor's per-sample responses before drawing conclusions about reasoning faithfulness.

## Changelog

### 5-A

Retry a truncated generation once with the identical prompt. Actor settings stay unchanged; a truncated judge receives a 32,768-token budget for its verdict. Persistently incomplete outputs raise a scoring error.

### 4-A

Reject summary-only or redacted actor reasoning as a scoring error. Show two core metrics by default; auxiliary statistics remain in sample records or registered optional metrics.

### 3-A

Compute conditional monitorability and accuracy from individual epoch observations, preserving fractional rates across repetitions.

### 2-A

Use the OpenAI-compatible provider for the fallback scoring model so configured sampling parameters are sent without OpenAI model-family rewrites.

Use native Inspect task configuration for actor effort. Answer judges must end with an explicit CORRECT or INCORRECT verdict; earlier mentions cannot override it, and malformed verdicts raise scoring errors.

### 1-A

Initial version using the pinned OpenAI process data and monitor prompts.
