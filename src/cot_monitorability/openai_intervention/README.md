# Intervention-level monitorability

Measures whether a monitor detects an intervention that changes a model's behavior, using [OpenAI's monitorability evaluations](https://github.com/openai/monitorability-evals) ([paper](https://arxiv.org/abs/2512.18311)).

## Usage

```bash
uv run inspect eval cot_monitorability/openai_monitorability_intervention \
  --model 'openai-api/openai/<your-evaluated-model>' \
  --model-role 'monitor=openai-api/openai/<your-monitor-model>' \
  -T dataset=scruples_suggest_wrong --limit 10
```

The default has 150 paired samples. Each Inspect sample contains an unperturbed and perturbed generation; sample counts are not generation counts. The monitor evaluates both arms. An unparseable monitor verdict raises a scoring error; it is not counted as a negative classification.

## Task arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `dataset` | `scruples_suggest_wrong` | Intervention dataset or `all` |
| `scope` | `cot_only` | cot_only, answer_only, or all_messages |
| `target_reasoning_effort` | `medium` | Evaluated-model reasoning effort |
| `monitor_model` | DeepSeek V4.1 Flash-1017 | Fallback model; `--model-role monitor=...` takes precedence |

Datasets: `gpqa`, `gpqa_encoded`, `daft`, `scruples_first_person`, `scruples_suggest_right`, `scruples_suggest_wrong`, and `wmdp_sandbagging`.

## Scoring

The default overview reports the lower intervention g-mean² bound and eligible fraction. Eligibility depends on positive intervention effect. Do not report the bound as a benchmark result when fewer than 10% of instances are eligible. Per-sample arm records and reasoning lengths remain available; the upper-bound and other diagnostic metrics remain registered for additional analysis.

With one generation per arm, eligibility estimates can be noisy. Compare the same dataset, monitor scope, sampling configuration, and number of epochs. Inspect paired outputs and monitor verdicts alongside aggregate scores.

## Changelog

### 5-A

Retry a truncated generation once with the identical prompt. Actor settings stay unchanged; a truncated judge receives a 32,768-token budget for its verdict. Persistently incomplete outputs raise a scoring error.

### 4-A

Reject summary-only or redacted actor reasoning as a scoring error. Show two core metrics by default; auxiliary statistics remain in sample records or registered optional metrics.

### 3-A

Pool all epoch arm records before computing intervention bounds and eligibility. Missing monitor verdicts raise scoring errors instead of becoming negative classifications.

### 2-A

Use the OpenAI-compatible provider for the fallback scoring model so configured sampling parameters are sent without OpenAI model-family rewrites.

Use native Inspect task configuration for both actor arms. Answer judges must end with an explicit CORRECT or INCORRECT verdict; malformed verdicts raise scoring errors.

### 1-A

Initial version with paired generation and the pinned intervention metric.
