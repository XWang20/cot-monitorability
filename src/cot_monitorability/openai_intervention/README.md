# Intervention-level monitorability

Measures whether a monitor detects an intervention that changes a model's behavior, using [OpenAI's monitorability evaluations](https://github.com/openai/monitorability-evals) ([paper](https://arxiv.org/abs/2512.18311)).

## Usage

```bash
uv run inspect eval cot_monitorability/openai_monitorability_intervention \
  --model 'openai/<evaluated-model>' \
  --model-role 'monitor=openai/<monitor-model>' \
  -T dataset=scruples_suggest_wrong --limit 10
```

The default has 150 paired samples. Each Inspect sample contains an unperturbed and perturbed generation; sample counts are not generation counts. The monitor evaluates both arms.

## Task arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `dataset` | `scruples_suggest_wrong` | Intervention dataset or `all` |
| `scope` | `cot_only` | cot_only, answer_only, or all_messages |
| `target_reasoning_effort` | `medium` | Evaluated-model reasoning effort |
| `monitor_model` | DeepSeek V4.1 Flash-1017 | Fallback model; `--model-role monitor=...` takes precedence |

Datasets: `gpqa`, `gpqa_encoded`, `daft`, `scruples_first_person`, `scruples_suggest_right`, `scruples_suggest_wrong`, and `wmdp_sandbagging`.

## Scoring

Logs report lower and upper intervention g-mean² bounds, effect shift, eligible fraction, reportability, and mean reasoning length. Eligibility depends on positive intervention effect. `intervention_reportable=0` means fewer than 10% of instances are eligible; do not report the bound as a benchmark result in that case.

With one generation per arm, eligibility estimates can be noisy. Compare the same dataset, monitor scope, sampling configuration, and number of epochs. Inspect paired outputs and monitor verdicts alongside aggregate scores.

## Changelog

### 1-A

Initial version with paired generation and the pinned intervention metric.
