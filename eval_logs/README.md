# Evaluation results

Index of 14 stored runs. Raw `.eval` files are kept by the operator; this directory versions their metadata and checksums.

Interpret each run using its recorded task namespace, version, configuration, and package revision. Limited runs do not establish full-dataset performance.

## Runs

| Task / subset | Version | Model | Effort | Samples | Status | Main metrics |
| --- | --- | --- | --- | --- | --- | --- |
| inspect_evals/cot_controllability_qa / gpqa/word_suppression | 0 | deepseek-v4.1-flash-1017 | high | 10/10 | success | cot_controllability_score=0.1 |
| inspect_evals/cot_controllability_qa / gpqa/word_suppression | 0 | kimi-k3.1-pro | high | 10/10 | success | cot_controllability_score=0.2 |
| inspect_evals/openai_monitorability_process / cot_only | 0 | deepseek-v4.1-flash-1017 | high | 10/10 | success | process_gmean2_cot=0.875 |
| inspect_evals/openai_monitorability_process / cot_only | 0 | kimi-k3.1-pro | high | 10/10 | success | process_gmean2_cot=0.875 |
| inspect_evals/openai_monitorability_intervention / gpqa | 0 | deepseek-v4.1-flash-1017 | high | 10/10 | success | Not reportable (eligible fraction below 10%) |
| inspect_evals/openai_monitorability_intervention / gpqa | 0 | kimi-k3.1-pro | high | 10/10 | success | intervention_gmean2_lower=0.5 |
| inspect_evals/openai_monitorability_intervention / scruples_suggest_wrong | 0 | deepseek-v4.1-flash-1017 | high | 10/10 | success | intervention_gmean2_lower=0.5 |
| inspect_evals/openai_monitorability_intervention / scruples_suggest_wrong | 0 | kimi-k3.1-pro | high | 10/10 | success | intervention_gmean2_lower=0.3333 |
| inspect_evals/cot_immediacy_2510 / gpqa | 0 | deepseek-v4.1-flash-1017 | high | 10/10 | success | legibility_pct=92.5; coverage_pct=97.5 |
| inspect_evals/cot_immediacy_2510 / gpqa | 0 | kimi-k3.1-pro | high | 10/10 | success | legibility_pct=85; coverage_pct=87.5 |
| inspect_evals/ncri_bench / arithmetic | 0 | deepseek-v4.1-flash-1017 | disabled | 10/10 | success | ncri_placement_score=105.1 |
| inspect_evals/ncri_bench / arithmetic | 0 | kimi-k3.1-pro | disabled | 10/10 | success | ncri_placement_score=132.5 |
| inspect_evals/monitorbench / cue_condition.preference | 0 | deepseek-v4.1-flash-1017 | high | 10/10 | success | mb_cot_monitorability=1 |
| inspect_evals/monitorbench / cue_condition.preference | 0 | kimi-k3.1-pro | high | 10/10 | success | mb_cot_monitorability=1 |

## Interpretation

- Counts refer to Inspect samples. Intervention samples contain paired generations.
- Runs, subsets, and repetitions are listed separately; their samples are not pooled.
- Effort comes from stored generation configuration or explicit no-CoT task settings.
- Metrics measure different quantities across tasks. Do not combine them into a ranking.
- A -1 value can denote an empty eligible window rather than a negative score.
- Some historical aggregate metrics were reconstructed from stored sample scores on 2026-10-30. The indexer reads headers and does not modify logs.

[index.json](index.json) contains file paths, SHA-256, task and package versions, timing, model usage, task arguments, and all recorded metrics.

## Inspecting and indexing logs

Run from the repository root:

```bash
uv run inspect view --log-dir eval_logs
python3 scripts/index_eval_logs.py
```

The indexer refuses to replace the index when no local logs are present.
