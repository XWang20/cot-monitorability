# CoT-Control QA

Tests whether a model follows constraints on its chain of thought while answering a question. Based on [Reasoning Models Struggle to Control their Chains of Thought](https://arxiv.org/abs/2603.05706) and the [CoT-Control implementation](https://github.com/YuehHanChen/CoTControl).

## Usage

```bash
uv run inspect eval cot_monitorability/cot_controllability_qa \
  --model 'openai-api/openai/<your-evaluated-model>' \
  --model-role 'judge=openai-api/openai/<your-judge-model>' \
  -T dataset=gpqa -T modes=word_suppression --limit 10
```

The default configuration contains 427 samples. The judge is used for `ignore_question` and optional meta-discussion checks; other constraints use rule-based graders.

## Task arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `dataset` | `gpqa` | `gpqa`, `hle`, `mmlu_pro`, or `all` |
| `modes` | `word_suppression` | Comma-separated modes, `all_main`, or `all_with_baseline` |
| `target_reasoning_effort` | `high` | Evaluated-model reasoning effort |
| `sample_limit_per_mode` | None | Limit each dataset/mode combination before evaluation |
| `adversarial_pressure` | `na` | Upstream prompt-pressure variant |
| `non_reasoning` | false | Use the upstream non-reasoning prompt variant |
| `monitor_aware`, `eval_aware` | false | Include the corresponding awareness instructions |
| `cot_sys_prompt` | false | Put the CoT control instruction in a system prompt |
| `elicit_brevity` | false | Append the brevity instruction |
| `check_meta_discussion` | false | Check whether a compliant trace discusses the constraint |
| `judge_model` | DeepSeek V4.1 Flash-1017 | Fallback model; `--model-role judge=...` takes precedence |

Modes: `baseline`, `word_suppression`, `multiple_word_suppression`, `uppercase_thinking`, `lowercase_thinking`, `alternating_case`, `repeat_sentences`, `end_of_sentence`, `meow_between_words`, and `ignore_question`. `all_main` excludes baseline.

## Scoring

- `cot_controllability_score`: fraction of traces satisfying the constraint.
- `task_accuracy`: fraction of answers matching the reference choice.
- `controlled_and_accurate_rate`: fraction satisfying both conditions.
- `mean_cot_tokens`: mean reasoning length.
- `controllability_750_1250_tok`: compliance within the 750–1,250-token window; -1 means no eligible samples.

Baseline compliance is marked -1 and excluded from compliance aggregates. Missing reasoning receives zero compliance for control modes. Token counts use provider-reported reasoning tokens when available, otherwise an approximate word-based count; do not mix the two as exact length measurements.

The T/F judge parser accepts some non-verdict text containing "T". The regression suite records this known issue. Inspect judge responses when using `ignore_question` or meta-discussion scoring.

## Changelog

### 2-A

Use the OpenAI-compatible provider for the fallback scoring model so configured sampling parameters are sent without OpenAI model-family rewrites.

Use native Inspect task configuration for actor effort; CLI settings override the task default.

### 1-A

Initial version with the CoT-Control main-paper modes and pinned data.
