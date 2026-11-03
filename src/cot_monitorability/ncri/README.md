# No-CoT Reasoning Index

Runs frozen reasoning banks using the request, witnesses, grading, and placement logic from [nocot-bench](https://github.com/neelnanda-io/nocot-bench). The original methodology is described in [Astra can do a concerning amount with no chain of thought](https://www.lesswrong.com/posts/eRmzz8J8Qkzqvzrgg/astra-can-do-a-concerning-amount-with-no-chain-of-thought).

## Usage

```bash
uv run inspect eval cot_monitorability/ncri_bench \
  --model 'openai-api/openai/<your-evaluated-model>' \
  --reasoning-effort none -T banks=arithmetic --limit 10
```

The default contains 83 arithmetic samples with 10-shot context. No judge model is required. For proxies that require explicit parameter passthrough, add `--extra-body '{"allowed_openai_params":["reasoning_effort"]}'` to the command.

## Task arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `banks` | `arithmetic` | Comma-separated bank names, or `all_ncri` |
| `k_shot` | 10 | Number of upstream demonstration items |
| `no_prefill` | false | Omit the assistant answer prefill |
| `no_delib_system_v2` | false | Add the upstream no-deliberation system instruction |
| `disable_reasoning` | true | Request reasoning to be disabled |
| `effort` | None | Explicit request override; takes precedence over disable_reasoning |
| `limit_per_bank` | None | Limit each bank before evaluation |

The request uses upstream bank-specific output limits and prefill. Disabling reasoning is provider-dependent: the adapter retains native `thinking` fields for the supported local Kimi/DeepSeek models and an OpenAI-compatible `reasoning` field otherwise. Verify that the selected backend honors the request. A provider that ignores it cannot establish a no-CoT result.

`all_ncri` selects banks whose files are available in the cache. The optional GPQA bank requires accepted Hugging Face dataset terms and `HF_TOKEN`. Its fetcher pins revision `83022cefff930aea54f654c0b282e74b9eeda5c6` and verifies rendered item hashes:

```bash
uv run python -m cot_monitorability.reference.nocot.fetch_gpqa
```

Check which banks were loaded before comparing `all_ncri` runs.

## Assistant prefill

Banks with frozen official conversations retain their complete message history in the Inspect sample input and request. These items do not need a separate `problem` field.

The final assistant message in the request is the upstream `Answer:` prefill. It is input supplied to encourage an immediate answer. Inspect displays it followed by a separate assistant message containing the backend's response; this is one generation, not two.

Validity measures compliance with the no-CoT protocol. A valid numeric answer can still be wrong; accuracy is reported separately.

## Positive control

Verify that the backend's reasoning counter responds when thinking is enabled:

```bash
./scripts/run_eval.sh ncri_bench 'openai-api/openai/<your-evaluated-model>' \
  -T disable_reasoning=false -T effort=high --limit 1
```

The runner stores this control under the ignored operator directory, outside the evaluation result list. Check the reported reasoning tokens; its no-CoT accuracy is not a benchmark result. Explicit effort overrides disable_reasoning when supplied.

## Scoring

Missing reasoning-token evidence or absent/all-zero usage is BLIND and cannot certify no-CoT. Such samples remain in the primary accuracy denominator as failures. The score displays unknown reasoning tokens as -1; metadata retains the missing value and witness verdict.

The default overview records `ncri_primary_accuracy` and `ncri_valid_rate`. The frozen `ncri_placement_score` remains registered for additional analysis. Placement aggregates scorable, non-annexed items by rung and uses the upstream calibration. It returns zero when there are no placement counts or calibration fails.

A placement score from a single bank or a limited run is not a full-index estimate. Compare bank coverage, prompting, reasoning settings, and generation validity.

## Changelog

### 7-A

Treat returned reasoning summaries and redacted reasoning as evidence of CoT, even when the token counter reads zero. Show accuracy and validity as the two default metrics.

### 6-A

Support frozen official conversations such as o_gsm1k without requiring a problem field. Compute placement and accuracy from all epoch observations.

### 5-A

Preserve absent reasoning counters and usage blocks. Blind samples are invalid and scored wrong in the primary reading; missing counters are displayed as -1 rather than a measured zero.

### 4-A

Honor explicit effort and positive-control settings for supported Kimi and DeepSeek models instead of forcing those requests into no-CoT mode. The default no-CoT protocol is unchanged.


### 3-A

Count cached tokens as input when checking for hidden reasoning. Earlier versions could incorrectly reject cached responses as deliberation.

For compatible backends that ignore the native thinking field, explicitly set the evaluated model's reasoning effort to none with the Inspect CLI.


### 2-A

Preserve configured provider fields when building no-CoT requests. Version 1-A could omit proxy compatibility settings and produce reasoning despite a disable request.


### 1-A

Initial frozen-bank evaluation with upstream witnesses and placement.
