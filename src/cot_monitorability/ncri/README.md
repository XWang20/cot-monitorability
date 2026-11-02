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
| `effort` | None | Optional upstream reasoning request override |
| `limit_per_bank` | None | Limit each bank before evaluation |

The request uses upstream bank-specific output limits and prefill. Disabling reasoning is provider-dependent: the adapter retains native `thinking` fields for the supported local Kimi/DeepSeek models and an OpenAI-compatible `reasoning` field otherwise. Verify that the selected backend honors the request. A provider that ignores it cannot establish a no-CoT result.

`all_ncri` selects banks whose files are available in the cache. The optional GPQA bank requires accepted Hugging Face dataset terms and `HF_TOKEN`. Its fetcher pins revision `83022cefff930aea54f654c0b282e74b9eeda5c6` and verifies rendered item hashes:

```bash
uv run python -m cot_monitorability.reference.nocot.fetch_gpqa
```

Check which banks were loaded before comparing `all_ncri` runs.

## Scoring

Logs record `ncri_primary_accuracy`, `ncri_valid_rate`, and the frozen `ncri_placement_score`. Placement aggregates scorable, non-annexed items by rung and uses the upstream calibration. It returns zero when there are no placement counts or calibration fails.

A placement score from a single bank or a limited run is not a full-index estimate. Compare bank coverage, prompting, reasoning settings, and generation validity.

## Changelog

### 3-A

Count cached tokens as input when checking for hidden reasoning. Earlier versions could incorrectly reject cached responses as deliberation.

For compatible backends that ignore the native thinking field, explicitly set the evaluated model's reasoning effort to none with the Inspect CLI.


### 2-A

Preserve configured provider fields when building no-CoT requests. Version 1-A could omit proxy compatibility settings and produce reasoning despite a disable request.


### 1-A

Initial frozen-bank evaluation with upstream witnesses and placement.
