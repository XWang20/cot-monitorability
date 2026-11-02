# NCRI: configurable frozen no-CoT reasoning banks

Runs the pinned nocot-bench request, witnesses, grading and frozen placement logic. The default bank is arithmetic; banks=all_ncri selects the configured NCRI bank set.

## Usage

Install the project with `uv sync`. The first task construction fetches its pinned source assets into the package cache. Set `COT_MONITORABILITY_CACHE_DIR` before importing tasks to select another absolute cache directory.

```bash
uv run inspect eval cot_monitorability/ncri_bench \
  --model openai/<evaluated-model> --limit 10
```

The same task can be selected by its source file:

```bash
uv run inspect eval src/cot_monitorability/ncri/ncri.py@ncri_bench --model openai/<evaluated-model> --limit 10
```

## Configuration and data

Task version: `1-A`. Default dataset size after filtering: 83 samples. Evaluated-model effort: `disabled`. Task arguments allow the documented dataset and generation variants.

All source revisions and archive SHA-256 values are in `src/cot_monitorability/asset_specs.json`. Reference implementations and licence notices are retained in the package and root NOTICE. Cached files are verified before reuse. The source archives are downloaded before model calls.

## Validation status

The publication format is tested with the public Inspect SDK. Dataset inputs, choices, targets and IDs are checked against the previous integration at the pinned assets. The existing 14 model-run logs belong to the earlier integration and are not claimed as full runs of this task version.

Human methodology review, published-baseline comparison, and two-model full-run logs remain separate requirements before an actual registration. This packaging change does not claim that the transferred scorers have completed those reviews.

## Changelog

### 1-A

Initial standalone task format. Runtime assets are pinned and task sources are directly available in this repository.

NCRI has no verified arXiv URL recorded here. A suitable methodology or replication paper URL must be supplied before submitting its registration issue.

## Original methodology

[Astra can do a concerning amount with no chain of thought](https://www.lesswrong.com/posts/eRmzz8J8Qkzqvzrgg/astra-can-do-a-concerning-amount-with-no-chain-of-thought). This is the source article provided by the user; it is recorded separately from the pending arXiv field.

The optional GPQA bank uses Hugging Face revision `83022cefff930aea54f654c0b282e74b9eeda5c6`, requires `HF_TOKEN` and accepted dataset terms, and validates rendered item hashes. It has not been downloaded or exercised in this format-only run.
