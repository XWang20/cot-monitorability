# CoT immediacy: legibility and coverage

Generates baseline QA reasoning and applies the current legibility and coverage judge based on the DeepMind monitorability paper. The default dataset is GPQA and correct_only is false; these settings are configurable.

## Usage

Install the project with `uv sync`. The first task construction fetches its pinned source assets into the package cache. Set `COT_MONITORABILITY_CACHE_DIR` before importing tasks to select another absolute cache directory.

```bash
uv run inspect eval cot_monitorability/cot_immediacy_2510 \
  --model openai/<evaluated-model> --limit 10 \
  --model-role judge=openai/<judge-or-monitor-model>
```

The same task can be selected by its source file:

```bash
uv run inspect eval src/cot_monitorability/immediacy/immediacy.py@cot_immediacy_2510 --model openai/<evaluated-model> --limit 10
```

## Configuration and data

Task version: `1-A`. Default dataset size after filtering: 445 samples. Evaluated-model effort: `high`. Task arguments allow the documented dataset and generation variants.

All source revisions and archive SHA-256 values are in `src/cot_monitorability/asset_specs.json`. Reference implementations and licence notices are retained in the package and root NOTICE. Cached files are verified before reuse. The source archives are downloaded before model calls.

## Validation status

The publication format is tested with the public Inspect SDK. Dataset inputs, choices, targets and IDs are checked against the previous integration at the pinned assets. The existing 14 model-run logs belong to the earlier integration and are not claimed as full runs of this task version.

Human methodology review, published-baseline comparison, and two-model full-run logs remain separate requirements before an actual registration. This packaging change does not claim that the transferred scorers have completed those reviews.

## Changelog

### 1-A

Initial standalone task format. Runtime assets are pinned and task sources are directly available in this repository.

The default `correct_only=false`, prompt wording, range handling, and invalid-judge handling require comparison with Appendix C before reporting paper-aligned results.
