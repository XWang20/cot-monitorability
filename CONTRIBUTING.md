# Contributing

## Development

```bash
uv sync --locked --group dev
make check
make test
```

`make check` runs Ruff, type checks for task entry points and asset handling, and the official Inspect Evals linter. `make test` runs network-independent tests. To include dataset construction and all six mocked evaluations:

```bash
make test-datasets
```

This downloads the pinned archives first, then runs tests offline. Tests use `mockllm` and do not call model APIs. CI runs these checks and builds the package.

## Task changes

Each task lives in `src/cot_monitorability/<task>/` with its implementation, README, and `eval.yaml`. Shared components belong beside the tasks that use them. Keep dataset loading, solver behavior, scoring, and metrics independently reviewable.

Use public Inspect APIs, including Model Roles for judges and monitors. Keep endpoint credentials and proxy configuration in the environment or run configuration. Framework defects should be reported to [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai/issues); contribute fixes through that project's process rather than shipping a patched framework.

Task READMEs should describe inputs, supported arguments, score interpretation, and method limitations. Add deterministic regression tests when changing prompts, filters, parsers, scoring, or aggregation. Compare affected sample scores as well as aggregate metrics; a successful model request alone does not validate scoring.

Follow [Inspect Evals task versioning](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/TASK_VERSIONING.md). Update the task version, `eval.yaml`, and its changelog together. Changes affecting score comparability require a comparability-version bump.

## Upstream code and data

`uv.lock` pins Python dependencies. Update them deliberately with `uv lock --upgrade-package <package>` and run the checks before committing the lock.

`asset_specs.json` pins source archives by full commit and SHA-256. To update a benchmark:

1. Review upstream data, prompts, graders, and licenses at the proposed revision.
2. Update the archive revision and checksum, and the corresponding `external_assets` in `eval.yaml`.
3. Review affected code in `reference/` against its source. Preserve license headers, canaries, and attribution; update `NOTICE` when the copied source revision changes.
4. Run dataset and scoring tests. Review changed dataset fingerprints rather than replacing them without inspection. Update the task version when results cease to be comparable.

Cached revisions have separate directories. Existing results retain their recorded task and dependency versions; they should not be relabeled after an update.

Reference modules contain the subset of upstream code needed by these tasks. Changes to imports, resource locations, and Model Roles are integrated there. Update these modules through reviewed source changes and regression tests.

## Inspect Evals

[Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/CONTRIBUTING.md) uses the [Register submission workflow](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/register/README.md) for new externally maintained evaluations. The task code stays in this repository.

Submit a separate issue for each task with its methodology paper and a GitHub source URL pinned to a full commit SHA. The source file must contain the direct `@task` definition. The upstream workflow derives Register metadata; this repository maintains only the task's `eval.yaml`.

Before submission, review methodology and scoring against the cited source, run all samples on two models, inspect trajectories, and prepare the required logs. Check existing listings for overlapping tasks. NCRI's original article is cited in its README; its arXiv reference remains unresolved. Several MonitorBench variants already have upstream listings.

See the official [best practices](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/BEST_PRACTICES.md) and [evaluation checklist](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/EVALUATION_CHECKLIST.md) for review criteria.
