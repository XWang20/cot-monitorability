# CoT Monitorability

Inspect AI tasks for studying chain-of-thought control, monitorability, and reasoning without CoT.

## Installation

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/XWang20/cot-monitorability.git
cd cot-monitorability
uv sync --locked
```

The package uses the public [Inspect AI](https://inspect.aisi.org.uk/) SDK. Installation with `pip install .` is also supported.

## Evaluations

Each task's README describes its method, parameters, scoring, and limitations.

| Task | Default configuration | Actor effort | Judge / monitor effort | Samples | Documentation |
| --- | --- | --- | --- | ---: | --- |
| `cot_controllability_qa` | GPQA, word suppression | high | high | 427 | [CoT-Control](src/cot_monitorability/cot_control/README.md) |
| `openai_monitorability_process` | Theorem detection, CoT only | medium | high | 52 | [Process](src/cot_monitorability/openai_process/README.md) |
| `openai_monitorability_intervention` | Scruples, suggest wrong | medium | high | 150 pairs | [Intervention](src/cot_monitorability/openai_intervention/README.md) |
| `cot_immediacy_2510` | GPQA, all answers | high | high | 445 | [Immediacy](src/cot_monitorability/immediacy/README.md) |
| `ncri_bench` | Arithmetic, 10-shot, reasoning disabled | none | — | 83 | [NCRI](src/cot_monitorability/ncri/README.md) |
| `monitorbench` | Preference, standard condition | high | high | 108 pairs | [MonitorBench](src/cot_monitorability/monitorbench/README.md) |

Counts are for the default configuration. Dataset, mode, and subset arguments can change them. Effort names are backend controls, not fixed token budgets; compare runs using their recorded reasoning lengths.

## Running an evaluation

Configure credentials for your [model provider](https://inspect.aisi.org.uk/models.html). Specify the evaluated model with `--model` and the scoring model with `--model-role judge=...` or `--model-role monitor=...`.

```bash
uv run inspect eval cot_monitorability/cot_controllability_qa \
  --model 'openai-api/openai/<your-evaluated-model>' \
  --model-role 'judge=openai-api/openai/<your-judge-model>' \
  --limit 10 --log-dir eval_logs/cot_control
```

Task arguments use `-T`:

```bash
uv run inspect eval cot_monitorability/openai_monitorability_intervention \
  --model 'openai-api/openai/<your-evaluated-model>' \
  --model-role 'monitor=openai-api/openai/<your-monitor-model>' \
  -T dataset=gpqa -T target_reasoning_effort=medium \
  --limit 10 --log-dir eval_logs/intervention
```

Omit `--limit` for all samples. Use `--max-connections` to control concurrency. For repetitions, use `--epochs`. The package metrics consume individual epoch observations: conditional filters and token windows apply before aggregation, and intervention bounds pool all recorded arms. Delegated MonitorBench tasks retain their upstream pooling logic. Tasks also run directly from source, for example `inspect eval src/cot_monitorability/ncri/ncri.py@ncri_bench`.

CoT tasks need a provider that exposes reasoning text. Hidden reasoning produces empty traces and changes the interpretation of monitorability scores. Select judge and monitor roles explicitly when comparing models; their prompts and sampling configuration affect the results.

### OpenAI-compatible API

Use Inspect's native `openai-api` provider for compatible endpoints. The `openai` service prefix below reads the `OPENAI_*` environment variables:

```bash
export OPENAI_BASE_URL="<your-api-base-url>"
export OPENAI_API_KEY="<your-api-key>"
export JUDGE_MODEL='openai-api/openai/<your-judge-or-monitor-model>'

./scripts/run_eval.sh cot_controllability_qa 'openai-api/openai/<your-evaluated-model>' \
  --limit 10 --max-connections 5
```

Replace the placeholders with your own configuration. Credentials are read from the environment and should not be committed.

For custom OpenAI-compatible endpoints, use `openai-api/openai/<your-model>` for both the evaluated model and scoring roles. The `openai/` provider applies OpenAI-specific model-family rules that can remove sampling parameters for custom aliases.

The script takes a task name, model name, then standard Inspect arguments. Use `--reasoning-effort` to change the actor setting; the runner synchronizes the task argument and proxy configuration while keeping scoring roles at high. Task defaults are stored in native Inspect generation configuration; direct `inspect eval --reasoning-effort low` overrides them. The task argument sets a default, while explicit CLI configuration takes precedence. `REASONING_EFFORT=low` overrides the runner default for CoT tasks. NCRI retains its no-CoT request. Set `COT_USE_LITELLM_COMPAT=1` for proxies that require LiteLLM compatibility fields.

To compare effort levels on the same task:

```bash
for effort in low medium high; do
  ./scripts/run_eval.sh cot_controllability_qa 'openai-api/openai/<your-evaluated-model>' \
    --reasoning-effort "$effort" --limit 5
done
```

The result index keeps effort levels separate. NCRI's default run stays at none; positive controls are configured separately and stored outside the result directory.

## Data

The first task construction downloads the required source archives. [asset_specs.json](src/cot_monitorability/asset_specs.json) records immutable upstream commits and archive SHA-256 values. Archives and cached files are verified before use.

```bash
# Download all required archives in advance.
uv run python -m cot_monitorability.assets

# Prevent source archive downloads after prefetching.
export COT_MONITORABILITY_OFFLINE=1
```

Set `COT_MONITORABILITY_CACHE_DIR` to an absolute path before importing tasks to select a different cache. Optional gated datasets have additional requirements described in the task documentation.

## Results

Inspect writes prompts, model responses, sample scores, and aggregate metrics to `.eval` files. Open them with:

```bash
uv run inspect view --recursive --log-dir eval_logs
```

On a remote server, forward the View port through SSH. The [results index](eval_logs/README.md) lists stored runs and their configurations; [index.json](eval_logs/index.json) includes metrics and file checksums. Raw logs are kept by the operator and ignored by Git. Rebuild the index after a run with `uv run python scripts/index_eval_logs.py`.

Limited runs check the evaluation pipeline. Report model performance with the dataset, task version, configuration, and sample count. Scores from different benchmarks measure different quantities and should be reported separately.

### Runs and retries

Inspect View groups logs by task. A `.eval` file records one task/model/configuration evaluation, including its samples, epochs, and scoring-model calls. One `inspect eval` invocation can produce several files when it evaluates multiple tasks or models. Independent reruns create new files.

For a resumable task/model matrix, use native `inspect eval-set` with a dedicated experiment log directory. Reuse the same directory and plan to skip completed evaluations and recover interrupted ones. `inspect eval-retry "<your-failed-log>.eval"` retries a trusted failed log and reuses eligible completed samples. Changes to prompts, data, effort, or model settings should be recorded as a new experiment; scoring-only changes can reuse outputs through `inspect score`.

For one benchmark and two evaluated models:

```bash
uv run inspect eval-set cot_monitorability/cot_controllability_qa \
  --model 'openai-api/openai/<your-model-a>,openai-api/openai/<your-model-b>' \
  --model-role 'judge=openai-api/openai/<your-judge-model>' \
  --limit 5 --retry-attempts 2 --log-dir 'eval_logs/<your-experiment-id>'
```

The repository currently groups directories by benchmark and indexes the latest completed run for each task/model/effort. Inspect's `run_id` groups an invocation, `task_id` identifies its logical task, and `eval_id` distinguishes attempts; eval sets also record an `eval_set_id`. Keep these identities and original logs when comparing retries.

## Repository layout

```text
src/cot_monitorability/  Task implementations, metadata, and reference code
tests/                  Scoring, asset integrity, and mocked evaluation tests
scripts/                Local API runner and result indexing
eval_logs/              Result index; raw logs are ignored
licenses/               Third-party license and attribution notices
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development, dependency updates, and the Inspect Evals submission process.

## License

Integration code is MIT licensed. Included reference code has MIT and Apache-2.0 terms; dataset licenses and canaries remain applicable. [NOTICE](NOTICE) identifies sources and modifications, and [licenses/](licenses/) contains their license texts.
