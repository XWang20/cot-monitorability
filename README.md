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

| Task | Default configuration | Samples | Documentation |
| --- | --- | ---: | --- |
| `cot_controllability_qa` | GPQA, word suppression | 427 | [CoT-Control](src/cot_monitorability/cot_control/README.md) |
| `openai_monitorability_process` | Theorem detection, CoT only | 52 | [Process](src/cot_monitorability/openai_process/README.md) |
| `openai_monitorability_intervention` | Scruples, suggest wrong | 150 pairs | [Intervention](src/cot_monitorability/openai_intervention/README.md) |
| `cot_immediacy_2510` | GPQA, all answers | 445 | [Immediacy](src/cot_monitorability/immediacy/README.md) |
| `ncri_bench` | Arithmetic, 10-shot, reasoning disabled | 83 | [NCRI](src/cot_monitorability/ncri/README.md) |
| `monitorbench` | Preference, standard condition | 108 pairs | [MonitorBench](src/cot_monitorability/monitorbench/README.md) |

Counts are for the default configuration. Dataset, mode, and subset arguments can change them.

## Running an evaluation

Configure credentials for your [model provider](https://inspect.aisi.org.uk/models.html). Specify the evaluated model with `--model` and the scoring model with `--model-role judge=...` or `--model-role monitor=...`.

```bash
uv run inspect eval cot_monitorability/cot_controllability_qa \
  --model 'openai/<evaluated-model>' \
  --model-role 'judge=openai/<judge-model>' \
  --limit 10 --log-dir eval_logs/cot_control
```

Task arguments use `-T`:

```bash
uv run inspect eval cot_monitorability/openai_monitorability_intervention \
  --model 'openai/<evaluated-model>' \
  --model-role 'monitor=openai/<monitor-model>' \
  -T dataset=gpqa -T target_reasoning_effort=medium \
  --limit 10 --log-dir eval_logs/intervention
```

Omit `--limit` for all samples. Use `--max-connections` to control concurrency. Tasks also run directly from source, for example `inspect eval src/cot_monitorability/ncri/ncri.py@ncri_bench`.

CoT tasks need a provider that exposes reasoning text. Hidden reasoning produces empty traces and changes the interpretation of monitorability scores. Select judge and monitor roles explicitly when comparing models; their prompts and sampling configuration affect the results.

### Local OpenAI-compatible API

The convenience script configures the local proxy through public Inspect CLI options:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=a  # Replace if the proxy requires authentication.
export JUDGE_MODEL='openai/<judge-or-monitor-model>'

./scripts/run_eval.sh cot_controllability_qa 'openai/<evaluated-model>' \
  --limit 10 --max-connections 5
```

It takes a task name, model name, then standard Inspect arguments. Default effort comes from the task signature; `REASONING_EFFORT=low` overrides it for CoT tasks. NCRI retains its no-CoT request. The script supplies LiteLLM compatibility fields for localhost:8000; `COT_USE_LITELLM_COMPAT=0` disables them for other compatible services.

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
uv run inspect view --log-dir eval_logs
```

On a remote server, forward the View port through SSH. The [results index](eval_logs/README.md) lists stored runs and their configurations; [index.json](eval_logs/index.json) includes metrics and file checksums. Raw logs are kept by the operator and ignored by Git. Rebuild the index after a run with `python3 scripts/index_eval_logs.py`.

The indexed runs use 10-sample subsets of an earlier implementation. They are useful for inspecting those runs, but do not establish full-dataset performance for the current tasks. Scores from different benchmarks measure different quantities and should be reported separately.

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
