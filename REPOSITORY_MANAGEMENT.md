# Package and environment management

The primary deliverable is the standalone cot-monitorability Python package under src/cot_monitorability. It installs with uv sync --locked and runs against the public Inspect SDK. uv.lock fixes the tested dependencies.

On this server, UV_PROJECT_ENVIRONMENT=.venv-package preserves the pre-existing .venv and Inspect View process. On a new checkout, uv uses .venv normally. The launcher prefers .venv-package when it exists; INSPECT_BIN can override it.

Task assets are immutable upstream tar archives with commit and SHA-256 records in asset_specs.json. They are prepared in the platform cache, not in the old repos directory. COT_MONITORABILITY_CACHE_DIR selects an absolute cache before task imports; COT_MONITORABILITY_OFFLINE=1 rejects cache misses. The asset helper verifies archives and cached files, rejects unsafe archive paths, and retains original licences and canaries.

The optional NCRI GPQA data source is pinned separately in the task metadata. It requires accepted dataset terms and HF_TOKEN; its rendered-item hashes are checked by the upstream reference fetcher. That gated branch was not downloaded or evaluated during this format migration.

The old dependencies.lock.json, patches/ and independent clones remain for reproducing the legacy server integration. They are not installation prerequisites for the standalone package. The legacy dependency helper can still check or restore that environment.

Raw eval_logs/*.eval remain on the server. The repository versions only its historical result index. The format tests and mock runs do not rewrite those logs or claim full model-performance results for task version 1-A.

scripts/prepare_registration.py reads committed task sources and creates local register metadata pinned to their full commit SHA. It performs no GitHub submission, no log upload, and no repository visibility change. Actual registration also requires public sources, suitable paper references, full model runs and human review.
