# Repository and dependency management

This project combines a CoT-Control-based application with seven independent
source repositories. The application owns the launcher, configuration, docs,
`dependencies.lock.json`, and `patches/`. Dependency clones and generated logs
remain ignored by the application Git repository.

## Source of truth

Each entry in `dependencies.lock.json` records the source URL, exact upstream
commit, and a SHA-256-checked patch containing the current local source changes.
The recorded `checkout_head` is provenance; restoration uses `base_commit`
plus the patch, so local commits do not need to be pushed to seven separate forks.
The Inspect patches use publicly accessible official upstream commits as their
bases and preserve the existing imported fork changes, including compatibility,
refusal, and benchmark extensions. They are not limited to the CoT adapters. Upstream history and license files
remain in the independent clones.

This preserves source trees, not virtual environments, credentials, generated
results, or original local commit sequences. Keep `.env` private and archive
evaluation results separately. NCRI item banks are shipped in the upstream's
password-protected archive. The lock also records its upstream checksum and the
hashes of the required extracted files; restore verifies and extracts them,
including the upstream canary. Plaintext banks stay ignored in the dependency.

## Daily workflow

Check whether all seven dependency source trees match the recorded state:

```bash
python3 scripts/dependency_repos.py check
```

After editing dependency sources, commit or stage any new files in the relevant
dependency repository, then update and review the application snapshot:

```bash
python3 scripts/dependency_repos.py snapshot
git diff -- dependencies.lock.json patches/
```

`snapshot` keeps the already-pinned upstream base and clone URL. It records tracked staged and
unstaged changes, but refuses untracked source files. Commit the lock, patches,
and related launcher/docs changes together in the application repository.
Run `git status` separately in any dependency whose local history you maintain.

## Restoring a new checkout

From a fresh application checkout with none of the dependency directories present:

```bash
python3 scripts/dependency_repos.py restore
python3 scripts/dependency_repos.py check
```

The restore command clones each recorded URL, checks out its pinned upstream
commit, verifies the patch checksum, and applies the patch to the index and working
tree, including newly added adapter files. It refuses existing
destination directories. NCRI data is prepared from its upstream release archive;
`python3 scripts/dependency_repos.py prepare-data` can prepare missing data in
existing clones without replacing changed files. If a clone fails partway through, retain it for
inspection and retry into a fresh checkout or an empty `--destination` directory.
The recorded upstream sources are accessible through public HTTPS URLs.
The inaccessible original Inspect fork URLs are not required for restoration.

Install `inspect_ai` and `inspect_evals` from these source directories in editable
mode in your environment. The current `.venv` already uses editable source paths.
Adapters resolve benchmark data relative to the project layout, so keep
`inspect_ai/`, `inspect_evals/`, and `repos/` at the recorded locations.

## Upstream updates and Git remotes

An upstream update is explicit: fetch the dependency, choose and record a new
`base_commit`, rebase/reapply the local changes, rerun relevant checks, and run
`snapshot`. Review the resulting patch before committing the application snapshot.

The application uses `https://github.com/XWang20/cot-monitorability.git` as
`origin` and the original `YuehHanChen/CoTControl` repository as `upstream`.
The application starts from its own initial commit. The upstream remote is kept
for source comparison; dependency clones retain their independent upstream histories.

If several people later maintain a dependency continuously, a shared fork plus
a pinned Git submodule can replace its patch entry. Push its custom commit to the
shared fork before introducing the submodule. A submodule pointing to a local,
unpublished commit cannot be restored by other contributors.

## Local API

The unified runner defaults to `http://127.0.0.1:8000/v1`. All models, including
`kimi-k3.1-pro`, use this API through the normal OpenAI-compatible provider.
There is no model-specific switch to DMXAPI. The runner does not source the
legacy `.env`; explicitly exported `OPENAI_BASE_URL` and `OPENAI_API_KEY` override
the defaults (`PROXY_API_KEY` is also accepted as a key fallback).

```bash
./run_inspect_bench.sh 1 openai/kimi-k3.1-pro 10 5 high
```

For direct Inspect CLI use, export the same API configuration first. The older
CSV evaluation utilities still offer their explicit `--provider dmxapi|proxy`
selection; the unified Inspect runner uses the local proxy by default.
