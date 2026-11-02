# Format validation

Validated on 2026-11-02 with public Inspect AI 0.3.287 and inspect-evals-lint 0.12.0.

- Native package installs with uv and builds a wheel plus source distribution.
- Six directly available task sources declare version 1-A and construct datasets and scorers.
- All six tasks run with mockllm models, without paid model calls.
- 49 tests pass; one strict xfail records the copied legacy T/F parser issue.
- Built-wheel task imports work from another directory, with no original dependency clones.
- Dataset IDs, inputs, choices and targets match the legacy fingerprints for all six default tasks.
- Official native eval metadata models accept all six eval.yaml files.
- Inspect task lint: 110 passes, 0 failures, 16 warnings. Warnings concern additional metric invariant tests and the encoding lint rule; they are not registration-format blockers.
- Fourteen historical raw logs retain their original SHA-256 and remain server-only.

Human methodology review, full model runs, paper-reference completion and repository publication are not claimed by these format receipts. The stored historical logs are not full results for task version 1-A.
