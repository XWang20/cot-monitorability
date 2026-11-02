# CoT Monitorability

A platform for evaluating chain-of-thought monitorability through a unified
Inspect AI workflow, with model access through a local OpenAI-compatible proxy.

The platform integrates six benchmark families:

- CoT-Control QA
- OpenAI process-level monitorability
- OpenAI intervention-level monitorability
- DeepMind CoT immediacy (legibility and coverage)
- NCRI (No-CoT Reasoning Index)
- MonitorBench

See [the unified benchmark guide](INSPECT_BENCHMARKS_README.md) for evaluation
commands and [repository management](REPOSITORY_MANAGEMENT.md) for source setup,
pinned upstream versions, patches, and restoration.

The default model API is on port 8000. Inspect View on port 5010 displays
evaluation logs. The older CSV platform is documented in
[PLATFORM_README.md](PLATFORM_README.md).

## Source provenance

This repository has its own commit history. It builds on
[YuehHanChen/CoTControl](https://github.com/YuehHanChen/CoTControl) and the
upstream benchmark sources recorded in the dependency lock.
The [original CoT-Control documentation](UPSTREAM_COT_CONTROL_README.md)
retains the paper citation and upstream usage instructions.
Original license files and dataset provenance remain with their sources.
