# Outcome-justification asset attribution

These assets support MonitorBench's `dual_objectives.summarization` task.
Runtime code, prompt construction, verification, and monitor behavior are
pinned to
[MonitorBench](https://github.com/ASTRAL-Group/MonitorBench) commit
`43dda5994bfb16d34b1c30d4b3482d78a714e640`.

## Asset provenance

| Distributed path | Source at the pinned MonitorBench revision | Transformation | SHA-256 |
| ---------------- | ------------------------------------------ | -------------- | ------- |
| `data/task_writing_summarization.json` | `datasets/dual_objective/summary/task_writing_summarization.json` | None; byte-identical copy | `e4c35bc3af0d0d66768d9f7cd54d3ff685d3586f6652719ca549a96802c923c7` |
| `prompts/monitor_dual_objectives.summarization.yaml` | `prompts/monitor_dual_objectives.summarization.yaml` | None; byte-identical copy | `a540a25cf72f86a8cc5f69a0efb816e48ded172b7838518556e17c6dc2e512c4` |

The dataset copy is 2,917,659 bytes and preserves all 50 upstream rows and
their order. The monitor prompt copy is 2,659 bytes. Checksums are also
recorded in [`SHA256SUMS`](SHA256SUMS).

## Code and prompt license

MonitorBench's code and original monitor prompt are MIT licensed, Copyright
(c) 2026 ASTRAL Group @ UIUC. The full notice is retained in the parent
[`ATTRIBUTION.md`](../ATTRIBUTION.md).

## GovReport lineage and unresolved rights

The MonitorBench paper states that it randomly sampled the 50 report texts
from
[`ccdv/govreport-summarization`](https://huggingface.co/datasets/ccdv/govreport-summarization)
and used Grok to generate a unique payload for each report. The pinned JSON
does not record source split/row IDs, sampling seed, or the dataset revision.
The ccdv revision inspected for this audit was
`4e21184e01ae8017e2c036e180fe5e541fef60a0`.

The ccdv dataset card cites the GovReport dataset and its paper but does not
declare a license. Huang et al. describe GovReport as reports published by the
U.S. Government Accountability Office and Congressional Research Service.
Government origin does not, by itself, establish rights for every embedded or
transformed component. MonitorBench's MIT license must not be treated as a
separate redistribution grant for those reports.

Redistribution rights for the report text therefore remain unresolved and
require human/legal review before registry submission. This package records
the source and uncertainty; it does not claim that MIT or another license
covers the underlying reports.

Citation:

> Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang.
> "Efficient Attentions for Long Document Summarization." NAACL 2021.
> <https://arxiv.org/abs/2104.02112>.
