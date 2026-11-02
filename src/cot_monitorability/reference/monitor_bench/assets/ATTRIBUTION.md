# Attribution

The files listed below are vendored byte-identically from
[MonitorBench](https://github.com/ASTRAL-Group/MonitorBench)
(arXiv: <https://arxiv.org/abs/2603.28590v2>) at upstream commit
`43dda5994bfb16d34b1c30d4b3482d78a714e640` (Copyright (c) 2026
ASTRAL Group @ UIUC). MonitorBench's source code and original prompt content
are MIT licensed. The steganography writing dataset also contains material
derived from Databricks Dolly 15k and retains that dataset's CC BY-SA 3.0
terms. The summarization data is derived from GovReport and has the separate,
unresolved rights status documented below.

The goal-sandbagging tasks add a minimized AIME 2024 problem/answer asset, a
byte-identical 120-record WMDP subset, a minimized 150-record MMLU-Pro
engineering subset, byte-identical MonitorBench monitor prompts, and a
verifier adapted from MonitorBench's vendored DeepScaleR checker. Their
provenance, checksums, WMDP MIT notice, MMLU-Pro dataset MIT declaration,
and unresolved AIME redistribution-rights audit are recorded in
[`intervention/ATTRIBUTION.md`](intervention/ATTRIBUTION.md). The engineering
audit preserves two formatting differences from the official MMLU-Pro
snapshot. Its dataset license declaration is separate from MonitorBench's
MIT license and the MMLU-Pro software repository's Apache-2.0 license.

## Provenance

| Vendored path | Upstream path | SHA-256 |
| ------------- | ------------- | ------- |
| `src/monitor_bench/assets/task_writing_steganography.json` | `datasets/dual_objective/databricks_dolly/task_writing_steganography.json` | `492f0d3c3dd1a60ae9c8ad8355719719890dee857c99f4d5df721c7e7b66c93b` |
| `src/monitor_bench/assets/monitor_dual_objectives.steganography.yaml` | `prompts/monitor_dual_objectives.steganography.yaml` | `4c0d977d8c93156549044d0b6efefa656d975774d93e9913a6978c029058ba7e` |
| `src/monitor_bench/assets/outcome/data/task_writing_summarization.json` | `datasets/dual_objective/summary/task_writing_summarization.json` | `e4c35bc3af0d0d66768d9f7cd54d3ff685d3586f6652719ca549a96802c923c7` |
| `src/monitor_bench/assets/outcome/prompts/monitor_dual_objectives.summarization.yaml` | `prompts/monitor_dual_objectives.summarization.yaml` | `a540a25cf72f86a8cc5f69a0efb816e48ded172b7838518556e17c6dc2e512c4` |

### Databricks Dolly-derived writing prompts

The `Input` prompts in `task_writing_steganography.json` are derived from
[Databricks Dolly 15k at revision `bdd27f4d`](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a),
Copyright (c) 2023 Databricks, Inc., licensed under
[Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/).
Forty-nine of the 50 input strings match records at that revision exactly; the
first adds the word "the" ("What are the top …" rather than "What are top …").
MonitorBench selected the prompts and added benchmark identifiers,
additional-objective metadata, and acrostic payloads. This repository has not
changed the MonitorBench copy.

### GovReport-derived summarization prompts

The MonitorBench paper states that its 50 summarization reports were randomly
sampled from
[`ccdv/govreport-summarization`](https://huggingface.co/datasets/ccdv/govreport-summarization)
and that Grok created a unique payload for each report. The pinned MonitorBench
file does not record source split/row IDs, a sampling seed, or a ccdv revision.
For reference, the current converted ccdv repository revision inspected during
this port is `4e21184e01ae8017e2c036e180fe5e541fef60a0`.

The ccdv dataset card cites Huang et al.'s GovReport paper and describes the
`report` and `summary` fields, but does not declare a license. The authors'
paper describes GovReport as GAO and CRS reports; that provenance does not by
itself establish redistribution rights for all embedded or transformed
content. MonitorBench's MIT license covers its code and original material, but
must not be treated as a separate license grant for the underlying reports.
Redistribution rights therefore remain unresolved pending human/legal review.

See: Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang,
"Efficient Attentions for Long Document Summarization," NAACL 2021,
<https://arxiv.org/abs/2104.02112>.

## MonitorBench license

The following is the full text of the upstream repository's `LICENSE` file:

```text
MIT License

Copyright (c) 2026 ASTRAL Group @ UIUC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
