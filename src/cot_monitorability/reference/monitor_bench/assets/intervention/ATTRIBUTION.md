# Goal-sandbagging asset attribution

These assets support MonitorBench's `goal_sandbag.math`,
`goal_sandbag.safety`, and `goal_sandbag.engineering` tasks. Runtime code, prompt construction, and
verification are pinned to
[MonitorBench](https://github.com/ASTRAL-Group/MonitorBench) commit
`43dda5994bfb16d34b1c30d4b3482d78a714e640`.

## Asset provenance

| Distributed path | Source at the pinned MonitorBench revision | Transformation | SHA-256 |
| ---------------- | ------------------------------------------ | -------------- | ------- |
| `datasets/goal_sandbag_engineering.json` | `datasets/goal_sandbag/mmlu_pro/MMLU_Pro.json` | Retained only `question_id`, `question`, `options`, and `answer`; preserved all row/option order and string values | `2abe0ed1133146a13bf78f31fe7a38c0680105af7563b94342d344c1df4f4d7b` |
| `datasets/goal_sandbag_math.jsonl` | `datasets/goal_sandbag/aime24/test.jsonl` | Retained only `id`, `problem`, and `answer`; minified each JSON object | `b19f92afb0c6593571a484fc81f12c3b7eaa89f08745887a85287e9f17bb01b1` |
| `datasets/goal_sandbag_safety.json` | `datasets/goal_sandbag/wmdp/WMDP.json` | None; byte-identical copy | `052e2e082c8c1872680fce44b2adb27307527df8fe24c25dd1d8d55f02b187e4` |
| `prompts/monitor_goal_sandbag.engineering.yaml` | `prompts/monitor_goal_sandbag.engineering.yaml` | None; byte-identical copy | `38865f0ee21f6acdaf7c4a07473f0c542b2fe2bb0c2a81aba6633bc45c794f09` |
| `prompts/monitor_goal_sandbag.math.yaml` | `prompts/monitor_goal_sandbag.math.yaml` | None; byte-identical copy | `38865f0ee21f6acdaf7c4a07473f0c542b2fe2bb0c2a81aba6633bc45c794f09` |
| `prompts/monitor_goal_sandbag.safety.yaml` | `prompts/monitor_goal_sandbag.safety.yaml` | None; byte-identical copy | `38865f0ee21f6acdaf7c4a07473f0c542b2fe2bb0c2a81aba6633bc45c794f09` |

The complete upstream JSONL has SHA-256
`16f87a954fa47871094ce09e0da56c4a2e3e15ed26c529ed6187a26e10d892dd`.
The derived file preserves all 30 source rows and their order. It omits the
upstream `solution`, duplicate `question`, `url`, and `membership` fields.
Consequently, this repository does not redistribute the Art of Problem
Solving community solutions, contributor names, or source URLs embedded in
the upstream file.

The checksums used for wheel and source-tree verification are also recorded
in [`SHA256SUMS`](SHA256SUMS).

## Code and prompt licenses

MonitorBench's code and original monitor prompt are MIT licensed, Copyright
(c) 2026 ASTRAL Group @ UIUC. Its full MIT notice is retained in the parent
[`ATTRIBUTION.md`](../ATTRIBUTION.md).

The shared boxed-answer verifier is adapted from the DeepScaleR checker
vendored by MonitorBench at
`pipeline/verification/deepscaler/rewards/math_utils/utils.py`. MonitorBench's
vendored `pipeline/verification/deepscaler/setup.py` identifies the Agentica
Team as author and classifies the package as MIT. The adaptation is based on
the pinned MonitorBench copy; it is not a byte-identical redistribution.

## MMLU-Pro engineering provenance and dataset rights

The engineering asset retains all 150 ordered records from MonitorBench's
`datasets/goal_sandbag/mmlu_pro/MMLU_Pro.json` (source SHA-256
`2f34301bbd16ca1b6bceba864dbba53fd372f26cc56337aa60500fde73f68f56`).
Only `question_id`, `question`, `options`, and `answer` are retained, in that
key order; JSON uses UTF-8, `ensure_ascii=False`, two-space indentation, and a
final LF. The unused `answer_index`, `category`, `cot_content`, and `src`
fields are omitted. No question text, option text/order, answer, or source row
order is changed. Source provenance is 147 `stemez-*` records and three
`ori_mmlu-electrical_engineering` records; their subject counts are retained
in [`goal_sandbag_engineering_audit.json`](goal_sandbag_engineering_audit.json).

On 2026-10-15, all 150 source `question_id` values were matched against the
12,032-record official
[`TIGER-Lab/MMLU-Pro` test split](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be)
at immutable revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`.
The official `data/test-00000-of-00001.parquet` SHA-256 is
`0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`,
matching its published LFS object ID. Every answer and answer index agrees.
Of the complete source records, 148 agree field-for-field. Two differences
are deliberately preserved for fidelity to the pinned MonitorBench revision:

| Source index | Question ID | Field | Difference from the official snapshot |
| ------------ | ----------- | ----- | ------------------------------------- |
| 0 | 11535 | `question` | The pinned text contains one space removed in the official snapshot. |
| 5 | 11572 | `options` | Pinned choices B, D, E, and J contain U+000B vertical tabs absent from the official snapshot. |

The audit JSON records canonical JSON SHA-256 hashes of both versions of
each differing field, source/derived/file hashes, distributions, and exact
source revisions. These checks establish lineage without replacing the
benchmark's frozen sample text with later upstream formatting changes.

The pinned
[official dataset card](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/b189ec765aa7ed75c8acfea42df31fdae71f97be/README.md)
declares the **dataset MIT licensed**. Its SHA-256 is
`4bd710f67da3fa359a33edce1b4b5816b3de416c823c2624ba5e89c2557d2a47`.
Wang et al., "MMLU-Pro: A More Robust and Challenging Multi-Task Language
Understanding Benchmark," NeurIPS 2024,
[Appendix A.5](https://papers.nips.cc/paper_files/paper/2024/file/ad236edc564f3e3156e1b2feafb99a24-Paper-Datasets_and_Benchmarks_Track.pdf)
also states that MMLU-Pro is MIT licensed and identifies the STEM website
source as open-licensed. Source website terms were not independently
reconstructed by this audit. This records the dataset publisher's license
and source-license statements, rather than inferring dataset rights from
MonitorBench's software license.

The dataset snapshot supplies no separate dataset-specific LICENSE file or
copyright holder/year. Its declaration and the referenced standard MIT terms
are retained in [`MMLU_PRO_LICENSE`](MMLU_PRO_LICENSE), without inventing a
copyright notice. The separate
[MMLU-Pro software LICENSE](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/c992affcc124fd7238f13221e8ab8262928342a3/LICENSE)
is Apache-2.0; no code from that repository is copied by this port.

### Reproduce the offline engineering comparison

With PyArrow available as audit tooling, download the official Parquet once
from the immutable URL above, then run this from the repository root. No
runtime dataset download or new task dependency is required. The first
argument is a local pinned MonitorBench checkout; the second is the downloaded
official test Parquet.

```bash
uv run --no-sync --frozen python - ../MonitorBench /path/to/test.parquet <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

source = Path(sys.argv[1]) / "datasets/goal_sandbag/mmlu_pro/MMLU_Pro.json"
parquet = Path(sys.argv[2])
assert hashlib.sha256(source.read_bytes()).hexdigest() == "2f34301bbd16ca1b6bceba864dbba53fd372f26cc56337aa60500fde73f68f56"
assert hashlib.sha256(parquet.read_bytes()).hexdigest() == "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
pinned = json.loads(source.read_text())
rows = pq.read_table(parquet).to_pylist()
by_id = {row["question_id"]: row for row in rows}
assert len(by_id) == len(rows) == 12032
assert len(pinned) == len({row["question_id"] for row in pinned}) == 150
changes = []
for index, row in enumerate(pinned):
    official = by_id[row["question_id"]]
    assert row["answer"] == official["answer"]
    assert row["answer_index"] == official["answer_index"]
    for field in row:
        if row[field] != official[field]:
            def field_hash(value):
                encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                return hashlib.sha256(encoded.encode()).hexdigest()
            changes.append((index, row["question_id"], field))
            print(changes[-1], field_hash(row[field]), field_hash(official[field]))
assert changes == [(0, 11535, "question"), (5, 11572, "options")]
fields = ("question_id", "question", "options", "answer")
derived = json.loads(Path("src/monitor_bench/assets/intervention/datasets/goal_sandbag_engineering.json").read_text())
assert derived == [{key: row[key] for key in fields} for row in pinned]
print("150 IDs and targets matched; 148 full records exact; two frozen formatting differences")
PY
```

## WMDP dataset rights

The 120 safety records are MonitorBench's ordered subset of the Weapons of
Mass Destruction Proxy (WMDP) benchmark: 40 WMDP-Bio, followed by 40
WMDP-Chem, then 40 WMDP-Cyber records. All 120 question, choice, and answer
triples were matched exactly and uniquely against the official
[`cais/wmdp`](https://huggingface.co/datasets/cais/wmdp/tree/7125571f22f032c56415e7980f48d877dd830ff8)
release at revision `7125571f22f032c56415e7980f48d877dd830ff8`.

The official
[`centerforaisafety/wmdp`](https://github.com/centerforaisafety/wmdp/tree/c0b6c12bb0de3decf9c13bb13f9f1aa15754a132)
repository and Hugging Face dataset card declare MIT licensing, Copyright (c)
2024 centerforaisafety. The complete notice is redistributed in
[`WMDP_LICENSE`](WMDP_LICENSE). This is a separate grant from MonitorBench's
repository license. See also Nathaniel Li et al., "The WMDP Benchmark:
Measuring and Reducing Malicious Use With Unlearning," arXiv:2403.03218.

## AIME problem-text rights

The 30 problem statements and answers are from the 2024 American Invitational
Mathematics Examination (AIME). MonitorBench's MIT license covers its software
and original material; it does not establish redistribution rights for the
AIME problem text.

Public dataset lineage was checked separately:

- [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
  does not declare a license and points to the AI-MO validation dataset.
- [`AI-MO/aimo-validation-aime`](https://huggingface.co/datasets/AI-MO/aimo-validation-aime/tree/1f8807dabb2265892a7c2cb5e73d95d25df4507b)
  declares Apache-2.0 in its dataset card (declaration added at revision
  `13f9e12f613e720c2a2b2f345dd04b998a29494d`; referenced data revision
  `1f8807dabb2265892a7c2cb5e73d95d25df4507b`, Parquet LFS object SHA-256
  `025484a99fea498e7d0c3b0ee42afcbec0176405c19c5dbf557b9f6ca6445675`).
