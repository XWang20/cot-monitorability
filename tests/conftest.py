import os
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("COT_MONITORABILITY_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(
    params=sorted((ROOT / "src/cot_monitorability").glob("*/eval.yaml")),
    ids=lambda path: path.parent.name,
)
def task_entry(request):
    metadata = yaml.safe_load(request.param.read_text(encoding="utf-8"))
    task = metadata["tasks"][0]
    source = Path(task["task_path"])
    return {
        "name": task["name"],
        "module": ".".join(source.with_suffix("").parts[1:]),
        "task_path": task["task_path"],
        "dataset_samples": task["dataset_samples"],
        "version": metadata["version"],
    }
