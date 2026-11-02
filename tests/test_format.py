import ast
import importlib
import json
import os
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "src/cot_monitorability/catalog.json").read_text())[
    "tasks"
]


@pytest.mark.parametrize("entry", CATALOG, ids=lambda entry: entry["name"])
def test_task_sources_and_metadata_are_direct(entry):
    path = ROOT / entry["task_path"]
    nodes = ast.parse(path.read_text()).body
    function = next(
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name == entry["name"]
    )
    assert any(
        isinstance(d, (ast.Name, ast.Call))
        and (d.id if isinstance(d, ast.Name) else getattr(d.func, "id", None)) == "task"
        for d in function.decorator_list
    )
    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Task"
        for n in ast.walk(function)
    )
    metadata = yaml.safe_load((path.parent / "eval.yaml").read_text())
    assert metadata["version"] == entry["version"] == "1-A"
    assert metadata["tasks"][0]["name"] == entry["name"]
    assert metadata["tasks"][0]["dataset_samples"] == entry["dataset_samples"]
    assert (path.parent / "__init__.py").is_file()
    assert (path.parent / "README.md").is_file()
    assert "sys.path" not in path.read_text()


@pytest.mark.dataset_download
@pytest.mark.skipif(
    os.environ.get("RUN_DATASET_TESTS") != "1",
    reason="prepare pinned assets and enable RUN_DATASET_TESTS",
)
@pytest.mark.parametrize("entry", CATALOG, ids=lambda entry: entry["name"])
def test_default_task_constructs_offline(entry):
    factory = getattr(importlib.import_module(entry["module"]), entry["name"])
    task = factory()
    assert task.version == "1-A"
    assert len(task.dataset) == entry["dataset_samples"]
    assert task.scorer is not None


def test_source_archives_are_pinned():
    specs = json.loads((ROOT / "src/cot_monitorability/asset_specs.json").read_text())
    for spec in specs.values():
        assert len(spec["commit"]) == 40
        assert len(spec["sha256"]) == 64
        assert spec["commit"] in spec["archive_url"]


@pytest.mark.dataset_download
@pytest.mark.skipif(
    os.environ.get("RUN_DATASET_TESTS") != "1", reason="prepare pinned assets first"
)
@pytest.mark.parametrize("entry", CATALOG, ids=lambda entry: entry["name"])
def test_migrated_dataset_matches_legacy_inputs_and_targets(entry):
    import hashlib

    factory = getattr(importlib.import_module(entry["module"]), entry["name"])
    task = factory()
    samples = [
        s.model_dump(include={"id", "input", "target", "choices"}, mode="json")
        for s in task.dataset
    ]
    encoded = json.dumps(
        samples, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    baseline = json.loads(
        (ROOT / "tests/fixtures/legacy_dataset_fingerprints.json").read_text()
    )[entry["name"]]
    assert len(samples) == baseline["samples"]
    assert hashlib.sha256(encoded).hexdigest() == baseline["sha256"]
