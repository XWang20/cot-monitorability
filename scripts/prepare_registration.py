#!/usr/bin/env python3
"""Prepare pinned Register metadata locally; this command does not submit it."""

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[1]


def metadata(entry, commit, repository):
    source = {
        "repository_url": repository,
        "repository_commit": commit,
        "maintainers": ["XWang20"],
        "comment": "Standalone task version 1-A. This is a format draft, not a registration submission.",
    }
    if entry.get("methodology_url"):
        source["comment"] += " Original methodology: " + entry["methodology_url"]
    result = {
        "id": entry["id"],
        "full_title": entry["title"],
        "common_title": entry["title"],
        "description": entry["description"],
        "contributors": ["XWang20"],
        "tasks": [
            {
                "name": entry["name"],
                "task_path": entry["task_path"],
                "dataset_samples": entry["dataset_samples"],
            }
        ],
        "source": source,
    }
    if entry.get("arxiv"):
        result["arxiv"] = entry["arxiv"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", help="Published source commit; defaults to HEAD")
    parser.add_argument(
        "--repository", default="https://github.com/XWang20/cot-monitorability"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "register")
    args = parser.parse_args()
    commit = (
        args.commit
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Source commit must be a full 40-character SHA")
    catalog_text = subprocess.check_output(
        ["git", "show", commit + ":src/cot_monitorability/catalog.json"],
        cwd=ROOT,
        text=True,
    )
    entries = json.loads(catalog_text)["tasks"]
    schema = json.loads(
        (ROOT / "schemas/inspect-evals-register.schema.json").read_text()
    )
    forms = []
    for entry in entries:
        code = subprocess.check_output(
            ["git", "show", commit + ":" + entry["task_path"]], cwd=ROOT, text=True
        )
        function = next(
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.FunctionDef) and node.name == entry["name"]
        )
        data = metadata(entry, commit, args.repository)
        jsonschema.validate(data, schema, format_checker=jsonschema.FormatChecker())
        target = args.output / entry["id"]
        target.mkdir(parents=True, exist_ok=True)
        (target / "eval.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False, width=100000, allow_unicode=True)
        )
        forms.append(
            {
                "task": entry["name"],
                "arxiv_url": entry.get("arxiv"),
                "source_url": args.repository
                + "/blob/"
                + commit
                + "/"
                + entry["task_path"]
                + "#L"
                + str(function.lineno),
                "methodology_url": entry.get("methodology_url"),
            }
        )
    (ROOT / "docs/registration-fields.json").write_text(
        json.dumps(
            {"source_commit": commit, "submitted": False, "forms": forms}, indent=2
        )
        + "\n"
    )
    print(
        f"Prepared {len(forms)} local registration drafts pinned to {commit}; nothing was submitted."
    )


if __name__ == "__main__":
    main()
