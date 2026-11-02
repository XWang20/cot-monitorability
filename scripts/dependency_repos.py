#!/usr/bin/env python3
"""Record and restore upstream Git revisions plus local source patches."""

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import zipfile

REPOSITORIES = (
    "inspect_ai",
    "inspect_evals",
    "repos/CoTControl",
    "repos/MonitorBench",
    "repos/inspect-evals-monitor-bench",
    "repos/monitorability-evals",
    "repos/nocot-bench",
)


def git(path, *args):
    return subprocess.check_output(
        ["git", "-c", "color.ui=false", "-c", "core.quotePath=true", "-C", str(path), *args],
        stderr=subprocess.PIPE,
    )


def source_diff(path, base):
    return git(path, "diff", "--no-ext-diff", "--no-textconv", "--binary", "--full-index", base, "--")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Path must be inside the project: {relative}")
    return path


def load_lock(root):
    lock = json.loads((root / "dependencies.lock.json").read_text())
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported dependencies.lock.json schema")
    for entry in lock["repositories"]:
        inside(root, entry["path"])
        inside(root, entry["patch"])
        if not re.fullmatch(r"[0-9a-f]{40}", entry["base_commit"]):
            raise ValueError(f"Invalid base commit: {entry['path']}")
    return lock


def data_archive(path):
    archive = path / "data/nocot_data.zip"
    expected = (path / "data/NOCOT_DATA_ZIP.sha256").read_text().split()[0]
    if digest(archive.read_bytes()) != expected:
        raise ValueError("NCRI archive does not match its upstream checksum")
    files = [path / "data/CANARY.txt"]
    for relative in ("data/ncri", "data/extras", "data/knowledge"):
        files.extend(p for p in (path / relative).rglob("*") if p.is_file())
    if not (path / "data/ncri/arithmetic.jsonl").is_file():
        raise ValueError("Prepare NCRI data before snapshot")
    return {
        "path": "data/nocot_data.zip",
        "sha256": expected,
        # Public release password from the pinned upstream README.
        "password": "lantern-orchard-pebble",
        "files": [{"path": str(p.relative_to(path)), "sha256": digest(p.read_bytes())} for p in sorted(files)],
    }


def prepare_data(entry, path):
    data = entry.get("data_archive")
    if data is None:
        return
    archive = inside(path, data["path"])
    if digest(archive.read_bytes()) != data["sha256"]:
        raise ValueError(f"Data archive checksum mismatch: {entry['path']}")
    with zipfile.ZipFile(archive) as zipped:
        for item in data["files"]:
            target = inside(path, item["path"])
            if target.exists():
                if digest(target.read_bytes()) != item["sha256"]:
                    raise ValueError(f"Refusing to overwrite changed data: {target}")
                continue
            content = zipped.read(item["path"], pwd=data["password"].encode())
            if digest(content) != item["sha256"]:
                raise ValueError(f"Extracted data checksum mismatch: {item['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print(f"Prepared {len(data['files'])} NCRI data files from the upstream archive.")


def snapshot(root):
    previous = {}
    if (root / "dependencies.lock.json").exists():
        previous = {entry["path"]: entry for entry in load_lock(root)["repositories"]}
    entries = []
    patches = []
    for relative in REPOSITORIES:
        path = inside(root, relative)
        untracked = git(path, "ls-files", "--others", "--exclude-standard").decode().strip()
        if untracked:
            raise ValueError(f"Commit or stage new dependency source files before snapshot: {relative}\n{untracked}")
        base = previous.get(relative, {}).get("base_commit")
        if base is None:
            base = git(path, "rev-parse", "@{upstream}").decode().strip()
        patch = source_diff(path, base)
        # Never carry the previously removed inline API key into a source patch.
        if any(re.search(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{16,}", line) for line in patch.splitlines() if line.startswith(b"+") and not line.startswith(b"+++")):
            raise ValueError(f"Inline API key found in patch additions: {relative}")
        patch_name = "patches/" + relative.replace("/", "-") + ".patch"
        entries.append({
            "path": relative,
            "url": previous.get(relative, {}).get("url") or git(path, "remote", "get-url", "origin").decode().strip(),
            "base_commit": base,
            "checkout_head": git(path, "rev-parse", "HEAD").decode().strip(),
            "patch": patch_name,
            "patch_sha256": digest(patch),
        })
        if relative == "repos/nocot-bench":
            entries[-1]["data_archive"] = data_archive(path)
        patches.append((patch_name, patch))
    for name, patch in patches:
        target = inside(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(patch)
    target = root / "dependencies.lock.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "repositories": entries}, indent=2) + "\n")
    temporary.replace(target)
    print(f"Recorded {len(entries)} repositories in dependencies.lock.json and patches/.")


def checked_patch(root, entry):
    patch = inside(root, entry["patch"]).read_bytes()
    if digest(patch) != entry["patch_sha256"]:
        raise ValueError(f"Patch checksum mismatch: {entry['patch']}")
    return patch


def check(root, destination):
    failed = False
    for entry in load_lock(root)["repositories"]:
        try:
            patch = checked_patch(root, entry)
            path = inside(destination, entry["path"])
            if source_diff(path, entry["base_commit"]) != patch:
                raise ValueError("source differs from the recorded patch")
            if git(path, "ls-files", "--others", "--exclude-standard").strip():
                raise ValueError("untracked dependency source files are not recorded")
            data = entry.get("data_archive")
            if data is not None:
                if digest(inside(path, data["path"]).read_bytes()) != data["sha256"]:
                    raise ValueError("data archive checksum mismatch")
                for item in data["files"]:
                    if digest(inside(path, item["path"]).read_bytes()) != item["sha256"]:
                        raise ValueError(f"data checksum mismatch: {item['path']}")
            print(f"OK {entry['path']}")
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"MISMATCH {entry['path']}: {exc}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


def restore(root, destination):
    entries = load_lock(root)["repositories"]
    prepared = [(entry, inside(destination, entry["path"]), checked_patch(root, entry)) for entry in entries]
    # Validate all destinations before cloning; never reset an existing checkout.
    for entry, path, patch in prepared:
        if path.exists():
            raise ValueError(f"Destination already exists; use a fresh directory: {path}")
    for entry, path, patch in prepared:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--", entry["url"], str(path)], check=True)
        base = entry["base_commit"]
        try:
            git(path, "cat-file", "-e", base + "^{commit}")
        except subprocess.CalledProcessError:
            git(path, "fetch", "origin", base)
        git(path, "checkout", "--detach", base)
        if patch:
            subprocess.run(["git", "-C", str(path), "apply", "--index", "--check", "-"], input=patch, check=True)
            subprocess.run(["git", "-C", str(path), "apply", "--index", "-"], input=patch, check=True)
        prepare_data(entry, path)
        print(f"Restored {entry['path']} at {base[:12]} plus recorded patch.")
    return check(root, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("snapshot", "check", "restore", "prepare-data"))
    parser.add_argument("--destination", type=pathlib.Path, help="Alternative dependency root for check/restore")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    destination = (args.destination or root).resolve()
    if args.command == "snapshot":
        if args.destination is not None:
            parser.error("snapshot reads the current project; --destination is for check/restore")
        snapshot(root)
        return 0
    if args.command == "check":
        return check(root, destination)
    if args.command == "prepare-data":
        for entry in load_lock(root)["repositories"]:
            prepare_data(entry, inside(destination, entry["path"]))
        return check(root, destination)
    return restore(root, destination)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
