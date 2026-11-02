"""Fetch immutable upstream assets into a verified, writable cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path, PurePosixPath

from filelock import FileLock
from platformdirs import user_cache_dir


class AssetIntegrityError(ValueError):
    """An upstream archive or cached file does not match its recorded digest."""


def specifications() -> dict:
    return json.loads(
        resources.files("cot_monitorability").joinpath("asset_specs.json").read_text()
    )


def cache_root() -> Path:
    root = Path(
        os.environ.get(
            "COT_MONITORABILITY_CACHE_DIR", user_cache_dir("cot-monitorability")
        )
    ).expanduser()
    if not root.is_absolute():
        raise ValueError("COT_MONITORABILITY_CACHE_DIR must be an absolute path")
    return root


def asset_path(name: str) -> Path:
    spec = specifications()[name]
    return cache_root() / f"{name}-{spec['commit']}"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        digest = (
            hashlib.file_digest(handle, "sha256")
            if hasattr(hashlib, "file_digest")
            else None
        )
        if digest is not None:
            return digest.hexdigest()
        fallback = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            fallback.update(block)
        return fallback.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise AssetIntegrityError(f"Unsafe archive path: {relative}")
    target = root.joinpath(*path.parts)
    if root.resolve() not in target.resolve().parents:
        raise AssetIntegrityError(f"Archive path escapes its cache: {relative}")
    return target


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise AssetIntegrityError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise AssetIntegrityError(
                    f"Archive links are not supported: {member.name}"
                )
            relative = PurePosixPath(*path.parts[1:])
            if not relative.parts:
                continue
            target = _inside(destination, str(relative))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = source.extractfile(member)
                if handle is None:
                    raise AssetIntegrityError(
                        f"Archive member is unreadable: {member.name}"
                    )
                with handle, target.open("wb") as output:
                    shutil.copyfileobj(handle, output)


def _prepare_ncri(root: Path, spec: dict) -> None:
    data = spec.get("data_archive")
    if data is None:
        return
    archive = _inside(root, data["path"])
    if sha256(archive) != data["sha256"]:
        raise AssetIntegrityError("NCRI release archive checksum mismatch")
    prefixes = ("data/ncri/", "data/extras/", "data/knowledge/")
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir() or not (
                member.filename.startswith(prefixes)
                or member.filename == "data/CANARY.txt"
            ):
                continue
            target = _inside(root, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(member, pwd=data["password"].encode()))


def _verify(root: Path, spec: dict) -> None:
    marker = root / ".asset-manifest.json"
    if not marker.is_file():
        raise AssetIntegrityError(f"Incomplete asset cache: {root}")
    manifest = json.loads(marker.read_text())
    if manifest["archive_sha256"] != spec["sha256"]:
        raise AssetIntegrityError("Cached asset belongs to another archive")
    for relative, expected in manifest["files"].items():
        path = _inside(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise AssetIntegrityError(f"Cached asset changed: {relative}")


def ensure_asset(name: str, archive_path: Path | None = None) -> Path:
    """Prepare an asset once; optional archive_path supports air-gapped setup."""
    spec = specifications()[name]
    destination = asset_path(name)
    parent = cache_root()
    parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock", timeout=120):
        if destination.exists():
            _verify(destination, spec)
            return destination
        if archive_path is None and os.environ.get("COT_MONITORABILITY_OFFLINE") == "1":
            raise FileNotFoundError(
                f"Asset {name} is not cached; prepare assets before using offline mode"
            )
        with tempfile.TemporaryDirectory(prefix=".asset-", dir=parent) as temporary:
            temp = Path(temporary)
            archive = archive_path
            if archive is None:
                archive = temp / "source.tar.gz"
                request = urllib.request.Request(
                    spec["archive_url"],
                    headers={"User-Agent": "cot-monitorability/0.1"},
                )
                with (
                    urllib.request.urlopen(request, timeout=120) as source,
                    archive.open("wb") as target,
                ):
                    shutil.copyfileobj(source, target)
            if sha256(archive) != spec["sha256"]:
                raise AssetIntegrityError(f"Archive checksum mismatch: {name}")
            staged = temp / "source"
            staged.mkdir()
            _extract_tar(archive, staged)
            _prepare_ncri(staged, spec)
            manifest = {
                str(path.relative_to(staged)): sha256(path)
                for path in sorted(staged.rglob("*"))
                if path.is_file()
            }
            (staged / ".asset-manifest.json").write_text(
                json.dumps(
                    {"archive_sha256": spec["sha256"], "files": manifest}, indent=2
                )
                + "\n"
            )
            staged.rename(destination)
    return destination


def prepare_all() -> None:
    for name in specifications():
        ensure_asset(name)


if __name__ == "__main__":
    prepare_all()
