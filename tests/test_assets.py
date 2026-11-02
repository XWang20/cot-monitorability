import hashlib
import io
import tarfile

import pytest

from cot_monitorability import assets


def archive(tmp_path, members):
    target = tmp_path / "asset.tar.gz"
    with tarfile.open(target, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return target


def configure(monkeypatch, tmp_path, source):
    spec = {
        "demo": {
            "commit": "a" * 40,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "archive_url": "https://invalid.example/asset",
        }
    }
    monkeypatch.setattr(assets, "specifications", lambda: spec)
    monkeypatch.setenv("COT_MONITORABILITY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("COT_MONITORABILITY_OFFLINE", "1")


def test_archive_and_cached_files_are_verified(monkeypatch, tmp_path):
    source = archive(tmp_path, {"source/data/item.txt": b"frozen"})
    configure(monkeypatch, tmp_path, source)
    root = assets.ensure_asset("demo", source)
    assert (root / "data/item.txt").read_bytes() == b"frozen"
    assert assets.ensure_asset("demo") == root
    (root / "data/item.txt").write_bytes(b"modified")
    with pytest.raises(assets.AssetIntegrityError, match="Cached asset changed"):
        assets.ensure_asset("demo")


def test_rejects_bad_archive_digest(monkeypatch, tmp_path):
    source = archive(tmp_path, {"source/item": b"frozen"})
    configure(monkeypatch, tmp_path, source)
    source.write_bytes(source.read_bytes() + b"corrupt")
    with pytest.raises(assets.AssetIntegrityError, match="checksum"):
        assets.ensure_asset("demo", source)


def test_rejects_path_traversal(monkeypatch, tmp_path):
    source = archive(tmp_path, {"source/../escape": b"bad"})
    configure(monkeypatch, tmp_path, source)
    with pytest.raises(assets.AssetIntegrityError, match="Unsafe archive path"):
        assets.ensure_asset("demo", source)
    assert not (tmp_path / "escape").exists()


def test_offline_cache_miss_does_not_download(monkeypatch, tmp_path):
    source = archive(tmp_path, {"source/item": b"frozen"})
    configure(monkeypatch, tmp_path, source)
    with pytest.raises(FileNotFoundError, match="not cached"):
        assets.ensure_asset("demo")


def test_cache_root_must_be_absolute(monkeypatch):
    monkeypatch.setenv("COT_MONITORABILITY_CACHE_DIR", "relative/cache")
    with pytest.raises(ValueError, match="absolute"):
        assets.cache_root()
