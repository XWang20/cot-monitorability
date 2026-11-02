# Adapted from semsorock/inspect-evals-monitor-bench at 9452b3ddcc1be81c4322e141dceb6f6c69b49434 (MIT); see NOTICE.
"""Shared text cleaning and sentence splitting for Outcome tasks.

Ported from ``pipeline/outcome.py`` in MonitorBench commit
``43dda5994bfb16d34b1c30d4b3482d78a714e640`` (Copyright (c) 2026 ASTRAL
Group @ UIUC, MIT License). The pinned Punkt acquisition replaces upstream's
import-time ``nltk.download('punkt')`` side effect.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import threading
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_ACTION_PREFIX_RE = re.compile(
    r"^\s*(?:<\|assistant\|>\s*)*(?:assistant\s*final|assistantfinal|assistant_final)\s*",
    flags=re.IGNORECASE,
)
_ACTION_MARKER_RE = re.compile(
    r"(?:<\|assistant\|>\s*)*(?:assistant\s*final|assistantfinal|assistant_final)\s*",
    flags=re.IGNORECASE,
)
_THINK_END_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_SENTENCE_FALLBACK_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n(?=\s*[-*•])|\n{2,}")
_BULLET_PREFIX_RE = re.compile(r"^\s*[-*•]+\s*")

_PUNKT_TAB_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
_PUNKT_TAB_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/"
    f"{_PUNKT_TAB_REVISION}/packages/tokenizers/punkt_tab.zip"
)
_PUNKT_TAB_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
_PUNKT_TAB_MAX_BYTES = 16 * 1024 * 1024
_PUNKT_RESOURCE = "tokenizers/punkt_tab/english"
_PUNKT_ENGLISH_SHA256 = {
    "abbrev_types.txt": (
        "92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779"
    ),
    "collocations.tab": (
        "8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf"
    ),
    "ortho_context.tab": (
        "4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c"
    ),
    "sent_starters.txt": (
        "f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac"
    ),
}


class _SentenceTokenizer(Protocol):
    """Structural type for the pinned sentence tokenizer."""

    def tokenize(self, text: str) -> list[str]:
        """Split text into sentences."""


@dataclass(frozen=True)
class _PunktLoadResult:
    """Cache either the tokenizer or its first load failure."""

    tokenizer: _SentenceTokenizer | None
    error: Exception | None


@dataclass
class _PunktLoadState:
    """Synchronize one process-wide tokenizer load attempt."""

    result: _PunktLoadResult | None = None


class PunktResourceError(RuntimeError):
    """Raised when the pinned Punkt model cannot be loaded safely."""


_PUNKT_LOAD_LOCK = threading.Lock()
_PUNKT_LOAD_STATE = _PunktLoadState()


def clean_action_text(action: str | None) -> str:
    """Strip reasoning residue and chat-template markers from action text."""
    if action is None:
        return ""
    text = action.replace("\ufeff", "").replace("\u200b", "").strip()
    think_end = _THINK_END_RE.search(text)
    if think_end:
        text = text[think_end.end() :].strip()

    markers = list(_ACTION_MARKER_RE.finditer(text))
    if markers:
        text = text[markers[-1].end() :].strip()

    return _ACTION_PREFIX_RE.sub("", text).strip()


def _sha256_file(path: Path) -> str:
    """Return a file SHA-256 without loading the whole file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _punkt_data_root_is_valid(data_root: Path) -> bool:
    """Check for the exact pinned English Punkt files."""
    english_root = data_root / _PUNKT_RESOURCE
    try:
        return all(
            (path := english_root / filename).is_file()
            and _sha256_file(path) == expected_sha256
            for filename, expected_sha256 in _PUNKT_ENGLISH_SHA256.items()
        )
    except OSError:
        return False


def _punkt_cache_root() -> Path:
    """Return the package-owned Punkt cache root."""
    override = os.environ.get("MONITOR_BENCH_NLTK_DATA")
    if override:
        return Path(override).expanduser()

    cache_home = Path(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    ).expanduser()
    return cache_home / "monitor_bench" / "nltk_data" / _PUNKT_TAB_REVISION


def _download_punkt_tab() -> bytes:
    """Download and checksum the immutable Punkt archive."""
    import httpx

    payload_chunks: list[bytes] = []
    payload_size = 0
    digest = hashlib.sha256()
    with httpx.stream(
        "GET", _PUNKT_TAB_URL, follow_redirects=True, timeout=60.0
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            payload_size += len(chunk)
            if payload_size > _PUNKT_TAB_MAX_BYTES:
                raise RuntimeError(
                    "NLTK punkt_tab archive exceeds the expected maximum size"
                )
            digest.update(chunk)
            payload_chunks.append(chunk)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != _PUNKT_TAB_SHA256:
        raise RuntimeError(
            "NLTK punkt_tab archive checksum mismatch: "
            f"expected {_PUNKT_TAB_SHA256}, got {actual_sha256}"
        )
    return b"".join(payload_chunks)


def _install_punkt_tab(data_root: Path) -> None:
    """Install only checksum-verified English files from the archive."""
    if _punkt_data_root_is_valid(data_root):
        return

    payload = _download_punkt_tab()
    english_root = data_root / _PUNKT_RESOURCE
    english_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".punkt-tab-", dir=english_root.parent
    ) as temporary_directory:
        staged_root = Path(temporary_directory)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for filename, expected_sha256 in _PUNKT_ENGLISH_SHA256.items():
                archive_name = f"punkt_tab/english/{filename}"
                try:
                    file_payload = archive.read(archive_name)
                except KeyError as exc:
                    raise RuntimeError(
                        f"NLTK punkt_tab archive is missing {archive_name}"
                    ) from exc
                if hashlib.sha256(file_payload).hexdigest() != expected_sha256:
                    raise RuntimeError(
                        f"NLTK punkt_tab file checksum mismatch for {filename}"
                    )
                (staged_root / filename).write_bytes(file_payload)

        for filename in _PUNKT_ENGLISH_SHA256:
            (staged_root / filename).replace(english_root / filename)

    if not _punkt_data_root_is_valid(data_root):
        raise RuntimeError("Installed NLTK punkt_tab data failed verification")


def _existing_punkt_data_root(nltk_paths: Iterable[str]) -> Path | None:
    """Find an NLTK root matching the pinned English file hashes."""
    for raw_path in nltk_paths:
        candidate = Path(raw_path).expanduser()
        if _punkt_data_root_is_valid(candidate):
            return candidate
    return None


def _load_punkt_tokenizer() -> _PunktLoadResult:
    """Load and cache the tokenizer, including the first failure."""
    if _PUNKT_LOAD_STATE.result is not None:
        return _PUNKT_LOAD_STATE.result

    with _PUNKT_LOAD_LOCK:
        if _PUNKT_LOAD_STATE.result is not None:
            return _PUNKT_LOAD_STATE.result
        try:
            import nltk
            from nltk.data import FileSystemPathPointer
            from nltk.tokenize.punkt import (
                PunktSentenceTokenizer,
                load_punkt_params,
            )

            data_root = _existing_punkt_data_root(nltk.data.path)
            if data_root is None:
                data_root = _punkt_cache_root()
                _install_punkt_tab(data_root)

            model_directory = FileSystemPathPointer(str(data_root / _PUNKT_RESOURCE))
            parameters = load_punkt_params(model_directory)
            tokenizer = PunktSentenceTokenizer(parameters)
            _PUNKT_LOAD_STATE.result = _PunktLoadResult(tokenizer, None)
        except Exception as exc:
            _PUNKT_LOAD_STATE.result = _PunktLoadResult(None, exc)
        return _PUNKT_LOAD_STATE.result


def _reset_punkt_tokenizer_cache() -> None:
    """Reset package-owned tokenizer state for regression tests."""
    with _PUNKT_LOAD_LOCK:
        _PUNKT_LOAD_STATE.result = None


def _verified_punkt_tokenizer() -> _SentenceTokenizer:
    """Return the verified tokenizer or raise one stable error."""
    result = _load_punkt_tokenizer()
    if result.error is not None:
        raise PunktResourceError(
            "MonitorBench could not load the pinned NLTK punkt_tab model; "
            "refusing the regex fallback because it changes verifier scores. "
            "Check network access or pre-populate MONITOR_BENCH_NLTK_DATA "
            f"from {_PUNKT_TAB_URL}."
        ) from result.error
    if result.tokenizer is None:
        raise AssertionError("Punkt load result has neither tokenizer nor error")
    return result.tokenizer


def _ensure_punkt() -> bool:
    """Ensure the pinned Punkt model is available and verified."""
    _verified_punkt_tokenizer()
    return True


def split_sentences(text: str) -> list[str]:
    """Split sentences using upstream Punkt/fallback control flow."""
    text = (text or "").strip()
    if not text:
        return []
    tokenizer = _verified_punkt_tokenizer()
    try:
        sentences = tokenizer.tokenize(text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if len(sentences) <= 1 and "\n" in text:
            raise ValueError("fallback")
        return sentences
    except Exception:
        parts = _SENTENCE_FALLBACK_SPLIT_RE.split(text)
        output = []
        for raw_part in parts:
            part = (raw_part or "").strip()
            if not part:
                continue
            part = _BULLET_PREFIX_RE.sub("", part).strip()
            if part:
                output.append(part)
        return output
