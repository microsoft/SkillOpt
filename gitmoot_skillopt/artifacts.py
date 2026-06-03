"""Artifact helpers for Gitmoot SkillOpt exchange packages."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

HASH_PREFIX = "sha256:"


class ArtifactError(ValueError):
    """Raised when an artifact hash or path is invalid."""


def content_hash(content: bytes) -> str:
    return HASH_PREFIX + hashlib.sha256(content).hexdigest()


def normalize_hash(value: str) -> str:
    value = value.strip().lower()
    if value.startswith(HASH_PREFIX):
        value = value[len(HASH_PREFIX) :]
    if len(value) != hashlib.sha256().digest_size * 2:
        raise ArtifactError(f"artifact hash must be {hashlib.sha256().digest_size * 2} hex characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ArtifactError("artifact hash is not valid hex") from exc
    return HASH_PREFIX + value


def _hex_hash(value: str) -> str:
    return normalize_hash(value)[len(HASH_PREFIX) :]


@dataclass(frozen=True)
class ResolvedArtifact:
    hash: str
    path: Path
    content: bytes


class GitmootArtifactResolver:
    """Read Gitmoot content-addressed SHA256 blobs from an artifact root."""

    def __init__(self, artifact_root: str | Path) -> None:
        if str(artifact_root).strip() == "":
            raise ArtifactError("artifact root is required")
        root = Path(artifact_root).expanduser()
        self.artifact_root = root

    def path_for_hash(self, hash_value: str) -> Path:
        hex_hash = _hex_hash(hash_value)
        return self.artifact_root / "sha256" / hex_hash[:2] / hex_hash

    def read(self, hash_value: str, *, expected_size: int | None = None) -> ResolvedArtifact:
        normalized = normalize_hash(hash_value)
        path = self.path_for_hash(normalized)
        if not path.is_file():
            raise ArtifactError(f"artifact blob {normalized} not found")
        content = path.read_bytes()
        actual_hash = content_hash(content)
        if actual_hash != normalized:
            raise ArtifactError(f"artifact blob {normalized} content hash mismatch: got {actual_hash}")
        if expected_size is not None and len(content) != expected_size:
            raise ArtifactError(f"artifact blob {normalized} has size {len(content)}, want {expected_size}")
        return ResolvedArtifact(hash=normalized, path=path, content=content)


@dataclass(frozen=True)
class CandidateArtifactManifestEntry:
    id: str
    hash: str
    media_type: str
    driver: str
    path: str
    size_bytes: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CandidateArtifactManifestEntry":
        if not isinstance(data, dict):
            raise ArtifactError("candidate artifact manifest entry must be an object")
        size_bytes = data.get("size_bytes")
        if size_bytes is not None and (isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0):
            raise ArtifactError("candidate artifact size_bytes must be a non-negative integer")
        return cls(
            id=_required_text(data.get("id"), "candidate artifact id"),
            hash=normalize_hash(_required_text(data.get("hash"), "candidate artifact hash")),
            media_type=_required_text(data.get("media_type"), "candidate artifact media_type"),
            driver=_required_text(data.get("driver"), "candidate artifact driver"),
            path=_required_relative_path(data.get("path")),
            size_bytes=size_bytes,
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "id": self.id,
            "hash": self.hash,
            "media_type": self.media_type,
            "driver": self.driver,
            "path": self.path,
        }
        if self.size_bytes is not None:
            data["size_bytes"] = self.size_bytes
        return data


class OutputArtifactWriter:
    """Write optimizer artifacts under an artifact directory and emit manifests."""

    def __init__(self, out_root: str | Path, artifact_dir: str | Path | None = None) -> None:
        if str(out_root).strip() == "":
            raise ArtifactError("output root is required")
        root = Path(out_root).expanduser()
        self.out_root = root
        if artifact_dir is None or str(artifact_dir).strip() == "":
            self.artifact_root = self.out_root / "artifacts"
        else:
            self.artifact_root = Path(artifact_dir).expanduser()

    def write_bytes(
        self,
        relative_path: str | Path,
        content: bytes,
        *,
        artifact_id: str,
        media_type: str,
        driver: str,
    ) -> CandidateArtifactManifestEntry:
        if not artifact_id.strip():
            raise ArtifactError("artifact id is required")
        if not media_type.strip():
            raise ArtifactError("artifact media_type is required")
        if not driver.strip():
            raise ArtifactError("artifact driver is required")
        destination = self._safe_destination(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return CandidateArtifactManifestEntry(
            id=artifact_id.strip(),
            hash=content_hash(content),
            media_type=media_type.strip(),
            size_bytes=len(content),
            driver=driver.strip(),
            path=destination.relative_to(self.artifact_root).as_posix(),
        )

    def _safe_destination(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise ArtifactError("artifact output path must be relative")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ArtifactError("artifact output path cannot contain empty, current, or parent segments")
        destination = self.artifact_root / path
        self._ensure_artifact_parent(path)
        if destination.is_symlink():
            raise ArtifactError("artifact output path cannot be a symlink")
        root = self.artifact_root.resolve()
        resolved = destination.resolve(strict=False)
        try:
            common = os.path.commonpath([root, resolved])
        except ValueError as exc:
            raise ArtifactError("artifact output path escapes artifact directory") from exc
        if common != str(root):
            raise ArtifactError("artifact output path escapes artifact directory")
        return destination

    def _ensure_artifact_parent(self, relative_path: Path) -> None:
        if self.artifact_root.is_symlink():
            raise ArtifactError("artifact output root cannot be a symlink")
        if self.artifact_root.exists() and not self.artifact_root.is_dir():
            raise ArtifactError("artifact output root must be a directory")
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        current = self.artifact_root
        for part in relative_path.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ArtifactError("artifact output path cannot traverse a symlink")
            if current.exists():
                if not current.is_dir():
                    raise ArtifactError("artifact output path parent must be a directory")
                continue
            current.mkdir()


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{label} is required")
    return value.strip()


def _required_relative_path(value: object) -> str:
    path_text = _required_text(value, "candidate artifact path")
    path = Path(path_text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactError("candidate artifact path must be relative and cannot traverse directories")
    return path.as_posix()
