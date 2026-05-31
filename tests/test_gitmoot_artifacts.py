from __future__ import annotations

import pytest

from gitmoot_skillopt.artifacts import (
    ArtifactError,
    GitmootArtifactResolver,
    OutputArtifactWriter,
    content_hash,
    normalize_hash,
)


def write_gitmoot_blob(root, content: bytes) -> str:
    hash_value = content_hash(content)
    hex_hash = hash_value.removeprefix("sha256:")
    path = root / "sha256" / hex_hash[:2] / hex_hash
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return hash_value


def test_resolver_reads_gitmoot_sha256_blob(tmp_path):
    content = b"hello artifact\n"
    hash_value = write_gitmoot_blob(tmp_path, content)

    resolved = GitmootArtifactResolver(tmp_path).read(hash_value, expected_size=len(content))

    assert resolved.hash == hash_value
    assert resolved.content == content
    assert resolved.path.name == hash_value.removeprefix("sha256:")


def test_resolver_rejects_missing_blob(tmp_path):
    missing = "sha256:" + "a" * 64

    with pytest.raises(ArtifactError, match="not found"):
        GitmootArtifactResolver(tmp_path).read(missing)


def test_resolver_rejects_empty_root():
    with pytest.raises(ArtifactError, match="artifact root is required"):
        GitmootArtifactResolver("")


def test_resolver_rejects_hash_mismatch(tmp_path):
    expected = "sha256:" + "b" * 64
    hex_hash = expected.removeprefix("sha256:")
    path = tmp_path / "sha256" / hex_hash[:2] / hex_hash
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong")

    with pytest.raises(ArtifactError, match="content hash mismatch"):
        GitmootArtifactResolver(tmp_path).read(expected)


def test_normalize_hash_accepts_prefixed_and_plain_hashes():
    assert normalize_hash("C" * 64) == "sha256:" + "c" * 64
    assert normalize_hash("sha256:" + "D" * 64) == "sha256:" + "d" * 64


def test_output_writer_writes_under_artifacts_and_returns_manifest(tmp_path):
    writer = OutputArtifactWriter(tmp_path)

    entry = writer.write_bytes(
        "reports/diff.md",
        b"# diff\n",
        artifact_id="candidate-diff",
        media_type="text/markdown",
        driver="gitmoot-skillopt",
    )

    assert (tmp_path / "artifacts" / "reports" / "diff.md").read_bytes() == b"# diff\n"
    assert entry.to_dict() == {
        "id": "candidate-diff",
        "hash": content_hash(b"# diff\n"),
        "media_type": "text/markdown",
        "size_bytes": 7,
        "driver": "gitmoot-skillopt",
        "path": "reports/diff.md",
    }


def test_output_writer_rejects_empty_root():
    with pytest.raises(ArtifactError, match="output root is required"):
        OutputArtifactWriter("")


@pytest.mark.parametrize("relative_path", ["../escape.md", "/tmp/escape.md", "nested/../../escape.md"])
def test_output_writer_rejects_escaping_paths(tmp_path, relative_path):
    writer = OutputArtifactWriter(tmp_path)

    with pytest.raises(ArtifactError, match="path"):
        writer.write_bytes(
            relative_path,
            b"bad",
            artifact_id="bad",
            media_type="text/plain",
            driver="test",
        )


def test_output_writer_rejects_symlink_parent_without_outside_mutation(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    artifacts = tmp_path / "out" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "link").symlink_to(outside, target_is_directory=True)
    writer = OutputArtifactWriter(tmp_path / "out")

    with pytest.raises(ArtifactError, match="symlink"):
        writer.write_bytes(
            "link/new/diff.md",
            b"bad",
            artifact_id="bad",
            media_type="text/plain",
            driver="test",
        )

    assert not (outside / "new").exists()
