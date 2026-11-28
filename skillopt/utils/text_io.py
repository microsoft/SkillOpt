"""Read run files, including ones older releases wrote with the locale codec.

Skills, histories, patches, trajectories and rollout records are written as
UTF-8. Earlier releases left the codec to ``open()``, so a run started on a
Windows machine holds cp1252 (or GBK, ...) bytes. These readers try UTF-8 and
fall back to that codec, so such a run can still be resumed and its skills
evaluated.
"""
from __future__ import annotations

import locale


def locale_codec() -> str:
    """The codec ``open()`` picks when no encoding is given.

    Kept behind a function so tests can stand in a non-UTF-8 locale on a
    runner whose locale is UTF-8.
    """
    return locale.getpreferredencoding(False)


def read_text(path: str) -> str:
    """Return the text of ``path``: UTF-8, or the locale codec for older files."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, encoding=locale_codec()) as f:
            return f.read()


def read_lines(path: str) -> list[str]:
    """Return the lines of a JSONL file, each decoded like :func:`read_text`.

    Lines are decoded one by one because a resumed rollout appends UTF-8
    records to a file an earlier release may have started. A line valid in
    neither codec, such as a record cut off mid-write, is decoded with
    replacement characters instead of failing the whole read.
    """
    lines: list[str] = []
    with open(path, "rb") as f:
        for raw in f:
            try:
                lines.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                lines.append(raw.decode(locale_codec(), errors="replace"))
    return lines
