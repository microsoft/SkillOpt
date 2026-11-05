from __future__ import annotations

import hashlib
import os
import re

_SAFE_TASK_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def is_safe_task_id(task_id: str) -> bool:
    """Return whether a task id is safe as one portable path segment."""
    value = str(task_id)
    return bool(_SAFE_TASK_ID.match(value)) and ".." not in value


def task_output_segment(task_id: str, *, map_unsafe: bool = False) -> str:
    """Return the stable directory segment for a task id."""
    value = str(task_id)
    if is_safe_task_id(value):
        return value
    if not map_unsafe:
        raise ValueError(f"unsafe task id: {value!r}")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"_mapped-{digest}"


def confined_task_output_dir(
    out_root: str,
    task_id: str,
    *,
    map_unsafe: bool = False,
) -> str:
    """Return a task directory confined below ``out_root/predictions``."""
    predictions = os.path.join(os.path.abspath(out_root), "predictions")
    destination = os.path.join(
        predictions,
        task_output_segment(task_id, map_unsafe=map_unsafe),
    )
    if os.path.commonpath(
        [os.path.realpath(predictions), os.path.realpath(destination)]
    ) != os.path.realpath(predictions):
        raise ValueError(f"task output escapes predictions: {destination!r}")
    return destination


def confined_legacy_task_output_dir(out_root: str, task_id: str) -> str:
    """Return an existing raw-id task directory without allowing an escape."""
    predictions = os.path.join(os.path.abspath(out_root), "predictions")
    destination = os.path.join(predictions, str(task_id))
    if os.path.commonpath(
        [os.path.realpath(predictions), os.path.realpath(destination)]
    ) != os.path.realpath(predictions):
        raise ValueError(f"legacy task output escapes predictions: {destination!r}")
    return destination
