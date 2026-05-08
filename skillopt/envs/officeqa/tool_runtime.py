from __future__ import annotations

import fnmatch
import os
from pathlib import Path

_MAX_READ_CHARS = 4000
_MAX_GREP_MATCHES = 20
_MAX_GLOB_MATCHES = 50


def _normalize_data_dirs(data_dirs: list[str] | tuple[str, ...] | str | None, project_root: Path) -> list[str]:
    if data_dirs is None:
        return []
    if isinstance(data_dirs, str):
        items = [part.strip() for chunk in data_dirs.split(os.pathsep) for part in chunk.split(",")]
    else:
        items = [str(item).strip() for item in data_dirs]
    resolved: list[str] = []
    for item in items:
        if not item:
            continue
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = project_root / path
        resolved.append(str(path))
    return resolved


def resolve_docs_roots(data_dirs: list[str] | tuple[str, ...] | str | None = None) -> list[str]:
    project_root = Path(__file__).resolve().parents[3]
    env_value = os.environ.get("OFFICEQA_DOCS_DIR", "").strip()
    candidates = _normalize_data_dirs(data_dirs, project_root)
    candidates.extend(_normalize_data_dirs(env_value, project_root))
    candidates.extend([
        str(project_root / "data" / "officeqa_docs_official"),
        str(project_root / "data" / "officeqa_smoke_docs"),
        os.path.expanduser("~/officeqa-sparse/treasury_bulletins_parsed"),
        os.path.expanduser("~/officeqa/treasury_bulletins_parsed"),
    ])
    roots: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_dir():
            continue
        transformed = path / "transformed"
        resolved = str((transformed if transformed.is_dir() else path).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    if not roots:
        raise FileNotFoundError("OfficeQA docs directory not found. Set OFFICEQA_DOCS_DIR or env.data_dirs.")
    return roots


def _is_allowed(path: str, allowed_roots: list[str], allowed_files: list[str]) -> bool:
    try:
        resolved = str(Path(path).resolve())
    except FileNotFoundError:
        return False
    if not any(resolved.startswith(root + os.sep) or resolved == root for root in allowed_roots):
        return False
    if not allowed_files:
        return True
    base = os.path.basename(resolved)
    return base in allowed_files


def resolve_candidate_files(source_files: list[str], allowed_roots: list[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for root in allowed_roots:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if source_files and filename not in source_files:
                    continue
                full = str(Path(dirpath, filename).resolve())
                if full in seen:
                    continue
                seen.add(full)
                resolved.append(full)
    return resolved


def run_tool(name: str, arguments: dict, *, allowed_roots: list[str], allowed_files: list[str]) -> tuple[str, str]:
    if name == "glob":
        pattern = str(arguments.get("pattern") or "*")
        matches: list[str] = []
        for root in allowed_roots:
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    if allowed_files and filename not in allowed_files:
                        continue
                    rel = os.path.relpath(os.path.join(dirpath, filename), root)
                    if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(filename, pattern):
                        matches.append(os.path.join(dirpath, filename))
                    if len(matches) >= _MAX_GLOB_MATCHES:
                        break
                if len(matches) >= _MAX_GLOB_MATCHES:
                    break
        return f"glob(pattern={pattern!r})", "\n".join(matches) if matches else "[no matches]"

    if name == "read":
        path = str(arguments.get("path") or "")
        if not path:
            return "read(path='')", "[read error: missing path]"
        if not _is_allowed(path, allowed_roots, allowed_files):
            return f"read(path={path!r})", "[read error: path not allowed]"
        start = max(int(arguments.get("start") or 1), 1)
        limit = max(int(arguments.get("limit") or 80), 1)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        excerpt = "".join(lines[start - 1:start - 1 + limit])
        return f"read(path={path!r}, start={start}, limit={limit})", excerpt[:_MAX_READ_CHARS] or "[empty file]"

    if name == "grep":
        pattern = str(arguments.get("pattern") or "").lower()
        path = str(arguments.get("path") or "")
        if not pattern or not path:
            return f"grep(pattern={pattern!r}, path={path!r})", "[grep error: missing pattern or path]"
        if not _is_allowed(path, allowed_roots, allowed_files):
            return f"grep(pattern={pattern!r}, path={path!r})", "[grep error: path not allowed]"
        matches: list[str] = []
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if pattern in line.lower():
                    matches.append(f"{idx}: {line.rstrip()}")
                if len(matches) >= _MAX_GREP_MATCHES:
                    break
        return f"grep(pattern={pattern!r}, path={path!r})", "\n".join(matches) if matches else "[no matches]"

    return name, f"[tool error: unknown tool {name}]"
