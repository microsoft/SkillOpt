"""SkillOpt-Sleep — resolve a discovered skill name to a local ``SKILL.md``.

Skill names observed in transcripts are untrusted strings, so resolution is
deliberately narrow: a name is normalized, matched only inside documented local
skill roots, and reported. Nothing here writes, edits, or creates files.

Resolution outcomes are distinguishable on purpose — ``missing`` (no root has
the skill) is a different signal from ``ambiguous`` (several roots do) and from
``rejected`` (the name itself is unusable), so callers can fall back to the
existing managed-skill behavior instead of guessing.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

SKILL_FILENAME = "SKILL.md"

FOUND = "found"
MISSING = "missing"
AMBIGUOUS = "ambiguous"
REJECTED = "rejected"


@dataclass(frozen=True)
class SkillResolution:
    """The outcome of resolving one skill name. Never a partial success."""

    name: str
    status: str
    path: str = ""
    candidates: Tuple[str, ...] = ()
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == FOUND


def normalize_skill_name(name: object) -> str:
    """Return a usable skill directory name, or "" when the name is unusable.

    Only a single path segment is accepted: no separators, no parent traversal,
    no absolute or home-relative paths, no control characters. The name is
    whitespace-trimmed but otherwise preserved, since skill directories are
    case- and punctuation-sensitive.
    """
    if not isinstance(name, str):
        return ""
    candidate = name.strip()
    if not candidate or candidate in {os.curdir, os.pardir}:
        return ""
    if candidate.startswith("~"):
        return ""
    if os.path.isabs(candidate) or os.path.splitdrive(candidate)[0]:
        return ""
    if "/" in candidate or "\\" in candidate or os.sep in candidate:
        return ""
    if os.altsep and os.altsep in candidate:
        return ""
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in candidate):
        return ""
    return candidate


def _listdir(path: str) -> List[str]:
    """Sorted directory entries, or [] when the directory is absent/unreadable."""
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def _version_sort_key(name: str) -> tuple:
    """Order version directory names newest-last, numerically where possible.

    ``1.10.0`` must sort above ``1.9.0``, so the leading numeric segments
    compare as ints rather than as strings. Anything after the first
    non-numeric segment is a prerelease suffix (``2.0.0-beta``), and a bare
    release outranks any prerelease sharing its numeric prefix — comparing the
    segment lists alone would do the opposite, because a shorter list that is a
    prefix of a longer one sorts lower. The name is the final tie-break so the
    order is always total and deterministic.
    """
    numeric: List[int] = []
    suffix: List[str] = []
    for part in re.split(r"[._\-+]", name):
        if part.isdigit() and not suffix:
            numeric.append(int(part))
        else:
            suffix.append(part)
    return (numeric, 0 if suffix else 1, suffix, name)


def _plugin_skills_root(plugin_dir: str) -> str:
    """The single skills root for one installed plugin, or "" if it has none.

    Claude marketplace installs are versioned —
    ``<plugin>/<version>/skills`` — and several versions of the same plugin can
    be present at once. Returning each of them as a peer root would make an
    ordinary upgraded plugin resolve AMBIGUOUS, so exactly one root is chosen:
    the newest version, falling back to the legacy unversioned
    ``<plugin>/skills`` layout when no version directory carries skills.
    """
    versioned = []
    for entry in _listdir(plugin_dir):
        candidate = os.path.join(plugin_dir, entry, "skills")
        if os.path.isdir(candidate):
            versioned.append((_version_sort_key(entry), candidate))
    if versioned:
        versioned.sort()
        return versioned[-1][1]

    legacy = os.path.join(plugin_dir, "skills")
    return legacy if os.path.isdir(legacy) else ""


def skill_search_roots(cfg: object) -> List[str]:
    """Return every configured native skill root in deterministic order.

    The resolver supports project-native Claude/Codex/Cursor/Devin layouts,
    configured user homes, explicit extra roots, and installed Claude plugins.
    Returning all matches is intentional: ``resolve_skill`` refuses ambiguity
    instead of silently choosing one agent's file over another.
    """
    roots: List[str] = []

    invoked_project = str(getattr(cfg, "invoked_project", "") or "").strip()
    if invoked_project:
        project = os.path.abspath(os.path.expanduser(invoked_project))
        roots.extend(
            os.path.join(project, relative)
            for relative in (
                os.path.join(".agents", "skills"),
                os.path.join(".claude", "skills"),
                os.path.join(".cursor", "skills"),
                os.path.join(".devin", "skills"),
            )
        )

    # Keep the established user-level Claude root. Other agents' user-level
    # layouts are not sufficiently standardized; callers can add them through
    # ``skill_roots`` without silently creating cross-agent ambiguity.
    configured_claude = str(getattr(cfg, "claude_home", "") or "").strip()
    if configured_claude:
        roots.append(
            os.path.join(
                os.path.abspath(os.path.expanduser(configured_claude)),
                "skills",
            )
        )

    explicit = getattr(cfg, "skill_roots", ())
    if isinstance(explicit, (list, tuple)):
        for value in explicit:
            if isinstance(value, str) and value.strip():
                path = os.path.expanduser(value.strip())
                if not os.path.isabs(path) and invoked_project:
                    path = os.path.join(invoked_project, path)
                roots.append(os.path.abspath(path))

    if configured_claude:
        claude_home = os.path.abspath(os.path.expanduser(configured_claude))
        cache = os.path.join(claude_home, "plugins", "cache")
        for marketplace in _listdir(cache):
            plugins_dir = os.path.join(cache, marketplace)
            if not os.path.isdir(plugins_dir):
                continue
            for plugin in _listdir(plugins_dir):
                root = _plugin_skills_root(os.path.join(plugins_dir, plugin))
                if root:
                    roots.append(root)

    existing: List[str] = []
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        canonical = os.path.realpath(root)
        key = os.path.normcase(canonical)
        if key not in seen:
            seen.add(key)
            existing.append(canonical)
    return existing


def _contained_skill_file(root: str, name: str) -> str:
    """Return the real ``SKILL.md`` path under ``root`` for ``name``, else "".

    Symlinks are followed and then re-checked against the real root, so a skill
    directory or file that points outside the root is refused rather than read.
    """
    try:
        real_root = os.path.realpath(root)
        skill_file = os.path.realpath(os.path.join(real_root, name, SKILL_FILENAME))
    except OSError:
        return ""
    if not os.path.isfile(skill_file):
        return ""
    try:
        contained = os.path.commonpath([real_root, skill_file]) == real_root
    except ValueError:
        # Different drives / mixed path flavours (Windows): by definition the
        # file is not inside this root, so refuse it rather than crash.
        return ""
    if not contained:
        return ""
    return skill_file


def _plugin_name_for_root(root: str) -> str:
    """Return the Claude plugin name for an installed plugin skills root.

    Installed Claude plugin roots have the versioned shape
    ``.../plugins/cache/<marketplace>/<plugin>/<version>/skills`` and a
    legacy unversioned shape ``.../plugins/cache/<marketplace>/<plugin>/skills``.
    Requiring one of those shapes prevents an explicit user skill root from
    accidentally being treated as a plugin root just because its directory
    happens to be named like one.
    """
    parts = os.path.normpath(os.path.realpath(root)).split(os.sep)
    if (
        len(parts) >= 8
        and parts[-1] == "skills"
        and parts[-6:-4] == ["plugins", "cache"]
    ):
        return parts[-3]
    if (
        len(parts) >= 7
        and parts[-1] == "skills"
        and parts[-5:-3] == ["plugins", "cache"]
    ):
        return parts[-2]
    return ""


def resolve_skill(name: object, roots: Sequence[str]) -> SkillResolution:
    """Resolve ``name`` against ``roots`` without touching any skill content."""
    original = name.strip() if isinstance(name, str) else ""
    plugin_name = ""
    if ":" in original:
        if original.count(":") != 1:
            return SkillResolution(
                name=original,
                status=REJECTED,
                reason="plugin-qualified skill name must contain one safe plugin and skill name",
            )
        raw_plugin, raw_skill = original.split(":", 1)
        plugin_name = normalize_skill_name(raw_plugin)
        normalized = normalize_skill_name(raw_skill)
        if not plugin_name or not normalized:
            normalized = ""
    else:
        normalized = normalize_skill_name(name)

    if not normalized:
        return SkillResolution(
            name=name if isinstance(name, str) else "",
            status=REJECTED,
            reason=(
                "plugin-qualified skill name must contain one safe plugin and skill name"
                if plugin_name or ":" in original
                else "skill name is empty or not a single safe path segment"
            ),
        )

    matches: List[str] = []
    for root in roots:
        if plugin_name and _plugin_name_for_root(root) != plugin_name:
            continue
        found = _contained_skill_file(root, normalized)
        if found and found not in matches:
            matches.append(found)

    if not matches:
        return SkillResolution(
            name=normalized,
            status=MISSING,
            reason=f"no {SKILL_FILENAME} for this skill in the configured skill roots",
        )
    if len(matches) > 1:
        return SkillResolution(
            name=normalized,
            status=AMBIGUOUS,
            candidates=tuple(matches),
            reason="several skill roots define this skill",
        )
    return SkillResolution(
        name=normalized, status=FOUND, path=matches[0], candidates=tuple(matches)
    )
