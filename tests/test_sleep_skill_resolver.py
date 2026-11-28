"""Tests for bounded skill-name resolution (issue #120).

Pure-stdlib (unittest), hermetic (tmpdir only), no API key, no network.
Run:  python -m pytest tests/test_sleep_skill_resolver.py
"""
from __future__ import annotations

import os
import tempfile
import unittest

from skillopt_sleep.config import load_config
from skillopt_sleep.skill_resolver import (
    AMBIGUOUS,
    FOUND,
    MISSING,
    REJECTED,
    normalize_skill_name,
    resolve_skill,
    skill_search_roots,
)


def _write_skill(root, name, body="# skill\n"):
    path = os.path.join(root, name, "SKILL.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


def _canonical(path):
    return os.path.realpath(os.path.abspath(path))


def _symlink(test, source, link_name):
    """Create a symlink, or skip the test where the platform refuses one.

    Windows needs admin or Developer Mode for symlinks, and some CI sandboxes
    disallow them outright. The behaviour under test is a security property
    that only exists where symlinks do, so skipping is correct rather than
    failing on an unrelated platform limitation.
    """
    try:
        os.symlink(source, link_name)
    except (OSError, NotImplementedError, AttributeError) as exc:
        test.skipTest(f"symlinks unavailable on this platform: {exc}")


class TestNormalizeSkillName(unittest.TestCase):
    def test_trims_but_preserves_case_and_punctuation(self):
        self.assertEqual(normalize_skill_name("  Brand-Voice.v2  "), "Brand-Voice.v2")

    def test_rejects_unusable_names(self):
        for bad in ["", "   ", ".", "..", "../escape", "a/b", "a\\b", "/abs/skill",
                    "~/skill", "bad\nname", "bad\x00name", None, 3]:
            self.assertEqual(normalize_skill_name(bad), "", repr(bad))


class TestResolveSkill(unittest.TestCase):
    def test_resolves_a_single_local_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _write_skill(tmp, "example-skill")
            res = resolve_skill("  example-skill  ", [tmp])
            self.assertEqual(res.status, FOUND)
            self.assertTrue(res.ok)
            self.assertEqual(res.path, os.path.realpath(expected))
            self.assertEqual(res.name, "example-skill")

    def test_missing_skill_is_distinct_from_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_skill(tmp, "other-skill")
            res = resolve_skill("example-skill", [tmp])
            self.assertEqual(res.status, MISSING)
            self.assertEqual(res.path, "")
            self.assertEqual(res.candidates, ())

    def test_directory_without_skill_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "example-skill"))
            self.assertEqual(resolve_skill("example-skill", [tmp]).status, MISSING)

    def test_same_skill_in_two_roots_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            local, cache = os.path.join(tmp, "local"), os.path.join(tmp, "cache")
            first = _write_skill(local, "example-skill")
            second = _write_skill(cache, "example-skill")
            res = resolve_skill("example-skill", [local, cache])
            self.assertEqual(res.status, AMBIGUOUS)
            self.assertEqual(res.path, "")
            self.assertEqual(res.candidates,
                             (os.path.realpath(first), os.path.realpath(second)))

    def test_repeated_root_is_not_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_skill(tmp, "example-skill")
            self.assertEqual(resolve_skill("example-skill", [tmp, tmp]).status, FOUND)

    def test_traversal_is_rejected_without_touching_the_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "roots")
            _write_skill(tmp, "outside-skill")
            os.makedirs(root, exist_ok=True)
            res = resolve_skill("../outside-skill", [root])
            self.assertEqual(res.status, REJECTED)
            self.assertEqual(res.path, "")

    def test_plugin_qualified_name_only_resolves_in_that_plugins_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            user = os.path.join(tmp, "claude", "skills")
            wanted = os.path.join(
                tmp, "claude", "plugins", "cache", "market", "toolbox", "0.6.4", "skills"
            )
            other = os.path.join(
                tmp, "claude", "plugins", "cache", "market", "other", "0.6.4", "skills"
            )
            _write_skill(user, "doc-truth")
            expected = _write_skill(wanted, "doc-truth")
            _write_skill(other, "doc-truth")

            result = resolve_skill("toolbox:doc-truth", [user, wanted, other])

            self.assertEqual(result.status, FOUND)
            self.assertEqual(result.name, "doc-truth")
            self.assertEqual(result.path, os.path.realpath(expected))

    def test_legacy_plugin_root_resolves_qualified_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(
                tmp, "claude", "plugins", "cache", "market", "toolbox", "skills"
            )
            expected = _write_skill(root, "doc-truth")

            result = resolve_skill("toolbox:doc-truth", [root])

            self.assertEqual(result.status, FOUND)
            self.assertEqual(result.path, os.path.realpath(expected))

    def test_plugin_qualified_name_across_marketplaces_remains_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(
                tmp, "claude", "plugins", "cache", "market-a", "toolbox", "1", "skills"
            )
            second = os.path.join(
                tmp, "claude", "plugins", "cache", "market-b", "toolbox", "1", "skills"
            )
            _write_skill(first, "doc-truth")
            _write_skill(second, "doc-truth")

            result = resolve_skill("toolbox:doc-truth", [first, second])

            self.assertEqual(result.status, AMBIGUOUS)
            self.assertEqual(len(result.candidates), 2)

    def test_malformed_plugin_qualified_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in (":doc-truth", "toolbox:", "toolbox:../outside", "toolbox:doc:truth"):
                with self.subTest(name=name):
                    self.assertEqual(resolve_skill(name, [tmp]).status, REJECTED)

    def test_symlinked_skill_dir_escaping_the_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "roots")
            os.makedirs(root)
            outside = os.path.join(tmp, "outside")
            _write_skill(outside, "example-skill")
            _symlink(self, os.path.join(outside, "example-skill"),
                     os.path.join(root, "example-skill"))
            self.assertEqual(resolve_skill("example-skill", [root]).status, MISSING)

    def test_symlinked_skill_file_escaping_the_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "roots")
            os.makedirs(os.path.join(root, "example-skill"))
            elsewhere = os.path.join(tmp, "elsewhere.md")
            with open(elsewhere, "w", encoding="utf-8") as f:
                f.write("# not in the root\n")
            _symlink(self, elsewhere, os.path.join(root, "example-skill", "SKILL.md"))
            self.assertEqual(resolve_skill("example-skill", [root]).status, MISSING)

    def test_symlinked_root_itself_still_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = os.path.join(tmp, "real")
            _write_skill(real, "example-skill")
            link = os.path.join(tmp, "link")
            _symlink(self, real, link)
            self.assertEqual(resolve_skill("example-skill", [link]).status, FOUND)

    def test_resolution_never_modifies_the_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_skill(tmp, "example-skill", "# original\n")
            with open(path, encoding="utf-8") as f:
                before = (os.stat(path).st_size, f.read())
            resolve_skill("example-skill", [tmp])
            with open(path, encoding="utf-8") as f:
                after = (os.stat(path).st_size, f.read())
            self.assertEqual(after, before)

    def test_no_roots_is_missing(self):
        self.assertEqual(resolve_skill("example-skill", []).status, MISSING)

    def test_candidates_are_an_immutable_tuple(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_skill(tmp, "example-skill")
            res = resolve_skill("example-skill", [tmp])
            self.assertIsInstance(res.candidates, tuple)
            with self.assertRaises(AttributeError):
                res.candidates.append("/injected")  # type: ignore[attr-defined]

    def test_skill_file_on_another_drive_is_refused_not_crashed(self):
        # os.path.commonpath raises ValueError for paths that share no root
        # (mixed drives on Windows). Resolution must treat that as "outside".
        with tempfile.TemporaryDirectory() as tmp:
            _write_skill(tmp, "example-skill")
            real_commonpath = os.path.commonpath

            def exploding_commonpath(paths):
                raise ValueError("paths don't have the same drive")

            os.path.commonpath = exploding_commonpath
            try:
                self.assertEqual(resolve_skill("example-skill", [tmp]).status, MISSING)
            finally:
                os.path.commonpath = real_commonpath


class TestSkillSearchRoots(unittest.TestCase):
    def test_user_skills_root_comes_first_then_plugin_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            skills = os.path.join(claude_home, "skills")
            os.makedirs(skills)
            plugin_skills = os.path.join(
                claude_home, "plugins", "cache", "marketplace", "plugin", "skills"
            )
            os.makedirs(plugin_skills)
            cfg = load_config(claude_home=claude_home)
            self.assertEqual(
                skill_search_roots(cfg), [_canonical(skills), _canonical(plugin_skills)]
            )

    def test_absent_roots_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(claude_home=os.path.join(tmp, ".claude"))
            self.assertEqual(skill_search_roots(cfg), [])

    def test_blank_claude_home_never_falls_back_to_the_cwd(self):
        # os.path.abspath("") is the CWD; a blank override must not turn the
        # working directory into a skill root.
        class _Cfg:
            def __init__(self, claude_home):
                self.claude_home = claude_home

        for blank in ["", "   ", None]:
            self.assertEqual(skill_search_roots(_Cfg(blank)), [], repr(blank))

    def test_project_native_agent_roots_and_explicit_roots_are_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = []
            for relative in (
                os.path.join(".agents", "skills"),
                os.path.join(".claude", "skills"),
                os.path.join(".cursor", "skills"),
                os.path.join(".devin", "skills"),
                "custom-skills",
            ):
                root = os.path.join(tmp, relative)
                os.makedirs(root)
                expected.append(os.path.realpath(root))
            cfg = load_config(
                invoked_project=tmp,
                claude_home="",
                codex_home="",
                cursor_home="",
                skill_roots=["custom-skills"],
            )
            self.assertEqual(skill_search_roots(cfg), expected)

    def test_native_roots_make_duplicate_agent_definitions_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            for relative in (".agents/skills", ".devin/skills"):
                _write_skill(os.path.join(tmp, relative), "shared-skill")
            cfg = load_config(
                invoked_project=tmp,
                claude_home="",
                codex_home="",
                cursor_home="",
            )
            result = resolve_skill("shared-skill", skill_search_roots(cfg))
            self.assertEqual(result.status, AMBIGUOUS)
            self.assertEqual(len(result.candidates), 2)

    def test_unreadable_plugin_cache_does_not_break_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            skills = os.path.join(claude_home, "skills")
            os.makedirs(skills)
            cache = os.path.join(claude_home, "plugins", "cache")
            os.makedirs(cache)
            try:
                os.chmod(cache, 0o000)
            except OSError as exc:  # pragma: no cover - platform dependent
                self.skipTest(f"cannot drop permissions on this platform: {exc}")
            if os.access(cache, os.R_OK):
                # Windows and some filesystems ignore mode bits, and root
                # bypasses them, so the unreadable precondition never holds.
                os.chmod(cache, 0o700)
                self.skipTest("directory is still readable after chmod 000")
            try:
                cfg = load_config(claude_home=claude_home)
                self.assertEqual(skill_search_roots(cfg), [_canonical(skills)])
            finally:
                try:
                    os.chmod(cache, 0o700)
                except OSError:  # pragma: no cover - best-effort restore
                    pass

    def test_resolution_through_config_roots_prefers_the_user_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            expected = _write_skill(os.path.join(claude_home, "skills"), "example-skill")
            cfg = load_config(claude_home=claude_home)
            res = resolve_skill("example-skill", skill_search_roots(cfg))
            self.assertEqual(res.status, FOUND)
            self.assertEqual(res.path, os.path.realpath(expected))

    def test_versioned_marketplace_layout_is_discovered(self):
        # Sanitized mirror of a real marketplace install. Observed layout on a
        # working machine: every plugin skills dir sits at
        # <cache>/<marketplace>/<plugin>/<version>/skills — the unversioned
        # layout did not occur at all, so discovery must handle this one.
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            cache = os.path.join(claude_home, "plugins", "cache")
            expected = _write_skill(
                os.path.join(cache, "claude-plugins-official", "superpowers", "5.0.7", "skills"),
                "brainstorming",
            )
            cfg = load_config(claude_home=claude_home)
            self.assertIn(
                _canonical(os.path.join(
                    cache, "claude-plugins-official", "superpowers", "5.0.7", "skills"
                )),
                skill_search_roots(cfg),
            )
            res = resolve_skill("brainstorming", skill_search_roots(cfg))
            self.assertEqual(res.status, FOUND)
            self.assertEqual(res.path, os.path.realpath(expected))

    def test_multiple_installed_versions_resolve_to_the_newest_not_ambiguous(self):
        # A plugin upgraded in place keeps older version dirs alongside the new
        # one (observed: three versions of the same plugin). Treating each as a
        # peer root would make an ordinary upgrade resolve AMBIGUOUS.
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            plugin = _canonical(os.path.join(
                claude_home, "plugins", "cache",
                "claude-plugins-official", "chrome-devtools-mcp"
            ))
            for version in ["1.1.1", "1.5.0", "1.6.0"]:
                _write_skill(os.path.join(plugin, version, "skills"), "chrome-devtools")
            newest = _canonical(os.path.join(plugin, "1.6.0", "skills"))

            cfg = load_config(claude_home=claude_home)
            roots = skill_search_roots(cfg)
            self.assertEqual([r for r in roots if r.startswith(plugin)], [newest])

            res = resolve_skill("chrome-devtools", roots)
            self.assertEqual(res.status, FOUND)
            self.assertEqual(res.path,
                             os.path.realpath(os.path.join(newest, "chrome-devtools", "SKILL.md")))

    def test_version_ordering_is_numeric_not_lexicographic(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            plugin = _canonical(os.path.join(
                claude_home, "plugins", "cache", "market", "plugin"
            ))
            for version in ["1.9.0", "1.10.0"]:
                _write_skill(os.path.join(plugin, version, "skills"), "example-skill")
            cfg = load_config(claude_home=claude_home)
            roots = [r for r in skill_search_roots(cfg) if r.startswith(plugin)]
            # "1.10.0" < "1.9.0" as strings; it must still win as a version.
            self.assertEqual(roots, [_canonical(os.path.join(plugin, "1.10.0", "skills"))])

    def test_stable_release_beats_an_installed_prerelease(self):
        # Segment lists alone would rank 2.0.0-beta above 2.0.0, because a
        # shorter list that prefixes a longer one sorts lower. A prerelease
        # must never be preferred over the stable release it precedes.
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            plugin = _canonical(os.path.join(
                claude_home, "plugins", "cache", "market", "plugin"
            ))
            for version in ["2.0.0", "2.0.0-beta"]:
                _write_skill(os.path.join(plugin, version, "skills"), "example-skill")
            cfg = load_config(claude_home=claude_home)
            roots = [r for r in skill_search_roots(cfg) if r.startswith(plugin)]
            self.assertEqual(roots, [_canonical(os.path.join(plugin, "2.0.0", "skills"))])

    def test_version_key_orders_release_forms_sensibly(self):
        from skillopt_sleep.skill_resolver import _version_sort_key as key
        self.assertGreater(key("2.0.0"), key("2.0.0-beta"))
        self.assertGreater(key("1.10.0"), key("1.9.0"))
        self.assertGreater(key("2.0.0-beta"), key("2.0.0-alpha"))
        self.assertGreater(key("1.0.0"), key("1.0"))
        self.assertGreater(key("1.0.0"), key("1.0.0rc1"))
        self.assertGreater(key("1.0.0"), key("main"))

    def test_legacy_unversioned_layout_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            plugin = os.path.join(claude_home, "plugins", "cache", "market", "plugin")
            expected = _write_skill(os.path.join(plugin, "skills"), "example-skill")
            cfg = load_config(claude_home=claude_home)
            self.assertIn(_canonical(os.path.join(plugin, "skills")), skill_search_roots(cfg))
            res = resolve_skill("example-skill", skill_search_roots(cfg))
            self.assertEqual(res.status, FOUND)
            self.assertEqual(res.path, os.path.realpath(expected))

    def test_two_marketplaces_each_contribute_a_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            cache = os.path.join(claude_home, "plugins", "cache")
            official = os.path.join(cache, "claude-plugins-official", "sentry", "1.0.0", "skills")
            cognee = os.path.join(cache, "cognee", "cognee-memory", "1.0.0", "skills")
            _write_skill(official, "sentry-skill")
            _write_skill(cognee, "cognee-remember")
            cfg = load_config(claude_home=claude_home)
            roots = skill_search_roots(cfg)
            self.assertIn(_canonical(official), roots)
            self.assertIn(_canonical(cognee), roots)

    def test_legacy_target_skill_path_behavior_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "repo", "SKILL.md")
            cfg = load_config(claude_home=os.path.join(tmp, ".claude"),
                              target_skill_path=target)
            self.assertEqual(cfg.managed_skill_path(), os.path.abspath(target))


if __name__ == "__main__":
    unittest.main()
