"""Tests for read-only Google Antigravity conversation harvesting.

Fixtures are built rather than checked in: Antigravity's trajectory stores are
protobuf blobs inside SQLite, so a synthetic encoder keeps the expected shape
visible in the test itself and lets each case vary one thing (a step type, a
workspace, a WAL state, a secret) without hand-editing binary files.
"""
from __future__ import annotations

import os
import sqlite3
import struct
import tempfile
import time
import unittest
from unittest import mock

from skillopt_sleep.config import load_config
from skillopt_sleep.harvest_antigravity import (
    _path_from_file_uri,
    _read_direct,
    _read_rows,
    _read_via_backup,
    digest_antigravity_db,
    harvest_antigravity,
)
from skillopt_sleep.harvest_sources import harvest_for_config


# ── minimal protobuf encoder (mirrors what Antigravity writes) ────────────────

def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _bytes_field(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _str_field(field: int, text: str) -> bytes:
    return _bytes_field(field, text.encode("utf-8"))


def _varint_field(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(value)


def _path_to_uri(path: str) -> str:
    from urllib.parse import quote
    normalized = path.replace("\\", "/")
    if not normalized.startswith("/"):
        normalized = "/" + normalized  # drive-letter form: /C:/Users/...
    return "file://" + quote(normalized, safe="/:")


def metadata_blob(
    *,
    workspace: str = "",
    branch: str = "",
    created: int = 1_785_483_868,
    session_id: str = "11111111-2222-3333-4444-555555555555",
) -> bytes:
    """Build a ``trajectory_metadata_blob`` payload.

    With ``workspace`` empty this produces the ``outside-of-project`` form
    Antigravity writes for conversations started without a workspace.
    """
    blob = b""
    if workspace:
        uri = _path_to_uri(workspace)
        repo = _str_field(1, uri) + _str_field(2, uri)
        if branch:
            repo += _str_field(4, branch)
        blob += _bytes_field(1, repo)
    blob += _bytes_field(2, _varint_field(1, created) + _varint_field(2, 0))
    blob += _str_field(3, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    blob += _str_field(6, session_id)
    if workspace:
        blob += _str_field(7, _path_to_uri(workspace))
        blob += _str_field(18, "351053ef-3b69-4a5b-b9e5-a16e705ba970")
    else:
        blob += _str_field(18, "outside-of-project")
    return blob


def step_payload(text: str) -> bytes:
    """A step payload with the text nested one message deep, as observed."""
    return _bytes_field(3, _bytes_field(1, _str_field(2, text)))


class AntigravityFixture:
    """Builds a conversations directory of synthetic trajectory databases."""

    USER, ARTIFACT, TOOL = 14, 5, 33

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def write(
        self,
        session_id: str,
        steps,
        *,
        workspace: str = "",
        branch: str = "",
        created: int = 1_785_483_868,
        journal_mode: str = "delete",
        keep_open: bool = False,
        with_metadata: bool = True,
        mtime: float = 0.0,
    ):
        path = os.path.join(self.root, f"{session_id}.db")
        con = sqlite3.connect(path)
        con.execute(f"PRAGMA journal_mode={journal_mode}")
        con.execute(
            "CREATE TABLE steps (idx integer, step_type integer NOT NULL DEFAULT 0,"
            " step_payload blob)"
        )
        con.execute(
            "CREATE TABLE trajectory_metadata_blob (id text DEFAULT 'main',"
            " data blob, PRIMARY KEY (id))"
        )
        for idx, (step_type, payload) in enumerate(steps):
            con.execute(
                "INSERT INTO steps VALUES (?, ?, ?)",
                (idx, step_type, payload if isinstance(payload, bytes) else step_payload(payload)),
            )
        if with_metadata:
            con.execute(
                "INSERT INTO trajectory_metadata_blob VALUES (?, ?)",
                ("main", metadata_blob(workspace=workspace, branch=branch,
                                       created=created, session_id=session_id)),
            )
        con.commit()
        if keep_open:
            # Leave the writer attached, exactly like a live Antigravity editor:
            # in WAL mode the committed rows still live in the -wal sidecar.
            self._open = getattr(self, "_open", [])
            self._open.append(con)
        else:
            con.close()
        if mtime:
            os.utime(path, (mtime, mtime))
        return path

    def close(self):
        for con in getattr(self, "_open", []):
            try:
                con.close()
            except sqlite3.Error:
                pass


# ── tests ─────────────────────────────────────────────────────────────────────

class TestAntigravityStepExtraction(unittest.TestCase):
    def test_extracts_user_artifact_and_tool_step_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = AntigravityFixture(os.path.join(tmp, "conversations"))
            path = fx.write("s1", [
                (fx.USER, "/goal Refactor the retry helper so it backs off."),
                (fx.ARTIFACT, "I refactored the helper to use exponential backoff."),
                (fx.TOOL, step_payload('{"toolSummary": "run_command", "x": 1}')),
                (fx.USER, "That is still broken on the third attempt, please fix it."),
            ], workspace=os.path.join(tmp, "proj"))

            digest = digest_antigravity_db(path)

        self.assertIsNotNone(digest)
        self.assertEqual(digest.session_id, "s1")
        # step 14 -> prompts, with the /goal wrapper stripped
        self.assertEqual(digest.user_prompts, [
            "Refactor the retry helper so it backs off.",
            "That is still broken on the third attempt, please fix it.",
        ])
        # step 5 -> assistant finals
        self.assertEqual(digest.assistant_finals,
                         ["I refactored the helper to use exponential backoff."])
        # step 33 -> tool names
        self.assertEqual(digest.tools_used, ["run_command"])
        self.assertEqual(digest.n_user_turns, 2)
        self.assertTrue(digest.feedback_signals)

    def test_preserves_unicode_prompts(self):
        prompt = "Peux-tu corriger l'accentuation ? 変換もお願いします — emoji 🎯 too."
        with tempfile.TemporaryDirectory() as tmp:
            fx = AntigravityFixture(os.path.join(tmp, "conversations"))
            path = fx.write("s1", [(fx.USER, prompt)], workspace=os.path.join(tmp, "p"))
            digest = digest_antigravity_db(path)

        self.assertIsNotNone(digest)
        self.assertEqual(digest.user_prompts, [prompt])

    def test_tolerates_schema_drift_and_unknown_wire_types(self):
        """Reshaped payloads degrade to fewer strings, never to an exception."""
        with tempfile.TemporaryDirectory() as tmp:
            fx = AntigravityFixture(os.path.join(tmp, "conversations"))
            drifted = (
                _varint_field(99, 7)                       # unknown scalar field
                + _bytes_field(41, b"\x00\x01\x02\xff\xfe")  # non-UTF8 blob
                + _tag(12, 1) + struct.pack("<d", 1.5)      # 64-bit wire type
                + _tag(13, 5) + struct.pack("<f", 2.5)      # 32-bit wire type
                + _bytes_field(7, _str_field(3, "Please regenerate the changelog entry."))
            )
            path = fx.write("s1", [
                (fx.USER, drifted),
                (fx.USER, b"\xff\xfe not a protobuf at all"),
                (fx.USER, "A perfectly ordinary follow-up question here."),
            ], workspace=os.path.join(tmp, "p"))

            digest = digest_antigravity_db(path)

        self.assertIsNotNone(digest)
        self.assertIn("Please regenerate the changelog entry.", digest.user_prompts)
        self.assertIn("A perfectly ordinary follow-up question here.", digest.user_prompts)

    def test_session_without_recoverable_prompts_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = AntigravityFixture(os.path.join(tmp, "conversations"))
            path = fx.write("s1", [(fx.ARTIFACT, "Only an artifact, no user prompt.")],
                            workspace=os.path.join(tmp, "p"))
            self.assertIsNone(digest_antigravity_db(path))


class TestAntigravityConsistentReads(unittest.TestCase):
    def test_reads_rows_still_living_in_the_wal(self):
        """The bug a plain .db copy had: WAL commits are invisible to it."""
        with tempfile.TemporaryDirectory() as tmp:
            conversations = os.path.join(tmp, "conversations")
            fx = AntigravityFixture(conversations)
            path = fx.write("s1", [
                (fx.USER, "Explain why the WAL snapshot matters for harvesting."),
            ], workspace=os.path.join(tmp, "proj"),
                journal_mode="wal", keep_open=True)
            try:
                self.assertTrue(os.path.exists(path + "-wal"),
                                "fixture should leave an un-checkpointed WAL")
                # A bare file copy loses the WAL contents entirely...
                copy = os.path.join(tmp, "copy.db")
                with open(path, "rb") as src, open(copy, "wb") as dst:
                    dst.write(src.read())
                stale = sqlite3.connect(copy)
                try:
                    with self.assertRaises(sqlite3.DatabaseError):
                        stale.execute("SELECT count(*) FROM steps").fetchone()
                finally:
                    stale.close()

                # ...while the harvester reads every committed row.
                digest = digest_antigravity_db(path)
            finally:
                fx.close()

        self.assertIsNotNone(digest)
        self.assertEqual(digest.user_prompts,
                         ["Explain why the WAL snapshot matters for harvesting."])

    def test_backup_fallback_matches_direct_read(self):
        """When the direct open fails, the backup API returns the same rows."""
        with tempfile.TemporaryDirectory() as tmp:
            fx = AntigravityFixture(os.path.join(tmp, "conversations"))
            path = fx.write("s1", [(fx.USER, "Check the fallback snapshot path.")],
                            workspace=os.path.join(tmp, "p"), journal_mode="wal")
            direct = _read_direct(path)
            with tempfile.TemporaryDirectory() as workdir:
                fallback = _read_via_backup(path, workdir)
                self.assertEqual(os.listdir(workdir), [],
                                 "snapshot must not outlive the read")

            with tempfile.TemporaryDirectory() as workdir:
                with mock.patch(
                    "skillopt_sleep.harvest_antigravity._read_direct",
                    side_effect=sqlite3.OperationalError("database is locked"),
                ):
                    routed = _read_rows(path, workdir)

        self.assertEqual(direct["steps"], fallback["steps"])
        self.assertEqual(direct["meta"], fallback["meta"])
        self.assertEqual(routed["steps"], direct["steps"])

    def test_corrupt_and_locked_databases_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversations = os.path.join(tmp, "conversations")
            fx = AntigravityFixture(conversations)
            workspace = os.path.join(tmp, "proj")
            good = fx.write("good", [(fx.USER, "A real prompt that should survive.")],
                            workspace=workspace)

            corrupt = os.path.join(conversations, "corrupt.db")
            with open(corrupt, "wb") as f:
                f.write(b"this is definitely not a sqlite database")
            truncated = os.path.join(conversations, "truncated.db")
            with open(good, "rb") as src:
                head = src.read(48)
            with open(truncated, "wb") as f:
                f.write(head)

            self.assertIsNone(digest_antigravity_db(corrupt))
            self.assertIsNone(digest_antigravity_db(truncated))
            # a bad store must not take the night down with it
            digests = harvest_antigravity(conversations, scope="invoked",
                                          invoked_project=workspace)

        self.assertEqual([d.session_id for d in digests], ["good"])

    def test_read_uses_no_predictable_temp_path(self):
        """Nothing is written to a guessable name in the shared temp dir."""
        with tempfile.TemporaryDirectory() as tmp:
            fx = AntigravityFixture(os.path.join(tmp, "conversations"))
            path = fx.write("abcdef", [(fx.USER, "Confirm the temp path is private.")],
                            workspace=os.path.join(tmp, "p"))
            legacy = os.path.join(tempfile.gettempdir(), "skillopt_agy_abcdef.db")
            self.assertFalse(os.path.exists(legacy))
            digest_antigravity_db(path)
            self.assertFalse(os.path.exists(legacy),
                             "harvester must not write a predictable temp file")


class TestAntigravityProvenanceAndScope(unittest.TestCase):
    def _corpus(self, tmp):
        conversations = os.path.join(tmp, "conversations")
        fx = AntigravityFixture(conversations)
        project = os.path.join(tmp, "workspaces", "alpha")
        other = os.path.join(tmp, "workspaces", "beta")
        fx.write("in-project", [(fx.USER, "Fix the alpha importer please.")],
                 workspace=project, branch="master", mtime=time.time() - 300)
        fx.write("other-project", [(fx.USER, "Fix the beta exporter please.")],
                 workspace=other, mtime=time.time() - 200)
        fx.write("outside", [(fx.USER, "A question with no workspace attached.")],
                 workspace="", mtime=time.time() - 100)
        fx.write("no-metadata", [(fx.USER, "A store whose metadata row is absent.")],
                 with_metadata=False, mtime=time.time() - 50)
        return conversations, project, other

    def test_invoked_scope_selects_only_the_matching_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversations, project, _other = self._corpus(tmp)
            digests = harvest_antigravity(conversations, scope="invoked",
                                          invoked_project=project)

        self.assertEqual([d.session_id for d in digests], ["in-project"])
        self.assertEqual(digests[0].project, os.path.normpath(project))
        self.assertEqual(digests[0].git_branch, "master")

    def test_unknown_and_outside_sessions_are_never_relabelled(self):
        """The core regression: unrelated conversations must not be ingested."""
        with tempfile.TemporaryDirectory() as tmp:
            conversations, project, _other = self._corpus(tmp)
            digests = harvest_antigravity(conversations, scope="invoked",
                                          invoked_project=project)
            ids = [d.session_id for d in digests]
            self.assertNotIn("outside", ids)
            self.assertNotIn("no-metadata", ids)
            self.assertNotIn("other-project", ids)
            self.assertFalse([d for d in digests if d.project != os.path.normpath(project)])

            # An unrelated cwd must select nothing at all, not everything.
            unrelated = harvest_antigravity(
                conversations, scope="invoked",
                invoked_project=os.path.join(tmp, "workspaces", "gamma"))
            self.assertEqual(unrelated, [])

    def test_all_scope_is_the_explicit_opt_in_and_keeps_real_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversations, project, other = self._corpus(tmp)
            digests = harvest_antigravity(conversations, scope="all",
                                          invoked_project=project)

        by_id = {d.session_id: d for d in digests}
        self.assertEqual(set(by_id), {"in-project", "other-project", "outside", "no-metadata"})
        self.assertEqual(by_id["in-project"].project, os.path.normpath(project))
        self.assertEqual(by_id["other-project"].project, os.path.normpath(other))
        # unknown provenance stays unknown — it does not borrow the caller's identity
        self.assertEqual(by_id["outside"].project, "")
        self.assertEqual(by_id["no-metadata"].project, "")

    def test_explicit_project_list_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversations, project, other = self._corpus(tmp)
            digests = harvest_antigravity(conversations, scope=[other],
                                          invoked_project=project)
        self.assertEqual([d.session_id for d in digests], ["other-project"])

    def test_subdirectory_of_workspace_still_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversations, project, _other = self._corpus(tmp)
            nested = os.path.join(project, "src", "importer")
            digests = harvest_antigravity(conversations, scope="invoked",
                                          invoked_project=nested)
        self.assertEqual([d.session_id for d in digests], ["in-project"])

    def test_missing_invoked_project_selects_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversations, _project, _other = self._corpus(tmp)
            self.assertEqual(
                harvest_antigravity(conversations, scope="invoked", invoked_project=""), [])

    def test_missing_directory_returns_empty_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                harvest_antigravity(os.path.join(tmp, "absent"), scope="all"), [])

    def test_since_and_limit_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversations = os.path.join(tmp, "conversations")
            fx = AntigravityFixture(conversations)
            workspace = os.path.join(tmp, "proj")
            old = time.time() - 86_400
            fx.write("old", [(fx.USER, "An older question about the parser.")],
                     workspace=workspace, mtime=old)
            fx.write("new", [(fx.USER, "A newer question about the parser.")],
                     workspace=workspace, mtime=time.time() - 60)

            since = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 3600))
            recent = harvest_antigravity(conversations, scope="invoked",
                                         invoked_project=workspace, since_iso=since)
            self.assertEqual([d.session_id for d in recent], ["new"])

            capped = harvest_antigravity(conversations, scope="invoked",
                                         invoked_project=workspace, limit=1)
            self.assertEqual(len(capped), 1)
            self.assertEqual(capped[0].session_id, "new")  # newest first

    def test_percent_encoded_and_drive_letter_uris_decode(self):
        self.assertEqual(_path_from_file_uri("file:///C:/Users/a/My%20Project"),
                         os.path.normpath("C:/Users/a/My Project"))
        self.assertEqual(_path_from_file_uri("file:///home/a/my%20project"),
                         os.path.normpath("/home/a/my project"))
        self.assertEqual(_path_from_file_uri("outside-of-project"), "")
        self.assertEqual(_path_from_file_uri(""), "")

    def test_workspace_with_spaces_round_trips_through_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversations = os.path.join(tmp, "conversations")
            fx = AntigravityFixture(conversations)
            workspace = os.path.join(tmp, "Parsing Latent Space _project")
            fx.write("spaced", [(fx.USER, "Does a spaced workspace path survive?")],
                     workspace=workspace)
            digests = harvest_antigravity(conversations, scope="invoked",
                                          invoked_project=workspace)
        self.assertEqual([d.session_id for d in digests], ["spaced"])
        self.assertEqual(digests[0].project, os.path.normpath(workspace))


class TestAntigravityRedaction(unittest.TestCase):
    def test_secrets_are_redacted_before_leaving_the_harvester(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA7k3mQe1vX2pLmNq8Rr4TtYuIoPaSdFgHjKlZxCvBnM1234567\n"
            "-----END RSA PRIVATE KEY-----"
        )
        token = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8" * 4
        with tempfile.TemporaryDirectory() as tmp:
            fx = AntigravityFixture(os.path.join(tmp, "conversations"))
            workspace = os.path.join(tmp, "proj")
            path = fx.write("secrets", [
                (fx.USER, f"Deploy with this key please: {token} and confirm."),
                (fx.USER, f"Here is the signing material to install:\n{pem}\nThanks."),
                (fx.ARTIFACT, f"Installed the credential {token} into the runner."),
            ], workspace=workspace)

            digest = digest_antigravity_db(path)
            blob = "\n".join(digest.user_prompts + digest.assistant_finals)

        self.assertNotIn(token, blob)
        self.assertNotIn("MIIEowIBAAKCAQEA7k3mQe1vX2pLmNq8Rr4TtYuIoPaSdFgHjKlZxCvBnM1234567", blob)
        # the surrounding human text is preserved, so the prompt stays minable
        self.assertIn("Deploy with this key please", blob)
        self.assertIn("Thanks.", blob)

    def test_redaction_applies_through_the_harvest_entry_point(self):
        token = "ghp_" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1Qq0Pp"
        with tempfile.TemporaryDirectory() as tmp:
            conversations = os.path.join(tmp, "conversations")
            fx = AntigravityFixture(conversations)
            workspace = os.path.join(tmp, "proj")
            fx.write("s1", [(fx.USER, f"Rotate the token {token} in the deploy job.")],
                     workspace=workspace)
            digests = harvest_antigravity(conversations, scope="invoked",
                                          invoked_project=workspace)

        self.assertEqual(len(digests), 1)
        self.assertNotIn(token, " ".join(digests[0].user_prompts))


class TestAntigravitySourceRouting(unittest.TestCase):
    """The user's own config file must not leak into these assertions."""

    def _isolated_config(self, **overrides):
        with mock.patch("skillopt_sleep.config._user_config_path", return_value=""):
            return load_config(**overrides)

    def test_config_and_source_router_select_antigravity(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = os.path.join(tmp, "antigravity")
            conversations = os.path.join(home, "conversations")
            fx = AntigravityFixture(conversations)
            workspace = os.path.join(tmp, "proj")
            fx.write("routed", [(fx.USER, "Route me through harvest_for_config.")],
                     workspace=workspace)

            cfg = self._isolated_config(
                transcript_source="antigravity",
                antigravity_home=home,
                projects="invoked",
                invoked_project=workspace,
            )
            self.assertEqual(cfg.antigravity_conversations_dir, conversations)
            digests = harvest_for_config(cfg)

        self.assertEqual([d.session_id for d in digests], ["routed"])

    def test_antigravity_source_respects_all_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = os.path.join(tmp, "antigravity")
            fx = AntigravityFixture(os.path.join(home, "conversations"))
            fx.write("outside", [(fx.USER, "No workspace was attached to this one.")],
                     workspace="")

            scoped = self._isolated_config(
                transcript_source="antigravity", antigravity_home=home,
                projects="invoked", invoked_project=os.path.join(tmp, "proj"))
            opted_in = self._isolated_config(
                transcript_source="antigravity", antigravity_home=home, projects="all")

            self.assertEqual(harvest_for_config(scoped), [])
            self.assertEqual([d.session_id for d in harvest_for_config(opted_in)],
                             ["outside"])

    def test_other_sources_are_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._isolated_config(
                transcript_source="claude",
                claude_home=os.path.join(tmp, "claude"),
                projects="invoked",
                invoked_project=tmp,
            )
            with mock.patch(
                "skillopt_sleep.harvest_sources.harvest_antigravity"
            ) as antigravity:
                self.assertEqual(harvest_for_config(cfg), [])
            antigravity.assert_not_called()


if __name__ == "__main__":
    unittest.main()
