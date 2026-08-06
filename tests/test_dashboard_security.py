"""Tests for the local dashboard's request-authorization boundary.

The dashboard's POST endpoints start real runs, adopt staged artifacts, and
rewrite config and prompts. Binding to loopback does not protect them: any
page in the user's browser can POST cross-origin to 127.0.0.1, and a hostile
name resolving to loopback (DNS rebinding) makes those requests look local.

Every negative case here asserts twice: that the request was refused, *and*
that the side effect did not happen — a 403 that still started a run would
pass a status-code-only test.

Pure stdlib, deterministic, no network beyond 127.0.0.1, no third-party deps.

Run:  python -m unittest tests.test_dashboard_security
"""
from __future__ import annotations

import http.client
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from skillopt_sleep import dashboard
from skillopt_sleep import prompts as prompt_registry

_TOKEN_HEADER = "X-SkillOpt-Dashboard-Token"


class _FakeRunState:
    """Records launches instead of spawning a real pipeline subprocess."""

    def __init__(self) -> None:
        self.starts = []

    def running(self) -> bool:
        return False

    def start(self, project, dry_run):
        self.starts.append((project, bool(dry_run)))
        return {"ok": True, "mode": "dry-run" if dry_run else "run"}

    def status(self):
        return {"running": False, "returncode": None, "mode": "", "tail": ""}


class DashboardSecurityTestCase(unittest.TestCase):
    """A live loopback dashboard with every side effect redirected to tmp."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name

        self.config_path = os.path.join(root, "config.json")
        self.prompts_path = os.path.join(root, "prompts.json")

        patches = [
            mock.patch.dict(os.environ,
                            {"SKILLOPT_SLEEP_PROMPTS_PATH": self.prompts_path}),
            mock.patch.object(dashboard, "_user_config_file",
                              return_value=self.config_path),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        self.adopted = []
        adopt_patch = mock.patch.object(
            dashboard, "adopt_staging",
            side_effect=lambda d: self.adopted.append(d) or {"skill": "ok"})
        adopt_patch.start()
        self.addCleanup(adopt_patch.stop)

        # A staged night that /api/adopt would accept if it got through.
        self.night = "20260801-000000"
        staging = os.path.join(root, ".skillopt-sleep", "staging", self.night)
        os.makedirs(staging)
        with open(os.path.join(staging, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"files": []}, f)

        self.httpd = dashboard.make_server(root, 0)
        self.handler = self.httpd.RequestHandlerClass
        self.run_state = _FakeRunState()
        self.handler.run_state = self.run_state
        self.port = self.httpd.server_address[1]
        self.token = self.handler.token
        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()

        def _shutdown():
            self.httpd.shutdown()
            self.httpd.server_close()
        self.addCleanup(_shutdown)

    # ── request helpers ───────────────────────────────────────────────────
    def request(self, method, path, *, body=None, headers=None,
                host=None, origin="", ctype="application/json", token=None):
        """Issue one request with full control over the security-relevant bits."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        sent = {"Host": host if host is not None else f"127.0.0.1:{self.port}"}
        if ctype:
            sent["Content-Type"] = ctype
        if origin:
            sent["Origin"] = origin
        resolved = self.token if token is None else token
        if resolved:
            sent[_TOKEN_HEADER] = resolved
        sent.update(headers or {})
        payload = body.encode("utf-8") if isinstance(body, str) else body
        conn.request(method, path, body=payload, headers=sent)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            parsed = raw
        return response.status, parsed, response

    def post(self, path, obj=None, **kw):
        body = kw.pop("body", None)
        if body is None:
            body = json.dumps({} if obj is None else obj)
        return self.request("POST", path, body=body,
                            origin=kw.pop("origin", self.origin()), **kw)

    def origin(self):
        return f"http://127.0.0.1:{self.port}"

    # ── side-effect probes ────────────────────────────────────────────────
    def saved_config(self):
        try:
            with open(self.config_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def miner_override(self):
        for entry in prompt_registry.describe():
            if entry["name"] == "miner":
                return entry.get("override")
        return None

    def assertNothingChanged(self, msg=""):
        self.assertEqual(self.saved_config(), {}, f"config was written {msg}")
        self.assertIn(self.miner_override(), (None, ""), f"prompt was written {msg}")
        self.assertEqual(self.run_state.starts, [], f"a run was started {msg}")
        self.assertEqual(self.adopted, [], f"a night was adopted {msg}")


class TestHappyPath(DashboardSecurityTestCase):
    """Same-origin JSON with the token still does everything it should."""

    def test_html_carries_a_substituted_token(self):
        status, _body, response = self.request("GET", "/", ctype="")
        html = _body if isinstance(_body, bytes) else json.dumps(_body).encode()
        self.assertEqual(status, 200)
        self.assertIn(b"Control Panel", html)
        self.assertIn(self.token.encode(), html)
        self.assertNotIn(b"__SKILLOPT_DASHBOARD_TOKEN__", html)
        self.assertNotIn("Access-Control-Allow-Origin", dict(response.getheaders()))

    def test_overview_and_night_reads(self):
        status, body, _r = self.request("GET", "/api/overview", ctype="")
        self.assertEqual(status, 200)
        self.assertEqual({p["name"] for p in body["prompts"]},
                         {"miner", "attempt", "judge", "reflect"})
        status, _body, _r = self.request("GET", "/api/night/nope", ctype="")
        self.assertEqual(status, 404)

    def test_config_write(self):
        status, body, _r = self.post("/api/config", {"updates": {"edit_budget": 7}})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(self.saved_config(), {"edit_budget": 7})

    def test_config_ignores_keys_outside_the_allowlist(self):
        status, _body, _r = self.post(
            "/api/config", {"updates": {"edit_budget": 3, "claude_home": "/etc"}})
        self.assertEqual(status, 200)
        self.assertEqual(self.saved_config(), {"edit_budget": 3})

    def test_prompt_roundtrip(self):
        status, body, _r = self.post("/api/prompts",
                                     {"updates": {"miner": "X __PROMPTS__"}})
        self.assertEqual(status, 200)
        mined = [p for p in body["prompts"] if p["name"] == "miner"][0]
        self.assertEqual(mined["override"], "X __PROMPTS__")
        self.post("/api/prompts", {"updates": {"miner": None}})
        self.assertIn(self.miner_override(), (None, ""))

    def test_run_and_adopt(self):
        status, body, _r = self.post("/api/run", {"dry_run": True})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(self.run_state.starts, [(self.tmp.name, True)])

        status, body, _r = self.post("/api/adopt", {"ts": self.night})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(len(self.adopted), 1)

    def test_localhost_origin_and_host_are_accepted(self):
        status, _body, _r = self.request(
            "POST", "/api/config", body=json.dumps({"updates": {"edit_budget": 5}}),
            host=f"localhost:{self.port}", origin=f"http://localhost:{self.port}")
        self.assertEqual(status, 200)
        self.assertEqual(self.saved_config(), {"edit_budget": 5})


class TestHostValidation(DashboardSecurityTestCase):
    """DNS rebinding: the browser sends the attacker's name, not ours."""

    def test_foreign_host_is_rejected_on_every_endpoint(self):
        for method, path in [
            ("GET", "/"), ("GET", "/api/overview"), ("GET", "/api/run/status"),
            ("POST", "/api/config"), ("POST", "/api/prompts"),
            ("POST", "/api/run"), ("POST", "/api/adopt"),
        ]:
            with self.subTest(method=method, path=path):
                status, _body, _r = self.request(
                    method, path,
                    body=json.dumps({"updates": {"edit_budget": 9}, "ts": self.night,
                                     "dry_run": True}) if method == "POST" else None,
                    host=f"attacker.example.com:{self.port}", origin=self.origin())
                self.assertEqual(status, 403)
        self.assertNothingChanged("after foreign Host")

    def test_host_with_wrong_port_is_rejected(self):
        status, _body, _r = self.post("/api/config", {"updates": {"edit_budget": 9}},
                                      host=f"127.0.0.1:{self.port + 1}")
        self.assertEqual(status, 403)
        self.assertNothingChanged("after wrong-port Host")

    def test_host_without_port_is_rejected(self):
        status, _body, _r = self.post("/api/config", {"updates": {"edit_budget": 9}},
                                      host="127.0.0.1")
        self.assertEqual(status, 403)
        self.assertNothingChanged("after portless Host")

    def test_rebinding_name_resolving_to_loopback_is_rejected(self):
        # The attacker's DNS points at 127.0.0.1, so the connection succeeds;
        # only the Host header distinguishes it from the real dashboard.
        status, _body, _r = self.post("/api/run", {"dry_run": False},
                                      host=f"rebind.attacker.test:{self.port}",
                                      origin=f"http://rebind.attacker.test:{self.port}")
        self.assertEqual(status, 403)
        self.assertNothingChanged("after DNS-rebinding request")


class TestOriginValidation(DashboardSecurityTestCase):
    MUTATING = [
        ("/api/config", {"updates": {"edit_budget": 9}}),
        ("/api/prompts", {"updates": {"miner": "pwned"}}),
        ("/api/run", {"dry_run": False}),
        ("/api/adopt", {"ts": "20260801-000000"}),
    ]

    def test_foreign_origin_is_rejected(self):
        for path, payload in self.MUTATING:
            with self.subTest(path=path):
                status, _body, _r = self.post(path, payload,
                                              origin="http://evil.example.com")
                self.assertEqual(status, 403)
        self.assertNothingChanged("after foreign Origin")

    def test_missing_origin_is_rejected(self):
        for path, payload in self.MUTATING:
            with self.subTest(path=path):
                status, _body, _r = self.post(path, payload, origin="")
                self.assertEqual(status, 403)
        self.assertNothingChanged("after missing Origin")

    def test_null_and_lookalike_origins_are_rejected(self):
        for origin in ("null",
                       f"https://127.0.0.1:{self.port}",
                       f"http://127.0.0.1.evil.com:{self.port}",
                       f"http://127.0.0.1:{self.port}.evil.com",
                       f"http://127.0.0.1:{self.port + 1}"):
            with self.subTest(origin=origin):
                status, _body, _r = self.post(
                    "/api/config", {"updates": {"edit_budget": 9}}, origin=origin)
                self.assertEqual(status, 403)
        self.assertNothingChanged("after lookalike Origin")


class TestContentTypeValidation(DashboardSecurityTestCase):
    """The body shapes a cross-origin <form> can send without a preflight."""

    def test_form_and_text_bodies_are_rejected(self):
        for ctype, body in [
            ("application/x-www-form-urlencoded", "dry_run=1"),
            ("text/plain", json.dumps({"dry_run": True})),
            ("text/plain;charset=UTF-8", json.dumps({"dry_run": True})),
            ("multipart/form-data; boundary=x", "--x--"),
            ("", json.dumps({"dry_run": True})),
        ]:
            with self.subTest(ctype=ctype):
                status, _body, _r = self.post("/api/run", body=body, ctype=ctype)
                self.assertEqual(status, 415)
        self.assertNothingChanged("after non-JSON content type")

    def test_the_documented_csrf_vector_cannot_start_a_run(self):
        """The reported case: an empty cross-origin form POST to /api/run."""
        status, _body, _r = self.request(
            "POST", "/api/run", body=b"",
            origin="http://evil.example.com",
            ctype="application/x-www-form-urlencoded", token="")
        self.assertEqual(status, 403)
        self.assertEqual(self.run_state.starts, [])

    def test_json_content_type_with_parameters_is_accepted(self):
        status, _body, _r = self.post("/api/config", {"updates": {"edit_budget": 4}},
                                      ctype="application/json; charset=utf-8")
        self.assertEqual(status, 200)
        self.assertEqual(self.saved_config(), {"edit_budget": 4})


class TestTokenValidation(DashboardSecurityTestCase):
    def test_missing_and_wrong_tokens_are_rejected(self):
        for token in ("", "not-the-token", "x" * len(self.token) if self.token else "x"):
            with self.subTest(token=token[:12]):
                status, _body, _r = self.post(
                    "/api/config", {"updates": {"edit_budget": 9}}, token=token)
                self.assertEqual(status, 403)
        self.assertNothingChanged("after bad token")

    def test_token_is_unguessable_and_per_process(self):
        other = dashboard.make_server(self.tmp.name, 0)
        try:
            self.assertNotEqual(other.RequestHandlerClass.token, self.token)
            self.assertGreaterEqual(len(self.token), 32)
            # A token from another dashboard process must not work here.
            status, _body, _r = self.post("/api/config", {"updates": {"edit_budget": 9}},
                                          token=other.RequestHandlerClass.token)
            self.assertEqual(status, 403)
        finally:
            other.server_close()
        self.assertNothingChanged("after cross-process token")


class TestBodyValidation(DashboardSecurityTestCase):
    def test_empty_body_is_not_coerced_to_an_empty_object(self):
        for path in ("/api/run", "/api/config", "/api/prompts", "/api/adopt"):
            with self.subTest(path=path):
                status, _body, _r = self.post(path, body=b"")
                self.assertEqual(status, 400)
        self.assertNothingChanged("after empty body")

    def test_malformed_and_non_object_bodies_are_rejected(self):
        for body in ("{not json", "[]", '"a string"', "null", "42", ""):
            with self.subTest(body=body):
                status, _body, _r = self.post("/api/config", body=body)
                self.assertEqual(status, 400)
        self.assertNothingChanged("after malformed body")

    def test_oversized_body_is_rejected(self):
        payload = json.dumps({"updates": {"preferences": "A" * (1 << 21)}})
        status, _body, _r = self.post("/api/config", body=payload)
        self.assertEqual(status, 413)
        self.assertNothingChanged("after oversized body")

    def test_non_object_updates_are_rejected(self):
        for path in ("/api/config", "/api/prompts"):
            with self.subTest(path=path):
                status, _body, _r = self.post(path, {"updates": "everything"})
                self.assertEqual(status, 400)
        self.assertNothingChanged("after non-object updates")


class TestStagingPathContainment(DashboardSecurityTestCase):
    """`os.path.basename` is not containment: basename("..") == "..".

    Before this was fixed, /api/night/.. read the staging parent and
    /api/adopt would copy whatever it found there over the live SKILL.md.
    """

    ESCAPES = ["..", "../..", "../" * 4, "/etc", "\\Windows",
               ".", "", "C:\\Windows", "%2e%2e", "..%2f.."]

    def test_night_read_cannot_escape_the_staging_root(self):
        for ts in self.ESCAPES:
            with self.subTest(ts=ts):
                status, body, _r = self.request(
                    "GET", "/api/night/" + ts, ctype="")
                self.assertEqual(status, 404)
                self.assertNotIn("report", body if isinstance(body, dict) else {})

    def test_adopt_cannot_escape_the_staging_root(self):
        for ts in self.ESCAPES:
            with self.subTest(ts=ts):
                status, _body, _r = self.post("/api/adopt", {"ts": ts})
                self.assertEqual(status, 404)
        self.assertEqual(self.adopted, [], "adopt escaped the staging root")

    def test_adopt_rejects_non_string_ts(self):
        for ts in [None, 42, ["a"], {"a": 1}, True]:
            with self.subTest(ts=ts):
                status, _body, _r = self.post("/api/adopt", {"ts": ts})
                self.assertEqual(status, 404)
        self.assertEqual(self.adopted, [])

    def test_linked_night_pointing_outside_is_rejected(self):
        """A link planted inside staging must not redirect the read out of it.

        The name is a plain component, so only resolving it catches this.
        Unprivileged Windows cannot create symlinks; a directory junction
        exercises the same containment check there.
        """
        outside = os.path.join(self.tmp.name, "outside")
        os.makedirs(outside, exist_ok=True)
        with open(os.path.join(outside, "report.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        link = os.path.join(self.tmp.name, ".skillopt-sleep", "staging", "sneaky")
        try:
            os.symlink(outside, link, target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError):
            if os.name != "nt":
                self.skipTest("symlink creation not permitted on this platform")
            import subprocess
            created = subprocess.run(["cmd", "/c", "mklink", "/J", link, outside],
                                     capture_output=True, text=True)
            if not os.path.isdir(link):
                self.skipTest(f"cannot create a link here: {created.stderr.strip()}")

        status, _body, _r = self.request("GET", "/api/night/sneaky", ctype="")
        self.assertEqual(status, 404)
        status, _body, _r = self.post("/api/adopt", {"ts": "sneaky"})
        self.assertEqual(status, 404)
        self.assertEqual(self.adopted, [])

    def test_the_real_night_is_still_reachable(self):
        status, body, _r = self.request("GET", "/api/night/" + self.night, ctype="")
        self.assertEqual(status, 200)
        self.assertEqual(body["ts"], self.night)
        status, body, _r = self.post("/api/adopt", {"ts": self.night})
        self.assertEqual(status, 200)
        self.assertEqual(len(self.adopted), 1)


class TestConfigValueTyping(DashboardSecurityTestCase):
    """`load_config` does not type-coerce, so the API must."""

    def test_numeric_strings_are_stored_as_numbers(self):
        status, _body, _r = self.post("/api/config", {"updates": {
            "edit_budget": "7", "gate_mixed_weight": "0.25",
            "max_tasks_per_night": "12"}})
        self.assertEqual(status, 200)
        saved = self.saved_config()
        self.assertEqual(saved["edit_budget"], 7)
        self.assertIsInstance(saved["edit_budget"], int)
        self.assertEqual(saved["gate_mixed_weight"], 0.25)
        self.assertIsInstance(saved["gate_mixed_weight"], float)
        self.assertIsInstance(saved["max_tasks_per_night"], int)

    def test_boolean_strings_are_stored_as_booleans(self):
        status, _body, _r = self.post("/api/config", {"updates": {
            "llm_mine": "false", "evolve_skill": "true", "auto_adopt": False}})
        self.assertEqual(status, 200)
        saved = self.saved_config()
        self.assertIs(saved["llm_mine"], False)
        self.assertIs(saved["evolve_skill"], True)
        self.assertIs(saved["auto_adopt"], False)

    def test_free_text_is_not_coerced_to_a_boolean(self):
        """The old blanket 'true' -> True turned house rules into a bool."""
        status, _body, _r = self.post(
            "/api/config", {"updates": {"preferences": "true"}})
        self.assertEqual(status, 200)
        self.assertEqual(self.saved_config()["preferences"], "true")

    def test_unparseable_numbers_are_rejected_and_nothing_is_written(self):
        for key, value in [("edit_budget", "not-a-number"),
                           ("gate_mixed_weight", "high"),
                           ("max_tasks_per_night", "12 tasks"),
                           ("llm_mine", "maybe")]:
            with self.subTest(key=key):
                status, body, _r = self.post(
                    "/api/config", {"updates": {key: value}})
                self.assertEqual(status, 400)
                self.assertIn(key, body["error"])
        self.assertEqual(self.saved_config(), {})

    def test_a_bad_field_does_not_half_apply_the_form(self):
        status, _body, _r = self.post("/api/config", {"updates": {
            "edit_budget": "5", "gate_mixed_weight": "nonsense"}})
        self.assertEqual(status, 400)
        self.assertEqual(self.saved_config(), {},
                         "a rejected form must not persist its valid fields")


class TestRunLauncherResilience(unittest.TestCase):
    """A failed spawn must not raise out of the request-handler thread."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        stub = mock.Mock()
        stub.state_dir = self.tmp.name
        patch = mock.patch.object(dashboard, "load_config", return_value=stub)
        patch.start()
        self.addCleanup(patch.stop)

    def test_spawn_failure_returns_a_structured_error(self):
        state = dashboard._RunState()
        with mock.patch.object(dashboard.subprocess, "Popen",
                               side_effect=OSError("no interpreter")):
            result = state.start(self.tmp.name, dry_run=True)
        self.assertFalse(result["ok"])
        self.assertIn("no interpreter", result["error"])
        self.assertFalse(state.running())
        # The launcher stays usable after a failure.
        self.assertEqual(state.status()["running"], False)

    def test_log_handle_is_released_to_the_child(self):
        state = dashboard._RunState()
        with mock.patch.object(dashboard.subprocess, "Popen") as popen:
            popen.return_value.poll.return_value = 0
            result = state.start(self.tmp.name, dry_run=True)
        self.assertTrue(result["ok"])
        # Windows refuses to remove a file the parent still holds open, so a
        # successful unlink proves the parent closed its copy.
        os.unlink(state.log_path)


class TestNoPermissiveCors(DashboardSecurityTestCase):
    def test_no_cors_headers_are_ever_emitted(self):
        probes = [
            ("GET", "/", None), ("GET", "/api/overview", None),
            ("POST", "/api/config", json.dumps({"updates": {"edit_budget": 2}})),
        ]
        for method, path, body in probes:
            with self.subTest(path=path):
                _status, _body, response = self.request(
                    method, path, body=body, origin=self.origin())
                headers = {k.lower() for k in dict(response.getheaders())}
                self.assertNotIn("access-control-allow-origin", headers)
                self.assertNotIn("access-control-allow-credentials", headers)
                self.assertNotIn("access-control-allow-headers", headers)

    def test_preflight_is_not_answered_permissively(self):
        status, _body, response = self.request(
            "OPTIONS", "/api/run", origin="http://evil.example.com", ctype="")
        headers = {k.lower() for k in dict(response.getheaders())}
        self.assertNotIn("access-control-allow-origin", headers)
        self.assertNotEqual(status, 204)
        self.assertGreaterEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
