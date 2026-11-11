"""Tests for AgenticReplayBackend and workspace lifecycle slice (Issue #155)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from skillopt_sleep.agentic_replay import (
    OUTCOME_FAIL,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_PASS,
    REASON_BUDGET_EXHAUSTED,
    REASON_FIXTURE_FAILURE,
    REASON_INFRA,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_TIMEOUT,
    AgenticReplayBackend,
    AgenticReplayConfig,
    AgenticWorkspace,
    FixtureSetupError,
    TaskVerifier,
)
from skillopt_sleep.backend import MockBackend, build_backend, get_backend
from skillopt_sleep.config import load_config
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.replay import aggregate_scores, pairwise_aggregate_scores, replay_batch, replay_one
from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord


def _init_git_repo(repo_dir: str, files: dict[str, str] | None = None) -> str:
    """Create a small standalone git repository and return its HEAD commit SHA."""
    os.makedirs(repo_dir, exist_ok=True)
    for rel, content in (files or {"app.txt": "baseline\n"}).items():
        full = os.path.join(repo_dir, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    git_cfg = ["-c", "user.name=Test", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-b", "main", repo_dir], check=True, capture_output=True)
    subprocess.run(["git", "-C", repo_dir, "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", repo_dir, *git_cfg, "commit", "-m", "initial"], check=True, capture_output=True)
    res = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return res.stdout.strip()


def _write_fake_agent_script(path: str, body: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in res.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class TestReplayResultOutcomes(unittest.TestCase):
    def test_default_pass_fail_inference(self):
        passed = ReplayResult(id="t1", hard=1.0)
        self.assertEqual(passed.outcome, OUTCOME_PASS)

        failed = ReplayResult(id="t2", hard=0.0)
        self.assertEqual(failed.outcome, OUTCOME_FAIL)

        mixed = ReplayResult(id="t3", hard=0.5)
        self.assertEqual(mixed.outcome, OUTCOME_FAIL)

    def test_explicit_inconclusive_outcome(self):
        res = ReplayResult(
            id="t1",
            hard=0.0,
            outcome=OUTCOME_INCONCLUSIVE,
            inconclusive_reason=REASON_TIMEOUT,
            diff="--- a/file\n+++ b/file\n",
        )
        self.assertEqual(res.outcome, OUTCOME_INCONCLUSIVE)
        self.assertEqual(res.inconclusive_reason, REASON_TIMEOUT)
        self.assertIn("--- a/file", res.diff)

        d = res.to_dict()
        self.assertEqual(d["outcome"], OUTCOME_INCONCLUSIVE)
        self.assertEqual(d["inconclusive_reason"], REASON_TIMEOUT)
        self.assertIn("--- a/file", d["diff"])


class TestAgenticOptInAndConfigWiring(unittest.TestCase):
    def test_disabled_by_default_blocks_execution(self):
        called = False

        def _runner(ws_path, env, ctx):
            nonlocal called
            called = True
            return {"response": "ran", "actions": 1}

        for flag in (False, "true", 1):
            cfg = AgenticReplayConfig(enabled=flag)  # type: ignore[arg-type]
            self.assertFalse(cfg.enabled)
            backend = AgenticReplayBackend(
                cfg,
                agent_runner=_runner,
                default_verifier=TaskVerifier(verifier_fn=lambda _p: (True, "ok")),
            )
            res = backend.replay_agentic_task(TaskRecord(id="t1", project="/p", intent="do task"))
            self.assertFalse(called)
            self.assertEqual(res.outcome, OUTCOME_INCONCLUSIVE)
            self.assertEqual(res.inconclusive_reason, REASON_INFRA)

    def test_config_keys_wired_through_load_config_build_backend_and_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_root = os.path.join(tmp, "fixtures")
            os.makedirs(fixture_root)
            cfg = load_config(
                backend="agentic",
                agentic_replay=True,
                agentic_max_turns=3,
                agentic_max_tokens=1500,
                agentic_timeout_sec=12.5,
                agentic_max_replays=7,
                agentic_isolate_env=False,
                agentic_min_actions=2,
                agentic_fixture_root=fixture_root,
                agentic_min_coverage=0.75,
            )
            payload = {k: cfg.get(k) for k in cfg.data if k.startswith("agentic_")}
            be_alias = get_backend("agentic_replay", agentic_config=payload)
            self.assertEqual(be_alias.name, "agentic")
            self.assertTrue(be_alias.config.enabled)

            be = build_backend(backend="agentic", agentic_config=payload)
            self.assertIsInstance(be, AgenticReplayBackend)
            # Disabled runner invocation via replay_one fails closed cleanly
            disabled_be = get_backend("agentic")
            res_dis = replay_one(disabled_be, TaskRecord(id="r1", project="/p", intent="hi"), "", "")
            self.assertEqual(res_dis.outcome, OUTCOME_INCONCLUSIVE)
            self.assertEqual(res_dis.inconclusive_reason, REASON_INFRA)
            self.assertTrue(be.config.enabled)
            self.assertEqual(be.config.max_turns, 3)
            self.assertEqual(be.config.max_tokens, 1500)
            self.assertEqual(be.config.timeout_sec, 12.5)
            self.assertEqual(be.config.max_replays_per_run, 7)
            self.assertFalse(be.config.isolate_env)
            self.assertEqual(be.config.min_decidable_actions, 2)
            self.assertEqual(be.config.fixture_root, fixture_root)
            self.assertEqual(be.config.min_coverage, 0.75)

            # Per-night reset clears replays_run at start of run_sleep_cycle
            be.replays_run = 6
            be.tokens_spent = 999
            proj = os.path.join(tmp, "proj")
            os.makedirs(proj)
            run_cfg = load_config(
                projects="invoked",
                invoked_project=proj,
                state_dir=os.path.join(tmp, "state"),
                claude_home=os.path.join(tmp, "claude"),
                evidence_log=False,
                evolve_skill=False,
                evolve_memory=False,
            )
            run_sleep_cycle(run_cfg, backend=be, seed_tasks=[], dry_run=True)
            self.assertEqual(be.replays_run, 0)
            self.assertEqual(be.tokens_spent, 0)

            # Verify run_sleep_cycle itself constructs AgenticReplayBackend from SleepConfig
            built_backends = []
            real_builder = build_backend

            def _capture_builder(**kwargs):
                res_be = real_builder(**kwargs)
                built_backends.append(res_be)
                return res_be

            agentic_cfg = load_config(
                backend="agentic",
                agentic_replay=True,
                agentic_max_turns=4,
                agentic_max_replays=9,
                projects="invoked",
                invoked_project=proj,
                state_dir=os.path.join(tmp, "state2"),
                claude_home=os.path.join(tmp, "claude"),
                evidence_log=False,
                evolve_skill=False,
                evolve_memory=False,
            )
            with mock.patch("skillopt_sleep.cycle.build_backend", side_effect=_capture_builder):
                run_sleep_cycle(agentic_cfg, seed_tasks=[], dry_run=True)
            self.assertGreaterEqual(len(built_backends), 1)
            self.assertIsInstance(built_backends[-1], AgenticReplayBackend)
            self.assertTrue(built_backends[-1].config.enabled)
            self.assertEqual(built_backends[-1].config.max_turns, 4)
            self.assertEqual(built_backends[-1].config.max_replays_per_run, 9)

    def test_lazy_import_of_agentic_backend_in_package_init(self):
        script = (
            "import sys, skillopt_sleep; "
            "assert 'skillopt_sleep.agentic_replay' not in sys.modules; "
            "from skillopt_sleep import AgenticReplayBackend; "
            "assert AgenticReplayBackend.name == 'agentic'"
        )
        res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)


class TestAgenticWorkspaceAndFakeAgentSubprocess(unittest.TestCase):
    def test_workspace_scratch_copy_env_allowlist_and_home_outside_diff(self):
        sensitive = {
            "TEST_SECRET_KEY": "supersecret",
            "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
            "GITHUB_TOKEN": "ghp_example",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/creds.json",
            "KUBECONFIG": "/tmp/kubeconfig",
            "DATABASE_URL": "postgres://user:pass@host/db",
            "GIT_ASKPASS": "/tmp/askpass.sh",
            "DOCKER_HOST": "tcp://127.0.0.1:2375",
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
            "IDENTITY_HEADER": "secret-hdr",
            "IDENTITY_ENDPOINT": "http://169.254.169.254",
            "CLAUDE_CONFIG_DIR": "/real/claude",
            "CODEX_HOME": "/real/codex",
            "HTTP_PROXY": "http://corp:8080",
            "https_proxy": "https://corp:8080",
            "no_proxy": "localhost",
        }
        with mock.patch.dict(os.environ, sensitive):
            with AgenticWorkspace(isolate_env=True) as ws:
                self.assertTrue(os.path.isdir(ws.workspace_path))
                self.assertFalse(
                    os.path.commonpath([
                        os.path.realpath(ws.workspace_path),
                        os.path.realpath(ws.home_dir),
                    ]) == os.path.realpath(ws.workspace_path),
                    "HOME must sit outside diffed workspace_path",
                )
                git_res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ws.workspace_path,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(git_res.returncode, 0)
                self.assertGreaterEqual(len(git_res.stdout.strip()), 7)

                env = ws.build_isolated_env()
                for forbidden in (
                    "TEST_SECRET_KEY", "AWS_ACCESS_KEY_ID", "GITHUB_TOKEN",
                    "GOOGLE_APPLICATION_CREDENTIALS", "KUBECONFIG", "DATABASE_URL",
                    "DOCKER_HOST", "DBUS_SESSION_BUS_ADDRESS",
                    "IDENTITY_HEADER", "IDENTITY_ENDPOINT",
                    "HTTP_PROXY", "https_proxy", "no_proxy",
                ):
                    self.assertNotIn(forbidden, env)

                self.assertEqual(env["GIT_ASKPASS"], "")
                self.assertEqual(env["GIT_CONFIG_NOSYSTEM"], "1")
                self.assertTrue(os.path.isfile(env["GIT_CONFIG_GLOBAL"]))
                self.assertEqual(os.path.getsize(env["GIT_CONFIG_GLOBAL"]), 0)
                self.assertEqual(env["HOME"], ws.home_dir)
                self.assertEqual(os.listdir(env["CLAUDE_CONFIG_DIR"]), [])
                self.assertEqual(os.listdir(env["CODEX_HOME"]), [])
                self.assertNotIn(".sandbox_home", env["HOME"])

                # Verify fake agent binary sees allowlisted env via subprocess and writes to HOME without dirtying diff
                script_dir = tempfile.mkdtemp(prefix="skillopt_fake_")
                try:
                    script_path = _write_fake_agent_script(
                        os.path.join(script_dir, "agent.py"),
                        (
                            "import json, os\n"
                            "assert 'KUBECONFIG' not in os.environ\n"
                            "assert 'DATABASE_URL' not in os.environ\n"
                            "assert 'https_proxy' not in os.environ\n"
                            "with open(os.path.join(os.environ['HOME'], 'cache.txt'), 'w') as hf:\n"
                            "    hf.write('temp-home-artifact\\n')\n"
                            "with open('sample.txt', 'w') as wf:\n"
                            "    wf.write('hello world\\n')\n"
                            "print(json.dumps({'home': os.environ['HOME']}))\n"
                        ),
                    )
                    timed_out, rc, out, err = ws.run_subprocess([sys.executable, script_path], env=env)
                    self.assertFalse(timed_out, err)
                    self.assertEqual(rc, 0, err)
                finally:
                    for name in os.listdir(script_dir):
                        os.remove(os.path.join(script_dir, name))
                    os.rmdir(script_dir)

                diff = ws.get_diff()
                self.assertIn("sample.txt", diff)
                self.assertIn("+hello world", diff)
                self.assertNotIn("cache.txt", diff)
                self.assertNotIn("temp-home-artifact", diff)

        self.assertFalse(os.path.exists(ws.workspace_path))

    def test_fake_agent_worktree_fixture_cannot_touch_real_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_root = os.path.join(tmp, "fixtures")
            real_repo = os.path.join(fixture_root, "real_repo")
            wt_dir = os.path.join(fixture_root, "wt")
            head_sha = _init_git_repo(real_repo, {"tracked.txt": "v1\n"})
            subprocess.run(
                ["git", "-C", real_repo, "worktree", "add", "--detach", wt_dir, head_sha],
                check=True,
                capture_output=True,
            )
            hook_sentinel = os.path.join(real_repo, ".git", "hooks", "pwn_marker")
            fake_script = _write_fake_agent_script(
                os.path.join(tmp, "wt_agent.py"),
                (
                    "import os, subprocess, sys\n"
                    "common = subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip()\n"
                    "main_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()\n"
                    "with open('edited.txt', 'w') as f:\n"
                    "    f.write('updated\\n')\n"
                    "subprocess.run(['git', 'add', 'edited.txt'], check=True)\n"
                    "subprocess.run(['git', '-c', 'user.name=A', '-c', 'user.email=a@a', 'commit', '-m', 'agent'], check=True)\n"
                    "new_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()\n"
                    "subprocess.run(['git', 'update-ref', 'refs/heads/main', new_sha], check=True)\n"
                    "with open(os.path.join(common, 'hooks', 'pwn_marker'), 'w') as hf:\n"
                    "    hf.write('touched')\n"
                ),
            )

            # 1. Unpinned worktree fixture must be rejected because .git is a linked file
            be_unpinned = AgenticReplayBackend(
                AgenticReplayConfig(enabled=True, fixture_root=fixture_root),
                agent_runner=lambda _ws, _env: {"response": "ok", "actions": 1},
                default_verifier=TaskVerifier(verifier_fn=lambda _p: (True, "ok")),
            )
            unpinned_res = be_unpinned.replay_agentic_task(
                TaskRecord(id="wt-unpinned", project=real_repo, intent="edit"),
                fixture_dir="wt",
            )
            self.assertEqual(unpinned_res.outcome, OUTCOME_INCONCLUSIVE)
            self.assertEqual(unpinned_res.inconclusive_reason, REASON_FIXTURE_FAILURE)

            # 2. Pinned worktree fixture materializes without .git pointer and isolates real_repo
            def _run_wt(ws_path, env, ctx):
                ws = ctx["workspace"]
                to, rc, out, err = ws.run_subprocess([sys.executable, fake_script], env=env)
                return {"response": out, "actions": 1, "returncode": rc}

            be_pinned = AgenticReplayBackend(
                AgenticReplayConfig(enabled=True, fixture_root=fixture_root, pinned_commit=head_sha),
                agent_runner=_run_wt,
                default_verifier=TaskVerifier(expected_files={"edited.txt": "updated"}),
            )
            pinned_res = be_pinned.replay_agentic_task(
                TaskRecord(id="wt-pinned", project=real_repo, intent="edit"),
                fixture_dir="wt",
            )
            self.assertEqual(pinned_res.outcome, OUTCOME_PASS, pinned_res.fail_reason)
            self.assertIn("edited.txt", pinned_res.diff)

            # Real repository's refs/heads/main and hooks dir are untouched
            real_main = subprocess.run(
                ["git", "-C", real_repo, "rev-parse", "refs/heads/main"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(real_main, head_sha)
            self.assertFalse(os.path.exists(hook_sentinel))
            subprocess.run(["git", "-C", real_repo, "worktree", "remove", "--force", wt_dir], capture_output=True)

    def test_alternates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            sha = _init_git_repo(repo)
            alt_file = os.path.join(repo, ".git", "objects", "info", "alternates")
            os.makedirs(os.path.dirname(alt_file), exist_ok=True)
            with open(alt_file, "w", encoding="utf-8") as f:
                f.write("/some/other/objects\n")

            with self.assertRaises(FixtureSetupError):
                AgenticWorkspace(base_dir=repo, fixture_root=tmp).setup()
            with self.assertRaises(FixtureSetupError):
                AgenticWorkspace(base_dir=repo, pinned_commit=sha, fixture_root=tmp).setup()

    def test_fake_agent_git_config_diff_external_and_host_gpgsign(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken_global_cfg = os.path.join(tmp, "host_gitconfig")
            with open(broken_global_cfg, "w", encoding="utf-8") as f:
                f.write("[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = nonexistent-gpg-bin\n")

            hijack_marker = os.path.join(tmp, "DIFF_EXTERNAL_EXECUTED")
            fake_agent = _write_fake_agent_script(
                os.path.join(tmp, "hijack_agent.py"),
                (
                    "import os, sys\n"
                    f"marker = {hijack_marker!r}\n"
                    "cfg_path = os.path.join('.git', 'config')\n"
                    "with open(cfg_path, 'a', encoding='utf-8') as f:\n"
                    "    f.write(f\"\\n[diff]\\n\\texternal = {sys.executable} -c \\\"open({marker!r}, 'w').write('pwnd')\\\"\\n\")\n"
                    "with open('feature.txt', 'w', encoding='utf-8') as f:\n"
                    "    f.write('real feature\\n')\n"
                ),
            )

            with mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": broken_global_cfg}):
                be = AgenticReplayBackend(
                    AgenticReplayConfig(enabled=True),
                    agent_runner=lambda _ws, env, ctx: (
                        lambda res: {"response": res[2], "actions": 1, "returncode": res[1]}
                    )(ctx["workspace"].run_subprocess([sys.executable, fake_agent], env=env)),
                    default_verifier=TaskVerifier(expected_files={"feature.txt": "real feature"}),
                )
                res = be.replay_agentic_task(TaskRecord(id="git-cfg", project=tmp, intent="add feature"))

            self.assertEqual(res.outcome, OUTCOME_PASS, res.fail_reason)
            self.assertIn("+real feature", res.diff)
            self.assertFalse(os.path.exists(hijack_marker))

    def test_fake_agent_mined_intent_flags_and_fixture_root_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_root = os.path.join(tmp, "fixtures")
            allowed_dir = os.path.join(fixture_root, "proj")
            outside_dir = os.path.join(tmp, "outside_secret")
            os.makedirs(allowed_dir)
            os.makedirs(outside_dir)
            with open(os.path.join(allowed_dir, "allowed.txt"), "w", encoding="utf-8") as f:
                f.write("ok\n")
            with open(os.path.join(outside_dir, "secret.txt"), "w", encoding="utf-8") as f:
                f.write("topsecret\n")

            fake_agent = _write_fake_agent_script(
                os.path.join(tmp, "prompt_reader.py"),
                (
                    "import json, os, sys\n"
                    "stdin_intent = sys.stdin.read()\n"
                    "with open('seen_intent.txt', 'w', encoding='utf-8') as f:\n"
                    "    f.write(stdin_intent)\n"
                    "payload_idx = sys.argv.index('--payload') + 1\n"
                    "payload = json.load(open(sys.argv[payload_idx], encoding='utf-8'))\n"
                    "with open('seen_skill.txt', 'w', encoding='utf-8') as f:\n"
                    "    f.write(payload['skill'] + '|' + payload['memory'])\n"
                ),
            )

            # 1. Task with flag-like intent ("--help") & context_excerpt pointing at existing outside_dir:
            #    outside_dir must NOT be treated as a fixture! Intent and skill must arrive via stdin/file.
            def _runner(_ws_path, env, ctx):
                ws = ctx["workspace"]
                to, rc, out, err = ws.run_subprocess(
                    [sys.executable, fake_agent, "--payload", ctx["prompt_file"]],
                    stdin_text=ctx["intent"],
                    env=env,
                )
                self.assertFalse(os.path.exists(os.path.join(ws.workspace_path, "secret.txt")))
                return {"response": out or "done", "actions": 1, "returncode": rc}

            be = AgenticReplayBackend(
                AgenticReplayConfig(enabled=True, fixture_root=fixture_root),
                agent_runner=_runner,
                default_verifier=TaskVerifier(expected_files={
                    "allowed.txt": "ok",
                    "seen_intent.txt": "--help --malicious",
                    "seen_skill.txt": "# Skill Body|# Memory Body",
                }),
            )
            task = TaskRecord(
                id="flag-intent",
                project=tmp,
                intent="--help --malicious",
                context_excerpt=outside_dir,
            )
            res = be.replay_agentic_task(
                task,
                skill="# Skill Body",
                memory="# Memory Body",
                fixture_dir="proj",
            )
            self.assertEqual(res.outcome, OUTCOME_PASS, res.fail_reason)

            # 2. Explicit fixture_dir outside fixture_root fails closed with REASON_FIXTURE_FAILURE
            escape_res = be.replay_agentic_task(
                task,
                fixture_dir=os.path.join("..", "outside_secret"),
            )
            self.assertEqual(escape_res.outcome, OUTCOME_INCONCLUSIVE)
            self.assertEqual(escape_res.inconclusive_reason, REASON_FIXTURE_FAILURE)

    def test_timeout_kills_process_tree_and_caps_output_and_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = os.path.join(tmp, "grandchild.pid")
            spawner = _write_fake_agent_script(
                os.path.join(tmp, "spawner.py"),
                (
                    "import subprocess, sys, time\n"
                    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                    f"open({pidfile!r}, 'w').write(str(p.pid))\n"
                    "time.sleep(30)\n"
                ),
            )
            noisy = _write_fake_agent_script(
                os.path.join(tmp, "noisy.py"),
                (
                    "import sys\n"
                    "sys.stdout.write('X' * 50000)\n"
                    "open('huge.txt', 'w').write('Y\\n' * 10000)\n"
                ),
            )

            with AgenticWorkspace(max_output_bytes=1024, max_diff_bytes=2048) as ws:
                timed_out, rc, _out, _err = ws.run_subprocess(
                    [sys.executable, spawner], timeout_sec=0.4
                )
                self.assertTrue(timed_out)
                self.assertNotEqual(rc, 0)
                if os.path.exists(pidfile):
                    gpid = int(open(pidfile, encoding="utf-8").read().strip())
                    self.assertFalse(_pid_alive(gpid), "grandchild process should be terminated")

                to2, rc2, out2, _ = ws.run_subprocess(
                    [sys.executable, noisy], timeout_sec=5.0
                )
                self.assertFalse(to2)
                self.assertEqual(rc2, 0)
                self.assertLess(len(out2), 2048)
                self.assertIn("truncated", out2)
                diff = ws.get_diff()
                self.assertLess(len(diff), 4096)
                self.assertIn("diff truncated", diff)


class TestTaskVerifierAndOutcomes(unittest.TestCase):
    def test_verifier_checks_and_no_verifier_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            # No verifier configured must be inconclusive (insufficient_evidence), NOT pass
            empty_v = TaskVerifier()
            outcome, hard, soft, _rat, reason = empty_v.verify(tmp)
            self.assertEqual(outcome, OUTCOME_INCONCLUSIVE)
            self.assertEqual(reason, REASON_INSUFFICIENT_EVIDENCE)
            self.assertEqual(hard, 0.0)
            self.assertEqual(soft, 0.0)

            verifier = TaskVerifier(verifier_fn=lambda _p: (True, "all checks passed"))
            self.assertEqual(verifier.verify(tmp)[:3], (OUTCOME_PASS, 1.0, 1.0))

            failing = TaskVerifier(verifier_fn=lambda _p: (False, "assertion failed"))
            self.assertEqual(failing.verify(tmp)[:3], (OUTCOME_FAIL, 0.0, 0.0))

            exp = TaskVerifier(expected_files={"out.json": '"ok"'})
            self.assertEqual(exp.verify(tmp)[0], OUTCOME_FAIL)
            with open(os.path.join(tmp, "out.json"), "w", encoding="utf-8") as f:
                f.write('{"status": "ok"}')
            self.assertEqual(exp.verify(tmp)[0], OUTCOME_PASS)

            timeout_v = TaskVerifier(test_cmd=[sys.executable, "-c", "import time; time.sleep(5)"], timeout_sec=0.3)
            t_out = timeout_v.verify(tmp)
            self.assertEqual(t_out[0], OUTCOME_INCONCLUSIVE)
            self.assertEqual(t_out[4], REASON_TIMEOUT)

    def test_nonzero_agent_exit_is_infra_and_bad_pin_is_fixture_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_root = os.path.join(tmp, "fixtures")
            repo = os.path.join(fixture_root, "repo")
            _init_git_repo(repo)

            be_bad_pin = AgenticReplayBackend(
                AgenticReplayConfig(
                    enabled=True,
                    fixture_root=fixture_root,
                    pinned_commit="deadbeefdeadbeefdeadbeef",
                ),
                agent_runner=lambda _ws, _env: {"response": "ok", "actions": 1},
                default_verifier=TaskVerifier(verifier_fn=lambda _p: (True, "ok")),
            )
            pairs = replay_batch(be_bad_pin, [TaskRecord(id="badpin", project=repo, intent="test")], "", "")
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][1].outcome, OUTCOME_INCONCLUSIVE)
            self.assertEqual(pairs[0][1].inconclusive_reason, REASON_FIXTURE_FAILURE)

            be_exit = AgenticReplayBackend(
                AgenticReplayConfig(enabled=True),
                agent_runner=lambda _ws, _env: {"response": "crash", "actions": 1, "returncode": 2},
                default_verifier=TaskVerifier(verifier_fn=lambda _p: (True, "ok")),
            )
            res = be_exit.replay_agentic_task(TaskRecord(id="crash", project=tmp, intent="fail"))
            self.assertEqual(res.outcome, OUTCOME_INCONCLUSIVE)
            self.assertEqual(res.inconclusive_reason, REASON_INFRA)

    def test_budgets_and_thread_safety(self):
        be = AgenticReplayBackend(
            AgenticReplayConfig(
                enabled=True,
                max_turns=2,
                max_tokens=100,
                max_replays_per_run=3,
                min_decidable_actions=1,
            ),
            agent_runner=lambda _ws, _env: {"response": "ok", "actions": 1, "turns": 1, "tokens": 0},
            default_verifier=TaskVerifier(verifier_fn=lambda _p: (True, "ok")),
        )
        # Parallel replay batch of 6 tasks against cap 3 must yield strictly 3 pass + 3 budget_exhausted
        tasks = [TaskRecord(id=f"p{i}", project="/p", intent="run") for i in range(6)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda t: be.replay_agentic_task(t), tasks))
        passes = [r for r in results if r.outcome == OUTCOME_PASS]
        budget = [r for r in results if r.inconclusive_reason == REASON_BUDGET_EXHAUSTED]
        self.assertEqual(len(passes), 3)
        self.assertEqual(len(budget), 3)
        self.assertEqual(sum(r.tokens for r in results), 0)  # No hardcoded fake 500-token constant

        # Check turn cap and token cap enforcement
        be2 = AgenticReplayBackend(
            AgenticReplayConfig(enabled=True, max_turns=2, max_tokens=50, max_replays_per_run=5),
            agent_runner=lambda _ws, _env: {"response": "ok", "actions": 1, "turns": 3, "tokens": 10},
            default_verifier=TaskVerifier(verifier_fn=lambda _p: (True, "ok")),
        )
        turn_over = be2.replay_agentic_task(TaskRecord(id="t-turn", project="/p", intent="run"))
        self.assertEqual(turn_over.inconclusive_reason, REASON_BUDGET_EXHAUSTED)

        be3 = AgenticReplayBackend(
            AgenticReplayConfig(enabled=True, max_turns=5, max_tokens=50, max_replays_per_run=5),
            agent_runner=lambda _ws, _env: {"response": "ok", "actions": 1, "turns": 1, "tokens": 60},
            default_verifier=TaskVerifier(verifier_fn=lambda _p: (True, "ok")),
        )
        tok_over = be3.replay_agentic_task(TaskRecord(id="t-tok", project="/p", intent="run"))
        self.assertEqual(tok_over.inconclusive_reason, REASON_BUDGET_EXHAUSTED)


class TestReflectionAndPairwiseGateSemantics(unittest.TestCase):
    def test_tiered_reflection_excludes_inconclusive_via_spy(self):
        base = MockBackend()
        spy_reflect = mock.MagicMock(wraps=base.reflect)
        base.reflect = spy_reflect  # type: ignore[method-assign]
        backend = AgenticReplayBackend(AgenticReplayConfig(enabled=True), base_backend=base)

        t_fail = TaskRecord(id="f1", project="/p", intent="fail deterministically")
        t_timeout = TaskRecord(id="to1", project="/p", intent="timed out")
        failures = [
            (t_fail, ReplayResult(id="f1", hard=0.0, outcome=OUTCOME_FAIL)),
            (t_timeout, ReplayResult(id="to1", hard=0.0, outcome=OUTCOME_INCONCLUSIVE, inconclusive_reason=REASON_TIMEOUT)),
        ]
        successes = [
            (TaskRecord(id="s1", project="/p", intent="ok"), ReplayResult(id="s1", hard=1.0, outcome=OUTCOME_PASS)),
            (TaskRecord(id="s_inc", project="/p", intent="bad"), ReplayResult(id="s_inc", hard=1.0, outcome=OUTCOME_INCONCLUSIVE)),
        ]

        backend.reflect(
            failures,
            successes,
            skill="# s",
            memory="# m",
            edit_budget=2,
            evolve_skill=True,
            evolve_memory=False,
        )
        spy_reflect.assert_called_once()
        passed_failures = spy_reflect.call_args.args[0]
        passed_successes = spy_reflect.call_args.args[1]
        self.assertEqual([t.id for t, _r in passed_failures], ["f1"])
        self.assertEqual([t.id for t, _r in passed_successes], ["s1"])

    def test_aggregate_and_pairwise_scores(self):
        t1 = TaskRecord(id="t1", project="/p", intent="1")
        t2 = TaskRecord(id="t2", project="/p", intent="2")
        t3 = TaskRecord(id="t3", project="/p", intent="3")
        pairs = [
            (t1, ReplayResult(id="t1", hard=1.0, soft=1.0, outcome=OUTCOME_PASS)),
            (t2, ReplayResult(id="t2", hard=0.0, soft=0.0, outcome=OUTCOME_FAIL)),
            (t3, ReplayResult(id="t3", hard=0.0, soft=0.0, outcome=OUTCOME_INCONCLUSIVE, inconclusive_reason=REASON_INFRA)),
        ]
        self.assertAlmostEqual(aggregate_scores(pairs)[0], 1 / 3)
        self.assertAlmostEqual(aggregate_scores(pairs, exclude_inconclusive=True)[0], 0.5)

        base_pairs = [
            (t1, ReplayResult(id="t1", hard=0.0, soft=0.0, outcome=OUTCOME_FAIL)),
            (t2, ReplayResult(id="t2", hard=1.0, soft=1.0, outcome=OUTCOME_PASS)),
            (t3, ReplayResult(id="t3", hard=1.0, soft=1.0, outcome=OUTCOME_PASS)),
        ]
        cand_pairs = [
            (t1, ReplayResult(id="t1", hard=1.0, soft=1.0, outcome=OUTCOME_PASS)),
            (t2, ReplayResult(id="t2", hard=1.0, soft=1.0, outcome=OUTCOME_PASS)),
            (t3, ReplayResult(id="t3", hard=0.0, soft=0.0, outcome=OUTCOME_INCONCLUSIVE, inconclusive_reason=REASON_BUDGET_EXHAUSTED)),
        ]
        b_pair, c_pair, n_dec, n_tot, cov = pairwise_aggregate_scores(base_pairs, cand_pairs)
        self.assertEqual((n_dec, n_tot), (2, 3))
        self.assertAlmostEqual(cov, 2 / 3)
        self.assertAlmostEqual(b_pair[0], 0.5)   # baseline scored ONLY on {t1, t2}
        self.assertAlmostEqual(c_pair[0], 1.0)   # candidate scored ONLY on {t1, t2}

    def test_consolidate_4_task_budget_exhaustion_abstains_pairwise(self):
        # Point 8 scenario: max_replays_per_run=5 on 4 val tasks + 1 train task.
        # Baseline val gets 4 passes (replays 1..4), train uses replay 5 ->
        # candidate trial & final val get inconclusive budget_exhausted.
        val_tasks = [TaskRecord(id=f"v{i}", project="/p", intent="solve", split="val") for i in range(4)]
        train_task = TaskRecord(id="tr0", project="/p", intent="learn", split="train")
        tasks = [train_task, *val_tasks]

        class _ProposedEditBackend(MockBackend):
            def reflect(
                self, failures, successes, skill, memory, *,
                edit_budget=4, evolve_skill=True, evolve_memory=True,
            ):
                if evolve_skill:
                    return [EditRecord(target="skill", op="add", content="New rule", rationale="fix")]
                return []

        def _runner(ws_path, _env, ctx):
            return {"response": "ok", "actions": 1}

        # Replays 1..4 = baseline val pass; replay 5 = train pass;
        # candidate trial runs v0 (replay 6) then v1..v3 budget_exhausted!
        be_partial = AgenticReplayBackend(
            AgenticReplayConfig(enabled=True, max_replays_per_run=6),
            base_backend=_ProposedEditBackend(),
            agent_runner=_runner,
            default_verifier=TaskVerifier(verifier_fn=lambda _p: (True, "pass")),
        )
        res = consolidate(
            be_partial,
            tasks,
            skill="# Base",
            memory="",
            evolve_memory=False,
            gate_metric="hard",
            min_decidable_coverage=0.5,
        )
        self.assertFalse(res.accepted)
        self.assertEqual(res.gate_action, "abstain_inconclusive")
        self.assertEqual(res.new_skill, "# Base")
        self.assertEqual(res.applied_edits, [])
        # Crucially: pairwise baseline/candidate never drop to 0.25 due to inconclusive zeros
        self.assertGreaterEqual(res.baseline_score, 0.0)
        trial_skill = [g for g in res.gate_trials if g["target"] == "skill"][0]
        # Candidate trial had v0 pass (replay 6) and v1..v3 budget_exhausted => baseline_score==1.0, candidate_score==1.0, abstained!
        self.assertAlmostEqual(trial_skill["baseline_score"], 1.0)
        self.assertAlmostEqual(trial_skill["candidate_score"], 1.0)
        self.assertTrue(trial_skill["abstained_inconclusive"])
        self.assertAlmostEqual(trial_skill["decidable_coverage"], 0.25)

    def test_consolidate_pairwise_accepts_above_coverage_floor_with_no_regression(self):
        val_tasks = [TaskRecord(id=f"v{i}", project="/p", intent=f"val {i}", split="val") for i in range(4)]
        train_task = TaskRecord(id="tr0", project="/p", intent="learn", split="train")
        tasks = [train_task, *val_tasks]

        class _EditBackend(MockBackend):
            def reflect(
                self, failures, successes, skill, memory, *,
                edit_budget=4, evolve_skill=True, evolve_memory=True,
            ):
                if evolve_skill:
                    return [EditRecord(target="skill", op="add", content="Improved rule", rationale="improve v0")]
                return []

        def _batch_side_effect(_backend, batch_tasks, skill, _memory, **_kw):
            out = []
            improved = "Improved rule" in skill
            for t in batch_tasks:
                if t.id == "v0":
                    out.append((t, ReplayResult(id=t.id, hard=1.0 if improved else 0.0, soft=1.0 if improved else 0.0)))
                elif t.id in {"v1", "v2"}:
                    out.append((t, ReplayResult(id=t.id, hard=1.0, soft=1.0)))
                elif t.id == "v3":
                    # Candidate hits timeout on v3; baseline passed v3. Pairwise excludes v3 from both sides!
                    if improved:
                        out.append((t, ReplayResult(id=t.id, hard=0.0, outcome=OUTCOME_INCONCLUSIVE, inconclusive_reason=REASON_TIMEOUT)))
                    else:
                        out.append((t, ReplayResult(id=t.id, hard=1.0, soft=1.0)))
                else:
                    out.append((t, ReplayResult(id=t.id, hard=0.0, outcome=OUTCOME_FAIL)))
            return out

        with mock.patch("skillopt_sleep.consolidate.replay_batch", side_effect=_batch_side_effect):
            res = consolidate(
                _EditBackend(),
                tasks,
                skill="# Base",
                memory="",
                evolve_memory=False,
                gate_metric="hard",
                gate_no_regression=True,
                min_decidable_coverage=0.5,
            )

        self.assertTrue(res.accepted)
        self.assertEqual(res.gate_action, "accept_new_best")
        # Baseline on {v0, v1, v2} = 2/3, Candidate on {v0, v1, v2} = 3/3 (coverage 0.75 >= 0.5)
        self.assertAlmostEqual(res.baseline_score, 2 / 3)
        self.assertAlmostEqual(res.candidate_score, 1.0)
        final_trial = res.gate_trials[-1]
        v3_row = [r for r in final_trial["task_deltas"] if r["task_id"] == "v3"][0]
        self.assertEqual(v3_row["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
