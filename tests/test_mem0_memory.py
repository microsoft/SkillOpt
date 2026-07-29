"""Mocked tests for the opt-in mem0 memory backend.

Run directly (no pytest needed):
    python tests/test_mem0_memory.py

No network is touched: a fake client records what *would* have been sent, so
the assertions are about the exact bytes leaving the process.

Covers:
1. Disabled by default — a ``MEM0_API_KEY`` in the environment must not, on its
   own, enable uploads. This is the central safety property.
2. Explicit opt-in does enable, and writes reach the client.
3. Redaction — credentials and home-directory paths are stripped from the
   payload, and the cap is applied to the redacted text.
4. Namespacing — stable per project, distinct across projects, and never the
   raw filesystem path.
5. Retrieval reaches the *actual* reflection prompt, verified end-to-end by
   capturing the user message handed to the optimizer.
6. Service failures (exceptions) degrade gracefully — training continues and
   the reflection context is returned unchanged.
7. Slow services are bounded by ``mem0_timeout_seconds``.
8. Malformed patches (``None``, strings, non-lists) are filtered, not raised.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

# Ensure THIS repo's skillopt is imported (not an installed copy) when the
# file is run directly: script mode puts tests/ on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillopt.memory.mem0_backend import SkillMemory  # noqa: E402
from skillopt.memory.redaction import redact_for_upload  # noqa: E402
from skillopt.memory.settings import (  # noqa: E402
    project_namespace,
    resolve_settings,
)
from skillopt.memory.trainer_hooks import (  # noqa: E402
    hook_post_reflect,
    hook_pre_reflect,
    maybe_init_mem0,
)

FAKE_KEY = "m0-abcdefghijklmnopqrstuvwxyz012345"


class FakeClient:
    """Stand-in for ``mem0.MemoryClient`` that records instead of sending."""

    def __init__(self, search_results=None, fail: bool = False, delay: float = 0.0):
        self.adds: list[tuple] = []
        self.searches: list[tuple] = []
        self._results = search_results or []
        self._fail = fail
        self._delay = delay

    def add(self, messages, **kwargs):
        if self._delay:
            time.sleep(self._delay)
        if self._fail:
            raise RuntimeError("mem0 service unavailable")
        self.adds.append((messages, kwargs))
        return {"ok": True}

    def search(self, query, **kwargs):
        if self._delay:
            time.sleep(self._delay)
        if self._fail:
            raise RuntimeError("mem0 service unavailable")
        self.searches.append((query, kwargs))
        return list(self._results)


class _EnvPatch:
    """Temporarily set environment variables."""

    def __init__(self, **kv):
        self._kv = kv
        self._old: dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self._kv.items():
            self._old[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, old in self._old.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        return False


def _enabled_settings(**overrides):
    cfg = {
        "mem0_enabled": True,
        "mem0_api_key": FAKE_KEY,
        "out_root": "/tmp/skillopt-test-project",
        "env": "alfworld",
    }
    cfg.update(overrides)
    return resolve_settings(cfg, env={})


# ── 1. Disabled by default ────────────────────────────────────────────────────

def test_api_key_alone_does_not_enable():
    """The maintainer's primary objection: a stray key must export nothing."""
    cfg = {"out_root": "/tmp/proj", "env": "alfworld"}
    settings = resolve_settings(cfg, env={"MEM0_API_KEY": FAKE_KEY})

    assert settings.enabled is False, "a bare env key must not enable memory"
    assert settings.usable is False
    assert settings.api_key == "", "a disabled run must not even read the key"

    with _EnvPatch(MEM0_API_KEY=FAKE_KEY):
        assert maybe_init_mem0(cfg) is None, "no backend may be constructed when disabled"

    print("ok  1. MEM0_API_KEY alone does not enable uploads")


def test_disabled_hooks_send_nothing():
    """With memory None, every hook is inert and returns context untouched."""
    original = "prior step context"
    assert hook_pre_reflect(None, "skill", original) == original
    hook_post_reflect(None, 0, 0, [{"patch": {}}])  # must not raise
    print("ok  1b. hooks are no-ops when memory is disabled")


# ── 2. Explicit opt-in ────────────────────────────────────────────────────────

def test_explicit_opt_in_writes():
    settings = _enabled_settings()
    assert settings.enabled and settings.usable

    client = FakeClient()
    memory = SkillMemory(settings, client=client)
    memory.store_skill_iteration(epoch=1, step=2, skill_text="do the thing", score=0.75)

    assert len(client.adds) == 1, "an opted-in run should write exactly once"
    messages, kwargs = client.adds[0]
    assert kwargs["user_id"] == settings.namespace
    assert "do the thing" in messages[0]["content"]
    memory.close()
    print("ok  2. explicit opt-in writes to the backend")


# ── 3. Redaction ──────────────────────────────────────────────────────────────

def test_redaction_strips_secrets_and_paths():
    settings = _enabled_settings(out_root="/home/alice/projects/run")
    client = FakeClient()
    memory = SkillMemory(settings, client=client)

    leaky = (
        "Use sk-abcdefghijklmnop1234 to authenticate.\n"
        "Config lives at /home/alice/projects/run/configs/train.yaml\n"
        "Also check /home/bob/other/notes.md\n"
        "api_key = supersecretvalue\n"
    )
    memory.store_skill_iteration(epoch=0, step=0, skill_text=leaky, score=1.0)

    sent = client.adds[0][0][0]["content"]
    assert "sk-abcdefghijklmnop1234" not in sent, "API key leaked"
    assert "supersecretvalue" not in sent, "assigned secret leaked"
    assert "/home/alice" not in sent, "home path (username) leaked"
    assert "/home/bob" not in sent, "third-party home path leaked"
    # Structure that makes the memory useful must survive.
    assert "configs/train.yaml" in sent, "redaction destroyed useful structure"
    memory.close()
    print("ok  3. secrets and home paths are stripped, structure preserved")


def test_truncation_measured_after_redaction():
    settings = _enabled_settings(mem0_max_chars=120)
    client = FakeClient()
    memory = SkillMemory(settings, client=client)
    memory.store_skill_iteration(epoch=0, step=0, skill_text="x" * 5000, score=0.0)

    sent = client.adds[0][0][0]["content"]
    assert len(sent) <= 120 + len("\n[...truncated]")
    memory.close()
    print("ok  3b. payload cap applied to the redacted text")


def test_redaction_helper_is_order_safe():
    """A key embedded in a home path must be scrubbed before the path collapses."""
    out = redact_for_upload("/home/alice/.config/sk-abcdefghijklmnop1234", "")
    assert "sk-abcdefghijklmnop1234" not in out
    assert "/home/alice" not in out
    print("ok  3c. credential inside a home path is still redacted")


# ── 4. Namespacing ────────────────────────────────────────────────────────────

def test_namespace_is_stable_and_project_specific():
    a1 = project_namespace("/home/alice/projA", "alfworld")
    a2 = project_namespace("/home/alice/projA", "alfworld")
    b = project_namespace("/home/alice/projB", "alfworld")
    other_env = project_namespace("/home/alice/projA", "searchqa")

    assert a1 == a2, "namespace must be stable across runs of one project"
    assert a1 != b, "distinct projects must not share a namespace"
    assert a1 != other_env, "distinct envs must not share a namespace"
    assert "/home/alice" not in a1, "namespace must not carry the raw path"
    assert a1.startswith("skillopt:alfworld:")
    print("ok  4. namespace stable, project-specific, and path-free")


def test_explicit_namespace_override_wins():
    settings = _enabled_settings(mem0_namespace="team-shared-pool")
    assert settings.namespace == "team-shared-pool"
    print("ok  4b. explicit mem0_namespace overrides the derived one")


# ── 4c. Structured configs must actually reach the trainer ────────────────────

def test_structured_config_keys_survive_flattening():
    """Regression: every shipped config is structured (inherits _base_/default).

    Structured configs are flattened through an explicit key map, so a
    top-level ``mem0_enabled`` would be silently dropped — the feature would
    appear inert with no error. The keys must be mapped under ``train``.
    """
    from skillopt.config import flatten_config

    flat = flatten_config({
        "train": {
            "mem0_enabled": True,
            "mem0_retrieval_limit": 3,
            "mem0_timeout_seconds": 2.5,
        },
    })
    assert flat.get("mem0_enabled") is True, "mem0_enabled dropped when flattening"
    assert flat.get("mem0_retrieval_limit") == 3
    assert flat.get("mem0_timeout_seconds") == 2.5

    settings = resolve_settings(flat, env={"MEM0_API_KEY": FAKE_KEY})
    assert settings.enabled and settings.retrieval_limit == 3
    print("ok  4c. structured-config mem0 keys survive flattening")


def test_shipped_example_config_enables_memory():
    """The opt-in smoke example must actually switch the feature on."""
    from skillopt.config import flatten_config, load_config

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example = os.path.join(repo_root, "configs", "features", "mem0_memory.yaml")
    assert os.path.exists(example), "shipped example config is missing"

    flat = flatten_config(load_config(example))
    settings = resolve_settings(flat, env={"MEM0_API_KEY": FAKE_KEY})

    assert settings.enabled, "the example config does not enable memory"
    assert settings.usable, "the example config does not yield a usable backend"
    assert settings.timeout_seconds == 5.0
    assert settings.max_chars == 4000
    print("ok  4d. shipped example config enables memory end to end")


# ── 5. Retrieval reaches the reflection prompt ────────────────────────────────

def test_retrieved_context_reaches_reflection_prompt():
    """End-to-end: stored memory → hook → the user message sent to the optimizer."""
    memo = "Open the drawer before searching it; searching first always failed."
    settings = _enabled_settings()
    client = FakeClient(search_results=[{"memory": memo}])
    memory = SkillMemory(settings, client=client)

    reflect_context = hook_pre_reflect(memory, "current skill text", "prior step context")
    assert memo in reflect_context, "retrieved memory missing from reflection context"
    assert "prior step context" in reflect_context, "existing context was dropped"

    # Now prove that context actually lands in the prompt the optimizer sees.
    from skillopt.gradient import reflect as reflect_mod

    captured: dict[str, str] = {}

    def fake_chat_optimizer(system, user, **kwargs):
        captured["user"] = user
        return ('{"patch": {"edits": []}}', None)

    original = reflect_mod.chat_optimizer
    reflect_mod.chat_optimizer = fake_chat_optimizer
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = os.path.join(tmp, "predictions")
            task_dir = os.path.join(pred_dir, "task-1")
            os.makedirs(task_dir)
            with open(os.path.join(task_dir, "conversation.json"), "w") as fh:
                json.dump([{"role": "user", "content": "go to the kitchen"}], fh)

            reflect_mod.run_error_analyst_minibatch(
                skill_content="current skill text",
                items=[{
                    "id": "task-1",
                    "task_description": "find the mug",
                    "task_type": "pick",
                    "fail_reason": "searched before opening",
                }],
                prediction_dir=pred_dir,
                step_buffer_context=reflect_context,
            )
    finally:
        reflect_mod.chat_optimizer = original

    assert "user" in captured, "optimizer was never called — test did not exercise the prompt"
    assert memo in captured["user"], "retrieved memory never reached the reflection prompt"
    memory.close()
    print("ok  5. retrieved memory reaches the actual reflection prompt")


def test_retrieval_can_be_disabled_independently():
    settings = _enabled_settings(mem0_retrieval_enabled=False)
    client = FakeClient(search_results=[{"memory": "should not be used"}])
    memory = SkillMemory(settings, client=client)

    out = hook_pre_reflect(memory, "skill", "ctx")
    assert out == "ctx"
    assert client.searches == [], "no search should be issued when retrieval is off"
    memory.close()
    print("ok  5b. retrieval can be turned off while writes stay on")


# ── 6. Failure handling ───────────────────────────────────────────────────────

def test_service_failure_degrades_gracefully():
    settings = _enabled_settings()
    client = FakeClient(fail=True)
    memory = SkillMemory(settings, client=client)

    # Writes must not raise.
    memory.store_skill_iteration(epoch=0, step=0, skill_text="s", score=0.0)
    hook_post_reflect(memory, 0, 0, [{"patch": {}}])

    # Reads must not raise, and must leave the context untouched.
    out = hook_pre_reflect(memory, "skill", "prior context")
    assert out == "prior context", "a failed retrieval must not alter reflection input"
    memory.close()
    print("ok  6. service failure degrades gracefully")


def test_slow_service_is_bounded():
    settings = _enabled_settings(mem0_timeout_seconds=0.2)
    client = FakeClient(search_results=[{"memory": "late"}], delay=3.0)
    memory = SkillMemory(settings, client=client)

    start = time.time()
    out = hook_pre_reflect(memory, "skill", "prior context")
    elapsed = time.time() - start

    assert elapsed < 2.0, f"retrieval blocked training for {elapsed:.1f}s despite a 0.2s bound"
    assert out == "prior context", "timed-out retrieval must not alter reflection input"
    memory.close()
    print(f"ok  7. slow service bounded ({elapsed:.2f}s, limit 0.2s)")


# ── 9. Copilot review follow-ups ──────────────────────────────────────────────

def test_retrieved_text_is_redacted_before_entering_the_prompt():
    """Inbound redaction: mem0 is an external store and may hold raw secrets.

    Anything it returns flows into the reflection prompt and on to the
    optimizer's model provider, so it must be scrubbed on the way *in* as well
    as on the way out.
    """
    poisoned = "Earlier run used sk-abcdefghijklmnop1234 from /home/alice/creds.txt"
    settings = _enabled_settings()
    client = FakeClient(search_results=[{"memory": poisoned}])
    memory = SkillMemory(settings, client=client)

    ctx = hook_pre_reflect(memory, "skill", "prior context")
    assert "sk-abcdefghijklmnop1234" not in ctx, "secret from mem0 reached the prompt"
    assert "/home/alice" not in ctx, "home path from mem0 reached the prompt"
    assert "Earlier run used" in ctx, "redaction destroyed the whole record"
    memory.close()
    print("ok  9. retrieved text is redacted before entering the prompt")


def test_repeated_timeouts_disable_the_backend():
    """A wedged service must cost a bounded total, not a bounded amount per step.

    With a single worker, a stuck call would otherwise make every later call
    queue behind it and time out again, so the cost would recur every step.
    """
    settings = _enabled_settings(mem0_timeout_seconds=0.1)
    client = FakeClient(search_results=[{"memory": "late"}], delay=5.0)
    memory = SkillMemory(settings, client=client)

    start = time.time()
    for _ in range(6):
        hook_pre_reflect(memory, "skill", "ctx")
    elapsed = time.time() - start

    assert memory._degraded, "backend should disable itself after repeated timeouts"
    # 3 timeouts at 0.1s, then short-circuited: well under 6 x 0.1s.
    assert elapsed < 1.0, f"repeated timeouts cost {elapsed:.2f}s — not bounded"
    memory.close()
    print(f"ok  9b. repeated timeouts disable the backend ({elapsed:.2f}s for 6 calls)")


def test_timeout_does_not_block_the_next_call():
    """The executor is retired on timeout, so a stuck thread is not serialising."""
    settings = _enabled_settings(mem0_timeout_seconds=0.1)
    client = FakeClient(search_results=[{"memory": "late"}], delay=5.0)
    memory = SkillMemory(settings, client=client)

    first_pool = memory._pool
    hook_pre_reflect(memory, "skill", "ctx")
    assert memory._pool is not first_pool, "executor was not retired after timeout"
    memory.close()
    print("ok  9c. executor is retired after a timeout")


def test_close_is_idempotent_and_unregisters():
    settings = _enabled_settings()
    memory = SkillMemory(settings, client=FakeClient())
    memory.close()
    memory.close()  # must not raise
    assert memory._closed
    # A closed backend refuses further calls rather than resurrecting the pool.
    assert memory.retrieve_relevant_context("anything") == []
    print("ok  9d. close() is idempotent and stops further calls")


def test_successful_call_resets_the_failure_counter():
    settings = _enabled_settings()
    client = FakeClient(fail=True)
    memory = SkillMemory(settings, client=client)

    memory.store_skill_iteration(epoch=0, step=0, skill_text="s", score=0.0)
    assert memory._consecutive_failures == 1
    client._fail = False
    memory.store_skill_iteration(epoch=0, step=1, skill_text="s", score=0.0)
    assert memory._consecutive_failures == 0, "a success should reset the counter"
    assert not memory._degraded
    memory.close()
    print("ok  9e. a successful call resets the failure counter")


# ── 8. Malformed patches ──────────────────────────────────────────────────────

def test_malformed_patches_are_filtered_not_raised():
    settings = _enabled_settings()
    client = FakeClient()
    memory = SkillMemory(settings, client=client)

    for bad in (None, "a string", 42, [None, "junk", {"patch": {"edits": []}}], {}):
        memory.store_reflection(epoch=0, step=0, patches=bad)

    # The mixed list contributes exactly one valid patch; nothing raised.
    mixed = [c for c in client.adds if "1 patch(es)" in c[0][0]["content"]]
    assert len(mixed) == 1, "the one valid dict patch should have been counted"
    memory.close()
    print("ok  8. malformed patches are filtered, not raised")


ALL_TESTS = [
    test_api_key_alone_does_not_enable,
    test_disabled_hooks_send_nothing,
    test_explicit_opt_in_writes,
    test_redaction_strips_secrets_and_paths,
    test_truncation_measured_after_redaction,
    test_redaction_helper_is_order_safe,
    test_namespace_is_stable_and_project_specific,
    test_explicit_namespace_override_wins,
    test_structured_config_keys_survive_flattening,
    test_shipped_example_config_enables_memory,
    test_retrieved_context_reaches_reflection_prompt,
    test_retrieval_can_be_disabled_independently,
    test_service_failure_degrades_gracefully,
    test_slow_service_is_bounded,
    test_retrieved_text_is_redacted_before_entering_the_prompt,
    test_repeated_timeouts_disable_the_backend,
    test_timeout_does_not_block_the_next_call,
    test_close_is_idempotent_and_unregisters,
    test_successful_call_resets_the_failure_counter,
    test_malformed_patches_are_filtered_not_raised,
]


def main() -> int:
    failures = 0
    for fn in ALL_TESTS:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, don't abort the suite
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
