"""Tests for CliBackend response caching + thread-safety on the parallel path."""

from __future__ import annotations

import threading

from skillopt_sleep.backend import CliBackend, Backend, DualBackend
from skillopt_sleep.replay import replay_one
from skillopt_sleep.types import TaskRecord


class _EchoBackend(CliBackend):
    """Minimal backend: echoes the prompt, records call count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.calls += 1
        return f"resp:{prompt}"


def test_cached_call_caches_and_counts_tokens():
    b = _EchoBackend()
    out = b._cached_call("k:1", "hello")
    assert out == "resp:hello"
    assert b.calls == 1
    assert b._tokens > 0

    # A cache hit returns the value and adds no tokens.
    tokens_before = b._tokens
    out2 = b._cached_call("k:1", "hello")
    assert out2 == "resp:hello"
    assert b.calls == 1  # no new _call on the hit
    assert b._tokens == tokens_before


def test_cached_call_distinct_keys_are_distinct():
    b = _EchoBackend()
    assert b._cached_call("k:1", "a") == "resp:a"
    assert b._cached_call("k:2", "b") == "resp:b"
    assert b.calls == 2


def test_cached_call_concurrent_access_does_not_corrupt():
    """Concurrent workers over one backend must not lose cache/token updates."""
    b = _EchoBackend()
    results: list[str] = []
    errors: list[Exception] = []

    def worker():
        try:
            results.append(b._cached_call("k:1", "hello"))
        except Exception as exc:  # pragma: no cover - safety net
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 8
    # The cache may duplicate a call on a concurrent miss, but every result is
    # a valid cached value and the cache/token state stays consistent.
    assert all(r == "resp:hello" for r in results)
    assert b._cache["k:1"] == "resp:hello"


def test_concurrent_misses_with_barrier_are_consistent():
    """Force overlapping cache misses with a barrier; state stays consistent."""
    n = 6
    barrier = threading.Barrier(n)

    class _BarrierBackend(_EchoBackend):
        def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
            barrier.wait(timeout=15)
            return super()._call(prompt, max_tokens=max_tokens)

    b = _BarrierBackend()
    results: list[str] = []
    errors: list[Exception] = []

    def worker():
        try:
            results.append(b._cached_call("k:1", "hello"))
        except Exception as exc:  # pragma: no cover - safety net
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert all(r == "resp:hello" for r in results)
    assert b._cache["k:1"] == "resp:hello"


def test_cache_pop_if_only_removes_own_value():
    """A failed caller must not delete another worker's successful result."""
    b = _EchoBackend()
    b._cache["k:1"] = "resp:hello"
    # Failed caller (pop-if with its empty value) must NOT remove a success.
    b._cache_pop_if("k:1", "")
    assert b._cache["k:1"] == "resp:hello"
    # But it does remove an entry that still holds the expected (empty) value.
    b._cache["k:2"] = ""
    b._cache_pop_if("k:2", "")
    assert "k:2" not in b._cache


def test_token_delta_is_call_local():
    """A single call records a positive per-call token delta on this thread."""
    b = _EchoBackend()
    b._cached_call("k:1", "hello")
    assert b.token_delta() > 0


class _LegacyBackend(Backend):
    """Implements only the older tokens_used() contract (no token_delta)."""

    def __init__(self, start: int = 100):
        self._tokens = start

    def tokens_used(self) -> int:
        return self._tokens

    def attempt(self, task, skill, memory, sample_id: int = 0):
        self._tokens += 37
        return "done"

    def judge(self, task, response):
        return 1.0, 1.0, ""


def _task(intent="done"):
    return TaskRecord(id="t1", project="p1", intent=intent, reference_kind="rule", judge=None)


def test_replay_one_legacy_backend_reports_total_diff():
    """A backend with only tokens_used() (no token_delta) must report the call-local
    cost as a same-thread before/after difference, not a text-length estimate."""
    b = _LegacyBackend(start=100)
    r = replay_one(b, _task(), "", "")
    assert r.tokens == 37, f"expected 37 (before/after diff), got {r.tokens}"


def test_dual_backend_legacy_target_reports_total_diff():
    """DualBackend wrapping a tokens_used()-only target must report the target's
    per-attempt cost, not a text-length estimate."""
    target = _LegacyBackend(start=100)
    optimizer = _LegacyBackend(start=0)
    db = DualBackend(target=target, optimizer=optimizer)
    r = replay_one(db, _task(), "", "")
    assert r.tokens == 37, f"expected 37 (target diff), got {r.tokens}"


def test_replay_one_cache_hit_reports_zero_tokens():
    """A replay served from cache has no new model call and must not fall back
    to a text-length estimate: known-zero call-local usage stays zero."""
    b = _EchoBackend()
    task = TaskRecord(
        id="t1",
        project="p1",
        intent="intent text here",
        reference_kind="exact",
        reference="expected",
    )

    first = replay_one(b, task, "skill", "memory")
    second = replay_one(b, task, "skill", "memory")

    assert b.calls == 1, "cache-hit replay must not invoke the model again"
    assert first.tokens > 0
    assert second.tokens == 0, (
        f"expected 0 tokens for a cache hit, got {second.tokens}"
    )


def test_token_delta_isolated_between_threads():
    """Per-thread token deltas do not leak across parallel workers."""
    b = _EchoBackend()
    deltas: dict[int, int] = {}

    def worker(i: int):
        b._cached_call(f"k:{i}", "hello")
        deltas[i] = b.token_delta()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each worker that actually made a call saw its own positive delta.
    for i in range(4):
        assert deltas[i] > 0, f"worker {i} got no call-local delta"


def test_cache_hit_resets_token_delta():
    """A cache hit must reset the per-thread delta (store-then-load reuse)."""
    b = _EchoBackend()
    b._cached_call("k:1", "hello")  # miss
    assert b.token_delta() > 0
    b._cached_call("k:1", "hello")  # hit
    assert b.token_delta() == 0, "cache hit leaked the previous call's delta"


def test_concurrent_missing_charges_every_real_call():
    """A barrier-forced same-key concurrent miss must charge every real call."""
    n = 6
    barrier = threading.Barrier(n)

    class _BarrierBackend(_EchoBackend):
        def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
            barrier.wait(timeout=15)
            return super()._call(prompt, max_tokens=max_tokens)

    b = _BarrierBackend()
    threads = [threading.Thread(target=lambda: b._cached_call("k:1", "hello")) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    per = len("hello") // 4 + len("resp:hello") // 4
    assert b.calls == n
    assert b._tokens == per * b.calls, "concurrent misses undercounted real calls"


def test_dual_backend_token_delta_returns_target():
    """DualBackend.token_delta() must report the target backend's call cost."""
    from skillopt_sleep.backend import DualBackend

    target = _EchoBackend()
    target._cached_call("k:1", "hello")
    optimizer = _EchoBackend()
    db = DualBackend(target=target, optimizer=optimizer)
    assert db.token_delta() == target.token_delta()


def test_dual_backend_modern_target_parallel_replay_is_call_local():
    """Parallel replay through a DualBackend over a modern target keeps each
    result's cost on its own task.

    ``token_delta()`` delegates to a target that has a thread-local delta, so
    this combination is safe to share across workers; the cumulative-only
    fallbacks are a different case and are deliberately not asserted here.

    The target holds every worker inside the call, so the spends really do
    overlap: without that, a backend that lost its thread-locality would still
    pass by never racing.
    """
    from skillopt_sleep.replay import replay_batch

    class _OverlappingBackend(_EchoBackend):
        """Blocks until every worker is in flight before spending tokens."""

        def __init__(self, parties: int) -> None:
            super().__init__()
            self._barrier = threading.Barrier(parties, timeout=15)

        def attempt(self, task, skill, memory, sample_id: int = 0):
            self._barrier.wait()
            return super().attempt(task, skill, memory, sample_id=sample_id)

    tasks = [
        TaskRecord(
            id=f"t{i}",
            project="p1",
            intent=f"intent-{i}-" + "x" * (25 * i),
            reference_kind="exact",
            reference="resp:ok",
        )
        for i in range(4)
    ]

    def costs(workers: int) -> dict[str, int]:
        backend = DualBackend(
            target=_OverlappingBackend(workers), optimizer=_EchoBackend()
        )
        return {
            t.id: r.tokens
            for t, r in replay_batch(backend, tasks, "skill", "memory", workers=workers)
        }

    sequential = costs(1)
    parallel = costs(2)

    assert len(set(sequential.values())) > 1, (
        "tasks must differ in cost, otherwise this check proves nothing"
    )
    assert parallel == sequential, "parallel workers leaked call-local cost across tasks"


def test_attempt_with_tools_sets_call_local_delta(monkeypatch):
    """Tool-aware replay must set the thread-local delta so replay_one() sees
    real call-local usage instead of falling back to a response-length estimate."""
    from types import SimpleNamespace

    import skillopt_sleep.backend as backend_mod
    from skillopt_sleep.backend import ClaudeCliBackend

    b = ClaudeCliBackend(model="", claude_path="claude", timeout=10)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(backend_mod.subprocess, "run", fake_run)

    task = SimpleNamespace(intent="intent", context_excerpt="ctx")
    response, called = b.attempt_with_tools(task, skill="s", memory="m", tools=["search"])
    assert response == "ok"
    assert b.token_delta() > 0, "attempt_with_tools did not set call-local delta"


def test_dual_backend_attempt_with_tools_sets_target_delta(monkeypatch):
    """The dual-backend tool-replay path must also surface the target's delta."""
    from types import SimpleNamespace

    import skillopt_sleep.backend as backend_mod
    from skillopt_sleep.backend import ClaudeCliBackend, DualBackend

    target = ClaudeCliBackend(model="", claude_path="claude", timeout=10)
    optimizer = ClaudeCliBackend(model="", claude_path="claude", timeout=10)
    db = DualBackend(target=target, optimizer=optimizer)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(backend_mod.subprocess, "run", fake_run)

    task = SimpleNamespace(intent="intent", context_excerpt="ctx")
    response, called = db.attempt_with_tools(task, skill="s", memory="m", tools=["search"])
    assert response == "ok"
    assert db.token_delta() > 0, "dual-backend tool replay did not set target delta"


def test_record_cost_concurrent_no_lost_updates():
    """Barrier-forced concurrent _record_cost calls must not lose updates."""
    n = 30
    barrier = threading.Barrier(n)

    class _BarrierEcho(_EchoBackend):
        def _cost_worker(self):
            barrier.wait(timeout=15)
            self._record_cost("hello", "world")

    b = _BarrierEcho()
    threads = [threading.Thread(target=b._cost_worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    per = len("hello") // 4 + len("world") // 4
    assert b._tokens == per * n, f"concurrent _record_cost lost updates: {b._tokens} != {per * n}"


def test_disabled_tool_replay_resets_call_delta():
    """A prior call's delta must not leak through a disabled-tool-replay path."""
    from types import SimpleNamespace

    from skillopt_sleep.backend import OpenCodeCliBackend

    b = OpenCodeCliBackend(model="", opencode_path="opencode", tool_replay=False)
    # A prior call on this thread set a nonzero call-local delta.
    b._record_cost("hello", "world")
    assert b.token_delta() > 0  # stale from the prior call
    # Disabled-tool-replay attempt_with_tools returns early; the delta must be
    # reset to 0 so a reused worker does not report the previous call's cost.
    task = SimpleNamespace(intent="intent", context_excerpt="ctx")
    out, called = b.attempt_with_tools(task, skill="s", memory="m", tools=["search"])
    assert out == "" and called == []
    assert b.token_delta() == 0, "disabled-tool-replay leaked the prior call's delta"


def test_cmd_harvest_redact_deep_is_key_aware():
    """_redact_deep must key-aware redact — `{"api_key": "x"}` used to leak."""
    from skillopt_sleep.__main__ import _redact_deep

    out = _redact_deep({"api_key": "x", "content": "keep me", "nested": {"token": "y"}})
    assert out == {"api_key": "[REDACTED]", "content": "keep me", "nested": {"token": "[REDACTED]"}}


def test_concurrent_pi_empty_and_success_preserve_success(monkeypatch):
    """A failed (empty) Pi worker must not clobber another worker's success."""
    from skillopt_sleep.backend import PiCliBackend

    b = PiCliBackend(model="x")
    n = 2
    barrier = threading.Barrier(n)
    state: dict[str, str] = {}

    def fake_call(prompt: str, *, max_tokens: int = 1024) -> str:
        barrier.wait(timeout=15)
        return state[threading.current_thread().name]

    monkeypatch.setattr(b, "_call", fake_call)
    results: list[str] = []

    def worker(ret: str):
        state[threading.current_thread().name] = ret
        results.append(b._cached_call("k:1", "p"))

    th_empty = threading.Thread(target=worker, args=("",))
    th_success = threading.Thread(target=worker, args=("resp:ok",))
    th_empty.start()
    th_success.start()
    th_empty.join()
    th_success.join()

    # The successful result is cached; the empty worker never removes it.
    assert b._cache["k:1"] == "resp:ok"
    assert "resp:ok" in results
