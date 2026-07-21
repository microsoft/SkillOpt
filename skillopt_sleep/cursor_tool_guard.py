"""Fail-closed full-command guard for synthetic Cursor replay tools."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Dict, Iterable, TextIO, Tuple


def command_verdict(payload: Any, allowed_commands: Iterable[str]) -> Tuple[bool, str]:
    """Return whether a hook payload contains one exact allowed command."""
    command = payload.get("command") if isinstance(payload, dict) else None
    if not isinstance(command, str):
        return False, ""
    normalized = command.strip()
    return normalized in set(allowed_commands), normalized


def command_digest(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8", errors="replace")).hexdigest()


def _response(allowed: bool) -> Dict[str, Any]:
    if allowed:
        return {"continue": True, "permission": "allow"}
    return {
        "continue": True,
        "permission": "deny",
        "user_message": "Command blocked by the SkillOpt replay boundary.",
        "agent_message": "Use only the exact synthetic tool command provided for this replay.",
    }


def evaluate_hook(
    *,
    policy_path: str,
    log_path: str,
    input_stream: TextIO,
    output_stream: TextIO,
) -> int:
    """Evaluate one Cursor hook request without ever emitting command text."""
    allowed = False
    command = ""
    try:
        with open(policy_path, encoding="utf-8") as f:
            policy = json.load(f)
        allowed_commands = policy.get("allowed_commands") if isinstance(policy, dict) else None
        if not isinstance(allowed_commands, list) or not all(
            isinstance(item, str) for item in allowed_commands
        ):
            raise ValueError("invalid command policy")
        payload = json.load(input_stream)
        allowed, command = command_verdict(payload, allowed_commands)
        event = {
            "allowed": allowed,
            "command_sha256": command_digest(command),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:
        # A valid deny response protects older Cursor versions too; the
        # failClosed hook setting remains the fallback if this process fails.
        allowed = False
    json.dump(_response(allowed), output_stream, sort_keys=True)
    output_stream.write("\n")
    output_stream.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args(argv)
    return evaluate_hook(
        policy_path=args.policy,
        log_path=args.log,
        input_stream=sys.stdin,
        output_stream=sys.stdout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
