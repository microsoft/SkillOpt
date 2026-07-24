"""Execute LLM-generated Python code against an input xlsx to produce an output xlsx."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap


RUNNER_TEMPLATE = textwrap.dedent(
    """
    import os, sys, traceback
    INPUT_PATH = {input_path!r}
    OUTPUT_PATH = {output_path!r}
    try:
    {user_code_indented}
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    """
)

# Regex to strip user-defined INPUT_PATH / OUTPUT_PATH assignments,
# since the runner template injects the correct values.
_PATH_ASSIGN_RE = re.compile(
    r'^\s*(INPUT_PATH|OUTPUT_PATH)\s*=\s*.+$', re.MULTILINE
)


def _strip_path_assignments(code: str) -> str:
    """Remove INPUT_PATH/OUTPUT_PATH assignments from user code."""
    return _PATH_ASSIGN_RE.sub("", code)


def run_generated_code(code: str, input_path: str, output_path: str, timeout: int | None = 120) -> tuple[bool, str]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned = _strip_path_assignments(code)
    indented = textwrap.indent(cleaned, "    ")
    script = RUNNER_TEMPLATE.format(
        input_path=input_path,
        output_path=output_path,
        user_code_indented=indented,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name
    # Build a minimal environment so the generated code cannot read API keys,
    # cloud credentials, or other secrets from the current process environment.
    import platform as _platform
    _safe_env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.path.dirname(output_path),
        "TMPDIR": tempfile.gettempdir(),
    }
    if _platform.system() == "Windows":
        _safe_env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        _safe_env["TEMP"] = tempfile.gettempdir()
        _safe_env["TMP"] = tempfile.gettempdir()
    # Drop empty entries (env dict values must be non-empty strings)
    _safe_env = {k: v for k, v in _safe_env.items() if v}
    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout if timeout and timeout > 0 else None,
            env=_safe_env,
        )
        if proc.returncode != 0:
            return False, (proc.stdout + "\n" + proc.stderr).strip()
        if not os.path.exists(output_path):
            return False, "output file was not created"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
