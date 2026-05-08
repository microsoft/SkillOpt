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


def run_generated_code(code: str, input_path: str, output_path: str, timeout: int = 120) -> tuple[bool, str]:
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
    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
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
