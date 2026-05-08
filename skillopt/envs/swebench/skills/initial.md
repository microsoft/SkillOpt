# SWE-bench Bug Fixing Skill

## Overview
This skill guides agents in resolving real-world GitHub issues by producing correct patches.

**Goal**: Given a repository and an issue description, produce a minimal, correct `git diff` patch that resolves the issue without modifying test files.

## Workflow

1. Understand the issue. Read the problem statement carefully and restate the expected behavior before editing code.
2. Locate relevant code. Use targeted search to identify the files, functions, and tests that encode the buggy behavior.
3. Reproduce the issue. Build a small, local reproduction before changing source files when feasible.
4. Implement the fix. Make the smallest source change that addresses the root cause.
5. Verify the fix. Re-run the reproduction and any focused checks needed to confirm the change.
6. Submit the patch. Generate a clean unified diff of only the source files you modified.

## Key Rules

- Keep changes minimal and directly tied to the bug.
- Do not modify tests, fixtures, or unrelated configuration unless the issue explicitly requires it.
- Prefer understanding the code path before patching.
- Verify behavior after editing instead of relying on intuition.
- The final submission must be a valid unified diff.
