# SkillOpt-Sleep

Use the bundled `skillopt-sleep` skill to run or manage SkillOpt-Sleep for the
current Cursor workspace.

Requested action: `$ARGUMENTS`

If no action was supplied, use `status`. Preserve all options supplied after
the action. For `harvest`, `dry-run`, and `run`, ensure the engine receives:

```text
--project <current workspace> --scope invoked --source cursor --target-skill-path .cursor/skills/skillopt-sleep-learned/SKILL.md
```

Do not add `--backend cursor` unless the user requested provider-backed replay;
the repository default remains the no-provider `mock` backend. Follow the
skill's runner selection, review, data-boundary, scheduling, and adoption rules.
Never edit the learned skill or `CLAUDE.md` directly as a substitute for the
engine's staged adoption flow.
