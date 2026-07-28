# Multi-skill staging and subset adoption

A night can stage a proposal for more than one skill. Adoption stays explicit:
staging only ever writes into the staging directory, and `adopt_skills()` copies
a **reviewed subset** over the live files, with a backup and a hash receipt per
skill.

Nothing here changes a single-managed-skill night. If a night stages no per-skill
proposals, the staging directory and `manifest.json` are exactly the legacy ones
and `skillopt-sleep adopt` keeps working unchanged.

## Staging layout

Legacy (single managed skill) — unchanged:

```text
.skillopt-sleep/staging/20260728-013000/
├── manifest.json          # live_skill_path, live_memory_path, has_skill, has_memory, accepted
├── proposed_SKILL.md
├── proposed_CLAUDE.md
├── report.json
└── report.md
```

Multi-skill night — one extra file and one manifest row per skill:

```text
.skillopt-sleep/staging/20260728-013000/
├── manifest.json          # …the legacy keys plus "skills": [ … ]
├── proposed_SKILL.alpha.md
├── proposed_SKILL.beta.md
├── report.json            # report.skill_groups carries each skill's gate evidence
└── report.md
```

```json
{
  "live_skill_path": "/home/dev/.claude/skills/alpha/SKILL.md",
  "has_skill": false,
  "accepted": true,
  "skills": [
    {
      "skill_name": "alpha",
      "proposed_file": "proposed_SKILL.alpha.md",
      "live_skill_path": "/home/dev/.claude/skills/alpha/SKILL.md"
    },
    {
      "skill_name": "beta",
      "proposed_file": "proposed_SKILL.beta.md",
      "live_skill_path": "/home/dev/.claude/skills/beta/SKILL.md"
    }
  ]
}
```

A skill name must be a single safe path segment and a live path must be an
absolute, traversal-free `*.md` file; two skills may not share a name or a target
file. A refused fan-out writes no `manifest.json`, so the folder is not adoptable.

## Adopting a reviewed subset

```python
from skillopt_sleep.staging import adopt_skills, latest_staging, staged_skills

staging = latest_staging("/path/to/project")
[row["skill_name"] for row in staged_skills(staging)]   # ['alpha', 'beta']

receipts = adopt_skills(staging, ["alpha"])             # beta is left alone
receipts[0].sha256_before, receipts[0].sha256_after
```

- `skill_names=None` adopts every staged skill; `[]` adopts nothing.
- An unknown or repeated name, an unsafe manifest row, or a missing proposal file
  raises `StagingError` **before** anything is written.
- Each live file is backed up to `backup/skills/<skill>/` and written atomically.
- If any write fails, every file in the selection is restored (and files that did
  not exist before are removed), so a partial adoption never survives.
- Receipts (`skill_name`, `live_skill_path`, `sha256_before`, `sha256_after`,
  `backup_path`) are returned and written to `adopted_skills.json` in the staging
  directory. An empty `sha256_before` means the skill had no live file yet.

## Migrating

- **Consumers of `manifest.json`**: treat `"skills"` as optional; when absent the
  night is a legacy single-proposal one.
- **Consumers of `report.json`**: `skill_groups` is `[]` on a single-skill night,
  and the flat `accepted` / `gate_action` / score fields keep their meaning.
- **Adoption tooling**: `adopt()` still adopts the legacy single proposal pair.
  Use `adopt_skills()` for per-skill nights; the two are independent, and neither
  runs implicitly.
