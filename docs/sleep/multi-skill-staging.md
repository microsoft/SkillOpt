# Multi-skill staging and reviewed adoption

Multi-skill nights separate learning from promotion. The cycle can consolidate
several hinted skills from their own live documents, but it never treats that
fan-out as permission to update every live file.

There are two independent proposal modes, and one night can contain both:

1. **Managed (legacy) proposal** — the aggregate cycle may stage
   `proposed_SKILL.md` and `proposed_CLAUDE.md` for the configured managed skill
   and project memory.
2. **Per-skill fan-out** — accepted hinted groups may stage one
   `proposed_SKILL.<name>.md` each. A reviewer chooses an explicit subset.

`auto_adopt` applies only an accepted managed proposal. It never promotes the
per-skill fan-out. Pending per-skill names remain visible after a managed
auto-adoption.

This feature landed after PyPI 0.2.0. Install from `main` until the next release.

## How the nightly fan-out works

Set the canonical `multi_skill_fanout` option to `true`. The earlier
`multi_skill_report` name remains a compatibility alias. When mined evidence
contains explicit skill hints, the cycle:

1. groups tasks by normalized hint;
2. resolves each name inside existing project-native `.agents/skills`,
   `.claude/skills`, `.cursor/skills`, and `.devin/skills` directories, the
   established Claude roots, and any repeatable `--skill-root PATH` overrides;
3. reads that skill's exact live `SKILL.md` bytes and canonical path;
4. runs the configured dream/consolidation pipeline independently for the
   group; and
5. stages a row only when that group's own gate accepted a non-empty update.

Missing, ambiguous, unreadable, aliased, unsafe, or colliding skills are skipped
with a note in both report formats. They never fall back to the managed skill's
document. The managed catch-all remains on `proposed_SKILL.md` and is not
duplicated as a per-skill row.

Each group inherits `recall_k`, `dream_rollouts`, `dream_factor`,
`dream_adversarial`, `dream_adversarial_blocking`,
`dream_adversarial_margin`, `dream_adversarial_rollouts`, `edit_budget`,
`gate_mode`, `gate_metric`, `gate_mixed_weight`, `gate_no_regression`, and
`evolve_skill`. Recalled archive tasks are restricted to the same skill hint,
and shared memory is read-only during group runs. Consequently,
`evolve_skill=false` disables managed and fan-out skill proposals.

The aggregate consolidation still runs once. Each usable hinted group adds one
independent dream/consolidation run, so provider calls and token use scale with
the number of groups, tasks, and configured rollouts.

## Staging layout

A mixed night can contain managed and per-skill artifacts together:

```text
.skillopt-sleep/staging/20260815-013000/
├── manifest.json
├── proposed_SKILL.md
├── proposed_CLAUDE.md
├── proposed_SKILL.alpha.md
├── proposed_SKILL.beta.md
├── report.json
├── report.md
└── evidence.jsonl
```

`manifest.json` is a versioned, fail-closed format. It retains the old top-level
field names only as safe compatibility sentinels and adds authoritative pinned
proposal rows:

```json
{
  "schema": "skillopt-sleep-staging",
  "schema_version": 2,
  "live_skill_path": "/repo/.agents/skills/managed/SKILL.md",
  "live_memory_path": "/repo/CLAUDE.md",
  "has_skill": false,
  "has_memory": false,
  "has_managed_skill": true,
  "has_managed_memory": true,
  "accepted": true,
  "legacy": {
    "skill": {
      "proposed_file": "proposed_SKILL.md",
      "live_path": "/repo/.agents/skills/managed/SKILL.md",
      "sha256": "<proposed raw-byte sha256>",
      "live_sha256": "<baseline raw-byte sha256>",
      "live_realpath": "/repo/.agents/skills/managed/SKILL.md"
    },
    "memory": {
      "proposed_file": "proposed_CLAUDE.md",
      "live_path": "/repo/CLAUDE.md",
      "sha256": "<proposed raw-byte sha256>",
      "live_sha256": "<baseline raw-byte sha256>",
      "live_realpath": "/repo/CLAUDE.md"
    }
  },
  "skills": [
    {
      "skill_name": "alpha",
      "proposed_file": "proposed_SKILL.alpha.md",
      "live_skill_path": "/home/dev/.claude/skills/alpha/SKILL.md",
      "sha256": "<proposed raw-byte sha256>",
      "live_sha256": "<baseline raw-byte sha256>",
      "live_realpath": "/home/dev/.claude/skills/alpha/SKILL.md"
    }
  ]
}
```

The top-level `has_skill` and `has_memory` compatibility fields are deliberately
always `false`. This makes the pre-feature PyPI 0.2.0 adopter treat a new night
as a no-op instead of bypassing the new validation and transaction engine.
`has_managed_skill` and `has_managed_memory` describe managed proposal presence;
the pinned `legacy` rows are authoritative for adoption. Top-level `accepted`
describes only the aggregate managed gate and does not summarize `skills`. An
aggregate gate may reject while an independently accepted group remains
reviewable in `skills`.

`sha256` pins the exact staged proposal bytes. `live_sha256` pins the raw live
bytes used as the consolidation baseline; an empty string means the file did
not exist. `live_realpath` pins the canonical destination identity. Staging
refuses publication if either live bytes or canonical identity changed after
the baseline read.

The writer reserves each staging directory atomically, writes a complete
artifact batch, and publishes its basename through a private mode-`0600`
`.latest` pointer. Invalid or symlinked pointers fall back only to contained,
reserved nights; adoption cannot reorder nights by changing a directory mtime.
Symlinked staging directories and manifests are ignored or refused.

## Reviewing and adopting

```text
python -m skillopt_sleep status --project PATH
python -m skillopt_sleep adopt --project PATH --staging NIGHT --skill alpha
python -m skillopt_sleep adopt --project PATH --staging NIGHT --skill alpha --skill beta
python -m skillopt_sleep adopt --project PATH --staging NIGHT --all-skills
python -m skillopt_sleep adopt --project PATH --staging NIGHT --legacy
```

Selection modes are mutually exclusive:

- `--skill NAME` is repeatable and promotes only those per-skill rows;
- `--all-skills` promotes every pending per-skill row;
- `--legacy` promotes only the co-staged managed skill/memory pair; and
- bare `adopt` remains convenient for a legacy-only night, but refuses a night
  with per-skill rows so it cannot imply “adopt everything.”

Use `--skill=--leading-dash` for a name beginning with `-`. Quote names with
spaces or shell metacharacters according to the active shell. Human guidance
lists names as data and never interpolates them into a copy/paste command.

The Python API uses the same transaction engine:

```python
from skillopt_sleep.staging import (
    adopt_skills,
    latest_staging,
    pending_staged_skills,
)

night = latest_staging("/path/to/project")
names = [row["skill_name"] for row in pending_staged_skills(night)]
receipts = adopt_skills(night, ["alpha"])
```

`skill_names=None` adopts all per-skill rows. An empty sequence adopts nothing.

## Integrity and recovery contract

Before any live mutation, adoption validates the entire relevant manifest and
selected proposal set, including:

- safe single-segment names and expected proposal filenames;
- unique names, staged files, case-folded paths, canonical paths, and live file
  identities, including hard-link aliases;
- regular, non-symlink proposal, manifest, receipt, backup, and journal files;
- proposal SHA-256 pins and valid UTF-8;
- live raw-byte hashes, file existence, canonical targets, file identities, and
  modes; and
- any prior receipt row against its derived immutable backup and hashes.

Old fan-out or managed manifests without live baseline pins are intentionally
refused. Discard and rerun the night; adoption does not guess a baseline for an
old proposal.

Adoption takes an exclusive staging lock plus stable per-target locks shared
across separate nights. A stale lock fails closed instead of being guessed away.
The locks cover manifest reload, full preflight, backup creation, final live
revalidation, all live replacements, and receipt publication.

Before the first mutation, the engine fsyncs a private mode-`0600`
`.adopt-transaction.json` version-2 write-ahead journal containing the recovery
state, including the identities of directories created by this transaction.
Backups are created without replacement:

```text
backup/skills/<name>/SKILL.md   # per-skill original
backup/SKILL.md                 # managed skill original
backup/CLAUDE.md                # managed memory original
```

Per-skill receipts accumulate in `adopted_skills.json`; managed receipts live in
`adopted_legacy.json`. A skill or managed target cannot be re-adopted from the
same night, and existing receipt/backup history must validate before another
subset can be appended.

The engine revalidates the complete target set before and after receipt
publication. The journal is removed only after every selected target and the
receipt are durably published; that removal is the commit point. A caught
failure triggers immediate rollback. If the process stops first, the next
adoption recovers the journal before it trusts the manifest. Recovery restores
only content still equal to this transaction's proposal and removes only empty
created directories whose identities still match the journal. If an external
editor changed content or replaced a directory, recovery preserves it, retains
the journal/backups, and raises `StagingRecoveryError` for manual resolution.

On POSIX, file and parent-directory changes are fsynced. Python's standard
library does not expose an equivalent portable directory flush on Windows, so
the journal and file contents are flushed there but power-loss durability of
directory entries remains filesystem/OS dependent.

The final byte/identity/mode check occurs immediately before atomic replacement,
and all SkillOpt adoption processes share target locks. Portable Python does not
provide a filesystem compare-and-swap against an unrelated process that ignores
those locks; such a process can still race in the final check/replace micro-gap.
Keep live skill editing and adoption coordinated when stronger OS-specific
locking is required.

## Machine interfaces and integrations

`run --json` includes additive `skill_groups` and `staged_skills` fields.
`status --json` always includes `staged_skills` (an empty array on a malformed
manifest) and adds `staging_error` when inspection failed. Adoption success and
failure are single JSON documents; selection-required failures return
`available_skills` as objects with `skill_name` and `live_skill_path`.

The Copilot and Devin MCP `sleep_adopt` tools expose `staging`, `skills`,
`all_skills`, and `legacy`. They forward names as subprocess argument-vector
elements, never shell text. The adapters validate actual JSON-RPC argument types
before launching a subprocess, preserve nonzero engine status, and do not copy
adopted content to a second unpinned destination. Native project skill roots are
adopted directly through the core transaction.

## Migration checklist

- Require `schema="skillopt-sleep-staging"` and `schema_version=2`; unknown or
  missing new-format versions fail closed.
- Treat `legacy` and `skills` as the authoritative managed and fan-out rows.
- Expect legacy `has_skill` / `has_memory` compatibility sentinels to be false;
  use `has_managed_skill` / `has_managed_memory` for managed presence.
- Interpret top-level `accepted` as aggregate-only.
- Require all proposal and live pins; restage older unpinned nights.
- Preserve unknown additive JSON fields when building external tooling.
- Use `--staging` when automating promotion so “latest” cannot change between
  review and adoption.
- Expect append-only receipts and fail-closed locks/recovery conflicts.
- Do not infer that a successful managed auto-adoption also promoted fan-out
  rows.
