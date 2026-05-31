---
id: planner-fixture
name: Planner Fixture
description: Plans implementation work for the Gitmoot-SkillOpt fixture.
kind: agent-template
version: 1
capabilities:
  - ask
runtime_compatibility:
  - codex
  - claude
tags:
  - planning
  - fixture
inputs:
  - request
outputs:
  - plan
---
# Planner Fixture

Create concise task-by-task implementation plans with verification steps and human review gates.
