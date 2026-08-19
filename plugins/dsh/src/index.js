// dsh-skillopt — Microsoft SkillOpt-Sleep integration for DeepSeek Harness.
//
// Gives the dsh agent a "sleep cycle": harvest past sessions -> mine recurring
// tasks -> replay via a backend -> consolidate validated skills behind a
// held-out gate. The heavy lifting is done by the upstream `skillopt_sleep`
// Python engine (https://github.com/microsoft/SkillOpt); this plugin exposes
// it to the agent as native dsh tools, plus a skill and configuration.

import Schema from '@deepseek-ai/schemastery'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'skillopt'

// Wait for the tool registry and the shell executor before applying.
export const inject = ['tools', 'shell']

// ---------------------------------------------------------------------------
// Config (Schemastery)
// ---------------------------------------------------------------------------

export const Config = Schema.object({
  pythonCmd: Schema.string()
    .default('python')
    .description('Python interpreter used to run the skillopt_sleep engine'),
  module: Schema.string()
    .default('skillopt_sleep')
    .description('Python module that implements the skillopt-sleep CLI'),
  project: Schema.string()
    .description('Default project directory for sleep cycles'),
  scope: Schema.union(['all', 'invoked']).description('Harvest scope'),
  backend: Schema.union([
    'mock', 'claude', 'codex', 'copilot', 'cursor', 'pi', 'opencode',
    'handoff', 'azure_openai',
  ]).description('Default backend (mock = no provider calls)'),
  model: Schema.string().description('Default backend model override'),
  source: Schema.union([
    'claude', 'codex', 'copilot', 'cursor', 'pi', 'opencode', 'auto',
  ]).description('Default transcript source'),
  maxTasks: Schema.number().description('Cap mined tasks (default 40)'),
  maxSessions: Schema.number().description('Cap harvested sessions'),
  editBudget: Schema.number().description('Max bounded edits per cycle (default 4)'),
  preferences: Schema.string().description('House rules injected into the reflection prior'),
  jsonOutput: Schema.boolean().default(false).description('Emit machine-readable JSON where supported'),
  timeoutMs: Schema.number()
    .default(600_000)
    .description('Per-call engine timeout in milliseconds (default 10 min)'),
})

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildCommand(config, action, explicit = {}, extras = []) {
  const python = config.pythonCmd || 'python'
  const module = config.module || 'skillopt_sleep'
  const parts = [python, '-m', module, action]
  const push = (flag, value) => {
    if (value !== undefined && value !== null && value !== '') parts.push(flag, String(value))
  }
  if (explicit.project !== undefined) push('--project', explicit.project)
  else push('--project', config.project)
  if (explicit.scope !== undefined) push('--scope', explicit.scope)
  else push('--scope', config.scope)
  if (explicit.source !== undefined) push('--source', explicit.source)
  else push('--source', config.source)
  if (explicit.backend !== undefined) push('--backend', explicit.backend)
  else push('--backend', config.backend)
  if (explicit.model !== undefined) push('--model', explicit.model)
  else push('--model', config.model)
  if (explicit.maxTasks !== undefined) push('--max-tasks', explicit.maxTasks)
  else push('--max-tasks', config.maxTasks)
  if (explicit.maxSessions !== undefined) push('--max-sessions', explicit.maxSessions)
  else push('--max-sessions', config.maxSessions)
  if (explicit.editBudget !== undefined) push('--edit-budget', explicit.editBudget)
  else push('--edit-budget', config.editBudget)
  if (explicit.preferences !== undefined) push('--preferences', explicit.preferences)
  else push('--preferences', config.preferences)
  if (config.jsonOutput || explicit.json) parts.push('--json')
  parts.push(...extras)
  return parts.join(' ')
}

function renderOutput(_args, value) {
  return [{ type: 'text', text: value }]
}

// ---------------------------------------------------------------------------
// Plugin entry
// ---------------------------------------------------------------------------

export function apply(ctx, config = {}) {
  const shell = ctx.shell

  const tools = [
    {
      name: 'skillopt_status',
      description:
        'Show SkillOpt-Sleep state: engine availability, latest staged proposal, last run report.',
      parameters: {
        project: { type: 'string', description: 'Project directory (defaults to config.project or cwd)' },
        json: { type: 'boolean', description: 'Emit machine-readable JSON' },
      },
      build: (a) => buildCommand(config, 'status', a),
    },
    {
      name: 'skillopt_dry_run',
      description:
        'Preview a full sleep cycle without staging anything: harvest, mine, replay, report.',
      parameters: {
        project: { type: 'string', description: 'Project directory' },
        source: { type: 'string', description: 'Transcript source: claude|codex|copilot|cursor|pi|opencode|auto' },
        backend: { type: 'string', description: 'Backend: mock|claude|codex|copilot|cursor|pi|opencode|handoff|azure_openai' },
        model: { type: 'string', description: 'Backend model override' },
        maxTasks: { type: 'number', description: 'Cap mined tasks (default 40)' },
        progress: { type: 'boolean', description: 'Print phase progress to stderr' },
      },
      build: (a) => buildCommand(config, 'dry-run', a, a.progress ? ['--progress'] : []),
    },
    {
      name: 'skillopt_run',
      description:
        'Run the full sleep cycle and stage a proposal. Nothing live changes until skillopt_adopt.',
      parameters: {
        project: { type: 'string', description: 'Project directory' },
        backend: { type: 'string', description: 'Backend for model calls' },
        source: { type: 'string', description: 'Transcript source' },
        preferences: { type: 'string', description: 'House rules for the reflection prior' },
        autoAdopt: { type: 'boolean', description: 'Auto-adopt if the gate passes' },
        progress: { type: 'boolean', description: 'Print phase progress to stderr' },
      },
      build: (a) => {
        const extra = []
        if (a.autoAdopt) extra.push('--auto-adopt')
        if (a.progress) extra.push('--progress')
        return buildCommand(config, 'run', a, extra)
      },
    },
    {
      name: 'skillopt_adopt',
      description:
        'Apply the latest staged proposal, backing up existing target files first. This is the live-change boundary.',
      parameters: {
        project: { type: 'string', description: 'Project directory' },
      },
      build: (a) => buildCommand(config, 'adopt', a),
    },
    {
      name: 'skillopt_harvest',
      description:
        'Harvest past sessions and show or export mined recurring tasks. Read-only.',
      parameters: {
        project: { type: 'string', description: 'Project directory' },
        source: { type: 'string', description: 'Transcript source' },
        output: { type: 'string', description: 'Export tasks JSON to this file' },
        maxTasks: { type: 'number', description: 'Cap mined tasks' },
      },
      build: (a) => {
        const extra = []
        if (a.output) extra.push('--output', a.output)
        return buildCommand(config, 'harvest', a, extra)
      },
    },
    {
      name: 'skillopt_schedule',
      description:
        'Install a nightly cron entry that runs the sleep cycle for this project.',
      parameters: {
        project: { type: 'string', description: 'Project directory' },
        hour: { type: 'number', description: 'Hour (0-23, default 3)' },
        minute: { type: 'number', description: 'Minute (default 17)' },
        backend: { type: 'string', description: 'Backend for scheduled runs' },
      },
      build: (a) => {
        const extra = []
        if (a.hour !== undefined) extra.push('--hour', String(a.hour))
        if (a.minute !== undefined) extra.push('--minute', String(a.minute))
        return buildCommand(config, 'schedule', a, extra)
      },
    },
    {
      name: 'skillopt_unschedule',
      description:
        'Remove the nightly cron entry for this project.',
      parameters: {
        project: { type: 'string', description: 'Project directory' },
        all: { type: 'boolean', description: 'Remove every managed entry' },
      },
      build: (a) => buildCommand(config, 'unschedule', a, a.all ? ['--all'] : []),
    },
  ]

  for (const t of tools) {
    ctx.tools.register(
      defineTool({
        name: t.name,
        description: t.description,
        parameters: t.parameters,
        output: { schema: { type: 'string' }, render: renderOutput },
        async execute(args, exec) {
          const command = t.build(args || {})
          try {
            const result = await shell.run({
              command,
              timeoutMs: config.timeoutMs,
              signal: exec?.signal,
            })
            const status = result?.exitCode ?? 'signal'
            const stdout = typeof result?.stdout === 'string' ? result.stdout : ''
            const stderr = typeof result?.stderr === 'string' ? result.stderr : ''
            const tail = [stdout, stderr].filter(Boolean).join('\n').trim()
            return [
              `[skillopt ${t.name}] exit=${status}`,
              tail ? tail.slice(0, 60_000) : '(no output)',
            ].join('\n')
          } catch (err) {
            return `[skillopt ${t.name}] engine call failed: ${err?.message || String(err)}`
          }
        },
      }),
    )
  }

  ctx.logger?.info?.('[dsh-skillopt] registered 7 skillopt tools (status/dry-run/run/adopt/harvest/schedule/unschedule)')
}
