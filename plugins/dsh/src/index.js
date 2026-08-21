// dsh-skillopt — Microsoft SkillOpt-Sleep integration for DeepSeek Harness.
//
// Gives the dsh agent a "sleep cycle": harvest past sessions -> mine recurring
// tasks -> replay via a backend -> consolidate validated skills behind a
// held-out gate. The heavy lifting is done by the upstream `skillopt_sleep`
// Python engine (https://github.com/microsoft/SkillOpt); this plugin exposes
// it to the agent as native dsh tools, plus a skill and configuration.

import Schema from '@deepseek-ai/schemastery'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

export const name = 'skillopt'

// Plugin directory (absolute) — used to resolve the bundled engine script so
// it works regardless of the dsh process cwd.
const PLUGIN_DIR = dirname(fileURLToPath(import.meta.url)) + '/..'

// Exported for the canary test (scripts/canary.mjs).
export { buildArgv, quoteArgv }

// Wait for the tool registry and the shell executor before applying.
export const inject = ['tools', 'shell']

// ---------------------------------------------------------------------------
// Config (Schemastery)
// ---------------------------------------------------------------------------

export const Config = Schema.object({
  pythonCmd: Schema.string()
    .default('python')
    .description('Python interpreter used to run the engine bootstrap (scripts/sleep.py)'),
  module: Schema.string()
    .description('Override: run `python -m <module>` directly instead of the bootstrap script'),
  engineScript: Schema.string()
    .description('Override: path to the engine bootstrap script (default: scripts/sleep.py)'),
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
  autoAdopt: Schema.boolean()
    .default(false)
    .description('OPERATOR-ONLY: auto-adopt a passed proposal without asking. The model cannot toggle this; set it in cordis.yml.'),
  timeoutMs: Schema.number()
    .default(600_000)
    .description('Per-call engine timeout in milliseconds (default 10 min)'),
})

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Quote one argv element for a POSIX shell (bash). Single quotes are literal;
// an embedded single quote is expressed as '\'' (close quote, escaped quote,
// reopen quote) — the only portable POSIX spelling. PowerShell is not a target
// here: dsh's ctx.shell executes via `bash -c` (LocalBashExecutor), so the
// quoting only needs to be bash-correct.
//
// Control characters are stripped as defense in depth: \r and \r\n inside a
// single-quoted word would otherwise split the value into multiple argv words
// (broken command, not RCE — quotes never execute), and \n would corrupt the
// engine's own arg parsing. Model-controlled values must arrive as exactly
// one argument.
function q(value) {
  const s = String(value).replace(/[\r\n\u0000-\u001f\u007f]/g, ' ')
  return `'${s.replace(/'/g, "'\\''")}'`
}

/**
 * Build the argv array for the engine with config defaults and per-call
 * overrides. Returns an ARRAY (not a joined string); execute() quotes each
 * element and lets shell.resolve() apply workdir/output-cap/sandbox defaults.
 *
 * The engine is invoked through scripts/sleep.py, which mirrors the official
 * SkillOpt runner: it resolves a source checkout (repo root), picks a
 * Python >= 3.10, and falls back to the `skillopt-sleep` CLI or an installed
 * package. `config.module` still works as a direct `python -m <module>` escape
 * hatch for users who prefer it.
 */
function buildArgv(config, action, explicit = {}, extras = [], allowed = null) {
  const parts = [config.pythonCmd || 'python']
  if (config.module) {
    // explicit escape hatch: python -m <module>
    parts.push('-m', config.module)
  } else {
    // default: the bundled bootstrap mirrors the official runner; resolve it
    // absolutely so it works no matter what cwd dsh was started from.
    parts.push(config.engineScript || join(PLUGIN_DIR, 'scripts', 'sleep.py'))
  }
  parts.push(action)
  const push = (flag, value) => {
    if (value !== undefined && value !== null && value !== '') parts.push(flag, String(value))
  }
  const has = (v) => v !== undefined && v !== null && v !== ''
  // `allowed` is the tool's declared parameter set (null = everything, the
  // pre-whitelist behavior). Both the model-supplied value AND the operator
  // config default are gated on it, so a tool like skillopt_adopt (declares
  // only `project`) never receives --backend/--model/--json/… from either
  // source — the config default must not leak into tools that do not declare
  // the key.
  const permits = (key) => !allowed || allowed.includes(key)
  const withDefault = (key, flag) => {
    if (!permits(key)) return
    if (has(explicit[key])) push(flag, explicit[key])
    else push(flag, config[key])
  }
  withDefault('project', '--project')
  withDefault('scope', '--scope')
  withDefault('source', '--source')
  withDefault('backend', '--backend')
  withDefault('model', '--model')
  withDefault('maxTasks', '--max-tasks')
  withDefault('maxSessions', '--max-sessions')
  withDefault('editBudget', '--edit-budget')
  withDefault('preferences', '--preferences')
  if (permits('json') && (config.jsonOutput || explicit.json)) parts.push('--json')
  parts.push(...extras)
  return parts
}

/** Join argv with safe quoting for the platform shell. */
function quoteArgv(argv) {
  return argv.map(q).join(' ')
}

// Pick exactly the parameters a tool declares. dsh's parameter schema does
// not reject undeclared properties by default (no additionalProperties:false),
// so without this filter the model could inject fields (backend, model, json,
// editBudget, …) that buildArgv would forward to the engine — crossing the
// per-tool surface and, for skillopt_adopt, the live-change boundary. Each
// tool's build() must pass through exactly its declared keys.
function pick(obj, keys) {
  const out = {}
  for (const key of keys) {
    if (obj[key] !== undefined) out[key] = obj[key]
  }
  return out
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
      build: (a) => buildArgv(config, 'status', pick(a, ['project', 'json']), [], ['project', 'json']),
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
      build: (a) => buildArgv(config, 'dry-run', pick(a, ['project', 'source', 'backend', 'model', 'maxTasks']), a.progress ? ['--progress'] : [], ['project', 'source', 'backend', 'model', 'maxTasks']),
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
        progress: { type: 'boolean', description: 'Print phase progress to stderr' },
      },
      build: (a) => {
        const extra = []
        // auto-adopt is OPERATOR-ONLY (config.autoAdopt); the model cannot set it.
        if (config.autoAdopt) extra.push('--auto-adopt')
        if (a.progress) extra.push('--progress')
        return buildArgv(config, 'run', pick(a, ['project', 'backend', 'source', 'preferences']), extra, ['project', 'backend', 'source', 'preferences'])
      },
    },
    {
      name: 'skillopt_adopt',
      description:
        'Apply the latest staged proposal, backing up existing target files first. This is the live-change boundary.',
      parameters: {
        project: { type: 'string', description: 'Project directory' },
      },
      build: (a) => buildArgv(config, 'adopt', pick(a, ['project']), [], ['project']),
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
        return buildArgv(config, 'harvest', pick(a, ['project', 'source', 'maxTasks']), extra, ['project', 'source', 'maxTasks'])
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
        return buildArgv(config, 'schedule', pick(a, ['project', 'backend']), extra, ['project', 'backend'])
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
      build: (a) => buildArgv(config, 'unschedule', pick(a, ['project']), a.all ? ['--all'] : [], ['project']),
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
          const argv = t.build(args || {})
          // `command` must be the shell-quoted form for the platform executor;
          // resolve() applies the executor's workdir/output-cap/sandbox defaults.
          const request = {
            command: quoteArgv(argv),
            timeoutMs: config.timeoutMs,
            signal: exec?.signal,
          }
          const spec = typeof shell.resolve === 'function' ? shell.resolve(request) : request
          try {
            const result = await shell.run(spec)
            // Distinguish the executor's timeout (timedOut: true, exitCode null)
            // from an abort/kill (exitCode null, no timedOut) so the marker is
            // honest instead of lumping both under "signal".
            const status = result?.timedOut
              ? 'timeout'
              : (result?.exitCode ?? 'signal')
            // rc.8 returns stdout/stderr as CollectedOutput { text, truncated, spillPath }
            const fmt = (co) => {
              if (co === undefined || co === null) return ''
              if (typeof co === 'string') return co
              const parts = []
              if (co.text) parts.push(co.text)
              if (co.truncated) {
                parts.push(`[truncated${co.spillPath ? ` — full output at ${co.spillPath}` : ''}]`)
              }
              return parts.join('\n')
            }
            const stdout = fmt(result?.stdout)
            const stderr = fmt(result?.stderr)
            const tail = [stdout, stderr].filter(Boolean).join('\n').trim()
            // Do NOT slice here: fmt() already carries the executor's truncation
            // marker + spill path when output was capped. A second slice would
            // hide data the executor already bounded and contradict the marker.
            return [
              `[skillopt ${t.name}] exit=${status}`,
              tail ? tail : '(no output)',
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
