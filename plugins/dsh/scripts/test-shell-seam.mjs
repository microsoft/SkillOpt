#!/usr/bin/env node
// Host-shell seam, both dsh generations.
//
// dsh 0.1.x exposes ShellExecutor.run(); 0.2.x exposes execute(), whose
// ShellExecution carries the outcome on result(). plugins/dsh declares
// @deepseek-ai/dsh-tools for both ranges, so the call must be feature-detected:
// a hard switch either way throws "<name> is not a function" on the other
// generation, and dsh 0.2.x users hit it on every skillopt_* tool call.
//
// Usage: node scripts/test-shell-seam.mjs
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const { runShell } = await import(pathToFileURL(join(root, 'src/index.js')).href)

let failures = 0
function check(name, cond, detail = '') {
  if (cond) console.log(`  ✅ ${name}`)
  else {
    failures++
    console.log(`  ❌ ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

const spec = { command: "'python' 'scripts/sleep.py' 'status'", timeoutMs: 1000 }

console.log('1. dsh 0.2.x: execute() returns a handle, outcome from result()')
{
  const seen = []
  const shell = {
    execute: async (s) => {
      seen.push(['execute', s])
      return { result: async () => ({ exitCode: 0, stdout: 'ok' }) }
    },
  }
  const r = await runShell(shell, spec)
  check('execute() used', seen.length === 1 && seen[0][0] === 'execute', JSON.stringify(seen))
  check('spec passed through unchanged', seen[0]?.[1] === spec)
  check('handle.result() outcome returned', r?.exitCode === 0 && r?.stdout === 'ok', JSON.stringify(r))
}

console.log('2. dsh 0.1.x: run() returns the outcome directly')
{
  const seen = []
  const shell = {
    run: async (s) => {
      seen.push(['run', s])
      return { exitCode: 7, stderr: 'boom' }
    },
  }
  const r = await runShell(shell, spec)
  check('run() used', seen.length === 1 && seen[0][0] === 'run', JSON.stringify(seen))
  check('outcome returned as-is', r?.exitCode === 7 && r?.stderr === 'boom', JSON.stringify(r))
}

console.log('3. transitional executor exposing both: execute() wins')
{
  const calls = []
  const shell = {
    execute: async () => {
      calls.push('execute')
      return { result: async () => ({ exitCode: 1 }) }
    },
    run: async () => {
      calls.push('run')
      return { exitCode: 2 }
    },
  }
  const r = await runShell(shell, spec)
  check('only execute() called', calls.join() === 'execute', calls.join())
  check('execute() outcome wins', r?.exitCode === 1, JSON.stringify(r))
}

console.log(failures === 0 ? '\nALL SHELL SEAM CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`)
process.exit(failures === 0 ? 0 : 1)
