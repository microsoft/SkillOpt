// Security audit: control characters/newlines cannot escape the quoting
// boundary of the HOST shell, and produce no side effects. Expected after the
// fix: control chars are folded to spaces, the value still arrives as a single
// argument, no file or command runs.
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
const m = await import('../src/index.js')
const { quoteArgv, IS_WINDOWS } = m

const payloads = [
  'x\n touch /tmp/nl-pwned',
  'x\r echo PWNED',
  'x\ttab',
  'x`id`',
  'x\x00null',
  "'; touch /tmp/semi-pwned;'",
  'normal\r\ntouch /tmp/crnl-pwned',
]

function expandArgs(quoted) {
  if (IS_WINDOWS) {
    const candidates = [
      process.env.PWSH_PATH,
      process.env.ProgramFiles ? join(process.env.ProgramFiles, 'PowerShell', '7', 'pwsh.exe') : '',
      join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'),
    ].filter(Boolean)
    const shell = candidates.find((p) => existsSync(p))
    if (!shell) throw new Error('no PowerShell executable found for control-char audit')
    const args = quoted.replace(/^& /, '')
    const script = `Write-Output ${args} | ForEach-Object { "[$_]" }`
    const out = execFileSync(shell, ['-NoProfile', '-Command', script], { encoding: 'utf8' })
    // PowerShell emits CRLF; strip \r before slicing the [..] markers.
    return out.trim().split('\n').map((l) => l.replace(/\r$/, '').slice(1, -1))
  }
  const script = `for a in ${quoted}; do printf '[%s]\\n' "$a"; done`
  const out = execFileSync(process.env.BASH_PATH || '/bin/bash', ['-c', script], { encoding: 'utf8' })
  return out.trim().split('\n').map((l) => l.slice(1, -1))
}

let fail = 0
for (const p of payloads) {
  const argv = ['python', '-m', 'skillopt_sleep', 'run', '--preferences', p]
  const quoted = quoteArgv(argv)
  const args = expandArgs(quoted)
  // Safety assertion 1: exactly one argument after --preferences (no split).
  const idx = args.indexOf('--preferences')
  const singleArg = idx >= 0 && idx + 1 < args.length && idx + 2 === args.length
  // Safety assertion 2: the arrived value has no raw control chars (folded to space).
  const arrived = idx >= 0 ? args[idx + 1] : ''
  const stripped = !/[\r\n\t\x00]/.test(arrived)
  // Safety assertion 3: no injected command outside the argument list.
  const noInjection = !args.some((a) => /touch|PWNED|rm\s/.test(a) && a !== arrived)
  if (!singleArg || !stripped || !noInjection) {
    fail++
    console.log('FAIL:', JSON.stringify(p), '-> args:', JSON.stringify(args))
  }
}
for (const f of ['/tmp/nl-pwned', '/tmp/semi-pwned', '/tmp/crnl-pwned']) {
  if (existsSync(f)) { fail++; console.log('FILE CREATED:', f) }
}
console.log(fail === 0 ? 'ALL CONTROL-CHAR PAYLOADS NEUTRALIZED' : `${fail} FAILURES`)
process.exit(fail === 0 ? 0 : 1)
