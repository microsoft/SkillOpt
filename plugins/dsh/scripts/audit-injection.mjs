// Injection audit: malicious payloads through quoteArgv -> the HOST shell ->
// verify none escape. On win32 the host shell is PowerShell (5.1 or pwsh 7);
// on POSIX it is bash. Each payload must arrive as exactly one argument, and
// no command may run.
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
const m = await import('../src/index.js')
const { quoteArgv, IS_WINDOWS } = m

const payloads = [
  'x; touch /tmp/pwned',
  'x$(touch /tmp/pwned2)',
  'x`touch /tmp/pwned3`',
  'x|cat /etc/passwd',
  'x&&rm -rf /',
  "' OR 1=1 --",
  'x > /tmp/redirected',
]

// Render `quoted` as one argument per line and return the array. The command
// is passed to the host shell as ONE string; each argv element arrives quoted,
// so the script only needs to echo every received argument back verbatim.
function expandArgs(quoted) {
  if (IS_WINDOWS) {
    // Windows PowerShell (5.1 or pwsh 7): single-quoted args parse the same on
    // both. Write-Output pipes each argument through ForEach-Object as one
    // pipeline object, so the round-trip is the real parse.
    const candidates = [
      process.env.PWSH_PATH,
      process.env.ProgramFiles ? join(process.env.ProgramFiles, 'PowerShell', '7', 'pwsh.exe') : '',
      join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'),
    ].filter(Boolean)
    const shell = candidates.find((p) => existsSync(p))
    if (!shell) throw new Error('no PowerShell executable found for injection audit')
    // quoteArgv now prepends "& " (the PS call operator) on win32. Strip it
    // before echoing the argument array; PowerShell would reject `&` in the
    // middle of a pipeline expression.
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
  const pref = args[args.indexOf('--preferences') + 1]
  const ok = pref === p
  if (!ok) { fail++; console.log('FAIL:', JSON.stringify(p), '->', JSON.stringify(pref)) }
}
for (const f of ['/tmp/pwned', '/tmp/pwned2', '/tmp/pwned3', '/tmp/redirected', '/tmp/pwnedx']) {
  if (existsSync(f)) { fail++; console.log('FILE CREATED:', f) }
}
console.log(fail === 0 ? 'ALL 7 INJECTION PAYLOADS INERT' : `${fail} FAILURES`)
process.exit(fail === 0 ? 0 : 1)
