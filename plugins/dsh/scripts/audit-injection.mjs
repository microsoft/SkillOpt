// 独立注入审计：各种恶意 payload 过 quoteArgv → 真实 bash → 验证不逃逸
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
const m = await import('../src/index.js')
const { quoteArgv } = m
const BASH = 'C:/Program Files/Git/bin/bash.exe'
const payloads = [
  'x; touch /tmp/pwned',
  'x$(touch /tmp/pwned2)',
  'x`touch /tmp/pwned3`',
  'x|cat /etc/passwd',
  'x&&rm -rf /',
  "' OR 1=1 --",
  'x > /tmp/redirected',
]
let fail = 0
for (const p of payloads) {
  const argv = ['python', '-m', 'skillopt_sleep', 'run', '--preferences', p]
  const quoted = quoteArgv(argv)
  const script = `for a in ${quoted}; do printf '[%s]\\n' "$a"; done`
  const out = execFileSync(BASH, ['-c', script], { encoding: 'utf8' })
  const args = out.trim().split('\n').map((l) => l.slice(1, -1))
  const pref = args[args.indexOf('--preferences') + 1]
  const ok = pref === p
  if (!ok) { fail++; console.log('FAIL:', JSON.stringify(p), '->', JSON.stringify(pref)) }
}
for (const f of ['/tmp/pwned', '/tmp/pwned2', '/tmp/pwned3', '/tmp/redirected', '/tmp/pwnedx']) {
  if (existsSync(f)) { fail++; console.log('FILE CREATED:', f) }
}
console.log(fail === 0 ? 'ALL 7 INJECTION PAYLOADS INERT' : `${fail} FAILURES`)
process.exit(fail === 0 ? 0 : 1)
