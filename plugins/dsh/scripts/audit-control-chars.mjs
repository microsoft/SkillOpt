// 安全审核：控制字符/换行不能逃逸单引号边界，且不产生副作用。
// 修复后的期望：控制字符被剥离为空格，值仍作为单个参数到达，无文件/命令执行。
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
const m = await import('../src/index.js')
const { quoteArgv } = m
const BASH = 'C:/Program Files/Git/bin/bash.exe'
const payloads = [
  'x\n touch /tmp/nl-pwned',
  'x\r echo PWNED',
  'x\ttab',
  'x`id`',
  'x\x00null',
  "'; touch /tmp/semi-pwned;'",
  'normal\r\ntouch /tmp/crnl-pwned',
]
let fail = 0
for (const p of payloads) {
  const argv = ['python', '-m', 'skillopt_sleep', 'run', '--preferences', p]
  const quoted = quoteArgv(argv)
  const script = `for a in ${quoted}; do printf '[%s]\\n' "$a"; done`
  const out = execFileSync(BASH, ['-c', script], { encoding: 'utf8' })
  const args = out.trim().split('\n').map((l) => l.slice(1, -1))
  // 安全断言 1：--preferences 后恰好一个参数（无参数分裂）
  const idx = args.indexOf('--preferences')
  const singleArg = idx >= 0 && idx + 1 < args.length && idx + 2 === args.length
  // 安全断言 2：到达的值不含原始控制字符（已被剥离为空格）
  const arrived = idx >= 0 ? args[idx + 1] : ''
  const stripped = !/[\r\n\t\x00]/.test(arrived)
  // 安全断言 3：没有注入命令出现在参数列表外
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
