#!/usr/bin/env node
// Real-bash quoting verification for dsh-skillopt.
//
// The review asked for "Bash and pwsh tests for spaces, quotes, and
// metacharacters". This runs the plugin's quoteArgv() output through a REAL
// bash (Git Bash on Windows) and asserts the shell sees exactly one argument
// per argv element — spaces stay inside one argument, embedded quotes are
// preserved, and metacharacters cannot break out.
//
// Usage: node scripts/test-quoting-bash.mjs   (requires Git Bash)
import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)
const root = dirname(dirname(fileURLToPath(import.meta.url)))
const BASH = process.env.BASH_PATH || 'C:/Program Files/Git/bin/bash.exe'

const { quoteArgv } = await import(pathToFileURL(join(root, 'src/index.js')).href)

let failures = 0
function check(name, cond, detail = '') {
  if (cond) console.log(`  ✅ ${name}`)
  else {
    failures++
    console.log(`  ❌ ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

// Run a bash snippet that prints each received argument on its own line,
// then compare what the shell received against what we intended.
function bashRoundtrip(argv) {
  const quoted = quoteArgv(argv)
  // bash: for each arg, print a delimiter + the arg; newlines in args are
  // escaped so the split stays unambiguous.
  const script = `for a in ${quoted}; do printf '[%s]\\n' "$a"; done`
  const out = execFileSync(BASH, ['-c', script], { encoding: 'utf8', cwd: root })
  return out.trim().split('\n').map((l) => l.replace(/^\[/, '').replace(/\]$/, ''))
}

console.log('1. spaces stay inside one argument')
const spaced = ['python', '-m', 'skillopt_sleep', 'run', '--project', '/tmp/my proj', '--preferences', 'use async always']
const got1 = bashRoundtrip(spaced)
check('path with space intact', got1[5] === '/tmp/my proj', JSON.stringify(got1))
check('preference with space intact', got1[7] === 'use async always', JSON.stringify(got1))

console.log('2. embedded single quotes are preserved')
const quoted = ['python', '-m', 'skillopt_sleep', 'run', '--preferences', "never ' rm -rf /"]
const got2 = bashRoundtrip(quoted)
check("embedded quote preserved", got2[5] === "never ' rm -rf /", JSON.stringify(got2))

console.log('3. metacharacters cannot break out (injection attempt)')
const inject = ['python', '-m', 'skillopt_sleep', 'run', '--preferences', 'x; touch /tmp/dsh-injected; echo PWNED']
const got3 = bashRoundtrip(inject)
// The injection must arrive as ONE literal argument, and the touch/echo must
// NOT have executed as shell commands.
check('injection stays one argument', got3[5] === 'x; touch /tmp/dsh-injected; echo PWNED', JSON.stringify(got3))
// The bash loop prints the arg verbatim, so PWNED appears in the ARG text —
// the real assertion is that NO extra output line was produced (which would
// mean the `;` broke out and echo executed).
check('no extra output line from executed echo', got3.length === 6, `got ${got3.length} lines`)
// ensure no file was created by the injection attempt
const { existsSync } = await import('node:fs')
check('no /tmp/dsh-injected file created', !existsSync('/tmp/dsh-injected') && !existsSync('C:/tmp/dsh-injected'))

console.log('4. double quotes and backticks are inert')
const backtick = ['python', '-m', 'skillopt_sleep', 'run', '--preferences', 'echo `id` $(whoami) "x"']
const got4 = bashRoundtrip(backtick)
check('backticks/dollar stay literal', got4[5] === 'echo `id` $(whoami) "x"', JSON.stringify(got4))

console.log('5. empty and numeric values')
const mixed = ['python', '-m', 'skillopt_sleep', 'run', '--max-tasks', '40', '--hour', '3']
const got5 = bashRoundtrip(mixed)
check('numbers intact', got5[5] === '40' && got5[7] === '3', JSON.stringify(got5))

console.log(failures === 0 ? '\nALL BASH QUOTING CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`)
process.exit(failures === 0 ? 0 : 1)
