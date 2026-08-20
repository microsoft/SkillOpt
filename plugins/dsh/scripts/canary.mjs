// dsh-skillopt canary — the clean-package check the SkillOpt review asked for.
//
// Packs the plugin with `npm pack --dry-run`, asserts the bundle manifest is
// complete (cordis.patch.yml present), loads the packed plugin into a mock
// Cordis context with a fake rc.8-shaped shell (CollectedOutput objects), and
// invokes every tool, asserting real stdout/exit/error behavior.
//
// Run: node scripts/canary.mjs

import { execSync } from 'node:child_process'
import { readFileSync, existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)
const root = dirname(dirname(fileURLToPath(import.meta.url)))
const { Context } = require('@deepseek-ai/cordis')

let failures = 0
function check(name, cond, detail = '') {
  if (cond) console.log(`  ✅ ${name}`)
  else {
    failures++
    console.log(`  ❌ ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

// ---------------------------------------------------------------------------
// 1. npm pack --dry-run: bundle must include cordis.patch.yml
// ---------------------------------------------------------------------------
console.log('1. bundle completeness (npm pack --dry-run)')
const packOut = execSync('npm pack --dry-run --json', { cwd: root, encoding: 'utf8' })
const packInfo = JSON.parse(packOut)
const packedFiles = packInfo.map((p) => p.files.map((f) => f.path)).flat()
check('cordis.patch.yml packed', packedFiles.includes('cordis.patch.yml'))
check('src/index.js packed', packedFiles.some((f) => f === 'src/index.js' || f.endsWith('/src/index.js')))
check('package.json packed', packedFiles.includes('package.json'))

// package.json declares dsh.bundle.patch → cordis.patch.yml
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'))
check('dsh.bundle.patch points at packed file', packedFiles.includes(pkg.dsh?.bundle?.patch))
check('schemastery declared as direct dependency', !!pkg.dependencies?.['@deepseek-ai/schemastery'])

// ---------------------------------------------------------------------------
// 2. load the plugin against a mock rc.8-shaped shell
// ---------------------------------------------------------------------------
console.log('2. plugin loads and registers 7 tools')
const ctx = new Context()
const defs = {}
ctx.tools = {
  register(def) {
    defs[def.name] = def
    return () => {}
  },
}
// rc.8-shaped fake shell: resolve() applies defaults, run() returns
// CollectedOutput objects for stdout/stderr.
const called = { resolve: 0, run: 0, commands: [] }
ctx.shell = {
  resolve(req) {
    called.resolve++
    return { ...req, workdir: '.', stdoutMaxBytes: 2_000_000, timeoutMs: req.timeoutMs ?? 600_000 }
  },
  async run(spec) {
    called.run++
    called.commands.push(spec.command)
    if (spec.command.includes("'status'")) {
      return {
        exitCode: 0,
        stdout: { text: '[sleep] nights so far: 0\n[sleep] no staged proposals yet.', truncated: false },
        stderr: { text: '', truncated: false },
      }
    }
    if (spec.command.includes("'dry-run'")) {
      return {
        exitCode: 0,
        stdout: { text: '[sleep] night 1: 0 sessions -> 0 tasks', truncated: false },
        stderr: { text: '', truncated: false },
      }
    }
    if (spec.command.includes("'run'") && spec.command.includes('--bad-model')) {
      // a real model value that makes the engine exit 2 (e.g. unknown provider)
      return {
        exitCode: 2,
        stdout: { text: '', truncated: false },
        stderr: { text: "error: unknown model '--bad-model'", truncated: false },
      }
    }
    if (spec.command.includes('--timeout-trigger')) {
      // executor timeout shape: exitCode null, timedOut flag, stderr explains
      return {
        exitCode: null,
        timedOut: true,
        stdout: { text: '', truncated: false },
        stderr: { text: 'command timed out after 600000ms', truncated: false },
      }
    }
    if (spec.signal?.aborted) {
      // executor abort shape: killed by signal, no exit code
      return {
        exitCode: null,
        killed: 'SIGTERM',
        stdout: { text: '', truncated: false },
        stderr: { text: 'process killed by signal SIGTERM', truncated: false },
      }
    }
    // truncated output with spill path (dry-run and others)
    return {
      exitCode: 0,
      stdout: { text: 'big output tail…', truncated: true, spillPath: 'C:/spill/stdout.log' },
      stderr: { text: '', truncated: false },
    }
  },
}
ctx.logger = { info: () => {} }

const { apply } = await import(pathToFileURL(join(root, 'src/index.js')).href)
apply(ctx, { backend: 'mock' })

check('7 tools registered', Object.keys(defs).length === 7, `got ${Object.keys(defs).length}`)

// ---------------------------------------------------------------------------
// 3. skillopt_status: real stdout surfaced (also proves resolve() is used)
// ---------------------------------------------------------------------------
console.log('3. skillopt_status surfaces real stdout')
const status = await defs['skillopt_status'].execute({}, {})
check('exit=0 reported', status.includes('exit=0'), status.slice(0, 120))
check('real stdout present', status.includes('nights so far'), status.slice(0, 200))
check('no "(no output)" for real output', !status.includes('(no output)'))
check('shell.resolve used', called.resolve > 0, 'execute must go through resolve()')

// ---------------------------------------------------------------------------
// 4. nonzero exit: stderr surfaced with exit code
// ---------------------------------------------------------------------------
console.log('4. nonzero exit surfaces stderr')
// model is a REAL parameter; a bogus model value makes the engine exit 2
const bad = await defs['skillopt_run'].execute({ model: '--bad-model' }, {})
check('exit code surfaced', bad.includes('exit=2'), bad.slice(0, 150))
check('stderr text surfaced', bad.includes('unknown model'), bad.slice(0, 200))

// ---------------------------------------------------------------------------
// 4b. timeout: executor timeout shape is surfaced, not swallowed as failure
// ---------------------------------------------------------------------------
console.log('4b. timeout surfaces executor timeout')
const to = await defs['skillopt_harvest'].execute({ output: '--timeout-trigger' }, {})
check('timeout reported', to.includes('exit=timeout') && to.includes('timed out'), to.slice(0, 200))
check('timeout stderr surfaced', to.includes('command timed out'), to.slice(0, 200))

// ---------------------------------------------------------------------------
// 4c. abort: signal-driven kill is surfaced as signal, not as a crash
// ---------------------------------------------------------------------------
console.log('4c. abort (signal) is surfaced')
const abortCtrl = { aborted: true, reason: 'user cancel' }
const ab = await defs['skillopt_adopt'].execute({}, { signal: abortCtrl })
check('abort run completed (no throw)', typeof ab === 'string')
check('abort marker surfaced', ab.includes('exit=null') || ab.includes('signal'), ab.slice(0, 120))
check('abort stderr surfaced', ab.includes('SIGTERM'), ab.slice(0, 200))

// ---------------------------------------------------------------------------
// 5. truncated output: spill path preserved
// ---------------------------------------------------------------------------
console.log('5. truncated output preserves spill path')
const trig = await defs['skillopt_adopt'].execute({}, {})
// adopt hits the fake shell's default branch (truncated + spill path)
check('truncated marker present', trig.includes('truncated'))
check('spill path present', trig.includes('C:/spill/stdout.log'))

// ---------------------------------------------------------------------------
// 6. argv quoting: spaces and metacharacters cannot break out
// ---------------------------------------------------------------------------
console.log('6. argv quoting is shell-safe')
const { buildArgv, quoteArgv } = await import(pathToFileURL(join(root, 'src/index.js')).href)
// Verify quoting directly: a preference with spaces and metacharacters must stay
// inside one argument (single-quoted, embedded quotes doubled).
const argv = buildArgv({}, 'run', { preferences: "never ' rm -rf /" })
const quoted = quoteArgv(argv)
const prefArg = argv[argv.indexOf('--preferences') + 1]
check('preference stays one argv element', argv.includes('--preferences') && argv[argv.indexOf('--preferences') + 1] === "never ' rm -rf /")
check('quoted form uses bash-safe escape', quoted.includes("'never '\\'' rm -rf /'"))
check('no unquoted shell metacharacters', !/;\s*rm\s+-rf/.test(quoted))

// ---------------------------------------------------------------------------
// 7. auto-adopt is OPERATOR-ONLY: the model cannot set it
// ---------------------------------------------------------------------------
console.log('7. auto-adopt is operator-only')
const before = called.commands.length
await defs['skillopt_run'].execute({ autoAdopt: true, backend: 'mock' }, {})
const runCmd = called.commands.slice(before).find((c) => c.includes("'run'"))
check('model-supplied autoAdopt ignored', runCmd ? !runCmd.includes('--auto-adopt') : true, runCmd || 'no run command')
// operator config enables it
const { apply: apply2 } = await import(pathToFileURL(join(root, 'src/index.js')).href)
// re-apply with a fresh capture to check config-driven --auto-adopt
const ctx2 = new Context()
const defs2 = {}
ctx2.tools = { register(d) { defs2[d.name] = d; return () => {} } }
const cmds2 = []
ctx2.shell = {
  resolve(req) { return req },
  async run(spec) { cmds2.push(spec.command); return { exitCode: 0, stdout: { text: 'ok', truncated: false }, stderr: { text: '', truncated: false } } },
}
ctx2.logger = { info: () => {} }
apply2(ctx2, { backend: 'mock', autoAdopt: true })
await defs2['skillopt_run'].execute({ backend: 'mock' }, {})
const runCmd2 = cmds2.find((c) => c.includes("'run'"))
check('operator config autoAdopt adds --auto-adopt', runCmd2 ? runCmd2.includes('--auto-adopt') : false, runCmd2 || 'no run command')

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`)
process.exit(failures === 0 ? 0 : 1)
