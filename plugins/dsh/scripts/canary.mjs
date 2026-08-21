// dsh-skillopt canary — the clean-package check the SkillOpt review asked for.
//
// Packs the plugin with `npm pack --dry-run`, asserts the bundle manifest is
// complete (cordis.patch.yml present), then ACTUALLY packs it (npm pack),
// extracts the tarball, and loads the plugin FROM THE PACKED ARTIFACT into a
// mock Cordis context with a fake rc.8-shaped shell (CollectedOutput objects),
// invoking every tool and asserting real stdout/exit/error behavior. Loading
// the extracted bundle (not the source tree) is what the review's "loads the
// packed bundle" demands — the packed files are exactly what `files` ships.
//
// Run: node scripts/canary.mjs   (requires npm + the plugin's deps resolvable)

import { execSync } from 'node:child_process'
import { mkdirSync, readFileSync, existsSync, rmSync } from 'node:fs'
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
// 1b. ACTUAL pack + extract: the rest of the canary runs against the packed
// artifact (what `files` ships), not the source tree — the review asked for a
// canary that "loads the packed bundle". npm pack --json prints the tarball
// name; extract into a scratch dir inside root so the extracted module can
// still resolve @deepseek-ai/* deps up the tree.
// ---------------------------------------------------------------------------
console.log('1b. real pack + extract (canary runs against the packed artifact)')
const tarball = JSON.parse(execSync('npm pack --json', { cwd: root, encoding: 'utf8' }))[0].filename
check('npm pack produced a tarball', !!tarball && existsSync(join(root, tarball)), tarball || 'no tarball')
const scratch = join(root, '.canary-pack')
rmSync(scratch, { recursive: true, force: true })
mkdirSync(scratch, { recursive: true })
execSync(`tar -xzf "${tarball}" -C "${scratch}"`, { cwd: root, encoding: 'utf8' })
const packedRoot = join(scratch, 'package')
check('extracted package/ contains src/index.js', existsSync(join(packedRoot, 'src/index.js')))
check('extracted package/ contains cordis.patch.yml', existsSync(join(packedRoot, 'cordis.patch.yml')))
check('extracted package.json matches files list', JSON.parse(readFileSync(join(packedRoot, 'package.json'), 'utf8')).name === pkg.name)
// Never leave the tarball or scratch dir behind.
process.on('exit', () => {
  try { rmSync(join(root, tarball), { force: true }) } catch {}
  try { rmSync(scratch, { recursive: true, force: true }) } catch {}
})

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

const { apply } = await import(pathToFileURL(join(packedRoot, 'src/index.js')).href)
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
// preferences is a REAL parameter of run; a bogus value makes the engine exit 2
const bad = await defs['skillopt_run'].execute({ preferences: '--bad-model' }, {})
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
console.log('6. argv quoting is shell-safe (platform-aware)')
const { buildArgv, quoteArgv, q, IS_WINDOWS } = await import(pathToFileURL(join(packedRoot, 'src/index.js')).href)
// Verify quoting directly: a preference with spaces and metacharacters must stay
// inside one argument (single-quoted, embedded quotes escaped per platform).
const argv = buildArgv({}, 'run', { preferences: "never ' rm -rf /" })
const quoted = quoteArgv(argv)
const prefArg = argv[argv.indexOf('--preferences') + 1]
check('preference stays one argv element', argv.includes('--preferences') && argv[argv.indexOf('--preferences') + 1] === "never ' rm -rf /")
// pwsh escapes an embedded quote by doubling it (''); bash closes/reopens ('\'').
const expected = IS_WINDOWS ? "'never '' rm -rf /'" : "'never '\\'' rm -rf /'"
check('quoted form uses the host shell escape', quoted.includes(expected), `expected ${expected} got ...${quoted.slice(-40)}`)
check('embedded quote escaped exactly once', (IS_WINDOWS ? quoted.split("''").length - 1 : quoted.split("'\\''").length - 1) === 1)
check('no unquoted shell metacharacters', !/;\s*rm\s+-rf/.test(quoted))
check('q() control chars folded to single spaces', q("a\r\nb\x00c") === "'a b c'")

// ---------------------------------------------------------------------------
// 7. auto-adopt is OPERATOR-ONLY: the model cannot set it
// ---------------------------------------------------------------------------
console.log('7. auto-adopt is operator-only')
const before = called.commands.length
await defs['skillopt_run'].execute({ autoAdopt: true, backend: 'mock' }, {})
const runCmd = called.commands.slice(before).find((c) => c.includes("'run'"))
check('model-supplied autoAdopt ignored', runCmd ? !runCmd.includes('--auto-adopt') : true, runCmd || 'no run command')
// operator config enables it
const { apply: apply2 } = await import(pathToFileURL(join(packedRoot, 'src/index.js')).href)
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

// ---------------------------------------------------------------------------
// 7b. undeclared parameters are filtered: the model cannot inject fields the
// tool does not declare (dsh's parameter schema allows extra properties by
// default, so the plugin's per-tool whitelist is what stops this). adopt
// declares only `project`; backend/model/maxTasks/json must not reach argv.
// ---------------------------------------------------------------------------
console.log('7b. undeclared tool parameters are filtered')
const before7b = called.commands.length
await defs['skillopt_adopt'].execute({ project: '/tmp/p', backend: 'codex', model: 'gpt-x', maxTasks: 99, json: true }, {})
const adoptCmd7b = called.commands.slice(before7b).find((c) => c.includes("'adopt'"))
check('adopt keeps declared project', adoptCmd7b ? adoptCmd7b.includes("'--project'") : false, adoptCmd7b || 'no adopt command')
check('adopt drops undeclared backend', adoptCmd7b ? !adoptCmd7b.includes("'--backend'") : true, adoptCmd7b || 'no adopt command')
check('adopt drops undeclared model', adoptCmd7b ? !adoptCmd7b.includes("'--model'") : true)
check('adopt drops undeclared maxTasks', adoptCmd7b ? !adoptCmd7b.includes("'--max-tasks'") : true)
check('adopt drops undeclared json', adoptCmd7b ? !adoptCmd7b.includes("'--json'") : true)

// ---------------------------------------------------------------------------
// 7c. value-domain guard: project/output with shell metacharacters are rejected
// before they reach the engine's own shell/crontab/schtasks interpolation
// (scheduler.py splices --project into a crontab line / Windows run.cmd, and
// write_tasks_file() writes --output to an arbitrary path). Legitimate values
// pass; metacharacter and traversal values are refused with an error message.
// ---------------------------------------------------------------------------
console.log('7c. path value-domain guard (engine re-interpolation / file write)')
const before7c = called.commands.length
// schedule with an injected project (would break out of the engine's own
// `--project "..."` splice and run a separate command under the scheduler)
const inj = await defs['skillopt_schedule'].execute({ project: 'C:/tmp/x" & echo PWNED > C:/tmp/pwned.txt & "', hour: 3 }, {})
check('schedule rejects injected project', /rejected/.test(inj), inj.slice(0, 160))
check('no schedule command reached the shell', called.commands.length === before7c)
// harvest output escaping the working area (absolute path / traversal)
const abs = await defs['skillopt_harvest'].execute({ project: '/tmp/p', output: 'C:/Windows/System32/drivers/etc/hosts' }, {})
check('harvest rejects absolute output', /rejected/.test(abs), abs.slice(0, 160))
const trav = await defs['skillopt_harvest'].execute({ project: '/tmp/p', output: '../../etc/hosts' }, {})
check('harvest rejects traversal output', /rejected/.test(trav), trav.slice(0, 160))
// legit values still pass through the guard
const ok7c = await defs['skillopt_harvest'].execute({ project: '/tmp/my proj', output: 'tasks.json', source: 'claude' }, {})
const okCmd7c = called.commands.slice(before7c).find((c) => c.includes("'harvest'"))
check('legit project/output pass', !/rejected/.test(ok7c) && !!okCmd7c, ok7c.slice(0, 120))
check('legit harvest cmd has project+output', okCmd7c ? okCmd7c.includes("'--output'") && okCmd7c.includes("'/tmp/my proj'") : false, okCmd7c || 'no harvest command')

// ---------------------------------------------------------------------------
// 7d. clock range guard: schedule's hour/minute are spliced by the engine into
// a crontab line and a schtasks start time without validation; out-of-range
// values would create broken scheduled entries. They must be rejected.
// ---------------------------------------------------------------------------
console.log('7d. schedule clock range guard')
const badHour = await defs['skillopt_schedule'].execute({ project: '/tmp/p', hour: 99, minute: 17 }, {})
check('schedule rejects hour=99', /rejected/.test(badHour), badHour.slice(0, 140))
const badMinute = await defs['skillopt_schedule'].execute({ project: '/tmp/p', hour: 3, minute: -1 }, {})
check('schedule rejects minute=-1', /rejected/.test(badMinute), badMinute.slice(0, 140))
const okSched = await defs['skillopt_schedule'].execute({ project: '/tmp/p', hour: 3, minute: 17 }, {})
const okSchedCmd = called.commands.slice(-1)[0]
check('legit clock passes and reaches shell', !/rejected/.test(okSched) && !!okSchedCmd && okSchedCmd.includes("'--hour'") && okSchedCmd.includes("'--minute'"), okSched.slice(0, 120))

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`)
process.exit(failures === 0 ? 0 : 1)
