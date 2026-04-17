const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const launcher = require('../bin/hermes-shim-http.js');

function makePy(version = '3.12.1') {
  return { command: 'python3', baseArgs: [], version, executable: '/usr/bin/python3' };
}

test('checkPythonVenvSupport reports missing ensurepip with install guidance', () => {
  const py = makePy('3.12.1');
  const result = launcher.checkPythonVenvSupport(py, {
    spawnSyncImpl: () => ({ status: 1, stderr: 'ensurepip is not available' }),
  });

  assert.equal(result.ok, false);
  assert.match(result.message, /python3\.12-venv/);
  assert.match(result.message, /sudo apt install python3\.12-venv/);
});

test('buildVenvHelpMessage falls back to generic python3-venv guidance', () => {
  const message = launcher.buildVenvHelpMessage({
    pyVersion: '3.13.0',
    detail: 'ensurepip is not available',
  });

  assert.match(message, /python3-venv/);
  assert.match(message, /ensurepip is not available/);
});

test('ensureEnv fails with a friendly message when venv support is missing', () => {
  const py = makePy('3.12.1');
  const errors = [];

  assert.throws(
    () => launcher.ensureEnv(py, {
      fsImpl: {
        existsSync: () => false,
        mkdirSync: () => {},
        rmSync: () => {},
        writeFileSync: () => {},
        readFileSync: () => '',
      },
      failImpl: (message, extra) => {
        errors.push({ message, extra });
        throw new Error('fail');
      },
      runImpl: () => {
        throw new Error('run should not be called when venv support is missing');
      },
      checkPythonVenvSupportImpl: () => ({
        ok: false,
        message: 'Python virtual environment support is missing. Install it with: sudo apt install python3.12-venv',
      }),
      packageRoot: '/tmp/pkg',
      cacheRoot: '/tmp/cache',
      envRoot: '/tmp/cache/env',
      markerPath: '/tmp/cache/env/.bootstrap-complete.json',
      requirementsHash: 'abc',
      pkg: { version: '0.1.2' },
      requirementsPath: '/tmp/pkg/requirements.txt',
    }),
    /fail/,
  );

  assert.equal(errors.length, 1);
  assert.match(errors[0].message, /Python virtual environment support is missing/);
  assert.match(errors[0].extra, /python3\.12-venv/);
});

test('doctor summary reports missing venv support clearly', () => {
  const summary = launcher.getDoctorSummary({
    py: makePy('3.12.1'),
    venvCheck: { ok: false, message: 'Install python3.12-venv' },
    shimCommand: 'claude',
    cwd: '/repo',
    existsSyncImpl: (value) => value === '/repo',
    statSyncImpl: () => ({ isDirectory: () => true }),
    commandExistsImpl: () => true,
    resolveCliArgsImpl: () => ['-p'],
  });

  assert.equal(summary.python.ok, true);
  assert.equal(summary.venv.ok, false);
  assert.equal(summary.cwd.ok, true);
  assert.equal(summary.command.ok, true);
  assert.deepEqual(summary.command.effective_args, ['-p']);
  assert.match(summary.venv.message, /python3\.12-venv/);
});

test('doctor summary reports missing wrapped CLI command', () => {
  const summary = launcher.getDoctorSummary({
    py: makePy('3.11.9'),
    venvCheck: { ok: true, message: 'ok' },
    shimCommand: 'claude',
    cwd: '/repo',
    existsSyncImpl: () => true,
    statSyncImpl: () => ({ isDirectory: () => true }),
    commandExistsImpl: () => false,
    resolveCliArgsImpl: () => ['-p'],
  });

  assert.equal(summary.command.ok, false);
  assert.match(summary.command.message, /claude/);
});

test('doctor summary rejects cwd paths that are not directories', () => {
  const summary = launcher.getDoctorSummary({
    py: makePy('3.11.9'),
    venvCheck: { ok: true, message: 'ok' },
    shimCommand: 'claude',
    cwd: '/tmp/not-a-dir',
    existsSyncImpl: () => true,
    statSyncImpl: () => ({ isDirectory: () => false }),
    commandExistsImpl: () => true,
    resolveCliArgsImpl: () => ['-p'],
  });

  assert.equal(summary.cwd.ok, false);
  assert.match(summary.cwd.message, /directory/i);
});

test('parseCliOptions keeps remainder args for effective arg reporting', () => {
  const parsed = launcher.parseCliOptions(['--command', 'claude', '--cwd', '/repo', '--', '--verbose', '--foo']);

  assert.deepEqual(parsed.providedArgs, ['--verbose', '--foo']);
});

test('parseCliOptions leaves new server feature flags in passthrough args', () => {
  const argv = [
    '--command', 'claude',
    '--cwd', '/repo',
    '--cache-path', '/tmp/sessions.sqlite',
    '--cache-ttl-seconds', '42',
    '--cache-max-entries', '7',
    '--compaction', 'window',
    '--compaction-threshold', '0.75',
    '--log-level', 'debug',
    '--log-format', 'json',
  ];

  const parsed = launcher.parseCliOptions(argv);

  assert.equal(parsed.command, 'claude');
  assert.equal(parsed.cwd, '/repo');
  assert.deepEqual(parsed.passthrough, argv);
  assert.deepEqual(parsed.providedArgs, []);
});
