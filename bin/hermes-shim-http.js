#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { spawnSync } = require('child_process');

const packageRoot = path.resolve(__dirname, '..');
const pkg = require(path.join(packageRoot, 'package.json'));
const requirementsPath = path.join(packageRoot, 'requirements.txt');
const requirementsHash = crypto.createHash('sha256').update(fs.readFileSync(requirementsPath)).digest('hex').slice(0, 12);
const cacheRoot = path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), 'hermes-shim-http');
const envRoot = path.join(cacheRoot, `${pkg.version}-${requirementsHash}`);
const markerPath = path.join(envRoot, '.bootstrap-complete.json');

function fail(message, extra) {
  console.error(`[hermes-shim-http] ${message}`);
  if (extra) console.error(extra);
  process.exit(1);
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch (_) {
    return null;
  }
}

function parseCliOptions(argv) {
  const args = Array.from(argv || []);
  const options = {
    command: 'claude',
    cwd: process.cwd(),
    profile: 'auto',
    doctor: false,
    passthrough: args,
    providedArgs: [],
  };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--') {
      options.providedArgs = args.slice(i + 1);
      break;
    }
    if (arg === '--doctor') {
      options.doctor = true;
      continue;
    }
    if (arg === '--command' && i + 1 < args.length) {
      options.command = args[i + 1];
      i += 1;
      continue;
    }
    if (arg.startsWith('--command=')) {
      options.command = arg.split('=', 2)[1];
      continue;
    }
    if (arg === '--cwd' && i + 1 < args.length) {
      options.cwd = args[i + 1];
      i += 1;
      continue;
    }
    if (arg.startsWith('--cwd=')) {
      options.cwd = arg.split('=', 2)[1];
      continue;
    }
    if (arg === '--profile' && i + 1 < args.length) {
      options.profile = args[i + 1];
      i += 1;
      continue;
    }
    if (arg.startsWith('--profile=')) {
      options.profile = arg.split('=', 2)[1];
    }
  }
  return options;
}

function findPython({ spawnSyncImpl = spawnSync, failImpl = fail } = {}) {
  const candidates = process.platform === 'win32'
    ? [['py', ['-3']], ['python', []], ['python3', []]]
    : [['python3', []], ['python', []]];
  const rejections = [];

  for (const [command, baseArgs] of candidates) {
    const probe = spawnSyncImpl(
      command,
      [
        ...baseArgs,
        '-c',
        'import json, sys; print(json.dumps({"major": sys.version_info[0], "minor": sys.version_info[1], "micro": sys.version_info[2], "executable": sys.executable}))',
      ],
      { encoding: 'utf8' },
    );

    if (probe.status !== 0) continue;
    const info = parseJson((probe.stdout || '').trim());
    if (!info) continue;
    if (info.major === 3 && info.minor >= 10) {
      return { command, baseArgs, version: `${info.major}.${info.minor}.${info.micro}`, executable: info.executable };
    }
    rejections.push(`${command} ${baseArgs.join(' ')} -> Python ${info.major}.${info.minor}.${info.micro}`.trim());
  }

  if (rejections.length) {
    failImpl('Python 3.10+ is required, but only incompatible interpreters were found in PATH.', rejections.join('\n'));
  }
  return null;
}

function resolveCliArgs({ command = 'claude', profile = 'auto', providedArgs = [] } = {}) {
  if (Array.isArray(providedArgs) && providedArgs.length) {
    return Array.from(providedArgs);
  }
  const basename = path.basename((command || '').trim()).toLowerCase();
  let effectiveProfile = profile || 'auto';
  if (effectiveProfile === 'auto') {
    effectiveProfile = { claude: 'claude', codex: 'codex', opencode: 'opencode' }[basename] || 'generic';
  }
  return { claude: ['-p'], codex: ['exec'], opencode: ['run'], generic: [] }[effectiveProfile] || [];
}

function commandExists(command, { spawnSyncImpl = spawnSync } = {}) {
  if (!command) return false;
  const checkCommand = process.platform === 'win32' ? 'where' : 'which';
  const result = spawnSyncImpl(checkCommand, [command], { encoding: 'utf8' });
  return result.status === 0;
}

function buildVenvHelpMessage({ pyVersion, detail }) {
  const parts = String(pyVersion || '').split('.');
  const majorMinor = parts.length >= 2 ? `${parts[0]}.${parts[1]}` : null;
  const versionedPkg = majorMinor ? `python${majorMinor}-venv` : null;
  const packageHint = versionedPkg ? `${versionedPkg} (or python3-venv)` : 'python3-venv';
  const installHint = versionedPkg
    ? `sudo apt install ${versionedPkg}`
    : 'sudo apt install python3-venv';
  return [
    'Python virtual environment support is missing on this machine.',
    `Install ${packageHint} and try again.`,
    `Suggested command: ${installHint}`,
    detail ? `Original error: ${detail}` : null,
  ].filter(Boolean).join('\n');
}

function checkPythonVenvSupport(py, { spawnSyncImpl = spawnSync } = {}) {
  const probe = spawnSyncImpl(
    py.command,
    [...py.baseArgs, '-c', 'import ensurepip, venv; print("ok")'],
    { encoding: 'utf8' },
  );

  if (probe.status === 0) {
    return { ok: true, message: `Python ${py.version} supports virtual environments.` };
  }

  const detail = (probe.stderr || probe.stdout || '').trim();
  return {
    ok: false,
    message: buildVenvHelpMessage({ pyVersion: py.version, detail }),
  };
}

function run(command, args, options = {}) {
  const result = (options.spawnSyncImpl || spawnSync)(command, args, {
    stdio: options.capture ? 'pipe' : 'inherit',
    encoding: 'utf8',
    env: options.env || process.env,
    cwd: options.cwd || packageRoot,
  });
  if (result.status !== 0) {
    (options.failImpl || fail)(`Command failed: ${command} ${args.join(' ')}`, result.stderr || result.stdout || `exit ${result.status}`);
  }
  return result;
}

function ensureEnv(py, options = {}) {
  const fsImpl = options.fsImpl || fs;
  const failImpl = options.failImpl || fail;
  const runImpl = options.runImpl || run;
  const checkVenvImpl = options.checkPythonVenvSupportImpl || checkPythonVenvSupport;
  const localCacheRoot = options.cacheRoot || cacheRoot;
  const localEnvRoot = options.envRoot || envRoot;
  const localMarkerPath = options.markerPath || markerPath;
  const localRequirementsPath = options.requirementsPath || requirementsPath;
  const localRequirementsHash = options.requirementsHash || requirementsHash;
  const localPkg = options.pkg || pkg;

  fsImpl.mkdirSync(localCacheRoot, { recursive: true });
  const pythonBin = process.platform === 'win32'
    ? path.join(localEnvRoot, 'Scripts', 'python.exe')
    : path.join(localEnvRoot, 'bin', 'python');
  const pipArgs = ['-m', 'pip'];

  let bootstrapNeeded = true;
  if (fsImpl.existsSync(localMarkerPath)) {
    try {
      const marker = JSON.parse(fsImpl.readFileSync(localMarkerPath, 'utf8'));
      bootstrapNeeded = !(marker.version === localPkg.version && marker.requirementsHash === localRequirementsHash && fsImpl.existsSync(pythonBin));
    } catch (_) {
      bootstrapNeeded = true;
    }
  }

  if (bootstrapNeeded) {
    if (!fsImpl.existsSync(pythonBin)) {
      const venvCheck = checkVenvImpl(py);
      if (!venvCheck.ok) {
        failImpl('Python virtual environment support is missing.', venvCheck.message);
      }
      fsImpl.rmSync(localEnvRoot, { recursive: true, force: true });
      fsImpl.mkdirSync(localEnvRoot, { recursive: true });
      runImpl(py.command, [...py.baseArgs, '-m', 'venv', localEnvRoot], { failImpl });
    }
    runImpl(pythonBin, [...pipArgs, 'install', '--upgrade', 'pip'], { failImpl });
    runImpl(pythonBin, [...pipArgs, 'install', '-r', localRequirementsPath], { failImpl });
    fsImpl.writeFileSync(localMarkerPath, JSON.stringify({ version: localPkg.version, requirementsHash: localRequirementsHash }, null, 2));
  }
  return pythonBin;
}

function getDoctorSummary({
  py,
  venvCheck,
  shimCommand,
  cwd,
  profile = 'auto',
  providedArgs = [],
  existsSyncImpl = fs.existsSync,
  statSyncImpl = fs.statSync,
  commandExistsImpl = commandExists,
  resolveCliArgsImpl = resolveCliArgs,
} = {}) {
  const effectiveArgs = resolveCliArgsImpl({ command: shimCommand, profile, providedArgs });
  const cwdExists = !!existsSyncImpl(cwd);
  const cwdIsDirectory = cwdExists && !!statSyncImpl(cwd).isDirectory();
  return {
    python: py
      ? { ok: true, version: py.version, executable: py.executable }
      : { ok: false, message: 'Python 3.10+ was not found in PATH.' },
    venv: py
      ? { ok: !!(venvCheck && venvCheck.ok), message: venvCheck ? venvCheck.message : 'unknown' }
      : { ok: false, message: 'Python was not found, so venv support could not be checked.' },
    cwd: {
      ok: cwdIsDirectory,
      value: cwd,
      message: !cwdExists
        ? `Working directory does not exist: ${cwd}`
        : (cwdIsDirectory ? 'Working directory exists and is a directory.' : `Working directory is not a directory: ${cwd}`),
    },
    command: {
      ok: commandExistsImpl(shimCommand),
      value: shimCommand,
      effective_args: effectiveArgs,
      message: commandExistsImpl(shimCommand)
        ? `Wrapped CLI command '${shimCommand}' is available in PATH.`
        : `Wrapped CLI command '${shimCommand}' was not found in PATH.`,
    },
  };
}

function printDoctorSummary(summary) {
  console.log('[hermes-shim-http] preflight check');
  console.log(JSON.stringify(summary, null, 2));
}

function main(argv = process.argv.slice(2)) {
  const options = parseCliOptions(argv);
  const py = findPython();
  if (!py) fail('Python 3 is required but was not found in PATH.');
  const venvCheck = checkPythonVenvSupport(py);

  if (options.doctor) {
    printDoctorSummary(getDoctorSummary({
      py,
      venvCheck,
      shimCommand: options.command,
      cwd: options.cwd,
      profile: options.profile,
      providedArgs: options.providedArgs,
    }));
    if (!venvCheck.ok || !fs.existsSync(options.cwd) || !fs.statSync(options.cwd).isDirectory() || !commandExists(options.command)) {
      process.exit(1);
    }
    return;
  }

  const pythonBin = ensureEnv(py, { checkPythonVenvSupportImpl: () => venvCheck });
  const env = { ...process.env };
  env.PYTHONPATH = env.PYTHONPATH ? `${packageRoot}${path.delimiter}${env.PYTHONPATH}` : packageRoot;
  const args = ['-m', 'hermes_shim_http.server', ...argv];
  const child = spawnSync(pythonBin, args, { stdio: 'inherit', env, cwd: process.cwd() });
  if (typeof child.status === 'number') {
    process.exit(child.status);
  }
  fail('Failed to launch shim process.');
}

module.exports = {
  buildVenvHelpMessage,
  checkPythonVenvSupport,
  commandExists,
  ensureEnv,
  fail,
  findPython,
  getDoctorSummary,
  main,
  parseCliOptions,
  resolveCliArgs,
  run,
};

if (require.main === module) {
  main();
}
