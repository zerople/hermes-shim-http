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

function findPython() {
  const candidates = process.platform === 'win32'
    ? [['py', ['-3']], ['python', []], ['python3', []]]
    : [['python3', []], ['python', []]];
  const rejections = [];

  for (const [command, baseArgs] of candidates) {
    const probe = spawnSync(
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
    fail('Python 3.10+ is required, but only incompatible interpreters were found in PATH.', rejections.join('\n'));
  }
  return null;
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: options.capture ? 'pipe' : 'inherit',
    encoding: 'utf8',
    env: options.env || process.env,
    cwd: options.cwd || packageRoot,
  });
  if (result.status !== 0) {
    fail(`Command failed: ${command} ${args.join(' ')}`, result.stderr || result.stdout || `exit ${result.status}`);
  }
  return result;
}

function ensureEnv(py) {
  fs.mkdirSync(cacheRoot, { recursive: true });
  const pythonBin = process.platform === 'win32'
    ? path.join(envRoot, 'Scripts', 'python.exe')
    : path.join(envRoot, 'bin', 'python');
  const pipArgs = ['-m', 'pip'];

  let bootstrapNeeded = true;
  if (fs.existsSync(markerPath)) {
    try {
      const marker = JSON.parse(fs.readFileSync(markerPath, 'utf8'));
      bootstrapNeeded = !(marker.version === pkg.version && marker.requirementsHash === requirementsHash && fs.existsSync(pythonBin));
    } catch (_) {
      bootstrapNeeded = true;
    }
  }

  if (bootstrapNeeded) {
    if (!fs.existsSync(pythonBin)) {
      fs.rmSync(envRoot, { recursive: true, force: true });
      fs.mkdirSync(envRoot, { recursive: true });
      run(py.command, [...py.baseArgs, '-m', 'venv', envRoot]);
    }
    run(pythonBin, [...pipArgs, 'install', '--upgrade', 'pip']);
    run(pythonBin, [...pipArgs, 'install', '-r', requirementsPath]);
    fs.writeFileSync(markerPath, JSON.stringify({ version: pkg.version, requirementsHash }, null, 2));
  }
  return pythonBin;
}

(function main() {
  const py = findPython();
  if (!py) fail('Python 3 is required but was not found in PATH.');
  const pythonBin = ensureEnv(py);
  const env = { ...process.env };
  env.PYTHONPATH = env.PYTHONPATH ? `${packageRoot}${path.delimiter}${env.PYTHONPATH}` : packageRoot;
  const args = ['-m', 'hermes_shim_http.server', ...process.argv.slice(2)];
  const child = spawnSync(pythonBin, args, { stdio: 'inherit', env, cwd: process.cwd() });
  if (typeof child.status === 'number') {
    process.exit(child.status);
  }
  fail('Failed to launch shim process.');
})();
