#!/usr/bin/env node
/**
 * Start the Orqis website *and* the backend it talks to, from one command.
 *
 * The website is useless without the FastAPI service: no login, no worklist, no
 * screening. Keeping them in two terminals meant the site was routinely "broken"
 * when it was really just unaccompanied. This supervises both.
 *
 *   node scripts/dev.mjs dev     -> next dev   (default)
 *   node scripts/dev.mjs start   -> next start (after `next build`)
 *
 * Behaviour worth knowing:
 *  - If something is already answering on the backend port, that process is left
 *    alone and reused. Running this twice does not fight an existing uvicorn.
 *  - The backend is found by walking up from this folder looking for
 *    `backend/main.py`, then by checking the usual sibling locations. Set
 *    `ORQIS_BACKEND_DIR` in `.env.local` to be explicit.
 *  - If the backend genuinely cannot be started, the website still comes up. It
 *    degrades to a clear "backend unreachable" state rather than refusing to run.
 *  - Ctrl+C stops both, including uvicorn's reloader children on Windows.
 */

import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEBSITE_DIR = path.resolve(HERE, '..');
const IS_WINDOWS = process.platform === 'win32';

/** Built at runtime so no control characters end up in this source file. */
const ESC = String.fromCharCode(27);
const COLOR = {
  reset: ESC + '[0m',
  dim: ESC + '[2m',
  bold: ESC + '[1m',
  teal: ESC + '[36m',
  amber: ESC + '[33m',
  green: ESC + '[32m',
};

function note(message) {
  process.stdout.write(`${COLOR.dim}[orqis]${COLOR.reset} ${message}\n`);
}
function warn(message) {
  process.stdout.write(`${COLOR.amber}[orqis]${COLOR.reset} ${message}\n`);
}

// ---------------------------------------------------------------- env loading

/**
 * Minimal `.env` reader. Node's own `--env-file` is not used because this script
 * must run the same way under `npm run dev` on every supported Node version.
 * Values already present in the real environment always win.
 */
function loadEnvFile(file) {
  if (!existsSync(file)) return;
  for (const rawLine of readFileSync(file, 'utf8').split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    if (!key || key in process.env) continue;
    let value = line.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    process.env[key] = value;
  }
}

loadEnvFile(path.join(WEBSITE_DIR, '.env.local'));
loadEnvFile(path.join(WEBSITE_DIR, '.env'));

// ------------------------------------------------------------ configuration

const MODE = process.argv[2] === 'start' ? 'start' : 'dev';
const WEB_PORT = process.env.PORT || '3000';
const API_URL = (process.env.ORQIS_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const API_PORT = (() => {
  try {
    return new URL(API_URL).port || '8000';
  } catch {
    return '8000';
  }
})();

/** Locate the Orqis repository that holds `backend/main.py`. */
function findBackendDir() {
  const explicit = process.env.ORQIS_BACKEND_DIR;
  if (explicit) {
    const resolved = path.resolve(WEBSITE_DIR, explicit);
    return existsSync(path.join(resolved, 'backend', 'main.py')) ? resolved : null;
  }

  const candidates = [];
  // Walk up: covers the website living inside the repo, or one level beside it.
  let dir = WEBSITE_DIR;
  for (let i = 0; i < 6; i += 1) {
    candidates.push(dir);
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  // The usual places a cloned Orqis checkout ends up on a dev machine.
  const home = process.env.USERPROFILE || process.env.HOME || '';
  if (home) {
    candidates.push(
      path.join(home, 'Downloads', 'Orqis-main', 'Orqis-main'),
      path.join(home, 'Downloads', 'Orqis-main'),
      path.join(home, 'Orqis'),
      path.join(home, 'Desktop', 'Orqis'),
    );
  }

  return (
    candidates.find((candidate) =>
      existsSync(path.join(candidate, 'backend', 'main.py')),
    ) || null
  );
}

/** First Python that exists. `py -3` is the Windows launcher fallback. */
function pythonCommand() {
  if (process.env.ORQIS_PYTHON) return { cmd: process.env.ORQIS_PYTHON, args: [] };
  const backendDir = findBackendDir();
  if (backendDir) {
    const venv = IS_WINDOWS
      ? path.join(backendDir, '.venv', 'Scripts', 'python.exe')
      : path.join(backendDir, '.venv', 'bin', 'python');
    if (existsSync(venv)) return { cmd: venv, args: [] };
  }
  return IS_WINDOWS ? { cmd: 'python', args: [] } : { cmd: 'python3', args: [] };
}

// -------------------------------------------------------- process management

const children = [];
let shuttingDown = false;

function pipe(child, label, color) {
  const prefix = `${color}[${label}]${COLOR.reset} `;
  const forward = (stream, sink) => {
    let buffer = '';
    stream.setEncoding('utf8');
    stream.on('data', (chunk) => {
      buffer += chunk;
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? '';
      for (const line of lines) sink.write(`${prefix}${line}\n`);
    });
    stream.on('end', () => {
      if (buffer) sink.write(`${prefix}${buffer}\n`);
    });
  };
  if (child.stdout) forward(child.stdout, process.stdout);
  if (child.stderr) forward(child.stderr, process.stderr);
}

function launch(label, color, cmd, args, options = {}) {
  const child = spawn(cmd, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    ...options,
  });
  children.push({ label, child });
  pipe(child, label, color);
  child.on('error', (error) => {
    warn(`${label} could not start: ${error.message}`);
  });
  return child;
}

function stopAll() {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const { child } of children) {
    if (child.exitCode !== null || child.signalCode !== null || !child.pid) continue;
    if (IS_WINDOWS) {
      // uvicorn's reloader and Next's compiler both fork. /T kills the tree.
      spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], {
        stdio: 'ignore',
      }).on('error', () => child.kill('SIGKILL'));
    } else {
      child.kill('SIGTERM');
    }
  }
}

for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(signal, () => {
    note('stopping…');
    stopAll();
    setTimeout(() => process.exit(0), 400);
  });
}
process.on('exit', stopAll);

// ------------------------------------------------------------------- startup

/** Is something already serving the Orqis health endpoint? */
async function backendIsUp(timeoutMs = 1500) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_URL}/health`, { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

async function startBackend() {
  if (await backendIsUp()) {
    note(`backend already running at ${API_URL} — reusing it`);
    return;
  }

  const backendDir = findBackendDir();
  if (!backendDir) {
    warn('could not find the Orqis backend (no folder containing backend/main.py).');
    warn('Set ORQIS_BACKEND_DIR in .env.local to the repository root.');
    warn('The website will start, but every screening call will report the backend as down.');
    return;
  }

  const { cmd, args } = pythonCommand();
  const uvicornArgs = [
    ...args,
    '-m',
    'uvicorn',
    'backend.main:app',
    '--host',
    '127.0.0.1',
    '--port',
    API_PORT,
  ];
  // Reload is opt-in: it is useful when editing Python, and it doubles the
  // process count, which makes shutdown noisier.
  if (process.env.ORQIS_API_RELOAD === '1') uvicornArgs.push('--reload');

  note(`backend  ${backendDir}`);
  const child = launch('api', COLOR.teal, cmd, uvicornArgs, {
    cwd: backendDir,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });

  child.on('exit', (code) => {
    if (shuttingDown) return;
    warn(
      `backend exited with code ${code}. The website keeps running; fix the backend ` +
        'and restart this command.',
    );
  });

  // Report readiness once, so the first thing in the log is an answerable question.
  const deadline = Date.now() + 60_000;
  const poll = async () => {
    if (shuttingDown) return;
    if (await backendIsUp(1000)) {
      process.stdout.write(
        `${COLOR.green}[orqis]${COLOR.reset} backend ready at ${API_URL}\n`,
      );
      return;
    }
    if (Date.now() < deadline) setTimeout(poll, 1000);
  };
  setTimeout(poll, 1200);
}

function startWeb() {
  // Invoked through Node directly rather than the `.cmd` shim, so there is no
  // shell between us and the process — signals and exit codes stay accurate.
  const nextBin = path.join(WEBSITE_DIR, 'node_modules', 'next', 'dist', 'bin', 'next');
  if (!existsSync(nextBin)) {
    warn('next is not installed. Run `npm install` first.');
    process.exitCode = 1;
    stopAll();
    return;
  }

  const child = launch(
    'web',
    COLOR.bold,
    process.execPath,
    [nextBin, MODE, '-p', WEB_PORT],
    { cwd: WEBSITE_DIR, env: { ...process.env } },
  );

  child.on('exit', (code) => {
    if (shuttingDown) return;
    process.exitCode = code ?? 0;
    note('website stopped — shutting the backend down too');
    stopAll();
    setTimeout(() => process.exit(process.exitCode ?? 0), 300);
  });
}

note(`mode ${MODE} • website :${WEB_PORT} • api ${API_URL}`);
await startBackend();
startWeb();
