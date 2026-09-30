/**
 * Same-origin proxy to the Orqis FastAPI backend.
 *
 * Why this exists
 * ---------------
 * The portal used to call `http://localhost:8000` straight from the browser. That
 * works on exactly one machine: open the site from a phone on the same Wi-Fi, or
 * deploy it anywhere, and `localhost` resolves to the *viewer's* machine and every
 * request fails. It also made the backend address a build-time constant baked into
 * the browser bundle.
 *
 * Routing through the Next server instead means:
 *  - the browser only ever talks to its own origin, so CORS never applies;
 *  - the backend address is a server-side variable (`ORQIS_API_URL`) that can be
 *    changed without rebuilding, and is never shipped to the client;
 *  - "is the backend up?" has a single answer the whole app agrees on.
 *
 * A backend that is down must not look like a mysterious 500. When the connection
 * is refused this returns 503 with `x-orqis-proxy: unreachable`, which `lib/api.ts`
 * turns back into an offline `ApiError` — the same state the UI already renders for
 * a failed direct fetch.
 */

import { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/** Server-side only. Not `NEXT_PUBLIC_*`, so it never reaches the browser. */
const BACKEND_URL = (process.env.ORQIS_API_URL || 'http://127.0.0.1:8000').replace(
  /\/+$/,
  '',
);

/**
 * Generous, because an upload plus a full inference pass is the slowest thing the
 * portal does. Still bounded: a hung backend should surface as an error a clinician
 * can act on rather than a spinner that never resolves.
 */
const TIMEOUT_MS = 120_000;

/**
 * Connection-level headers belong to a single hop and must not be relayed.
 * `accept-encoding` and `content-length` are dropped so the fetch implementation
 * can set them correctly for the body it actually sends.
 */
const HOP_BY_HOP = new Set([
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
  'host',
  'content-length',
  'accept-encoding',
]);

function forwardableHeaders(source: Headers): Headers {
  const headers = new Headers();
  source.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) headers.set(key, value);
  });
  return headers;
}

/** The backend did not answer. Shaped like a FastAPI error so the client can read it. */
function unreachable(detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status: 503,
    headers: {
      'Content-Type': 'application/json',
      // The marker `lib/api.ts` looks for to classify this as offline rather than
      // as a backend that answered with a 503 of its own.
      'x-orqis-proxy': 'unreachable',
      'Cache-Control': 'no-store',
    },
  });
}

async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path?: string[] }> },
): Promise<Response> {
  const { path = [] } = await context.params;
  const target = `${BACKEND_URL}/${path.map(encodeURIComponent).join('/')}${
    request.nextUrl.search
  }`;

  const method = request.method.toUpperCase();
  const hasBody = method !== 'GET' && method !== 'HEAD';

  // Buffered rather than streamed: uploads are capped at 25 MB by the backend, and
  // buffering avoids the half-duplex streaming support that varies across runtimes.
  let body: ArrayBuffer | undefined;
  if (hasBody) {
    try {
      body = await request.arrayBuffer();
    } catch {
      return new Response(
        JSON.stringify({ detail: 'The request body could not be read.' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } },
      );
    }
  }

  // Abort on either the client giving up or the deadline passing.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  const onClientAbort = () => controller.abort();
  request.signal.addEventListener('abort', onClientAbort);

  try {
    const upstream = await fetch(target, {
      method,
      headers: forwardableHeaders(request.headers),
      body,
      signal: controller.signal,
      redirect: 'manual',
      cache: 'no-store',
    });

    const headers = new Headers();
    upstream.headers.forEach((value, key) => {
      const name = key.toLowerCase();
      // `content-encoding`/`content-length` describe the body fetch already decoded.
      if (HOP_BY_HOP.has(name) || name === 'content-encoding') return;
      headers.set(key, value);
    });
    headers.set('Cache-Control', 'no-store');

    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers,
    });
  } catch (error) {
    if (request.signal.aborted) {
      // The browser navigated away or cancelled. Nothing to report.
      return new Response(null, { status: 499 });
    }
    if (error instanceof DOMException && error.name === 'AbortError') {
      return unreachable(
        `The Orqis backend at ${BACKEND_URL} did not respond within ${
          TIMEOUT_MS / 1000
        } seconds.`,
      );
    }
    return unreachable(
      `Could not reach the Orqis backend at ${BACKEND_URL}. Start it with ` +
        '`npm run dev` from the website folder, or ' +
        '`python -m uvicorn backend.main:app --port 8000` from the Orqis repo.',
    );
  } finally {
    clearTimeout(timer);
    request.signal.removeEventListener('abort', onClientAbort);
  }
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const HEAD = proxy;
export const OPTIONS = proxy;
