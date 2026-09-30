/**
 * HTTP client for the Orqis FastAPI backend.
 *
 * Design notes
 * ------------
 * - By default every call goes to this site's own origin, through the proxy in
 *   `app/api/orqis/[...path]/route.ts`, which forwards to the backend server-side.
 *   That removes CORS entirely and keeps the backend address off the client, so the
 *   site works from a phone on the same network or from a real deployment — not
 *   only from the one machine where `localhost:8000` happens to mean something.
 *   Setting `NEXT_PUBLIC_API_BASE_URL` opts back into calling the backend directly.
 * - Every failure becomes an {@link ApiError} carrying the backend's own `detail`
 *   string. Those strings are written to be shown to a user (a poor photograph
 *   reads as "retake it"), so they are surfaced verbatim rather than replaced with
 *   a generic message. Only genuinely unexpected faults get a fallback.
 * - The auth token lives in `sessionStorage`, not `localStorage`: a clinical
 *   portal session should not outlive the browser tab.
 * - Nothing here retries. A screening that failed should be re-submitted by a
 *   person who saw the error, not silently re-sent.
 */

import type {
  AnalyzeRequest,
  Assessment,
  AssessmentResult,
  ClinicPatientsResponse,
  ClinicStats,
  FhirBundle,
  HealthResponse,
  HistoryEntry,
  LocalizationResult,
  LoginResponse,
  ModelInfo,
  UploadResponse,
  WorklistResponse,
} from './apiTypes';

/**
 * Where the backend lives, from the browser's point of view.
 *
 * Unset (the default) means "this origin, via the server-side proxy". A value
 * means "talk to that host directly" — useful when the backend is already exposed
 * on a public URL and the extra hop is pointless.
 */
export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || '')
  .trim()
  .replace(/\/+$/, '') || '/api/orqis';

/** True when requests take the same-origin proxy rather than a direct host. */
export const USES_PROXY = API_BASE_URL === '/api/orqis';

const TOKEN_KEY = 'orqis_portal_token';
const USER_KEY = 'orqis_portal_user';

/** How the proxy labels "the backend never answered", as opposed to a real 503. */
const PROXY_UNREACHABLE_HEADER = 'x-orqis-proxy';

/** Human-readable description of the backend target, for error copy. */
function backendLabel(): string {
  return USES_PROXY ? 'the Orqis backend' : `the Orqis backend at ${API_BASE_URL}`;
}

const OFFLINE_HINT =
  'Start it with `npm run dev` from the website folder — that boots the backend too.';

/** An error carrying the backend's status code and its user-facing detail. */
export class ApiError extends Error {
  readonly status: number;
  /** True when the request never reached the server (backend down, CORS, DNS). */
  readonly isNetworkError: boolean;

  constructor(message: string, status: number, isNetworkError = false) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.isNetworkError = isNetworkError;
  }

  /** The backend is not running or not reachable from the browser. */
  get isOffline(): boolean {
    return this.isNetworkError;
  }
}

// ------------------------------------------------------------------- tokens

export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string, user?: unknown): void {
  if (typeof window === 'undefined') return;
  try {
    window.sessionStorage.setItem(TOKEN_KEY, token);
    if (user !== undefined) {
      window.sessionStorage.setItem(USER_KEY, JSON.stringify(user));
    }
  } catch {
    // Private-mode storage failure. The token stays in memory for this page only.
  }
}

export function getStoredUser<T>(): T | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.sessionStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

export function clearToken(): void {
  if (typeof window === 'undefined') return;
  try {
    window.sessionStorage.removeItem(TOKEN_KEY);
    window.sessionStorage.removeItem(USER_KEY);
  } catch {
    // ignore
  }
}

// -------------------------------------------------------------- core fetch

/**
 * Pull the human-readable message out of a FastAPI error body.
 *
 * FastAPI returns `{"detail": "..."}` for raised HTTPExceptions and
 * `{"detail": [{...}, ...]}` for Pydantic validation failures. The second shape
 * is developer-facing, so it collapses to a short generic line rather than
 * dumping a validation trace at a clinician.
 */
async function extractDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    const detail = (body as { detail?: unknown })?.detail;
    if (typeof detail === 'string' && detail.trim()) return detail;
    if (Array.isArray(detail)) return 'The request was rejected as invalid.';
  } catch {
    // Not JSON, or an empty body. Fall through to the status-based message.
  }
  return `Request failed (HTTP ${response.status}).`;
}

/**
 * Turn a non-2xx response into the {@link ApiError} to throw.
 *
 * A 503 stamped by our own proxy means the request never reached the backend, so
 * it is reported as an offline error — the UI already knows how to render that,
 * and "the service is down" is a very different instruction to a clinician than
 * "the server rejected this".
 */
async function errorFor(response: Response): Promise<ApiError> {
  const proxied = response.headers.get(PROXY_UNREACHABLE_HEADER) === 'unreachable';
  return new ApiError(await extractDetail(response), response.status, proxied);
}

/** The request never left the browser: the origin itself is unreachable. */
function networkError(): ApiError {
  return new ApiError(`Could not reach ${backendLabel()}. ${OFFLINE_HINT}`, 0, true);
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  /** Attach the stored bearer token. Default true. */
  auth?: boolean;
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, auth = true, signal } = options;

  const headers: Record<string, string> = { Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';

  if (auth) {
    const token = getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
      cache: 'no-store',
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw networkError();
  }

  if (response.status === 401 && auth) {
    // The token is gone or expired. Drop it so the UI falls back to signed-out
    // state instead of looping on a credential the server has already rejected.
    clearToken();
    notifyUnauthorized();
  }

  if (!response.ok) {
    throw await errorFor(response);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// ------------------------------------------------------- session expiry signal

/**
 * A 401 can arrive from any screen at any time — a token quietly expires mid-shift
 * and the next fetch fails. Rather than every caller reimplementing "sign out and
 * send them to the login form", the client broadcasts it once and the auth context
 * listens. Nothing else in this module knows the portal exists.
 */
type UnauthorizedListener = () => void;
const unauthorizedListeners = new Set<UnauthorizedListener>();

export function onUnauthorized(listener: UnauthorizedListener): () => void {
  unauthorizedListeners.add(listener);
  return () => unauthorizedListeners.delete(listener);
}

function notifyUnauthorized(): void {
  unauthorizedListeners.forEach((listener) => {
    try {
      listener();
    } catch {
      // A broken listener must not take down the request that triggered it.
    }
  });
}

// ------------------------------------------------------------------ system

export const api = {
  health: (signal?: AbortSignal) =>
    request<HealthResponse>('/health', { auth: false, signal }),

  modelInfo: (signal?: AbortSignal) =>
    request<ModelInfo>('/api/model/info', { auth: false, signal }),

  /**
   * Health check with its own deadline.
   *
   * The status poller must always settle: a backend that accepts the connection
   * and then stops talking would otherwise leave the indicator on "checking"
   * forever, which reads as "fine" to anyone glancing at it.
   */
  async ping(timeoutMs = 6000): Promise<HealthResponse> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await request<HealthResponse>('/health', {
        auth: false,
        signal: controller.signal,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new ApiError(
          `${backendLabel()} did not answer a health check within ${Math.round(
            timeoutMs / 1000,
          )}s.`,
          0,
          true,
        );
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  },

  // ------------------------------------------------------------ screening

  /**
   * Upload an intra-oral photograph and get back a server-side path.
   *
   * Sent as multipart/form-data, so `Content-Type` is deliberately left unset --
   * the browser has to add its own multipart boundary.
   */
  async upload(
    file: File,
    patientId?: string | null,
    signal?: AbortSignal,
  ): Promise<UploadResponse> {
    const form = new FormData();
    form.append('file', file);
    if (patientId) form.append('patient_id', patientId);

    const headers: Record<string, string> = {};
    const token = getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    let response: Response;
    try {
      response = await fetch(`${API_BASE_URL}/api/screening/upload`, {
        method: 'POST',
        headers,
        body: form,
        signal,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') throw error;
      throw networkError();
    }
    if (response.status === 401) {
      clearToken();
      notifyUnauthorized();
    }
    if (!response.ok) throw await errorFor(response);
    return (await response.json()) as UploadResponse;
  },

  analyze: (payload: AnalyzeRequest, signal?: AbortSignal) =>
    request<AssessmentResult>('/api/screening/analyze', {
      method: 'POST',
      body: payload,
      signal,
    }),

  /**
   * Locate the lesion ROI for a visual overlay. Advisory: a 503 here means no
   * localiser artifact is loaded, which must never block a screening.
   */
  async localize(file: File, signal?: AbortSignal): Promise<LocalizationResult> {
    const form = new FormData();
    form.append('file', file);

    let response: Response;
    try {
      response = await fetch(`${API_BASE_URL}/api/localize`, {
        method: 'POST',
        body: form,
        signal,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') throw error;
      throw networkError();
    }
    if (!response.ok) throw await errorFor(response);
    return (await response.json()) as LocalizationResult;
  },

  screening: (id: string, signal?: AbortSignal) =>
    request<Assessment>(`/api/screening/${encodeURIComponent(id)}`, { signal }),

  result: (id: string, signal?: AbortSignal) =>
    request<AssessmentResult>(`/api/results/${encodeURIComponent(id)}`, { signal }),

  history: (patientId: string, signal?: AbortSignal) =>
    request<HistoryEntry[]>(
      `/api/patients/${encodeURIComponent(patientId)}/history`,
      { signal },
    ),

  fhir: (screeningId: string, signal?: AbortSignal) =>
    request<FhirBundle>(
      `/api/screening/${encodeURIComponent(screeningId)}/fhir`,
      { signal },
    ),

  /** URL of the stored capture. Used as an `<img src>`, not fetched here. */
  imageUrl: (screeningId: string) =>
    `${API_BASE_URL}/api/screening/${encodeURIComponent(screeningId)}/image`,

  // --------------------------------------------------------------- portal

  login: (email: string, password: string) =>
    request<LoginResponse>('/api/auth/login', {
      method: 'POST',
      body: { email, password },
      auth: false,
    }),

  me: (signal?: AbortSignal) => request<LoginResponse['user']>('/api/auth/me', { signal }),

  worklist: (
    clinicId: string,
    params: { limit?: number; offset?: number; risk?: string } = {},
    signal?: AbortSignal,
  ) => {
    const query = new URLSearchParams();
    if (params.limit != null) query.set('limit', String(params.limit));
    if (params.offset != null) query.set('offset', String(params.offset));
    if (params.risk) query.set('risk', params.risk);
    const suffix = query.toString() ? `?${query}` : '';
    return request<WorklistResponse>(
      `/api/clinics/${encodeURIComponent(clinicId)}/screenings${suffix}`,
      { signal },
    );
  },

  clinicPatients: (
    clinicId: string,
    params: { limit?: number; offset?: number } = {},
    signal?: AbortSignal,
  ) => {
    const query = new URLSearchParams();
    if (params.limit != null) query.set('limit', String(params.limit));
    if (params.offset != null) query.set('offset', String(params.offset));
    const suffix = query.toString() ? `?${query}` : '';
    return request<ClinicPatientsResponse>(
      `/api/clinics/${encodeURIComponent(clinicId)}/patients${suffix}`,
      { signal },
    );
  },

  clinicStats: (clinicId: string, signal?: AbortSignal) =>
    request<ClinicStats>(`/api/clinics/${encodeURIComponent(clinicId)}/stats`, { signal }),
};
