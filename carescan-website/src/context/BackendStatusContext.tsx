'use client';

/**
 * One shared answer to "is the Orqis backend up?".
 *
 * Every screen in this app is a view onto a FastAPI service. Before this existed,
 * each one discovered a dead backend on its own, at the moment a clinician pressed
 * something, and recovering meant reloading the page by hand. That is the wrong way
 * round: the connection is ambient state, so it is polled in one place and
 * everything else reads it.
 *
 * Two behaviours matter more than the polling itself:
 *
 *  - **Recovery is automatic.** `recoveryToken` increments on every offline → online
 *    transition. Data screens include it in their fetch dependencies, so the moment
 *    uvicorn comes back the worklist refills without anyone touching the keyboard.
 *  - **A hidden tab costs nothing.** After one initial check, *repeat* polling pauses
 *    while the document is hidden and fires immediately when it is shown again, so
 *    returning to a parked tab shows current state rather than a stale badge. The
 *    first check always runs, so the status is never left indeterminate.
 *
 * This is liveness only. It never claims anything about model quality, and
 * `health.model_ready === false` is a perfectly healthy backend — see the System
 * screen, which states that distinction explicitly.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { api, ApiError } from '@/lib/api';
import type { HealthResponse } from '@/lib/apiTypes';

export type BackendStatus = 'checking' | 'online' | 'offline';

/** Steady state: often enough to notice, rare enough to ignore. */
const ONLINE_INTERVAL_MS = 30_000;
/** While down, poll hard — someone is almost certainly starting the service. */
const OFFLINE_INTERVAL_MS = 5_000;

interface BackendStatusValue {
  status: BackendStatus;
  /** Last successful `/health` payload. Kept while offline so the UI can still
   *  show which backend it lost contact with. */
  health: HealthResponse | null;
  /** Why the last check failed, with the backend's own wording where there is any. */
  error: ApiError | null;
  /** Increments on each offline → online transition. Use as a fetch dependency. */
  recoveryToken: number;
  lastCheckedAt: number | null;
  /** Force an immediate check. Safe to call from a click handler. */
  refresh: () => void;
}

const BackendStatusContext = createContext<BackendStatusValue | null>(null);

export const BackendStatusProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [status, setStatus] = useState<BackendStatus>('checking');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [recoveryToken, setRecoveryToken] = useState(0);
  const [lastCheckedAt, setLastCheckedAt] = useState<number | null>(null);

  // Refs, not state: the scheduler reads these on every tick and must not be a
  // reason to re-run effects or rebuild callbacks.
  const statusRef = useRef<BackendStatus>('checking');
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlightRef = useRef(false);
  const aliveRef = useRef(true);
  const runRef = useRef<() => void>(() => {});
  /** Whether any check has completed. Gates the hidden-tab pause — see `check`. */
  const everCheckedRef = useRef(false);

  const schedule = useCallback((delayMs?: number) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (!aliveRef.current) return;
    const delay =
      delayMs ??
      (statusRef.current === 'online' ? ONLINE_INTERVAL_MS : OFFLINE_INTERVAL_MS);
    timerRef.current = setTimeout(() => runRef.current(), delay);
  }, []);

  const check = useCallback(
    async (force = false) => {
      // Pausing on a hidden tab is about not *repeating* round trips on something
      // nobody is looking at. It must not swallow the first one: a page opened in
      // a background tab would then sit at "Checking…" forever — an indeterminate
      // label that never resolves — and every screen waiting on the status would
      // wait with it. One request on mount costs nothing; the repeats are what
      // this saves. `force` is the manual refresh, which must always do what the
      // button says.
      if (
        !force &&
        everCheckedRef.current &&
        typeof document !== 'undefined' &&
        document.hidden
      ) {
        schedule();
        return;
      }
      if (inFlightRef.current) return;
      inFlightRef.current = true;

      try {
        const next = await api.ping();
        if (!aliveRef.current) return;
        const recovered = statusRef.current === 'offline';
        statusRef.current = 'online';
        setHealth(next);
        setError(null);
        setStatus('online');
        setLastCheckedAt(Date.now());
        if (recovered) setRecoveryToken((n) => n + 1);
      } catch (caught) {
        if (!aliveRef.current) return;
        statusRef.current = 'offline';
        setStatus('offline');
        setLastCheckedAt(Date.now());
        setError(
          caught instanceof ApiError
            ? caught
            : new ApiError('The Orqis backend did not answer.', 0, true),
        );
      } finally {
        everCheckedRef.current = true;
        inFlightRef.current = false;
        schedule();
      }
    },
    [schedule],
  );

  useEffect(() => {
    aliveRef.current = true;
    // `check` is stable, but the indirection is what lets `schedule` fire the
    // latest one without the two callbacks depending on each other.
    runRef.current = () => void check();

    // Deferred rather than awaited here: the first poll is a side effect on an
    // external service, not part of rendering this provider.
    schedule(0);

    const onVisible = () => {
      if (!document.hidden) {
        // Whatever was shown while parked is now suspect. Re-check at once.
        schedule(0);
      }
    };
    const onOnline = () => schedule(0);

    document.addEventListener('visibilitychange', onVisible);
    window.addEventListener('online', onOnline);
    window.addEventListener('focus', onVisible);

    return () => {
      aliveRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
      document.removeEventListener('visibilitychange', onVisible);
      window.removeEventListener('online', onOnline);
      window.removeEventListener('focus', onVisible);
    };
    // `check` and `schedule` are stable; this runs once per mount on purpose.
  }, [check, schedule]);

  // A click on the pill is an explicit "check now", so it bypasses the hidden-tab
  // pause rather than being quietly dropped.
  const refresh = useCallback(() => void check(true), [check]);

  const value = useMemo<BackendStatusValue>(
    () => ({ status, health, error, recoveryToken, lastCheckedAt, refresh }),
    [status, health, error, recoveryToken, lastCheckedAt, refresh],
  );

  return (
    <BackendStatusContext.Provider value={value}>{children}</BackendStatusContext.Provider>
  );
};

export function useBackendStatus(): BackendStatusValue {
  const context = useContext(BackendStatusContext);
  if (!context) {
    throw new Error('useBackendStatus must be used inside <BackendStatusProvider>.');
  }
  return context;
}
