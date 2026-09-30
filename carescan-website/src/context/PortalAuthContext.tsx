'use client';

/**
 * Session state for the clinic portal.
 *
 * The token lives in `sessionStorage` (see `lib/api.ts`), so a portal session dies
 * with the browser tab. On mount the stored token is re-validated against
 * `GET /api/auth/me` rather than trusted, because a token that expired overnight
 * should surface as a sign-in screen, not as a worklist that 401s halfway through
 * rendering.
 *
 * One deliberate exception: if that check fails because the *backend is unreachable*
 * the session is kept and `offline` is set. A clinician whose uvicorn process is
 * down should see "the backend is not running", not be silently signed out and left
 * guessing at their password. When the backend comes back the session is re-checked
 * automatically, so the portal heals without a reload.
 *
 * A token that expires mid-shift is handled the same way everywhere: `lib/api.ts`
 * broadcasts the 401 and this context signs out once, rather than each screen
 * inventing its own recovery from a rejected credential.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import {
  api,
  ApiError,
  clearToken,
  getStoredUser,
  getToken,
  onUnauthorized,
  setToken,
} from '@/lib/api';
import { useBackendStatus } from '@/context/BackendStatusContext';
import type { ClinicUser } from '@/lib/apiTypes';

export type AuthStatus = 'loading' | 'authenticated' | 'anonymous';

interface PortalAuthValue {
  status: AuthStatus;
  user: ClinicUser | null;
  /** True when the session was kept across an unreachable backend. */
  offline: boolean;
  /** Set when the server rejected a stored token, so the login form can say why. */
  expiredMessage: string | null;
  signIn: (email: string, password: string) => Promise<ClinicUser>;
  signOut: () => void;
  /** Clear `expiredMessage` once it has been shown. */
  acknowledgeExpiry: () => void;
}

const PortalAuthContext = createContext<PortalAuthValue | null>(null);

export const PortalAuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  // Not a lazy `useState` initializer: `sessionStorage` does not exist while this
  // route is prerendered, so seeding from it would make the first client render
  // disagree with the server HTML. The token can only be read once mounted.
  const [status, setStatus] = useState<AuthStatus>('loading');
  const [user, setUser] = useState<ClinicUser | null>(null);
  const [offline, setOffline] = useState(false);
  const [expiredMessage, setExpiredMessage] = useState<string | null>(null);
  const { recoveryToken } = useBackendStatus();

  useEffect(() => {
    if (!getToken()) {
      // Reading browser-only storage on mount is exactly what an effect is for;
      // there is no external system to subscribe to and no render to defer to.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setStatus('anonymous');
      return;
    }

    // Render the cached account immediately so the shell does not flash empty
    // while the round-trip completes.
    const cached = getStoredUser<ClinicUser>();
    if (cached) setUser(cached);

    const controller = new AbortController();

    api
      .me(controller.signal)
      .then((fresh) => {
        setUser(fresh);
        setOffline(false);
        setStatus('authenticated');
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return;

        if (error instanceof ApiError && error.isOffline && cached) {
          setOffline(true);
          setStatus('authenticated');
          return;
        }

        // A real rejection: the token is gone, expired, or the account was
        // deactivated. `request()` has already dropped it on a 401; clear again
        // so the other failure modes converge on the same signed-out state.
        clearToken();
        setUser(null);
        setStatus('anonymous');
        if (error instanceof ApiError && error.status === 401) {
          setExpiredMessage('Your session expired. Sign in again to continue.');
        }
      });

    return () => controller.abort();
    // `recoveryToken` re-validates a session that was kept across an outage.
  }, [recoveryToken]);

  // A 401 from any screen, not just the initial check. One listener, one sign-out.
  useEffect(
    () =>
      onUnauthorized(() => {
        setUser(null);
        setOffline(false);
        setStatus('anonymous');
        setExpiredMessage('Your session expired. Sign in again to continue.');
      }),
    [],
  );

  const signIn = useCallback(async (email: string, password: string) => {
    const response = await api.login(email, password);
    setToken(response.accessToken, response.user);
    setUser(response.user);
    setOffline(false);
    setExpiredMessage(null);
    setStatus('authenticated');
    return response.user;
  }, []);

  const signOut = useCallback(() => {
    clearToken();
    setUser(null);
    setOffline(false);
    setExpiredMessage(null);
    setStatus('anonymous');
  }, []);

  const acknowledgeExpiry = useCallback(() => setExpiredMessage(null), []);

  const value = useMemo<PortalAuthValue>(
    () => ({
      status,
      user,
      offline,
      expiredMessage,
      signIn,
      signOut,
      acknowledgeExpiry,
    }),
    [status, user, offline, expiredMessage, signIn, signOut, acknowledgeExpiry],
  );

  return <PortalAuthContext.Provider value={value}>{children}</PortalAuthContext.Provider>;
};

export function usePortalAuth(): PortalAuthValue {
  const context = useContext(PortalAuthContext);
  if (!context) {
    throw new Error('usePortalAuth must be used inside <PortalAuthProvider>.');
  }
  return context;
}
