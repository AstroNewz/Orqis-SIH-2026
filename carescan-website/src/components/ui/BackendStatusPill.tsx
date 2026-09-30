'use client';

/**
 * Live backend connection indicator.
 *
 * Deliberately small and always present rather than a modal that appears on
 * failure: the honest answer to "is this thing talking to the server?" should be
 * visible before something goes wrong, not only after. Clicking re-checks
 * immediately, which is what everyone tries anyway.
 */

import React from 'react';
import { Loader2, PlugZap, Wifi } from 'lucide-react';
import { useBackendStatus } from '@/context/BackendStatusContext';

export const BackendStatusPill: React.FC<{
  /** Hide the label under `sm`, for tight headers. */
  compact?: boolean;
  className?: string;
}> = ({ compact = false, className = '' }) => {
  const { status, health, error, lastCheckedAt, refresh } = useBackendStatus();

  const label =
    status === 'online' ? 'Backend live' : status === 'offline' ? 'Backend down' : 'Checking…';

  const tone =
    status === 'online'
      ? 'border-emerald-200 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950/50 text-emerald-800 dark:text-emerald-300'
      : status === 'offline'
        ? 'border-rose-200 dark:border-rose-900 bg-rose-50 dark:bg-rose-950/50 text-rose-800 dark:text-rose-300'
        : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-600 dark:text-slate-400';

  const detail =
    status === 'online'
      ? `${health?.service ?? 'Orqis backend'} ${health?.version ?? ''} · ${
          health?.environment ?? 'unknown environment'
        }`
      : status === 'offline'
        ? (error?.message ?? 'The backend did not answer.')
        : 'Contacting the Orqis backend…';

  const checked = lastCheckedAt
    ? new Date(lastCheckedAt).toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
    : null;

  return (
    <button
      type="button"
      onClick={refresh}
      title={checked ? `${detail}\nLast checked ${checked}. Click to re-check.` : detail}
      aria-label={`${label}. ${detail}`}
      className={`inline-flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-full border transition-colors cursor-pointer ${tone} ${className}`}
    >
      {status === 'checking' ? (
        <Loader2 className="w-3 h-3 animate-spin shrink-0" />
      ) : status === 'online' ? (
        <Wifi className="w-3 h-3 shrink-0" />
      ) : (
        <PlugZap className="w-3 h-3 shrink-0" />
      )}
      <span className={compact ? 'hidden sm:inline' : ''}>{label}</span>
      {/* A pulsing dot only while down — a steady state should not blink at anyone. */}
      {status === 'offline' && (
        <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse shrink-0" />
      )}
    </button>
  );
};
