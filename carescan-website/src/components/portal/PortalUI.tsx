'use client';

/**
 * Small shared pieces for the clinic portal.
 *
 * Loading, empty and error are first-class states here rather than afterthoughts:
 * every list screen in the portal renders one of them, so they are defined once and
 * behave identically everywhere. An error never shows a raw exception -- `ApiError`
 * already carries the backend's own user-facing `detail` string.
 */

import React from 'react';
import Link from 'next/link';
import { AlertTriangle, PlugZap, RefreshCw } from 'lucide-react';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';
import { ApiError } from '@/lib/api';
import { badgeVariantForTone, bandTone } from '@/lib/verdict';

// ------------------------------------------------------------------ band chip

/** The displayed risk band. `isMock` is loud on purpose -- it is not real data. */
export const BandChip: React.FC<{
  band: string | null | undefined;
  isMock?: boolean;
  size?: 'sm' | 'md';
}> = ({ band, isMock = false, size = 'sm' }) => {
  if (!band) {
    return (
      <Badge variant="neutral" size={size}>
        No result
      </Badge>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5">
      <Badge variant={badgeVariantForTone(bandTone(band))} size={size}>
        {band}
      </Badge>
      {isMock && (
        <Badge variant="warning" size="sm">
          TEST DATA
        </Badge>
      )}
    </span>
  );
};

// ----------------------------------------------------------------- short ids

/**
 * Patient and screening identifiers are UUIDs. Showing the whole thing in a table
 * cell is unreadable, so the first segment is shown and the full value stays in the
 * title attribute and on the detail page.
 */
export const ShortId: React.FC<{ value: string; className?: string }> = ({
  value,
  className = '',
}) => (
  <span
    title={value}
    className={`font-mono text-xs text-slate-700 dark:text-slate-300 ${className}`}
  >
    {value.slice(0, 8)}
  </span>
);

// --------------------------------------------------------------- date format

/** Locale-stable timestamp. Invalid or absent input renders an em dash, not "Invalid Date". */
export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// --------------------------------------------------------------- list states

export const SkeletonRows: React.FC<{ rows?: number; className?: string }> = ({
  rows = 5,
  className = '',
}) => (
  <div className={`space-y-2 ${className}`} aria-busy="true" aria-live="polite">
    <span className="sr-only">Loading…</span>
    {Array.from({ length: rows }).map((_, i) => (
      <div
        key={i}
        className="h-14 rounded-2xl bg-slate-100 dark:bg-slate-800/70 animate-pulse"
      />
    ))}
  </div>
);

export const ErrorState: React.FC<{
  error: unknown;
  onRetry?: () => void;
  className?: string;
}> = ({ error, onRetry, className = '' }) => {
  const offline = error instanceof ApiError && error.isOffline;
  const message =
    error instanceof Error ? error.message : 'Something went wrong loading this view.';

  return (
    <Card
      variant="white"
      padding="md"
      className={`border-rose-200 dark:border-rose-900/70 ${className}`}
    >
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-2xl bg-rose-100 dark:bg-rose-950/70 text-rose-700 dark:text-rose-300 flex items-center justify-center shrink-0">
          {offline ? <PlugZap className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
        </div>
        <div className="min-w-0 space-y-2">
          <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
            {offline ? 'The Orqis backend is not reachable' : 'This view could not load'}
          </h3>
          <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed break-words">
            {message}
          </p>
          {onRetry && (
            <Button
              variant="outline"
              size="sm"
              icon={<RefreshCw className="w-3.5 h-3.5" />}
              iconPosition="left"
              onClick={onRetry}
            >
              Try again
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
};

export const EmptyState: React.FC<{
  icon: React.ReactNode;
  title: string;
  message: string;
  actionLabel?: string;
  actionHref?: string;
}> = ({ icon, title, message, actionLabel, actionHref }) => (
  <Card variant="stone" padding="lg" className="text-center">
    <div className="w-12 h-12 rounded-3xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-teal-700 dark:text-teal-300 flex items-center justify-center mx-auto mb-3">
      {icon}
    </div>
    <h3 className="text-base font-bold text-slate-900 dark:text-slate-100 mb-1">{title}</h3>
    <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 max-w-md mx-auto leading-relaxed">
      {message}
    </p>
    {actionLabel && actionHref && (
      <Link href={actionHref} className="inline-block mt-4">
        <Button variant="secondary" size="sm">
          {actionLabel}
        </Button>
      </Link>
    )}
  </Card>
);

// ---------------------------------------------------------------- stat tiles

/**
 * A dashboard counter. These describe what a clinic has captured -- they are
 * throughput, never model performance, and the portal must not imply otherwise.
 */
export const StatTile: React.FC<{
  label: string;
  value: React.ReactNode;
  hint?: string;
  accent?: 'teal' | 'indigo' | 'amber' | 'rose' | 'emerald' | 'slate';
}> = ({ label, value, hint, accent = 'slate' }) => {
  const accents: Record<string, string> = {
    teal: 'text-teal-700 dark:text-teal-300',
    indigo: 'text-indigo-700 dark:text-indigo-300',
    amber: 'text-amber-700 dark:text-amber-300',
    rose: 'text-rose-700 dark:text-rose-300',
    emerald: 'text-emerald-700 dark:text-emerald-300',
    slate: 'text-slate-900 dark:text-slate-100',
  };

  return (
    <Card variant="white" padding="sm">
      <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        {label}
      </p>
      <p className={`text-2xl font-bold mt-1 ${accents[accent]}`}>{value}</p>
      {hint && (
        <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-snug">
          {hint}
        </p>
      )}
    </Card>
  );
};

// -------------------------------------------------------------- page heading

export const PageHeading: React.FC<{
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}> = ({ title, subtitle, actions }) => (
  <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
    <div>
      <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-slate-50 tracking-tight">
        {title}
      </h1>
      {subtitle && (
        <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 mt-1 max-w-2xl leading-relaxed">
          {subtitle}
        </p>
      )}
    </div>
    {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
  </div>
);
