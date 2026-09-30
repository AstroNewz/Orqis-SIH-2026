'use client';

/**
 * Chrome and route guard for every signed-in portal screen.
 *
 * The guard redirects rather than rendering a "403" page: a clinician who has been
 * signed out by an expired token wants the sign-in form, not an explanation. While
 * the stored token is being re-validated nothing is rendered except a spinner, so a
 * worklist never flashes before being replaced by a login screen.
 */

import React, { useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname, useRouter } from 'next/navigation';
import {
  Activity,
  ArrowLeft,
  LayoutDashboard,
  Loader2,
  LogOut,
  PlugZap,
  ScanLine,
  Users,
} from 'lucide-react';
import { usePortalAuth } from '@/context/PortalAuthContext';
import { useBackendStatus } from '@/context/BackendStatusContext';
import { BackendStatusPill } from '@/components/ui/BackendStatusPill';

const NAV_ITEMS = [
  { href: '/portal/worklist', label: 'Worklist', icon: LayoutDashboard },
  { href: '/portal/screening/new', label: 'New screening', icon: ScanLine },
  { href: '/portal/patients', label: 'Patients', icon: Users },
  { href: '/portal/system', label: 'System', icon: Activity },
];

export const PortalShell: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { status, user, offline, signOut } = usePortalAuth();
  const { status: backendStatus, error: backendError } = useBackendStatus();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (status === 'anonymous') router.replace('/portal/login');
  }, [status, router]);

  if (status !== 'authenticated') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-stone-50 dark:bg-slate-950">
        <div className="flex items-center gap-2.5 text-slate-500 dark:text-slate-400">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm font-medium">
            {status === 'loading' ? 'Checking your session…' : 'Redirecting to sign in…'}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-stone-50 dark:bg-slate-950">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="h-16 flex items-center justify-between gap-3">
            <Link href="/portal/worklist" className="flex items-center gap-2.5 min-w-0">
              <Image
                src="/orqis-logo.png"
                alt=""
                width={32}
                height={32}
                className="w-8 h-8 shrink-0"
              />
              <span className="min-w-0">
                <span className="block text-sm font-bold text-slate-900 dark:text-slate-50 leading-tight">
                  Orqis Clinic Portal
                </span>
                <span className="block text-[11px] text-slate-500 dark:text-slate-400 font-mono truncate">
                  {user?.clinicId ?? '—'}
                </span>
              </span>
            </Link>

            <div className="flex items-center gap-2 sm:gap-3 shrink-0">
              <BackendStatusPill compact />

              <Link
                href="/"
                className="hidden lg:inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 dark:text-slate-400 hover:text-teal-700 dark:hover:text-teal-300 transition-colors"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                Research site
              </Link>

              <span className="hidden md:block text-right">
                <span className="block text-xs font-semibold text-slate-900 dark:text-slate-100 leading-tight">
                  {user?.fullName || user?.email}
                </span>
                <span className="block text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  {user?.role}
                </span>
              </span>

              <button
                type="button"
                onClick={signOut}
                className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors cursor-pointer"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Sign out</span>
              </button>
            </div>
          </div>

          {/* Nav
              Wraps rather than scrolls. `overflow-x-auto` put the fourth tab
              ("System") entirely off-screen at 375px with no visual cue that the
              strip scrolled, so on a phone it was undiscoverable. Wrapping costs
              one extra row at narrow widths and stays a single row wherever the
              tabs fit, which is every width above ~430px. */}
          <nav className="flex flex-wrap items-center gap-1 -mx-1 px-1 pb-2">
            {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
              const active = pathname === href || pathname.startsWith(`${href}/`);
              return (
                <Link
                  key={href}
                  href={href}
                  aria-current={active ? 'page' : undefined}
                  className={`inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-2 rounded-xl whitespace-nowrap transition-colors ${
                    active
                      ? 'bg-teal-50 dark:bg-teal-950/60 text-teal-800 dark:text-teal-200 border border-teal-200 dark:border-teal-800'
                      : 'text-slate-600 dark:text-slate-400 border border-transparent hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>

      {(backendStatus === 'offline' || offline) && (
        <div className="bg-amber-50 dark:bg-amber-950/50 border-b border-amber-200 dark:border-amber-900">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2 flex items-start gap-2 text-xs text-amber-900 dark:text-amber-200">
            <PlugZap className="w-4 h-4 shrink-0 mt-0.5" />
            <span className="leading-relaxed">
              <strong className="font-bold">The Orqis backend is not answering.</strong>{' '}
              Your session is kept and this page is retrying every few seconds — it will
              refill on its own once the service is back. Nothing you see below is newer
              than the last successful fetch.
              {backendError?.message ? (
                <span className="block mt-0.5 opacity-80">{backendError.message}</span>
              ) : null}
            </span>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6">
        {children}
      </main>

      <footer className="border-t border-slate-200 dark:border-slate-800 mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed max-w-3xl">
            Orqis is AI-assisted preliminary decision support for oral lesion triage. It
            does not provide a diagnosis and does not replace clinical examination or
            histopathology. Every band shown here is produced by the backend; nothing is
            scored in this browser.
          </p>
        </div>
      </footer>
    </div>
  );
};
