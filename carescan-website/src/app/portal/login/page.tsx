'use client';

/**
 * Clinic sign-in.
 *
 * There is no sign-up link because the backend has no sign-up route: the first
 * account is seeded from `CLINIC_ADMIN_EMAIL` / `CLINIC_ADMIN_PASSWORD` at startup
 * and further accounts are an operator task. The failure text repeats the backend's
 * own wording, which is deliberately identical for an unknown email, a wrong
 * password and a deactivated account so the form cannot be used to discover which
 * addresses have accounts.
 */

import React, { useEffect, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  Eye,
  EyeOff,
  Loader2,
  Lock,
  Mail,
  TimerOff,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { BackendStatusPill } from '@/components/ui/BackendStatusPill';
import { usePortalAuth } from '@/context/PortalAuthContext';
import { ApiError } from '@/lib/api';

export default function PortalLoginPage() {
  const { status, signIn, expiredMessage, acknowledgeExpiry } = usePortalAuth();
  const router = useRouter();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [revealPassword, setRevealPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (status === 'authenticated') router.replace('/portal/worklist');
  }, [status, router]);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (submitting) return;

    setError(null);
    // The expiry notice explained why this form is on screen. Once the clinician
    // acts on it, it has done its job.
    acknowledgeExpiry();
    setSubmitting(true);
    try {
      await signIn(email.trim(), password);
      router.replace('/portal/worklist');
    } catch (caught) {
      // `ApiError.message` already carries the backend's own wording for a bad
      // credential and the client's own wording for an unreachable service, so
      // it is surfaced as-is rather than rewritten here.
      setError(
        caught instanceof ApiError ? caught.message : 'Sign-in failed. Please try again.',
      );
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen grid lg:grid-cols-2 bg-stone-50 dark:bg-slate-950">
      {/* Left: identity panel */}
      <div className="relative hidden lg:flex flex-col justify-between p-10 xl:p-14 bg-gradient-to-br from-teal-700 via-teal-800 to-slate-900 text-white overflow-hidden">
        <div className="absolute -top-24 -right-24 w-96 h-96 rounded-full bg-teal-400/20 blur-3xl" />
        <div className="absolute -bottom-32 -left-20 w-96 h-96 rounded-full bg-indigo-500/20 blur-3xl" />

        <div className="relative flex items-center gap-3">
          <Image
            src="/orqis-logo.png"
            alt=""
            width={40}
            height={40}
            className="w-10 h-10 rounded-xl bg-white/10 p-1"
          />
          <span className="text-lg font-bold tracking-tight">Orqis</span>
        </div>

        <div className="relative space-y-5 max-w-md">
          <Badge variant="teal" className="bg-white/15 text-white border-white/25">
            Clinic &amp; Hospital Portal
          </Badge>
          <h1 className="text-3xl xl:text-4xl font-bold leading-tight tracking-tight">
            Triage support that shows its working.
          </h1>
          <p className="text-sm text-teal-50/90 leading-relaxed">
            Capture an intra-oral photograph, let the frozen localizer find the lesion,
            and read a band set by the validated classical baseline. The quantum feature
            map is reported alongside it — clearly labelled, never as the verdict.
          </p>
          <ul className="space-y-2 text-xs text-teal-50/80">
            <li className="flex gap-2">
              <span className="text-teal-300">—</span>
              Every score is computed by the backend, never in the browser
            </li>
            <li className="flex gap-2">
              <span className="text-teal-300">—</span>
              Patients are pseudonymous: an identifier, counts and timestamps
            </li>
            <li className="flex gap-2">
              <span className="text-teal-300">—</span>
              Results export as HL7 FHIR R4 Observation and RiskAssessment
            </li>
          </ul>
        </div>

        <p className="relative text-[11px] text-teal-100/70 max-w-md leading-relaxed">
          Investigational decision support. Not a diagnosis, and not a replacement for
          clinical examination or histopathology.
        </p>
      </div>

      {/* Right: form */}
      <div className="flex flex-col justify-center px-5 sm:px-10 lg:px-14 py-12">
        <div className="w-full max-w-sm mx-auto">
          <div className="flex items-center justify-between gap-3 mb-8">
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 dark:text-slate-400 hover:text-teal-700 dark:hover:text-teal-300 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Back to the research site
            </Link>
            {/* Whether the service is up is the first thing worth knowing here: a
                failed sign-in against a dead backend is not a wrong password. */}
            <BackendStatusPill />
          </div>

          <div className="lg:hidden flex items-center gap-2.5 mb-6">
            <Image src="/orqis-logo.png" alt="" width={32} height={32} className="w-8 h-8" />
            <span className="text-base font-bold text-slate-900 dark:text-slate-50">
              Orqis Clinic Portal
            </span>
          </div>

          <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-50 tracking-tight">
            Sign in
          </h2>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1.5 mb-5 leading-relaxed">
            Use the account your operator provisioned for this clinic.
          </p>

          {/* Why the form is on screen, when the answer is "your token expired"
              rather than "you clicked sign in". */}
          {expiredMessage && !error && (
            <div
              role="status"
              className="flex items-start gap-2 text-xs text-amber-900 dark:text-amber-200 bg-amber-50 dark:bg-amber-950/50 border border-amber-200 dark:border-amber-900 rounded-2xl p-3 mb-5 leading-relaxed"
            >
              <TimerOff className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{expiredMessage}</span>
            </div>
          )}

          {/* Quick autofill for demonstration credentials */}
          <button
            type="button"
            onClick={() => {
              setEmail('clinician@orqis.local');
              setPassword('qqONIZ2T4EYTwau4');
            }}
            className="w-full mb-4 p-2.5 rounded-xl border border-teal-200 dark:border-teal-800 bg-teal-50/80 dark:bg-teal-950/40 text-teal-800 dark:text-teal-300 text-xs font-medium hover:bg-teal-100 dark:hover:bg-teal-900/50 transition-colors flex items-center justify-between group cursor-pointer"
          >
            <span>
              Quick Fill: <strong className="font-semibold">clinician@orqis.local</strong>
            </span>
            <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-teal-600 text-white group-hover:bg-teal-700">
              Autofill
            </span>
          </button>

          <form onSubmit={handleSubmit} className="space-y-4" noValidate>
            <div>
              <label
                htmlFor="portal-email"
                className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5"
              >
                Work email or username
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                <input
                  id="portal-email"
                  // Not `type="email"`: operator-created accounts may be plain
                  // usernames rather than addresses, and an email input asks mobile
                  // browsers for an email keyboard the user then has to fight.
                  // The backend matches this field case-insensitively either way.
                  type="text"
                  autoComplete="username"
                  autoCapitalize="none"
                  spellCheck={false}
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="clinician@hospital.org"
                  className="w-full pl-9 pr-3 py-2.5 text-sm rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/40 focus:border-teal-500 transition-all"
                />
              </div>
            </div>

            <div>
              <label
                htmlFor="portal-password"
                className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5"
              >
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                <input
                  id="portal-password"
                  type={revealPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full pl-9 pr-11 py-2.5 text-sm rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/40 focus:border-teal-500 transition-all"
                />
                {/* A mistyped password on a clinic workstation is the most common
                    reason sign-in fails; let it be checked without retyping. */}
                <button
                  type="button"
                  onClick={() => setRevealPassword((shown) => !shown)}
                  aria-label={revealPassword ? 'Hide password' : 'Show password'}
                  aria-pressed={revealPassword}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-xl text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500/40 transition-colors cursor-pointer"
                >
                  {revealPassword ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>

            {error && (
              <div
                role="alert"
                className="flex items-start gap-2 text-xs text-rose-800 dark:text-rose-300 bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-900 rounded-2xl p-3 leading-relaxed"
              >
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span className="break-words">{error}</span>
              </div>
            )}

            <Button
              type="submit"
              variant="secondary"
              size="md"
              disabled={submitting || !email.trim() || !password}
              className="w-full"
              icon={
                submitting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <ArrowRight className="w-4 h-4" />
                )
              }
            >
              {submitting ? 'Signing in…' : 'Sign in'}
            </Button>
          </form>

          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-6 leading-relaxed">
            There is no public sign-up. The first account is seeded at backend startup
            from the <code className="font-mono">CLINIC_ADMIN_EMAIL</code> and{' '}
            <code className="font-mono">CLINIC_ADMIN_PASSWORD</code> environment
            variables; if neither is set, sign-in is unavailable by design.
          </p>
        </div>
      </div>
    </div>
  );
}
