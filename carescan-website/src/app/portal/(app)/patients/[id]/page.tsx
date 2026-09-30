'use client';

/**
 * A patient's screening history.
 *
 * Next 16 hands dynamic route params to the client as a Promise, so they are read
 * with React's `use()` rather than destructured directly.
 *
 * `quantumVisual` is deliberately absent on every entry here: it is live-only
 * telemetry (DEC-035) that exists on the response to the analyse call and is null
 * whenever a result is re-read. The screen must not present that as data loss.
 */

import React, { use } from 'react';
import Link from 'next/link';
import { ArrowLeft, FileClock, ScanLine } from 'lucide-react';
import { api } from '@/lib/api';
import type { HistoryEntry } from '@/lib/apiTypes';
import {
  asPercent,
  bandTone,
  displayedBand,
  primaryScoreLine,
  TONE_CLASSES,
} from '@/lib/verdict';
import { useBackendResource } from '@/hooks/useBackendResource';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import {
  BandChip,
  EmptyState,
  ErrorState,
  PageHeading,
  SkeletonRows,
  formatTimestamp,
} from '@/components/portal/PortalUI';

export default function PatientDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: patientId } = use(params);

  const {
    data: entries,
    error,
    loading,
    reload,
  } = useBackendResource<HistoryEntry[]>(
    (signal) => api.history(patientId, signal),
    [patientId],
  );

  return (
    <>
      <Link
        href="/portal/patients"
        className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 dark:text-slate-400 hover:text-teal-700 dark:hover:text-teal-300 transition-colors"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        All patients
      </Link>

      <PageHeading
        title="Patient history"
        subtitle="Every screening recorded for this patient, newest first. Bands come from the validated classical baseline."
        actions={
          <Link href="/portal/screening/new">
            <Button variant="secondary" size="sm" icon={<ScanLine className="w-4 h-4" />} iconPosition="left">
              New screening
            </Button>
          </Link>
        }
      />

      <Card variant="stone" padding="sm">
        <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          Patient identifier
        </p>
        <p className="font-mono text-sm text-slate-900 dark:text-slate-100 mt-1 break-all">
          {patientId}
        </p>
        <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1.5 leading-snug">
          Pseudonymous. No name, age or sex is stored against this record.
        </p>
      </Card>

      {error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : loading && !entries ? (
        <SkeletonRows rows={4} />
      ) : entries && entries.length === 0 ? (
        <EmptyState
          icon={<FileClock className="w-6 h-6" />}
          title="No screenings for this patient"
          message="Either no capture has been analysed yet, or the identifier does not belong to this clinic."
          actionLabel="Start a screening"
          actionHref="/portal/screening/new"
        />
      ) : (
        <div className="space-y-3">
          {entries?.map((entry) => {
            const band = entry.result ? displayedBand(entry.result) : null;
            const tone = TONE_CLASSES[bandTone(band)];

            return (
              <Card key={entry.assessment.id} variant="white" padding="sm">
                <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-5">
                  <span className={`w-1.5 self-stretch rounded-full ${tone.bar} hidden sm:block`} />

                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2 mb-1">
                      <BandChip band={band} isMock={entry.result?.isMock} />
                      <span className="text-xs text-slate-500 dark:text-slate-400">
                        {formatTimestamp(entry.assessment.timestamp)}
                      </span>
                      <span className="text-[11px] text-slate-400 dark:text-slate-500">
                        · {entry.assessment.type}
                      </span>
                    </div>

                    {entry.result ? (
                      // Not `result.details`: that string narrates the quantum
                      // band, so under this band chip it routinely contradicts it.
                      <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                        {primaryScoreLine(entry.result) ??
                          'No primary score was recorded for this screening.'}
                      </p>
                    ) : (
                      <p className="text-xs text-amber-700 dark:text-amber-400">
                        Captured but never scored — no result was written for this screening.
                      </p>
                    )}
                  </div>

                  <div className="flex items-center gap-4 shrink-0">
                    <span className="text-right">
                      <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                        Quantum p
                      </span>
                      <span className="block text-xs font-mono font-bold text-slate-700 dark:text-slate-300">
                        {asPercent(entry.result?.finalProbability) ?? '—'}
                      </span>
                    </span>

                    {entry.result && (
                      <Link
                        href={`/portal/result/${encodeURIComponent(entry.assessment.id)}`}
                        className="text-xs font-semibold text-teal-700 dark:text-teal-300 hover:underline whitespace-nowrap"
                      >
                        Open result
                      </Link>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}

          <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed px-1">
            The quantum column is the calibrated quantum probability — a clearly
            labelled secondary readout, not the basis for the band. Experimental
            quantum-visual telemetry is live-only and is intentionally absent when a
            result is re-read from history.
          </p>
        </div>
      )}
    </>
  );
}
