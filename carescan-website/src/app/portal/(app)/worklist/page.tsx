'use client';

/**
 * The clinic worklist.
 *
 * Two things this screen must not do, both from DEC-034:
 *  - present `finalProbability` as the verdict. It is the calibrated *quantum*
 *    probability; the band in the Verdict column comes from the classical baseline,
 *    and the column header says so.
 *  - hide screenings that never produced a result. A capture that failed is exactly
 *    what a clinician needs to see, so `hasResult: false` rows stay in the list.
 */

import React, { useCallback, useMemo, useState } from 'react';
import Link from 'next/link';
import { ChevronLeft, ChevronRight, ClipboardList, Inbox, ScanLine } from 'lucide-react';
import { api } from '@/lib/api';
import type { ClinicStats, WorklistResponse } from '@/lib/apiTypes';
import { asPercent } from '@/lib/verdict';
import { usePortalAuth } from '@/context/PortalAuthContext';
import { useBackendResource } from '@/hooks/useBackendResource';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import {
  BandChip,
  EmptyState,
  ErrorState,
  PageHeading,
  ShortId,
  SkeletonRows,
  StatTile,
  formatTimestamp,
} from '@/components/portal/PortalUI';

const PAGE_SIZE = 20;

const BAND_FILTERS = [
  { label: 'All', value: '' },
  { label: 'High', value: 'HIGH RISK' },
  { label: 'Moderate', value: 'MODERATE RISK' },
  { label: 'Low', value: 'LOW RISK' },
];

export default function WorklistPage() {
  const { user } = usePortalAuth();
  const clinicId = user?.clinicId ?? '';

  const [risk, setRisk] = useState('');
  const [offset, setOffset] = useState(0);

  // Counters and rows are fetched together: a page showing 12 screenings above a
  // table of 20 is worse than a page that waits. `useBackendResource` refetches
  // on its own when the backend comes back.
  const { data, error, loading, reload } = useBackendResource<
    [ClinicStats, WorklistResponse]
  >(
    (signal) =>
      Promise.all([
        api.clinicStats(clinicId, signal),
        api.worklist(clinicId, { limit: PAGE_SIZE, offset, risk: risk || undefined }, signal),
      ]),
    [clinicId, offset, risk],
    { enabled: Boolean(clinicId) },
  );

  const [stats, worklist] = data ?? [null, null];

  const handleFilter = useCallback((value: string) => {
    setRisk(value);
    setOffset(0);
  }, []);

  const total = worklist?.total ?? 0;
  const pageEnd = Math.min(offset + PAGE_SIZE, total);

  const bandEntries = useMemo(
    () => Object.entries(stats?.bandCounts ?? {}).sort((a, b) => b[1] - a[1]),
    [stats],
  );

  return (
    <>
      <PageHeading
        title="Worklist"
        subtitle="Screenings captured by this clinic, newest first. The verdict column shows the band set by the validated classical baseline."
        actions={
          <Link href="/portal/screening/new">
            <Button variant="secondary" size="sm" icon={<ScanLine className="w-4 h-4" />} iconPosition="left">
              New screening
            </Button>
          </Link>
        }
      />

      {/* Counters */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatTile
          label="Screenings"
          value={loading && !stats ? '—' : stats?.totalScreenings ?? 0}
          hint="Captures recorded for this clinic"
        />
        <StatTile
          label="Patients"
          value={loading && !stats ? '—' : stats?.totalPatients ?? 0}
          hint="Pseudonymous records"
          accent="teal"
        />
        <StatTile
          label="Awaiting a result"
          value={loading && !stats ? '—' : stats?.pendingScreenings ?? 0}
          hint="Captured but not scored"
          accent="amber"
        />
        <StatTile
          label="Test-data results"
          value={loading && !stats ? '—' : stats?.mockResults ?? 0}
          hint="Mock stubs — no clinical meaning"
          accent="indigo"
        />
      </div>

      {bandEntries.length > 0 && (
        <Card variant="stone" padding="sm">
          <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-2">
            Band distribution
          </p>
          <div className="flex flex-wrap items-center gap-2">
            {bandEntries.map(([band, count]) => (
              <span key={band} className="inline-flex items-center gap-1.5">
                <BandChip band={band} />
                <span className="text-xs font-bold text-slate-700 dark:text-slate-300">
                  {count}
                </span>
              </span>
            ))}
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-2 leading-snug">
            Throughput for this clinic. These counts describe what was captured — they
            are not a measure of model accuracy.
          </p>
        </Card>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-1.5">
        {BAND_FILTERS.map((option) => (
          <button
            key={option.value || 'all'}
            type="button"
            onClick={() => handleFilter(option.value)}
            className={`text-xs font-semibold px-3 py-1.5 rounded-xl border transition-colors cursor-pointer ${
              risk === option.value
                ? 'bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 border-slate-900 dark:border-slate-100'
                : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800'
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>

      {/* List */}
      {error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : loading && !worklist ? (
        <SkeletonRows rows={6} />
      ) : worklist && worklist.items.length === 0 ? (
        <EmptyState
          icon={risk ? <Inbox className="w-6 h-6" /> : <ClipboardList className="w-6 h-6" />}
          title={risk ? 'No screenings in this band' : 'No screenings yet'}
          message={
            risk
              ? 'Nothing in this clinic currently carries that band. Clear the filter to see everything.'
              : 'Once a capture is analysed it appears here with the band the classical baseline assigned to it.'
          }
          actionLabel={risk ? undefined : 'Start a screening'}
          actionHref={risk ? undefined : '/portal/screening/new'}
        />
      ) : (
        <Card variant="white" padding="none" className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left min-w-[760px]">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-900/60">
                  {[
                    'Patient',
                    'Captured',
                    'Type',
                    'Verdict (classical baseline)',
                    'Quantum p — secondary',
                    'Model',
                    '',
                  ].map((heading, i) => (
                    <th
                      key={i}
                      scope="col"
                      className="px-4 py-3 text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 whitespace-nowrap"
                    >
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {worklist?.items.map((item) => (
                  <tr
                    key={item.screeningId}
                    className="border-b border-slate-100 dark:border-slate-800/70 last:border-0 hover:bg-slate-50/70 dark:hover:bg-slate-800/40 transition-colors"
                  >
                    <td className="px-4 py-3">
                      <Link
                        href={`/portal/patients/${encodeURIComponent(item.patientId)}`}
                        className="hover:underline"
                      >
                        <ShortId value={item.patientId} />
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-600 dark:text-slate-400 whitespace-nowrap">
                      {formatTimestamp(item.createdAt)}
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-600 dark:text-slate-400 whitespace-nowrap">
                      {item.scanType}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      {item.hasResult ? (
                        <BandChip band={item.riskLevel} isMock={item.isMock} />
                      ) : (
                        <span className="text-xs font-medium text-amber-700 dark:text-amber-400">
                          Awaiting analysis
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-xs font-mono text-slate-600 dark:text-slate-400 whitespace-nowrap">
                      {asPercent(item.finalProbability) ?? '—'}
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-500 dark:text-slate-400 whitespace-nowrap">
                      {item.primaryModel || item.modelVersion || '—'}
                    </td>
                    <td className="px-4 py-3 text-right whitespace-nowrap">
                      {item.hasResult && (
                        <Link
                          href={`/portal/result/${encodeURIComponent(item.screeningId)}`}
                          className="text-xs font-semibold text-teal-700 dark:text-teal-300 hover:underline"
                        >
                          Open
                        </Link>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Paging */}
          <div className="flex items-center justify-between gap-3 px-4 py-3 border-t border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-900/50">
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              {total === 0 ? '0' : `${offset + 1}–${pageEnd}`} of {total}
            </p>
            <div className="flex items-center gap-1.5">
              <Button
                variant="outline"
                size="sm"
                disabled={offset === 0 || loading}
                onClick={() => setOffset((value) => Math.max(0, value - PAGE_SIZE))}
                icon={<ChevronLeft className="w-3.5 h-3.5" />}
                iconPosition="left"
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={pageEnd >= total || loading}
                onClick={() => setOffset((value) => value + PAGE_SIZE)}
                icon={<ChevronRight className="w-3.5 h-3.5" />}
              >
                Next
              </Button>
            </div>
          </div>
        </Card>
      )}
    </>
  );
}
