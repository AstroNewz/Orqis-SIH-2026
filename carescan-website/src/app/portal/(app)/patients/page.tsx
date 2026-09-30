'use client';

/**
 * Patient roster.
 *
 * Patients are pseudonymous by design: the backend stores an identifier, counts and
 * timestamps, and nothing else. Age and sex are accepted for inference but never
 * persisted (`backend/models/screening.py`), so there is no demographic column to
 * show and none should be invented here.
 */

import React, { useState } from 'react';
import Link from 'next/link';
import { ChevronLeft, ChevronRight, ScanLine, Users } from 'lucide-react';
import { api } from '@/lib/api';
import type { ClinicPatientsResponse } from '@/lib/apiTypes';
import { usePortalAuth } from '@/context/PortalAuthContext';
import { useBackendResource } from '@/hooks/useBackendResource';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import {
  BandChip,
  EmptyState,
  ErrorState,
  PageHeading,
  ShortId,
  SkeletonRows,
  formatTimestamp,
} from '@/components/portal/PortalUI';

const PAGE_SIZE = 25;

export default function PatientsPage() {
  const { user } = usePortalAuth();
  const clinicId = user?.clinicId ?? '';

  const [offset, setOffset] = useState(0);

  const { data, error, loading, reload } = useBackendResource<ClinicPatientsResponse>(
    (signal) => api.clinicPatients(clinicId, { limit: PAGE_SIZE, offset }, signal),
    [clinicId, offset],
    { enabled: Boolean(clinicId) },
  );

  const total = data?.total ?? 0;
  const pageEnd = Math.min(offset + PAGE_SIZE, total);

  return (
    <>
      <PageHeading
        title="Patients"
        subtitle="Every patient record this clinic has screened. Records are pseudonymous — an identifier, screening counts and timestamps, with no demographic data stored."
        actions={
          <Link href="/portal/screening/new">
            <Button variant="secondary" size="sm" icon={<ScanLine className="w-4 h-4" />} iconPosition="left">
              New screening
            </Button>
          </Link>
        }
      />

      {error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : loading && !data ? (
        <SkeletonRows rows={6} />
      ) : data && data.items.length === 0 ? (
        <EmptyState
          icon={<Users className="w-6 h-6" />}
          title="No patients yet"
          message="A patient record is created the first time a screening is captured for them."
          actionLabel="Start a screening"
          actionHref="/portal/screening/new"
        />
      ) : (
        <Card variant="white" padding="none" className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left min-w-[640px]">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-900/60">
                  {['Patient', 'Screenings', 'Latest band', 'Last screened', 'First seen', ''].map(
                    (heading, i) => (
                      <th
                        key={i}
                        scope="col"
                        className="px-4 py-3 text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 whitespace-nowrap"
                      >
                        {heading}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {data?.items.map((patient) => (
                  <tr
                    key={patient.patientId}
                    className="border-b border-slate-100 dark:border-slate-800/70 last:border-0 hover:bg-slate-50/70 dark:hover:bg-slate-800/40 transition-colors"
                  >
                    <td className="px-4 py-3">
                      <span className="inline-flex items-center gap-2">
                        <ShortId value={patient.patientId} />
                        {!patient.isActive && (
                          <Badge variant="neutral" size="sm">
                            Inactive
                          </Badge>
                        )}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs font-bold text-slate-800 dark:text-slate-200">
                      {patient.screeningCount}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <BandChip band={patient.latestRiskLevel} />
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-600 dark:text-slate-400 whitespace-nowrap">
                      {formatTimestamp(patient.lastScreeningAt)}
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-600 dark:text-slate-400 whitespace-nowrap">
                      {formatTimestamp(patient.createdAt)}
                    </td>
                    <td className="px-4 py-3 text-right whitespace-nowrap">
                      <Link
                        href={`/portal/patients/${encodeURIComponent(patient.patientId)}`}
                        className="text-xs font-semibold text-teal-700 dark:text-teal-300 hover:underline"
                      >
                        History
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

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
