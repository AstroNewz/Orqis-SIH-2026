'use client';

/**
 * Operator view of the backend.
 *
 * Everything here is read straight from `/health` and `/api/model/info`. Two
 * honesty rules apply:
 *
 *  - `model_ready: false` is a real, expected state — no trained artifacts are
 *    loaded. The process is still healthy, and the page says exactly that instead
 *    of dressing it up as an outage or hiding it.
 *  - `quantum_execution_mode` is the *effective* mode. When it differs from
 *    `quantum_execution_mode_configured`, the backend fell back, and an operator
 *    has to be able to see that at a glance.
 */

import React from 'react';
import {
  Activity,
  AlertTriangle,
  Atom,
  CheckCircle2,
  Cpu,
  RefreshCw,
  ServerCog,
} from 'lucide-react';
import { api } from '@/lib/api';
import type { HealthResponse, ModelInfo } from '@/lib/apiTypes';
import { executionModeLabel, summariseModelInfo } from '@/lib/verdict';
import { useBackendStatus } from '@/context/BackendStatusContext';
import { useBackendResource } from '@/hooks/useBackendResource';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import {
  ErrorState,
  PageHeading,
  SkeletonRows,
  StatTile,
} from '@/components/portal/PortalUI';

/** One field of backend-reported state. Absent values stay visibly absent. */
const Row: React.FC<{ label: string; value: React.ReactNode; mono?: boolean }> = ({
  label,
  value,
  mono = false,
}) => (
  <div className="py-2 border-b border-slate-100 dark:border-slate-800/70 last:border-0">
    <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
      {label}
    </dt>
    <dd
      className={`text-xs text-slate-800 dark:text-slate-200 mt-0.5 break-words ${
        mono ? 'font-mono' : ''
      }`}
    >
      {value === null || value === undefined || value === '' ? '—' : value}
    </dd>
  </div>
);

export default function SystemPage() {
  const { lastCheckedAt } = useBackendStatus();

  const { data, error, loading, reload } = useBackendResource<{
    health: HealthResponse;
    model: ModelInfo | null;
  }>(async (signal) => {
    const health = await api.health(signal);
    // Model info is secondary: a failure here must not blank out the health
    // panel, which is the part an operator needs when something is wrong.
    let model: ModelInfo | null = null;
    try {
      model = await api.modelInfo(signal);
    } catch {
      model = null;
    }
    return { health, model };
  }, []);

  const health = data?.health ?? null;
  const model = data?.model ?? null;

  const refresh = reload;

  const modeFellBack =
    health != null &&
    health.quantum_execution_mode !== health.quantum_execution_mode_configured;

  return (
    <>
      <PageHeading
        title="System"
        subtitle="Live state of the Orqis backend this portal is talking to. Every value is reported by the server; nothing on this page is inferred."
        actions={
          <>
            <span className="hidden sm:inline text-[11px] text-slate-500 dark:text-slate-400">
              {lastCheckedAt
                ? `Polled ${new Date(lastCheckedAt).toLocaleTimeString()}`
                : 'Not yet polled'}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={refresh}
              disabled={loading}
              icon={<RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />}
              iconPosition="left"
            >
              Refresh
            </Button>
          </>
        }
      />

      {error ? (
        <ErrorState error={error} onRetry={refresh} />
      ) : loading && !health ? (
        <SkeletonRows rows={5} />
      ) : (
        health && (
          <>
            {/* Headline state */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <StatTile
                label="Service"
                value={health.status}
                hint={health.service}
                accent="emerald"
              />
              <StatTile
                label="Version"
                value={health.version}
                hint={health.environment}
              />
              <StatTile
                label="Model"
                value={health.model_ready ? 'Ready' : 'Not loaded'}
                hint={health.model_version ?? 'No artifact version reported'}
                accent={health.model_ready ? 'teal' : 'amber'}
              />
              <StatTile
                label="Qubits"
                value={health.quantum_qubits}
                hint={health.quantum_backend}
                accent="indigo"
              />
            </div>

            {/* Model readiness, stated plainly either way */}
            {health.model_ready ? (
              <Card variant="white" padding="sm" className="border-teal-200 dark:border-teal-900">
                <div className="flex items-start gap-2.5">
                  <CheckCircle2 className="w-5 h-5 text-teal-700 dark:text-teal-400 shrink-0 mt-0.5" />
                  <div className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
                    <strong className="block text-sm font-bold text-slate-900 dark:text-slate-100 mb-0.5">
                      Trained artifacts are loaded
                    </strong>
                    Screenings submitted from this portal run the real inference
                    pipeline. Results are scored, calibrated and recorded.
                  </div>
                </div>
              </Card>
            ) : (
              <Card variant="white" padding="sm" className="border-amber-300 dark:border-amber-800">
                <div className="flex items-start gap-2.5">
                  <AlertTriangle className="w-5 h-5 text-amber-700 dark:text-amber-400 shrink-0 mt-0.5" />
                  <div className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
                    <strong className="block text-sm font-bold text-slate-900 dark:text-slate-100 mb-0.5">
                      No trained model is loaded
                    </strong>
                    The service itself is healthy — this is a deployment state, not a
                    crash. Until artifacts are present in{' '}
                    <code className="font-mono text-[11px] bg-slate-100 dark:bg-slate-800 px-1 py-0.5 rounded">
                      backend/artifacts/models/current/
                    </code>
                    , a screening cannot be scored and any result you see came from the
                    labelled test stub.
                  </div>
                </div>
              </Card>
            )}

            {/* Execution mode, and whether it fell back */}
            {modeFellBack && (
              <Card variant="white" padding="sm" className="border-amber-300 dark:border-amber-800">
                <div className="flex items-start gap-2.5">
                  <AlertTriangle className="w-5 h-5 text-amber-700 dark:text-amber-400 shrink-0 mt-0.5" />
                  <div className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
                    <strong className="block text-sm font-bold text-slate-900 dark:text-slate-100 mb-0.5">
                      Quantum execution fell back
                    </strong>
                    Configured as{' '}
                    <span className="font-mono">
                      {health.quantum_execution_mode_configured}
                    </span>{' '}
                    but currently running as{' '}
                    <span className="font-mono">{health.quantum_execution_mode}</span>.
                    Results are still produced; they simply did not come from the
                    configured target.
                  </div>
                </div>
              </Card>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Service */}
              <Card variant="white" padding="md">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 flex items-center justify-center">
                    <ServerCog className="w-4 h-4" />
                  </div>
                  <div>
                    <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                      Service
                    </h2>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400">
                      From <span className="font-mono">GET /health</span>
                    </p>
                  </div>
                </div>

                <dl>
                  <Row label="Status" value={health.status} />
                  <Row label="Service name" value={health.service} mono />
                  <Row label="Version" value={health.version} mono />
                  <Row label="Environment" value={health.environment} mono />
                  <Row
                    label="Model ready"
                    value={
                      <Badge
                        variant={health.model_ready ? 'teal' : 'neutral'}
                        size="sm"
                      >
                        {health.model_ready ? 'Yes' : 'No'}
                      </Badge>
                    }
                  />
                  <Row label="Model version" value={health.model_version} mono />
                </dl>
              </Card>

              {/* Quantum stage */}
              <Card variant="white" padding="md">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-xl bg-purple-100 dark:bg-purple-950/70 text-purple-700 dark:text-purple-300 flex items-center justify-center">
                    <Atom className="w-4 h-4" />
                  </div>
                  <div>
                    <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                      Quantum stage
                    </h2>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400">
                      Effective configuration, not a claim about performance
                    </p>
                  </div>
                </div>

                <dl>
                  <Row
                    label="Execution mode (effective)"
                    value={executionModeLabel(health.quantum_execution_mode)}
                  />
                  <Row
                    label="Execution mode (configured)"
                    value={executionModeLabel(health.quantum_execution_mode_configured)}
                  />
                  <Row label="Backend" value={health.quantum_backend} mono />
                  <Row label="Qubits" value={health.quantum_qubits} mono />
                </dl>

                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-3 leading-relaxed">
                  The quantum stage is a measured demonstrator running beside the
                  classical baseline that sets the band. It has not been shown to beat
                  that baseline on this dataset.
                </p>
              </Card>
            </div>

            {/* Model artifact */}
            <Card variant="stone" padding="md">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Loaded model artifact
                </h2>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-3 leading-relaxed">
                From <span className="font-mono">GET /api/model/info</span>. The shape of
                this response varies with readiness, so a blank field means the backend
                did not report it.
              </p>

              {model ? (
                (() => {
                  const summary = summariseModelInfo(model);
                  return (
                    <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6">
                      <Row
                        label="Ready"
                        value={
                          summary.ready === undefined ? null : summary.ready ? 'Yes' : 'No'
                        }
                      />
                      <Row label="Model version" value={summary.modelVersion} mono />
                      <Row label="Qubits" value={summary.qubits} mono />
                      <Row label="Circuit depth" value={summary.circuitDepth} mono />
                      <Row
                        label="Calibration method"
                        value={summary.calibrationMethod}
                        mono
                      />
                      <Row
                        label="Execution mode"
                        value={
                          summary.executionMode
                            ? executionModeLabel(summary.executionMode)
                            : null
                        }
                      />
                      <Row label="Backend name" value={summary.backendName} mono />
                      <Row label="Feature mode" value={summary.featureMode} mono />
                      <Row
                        label="Descriptor dimension"
                        value={summary.descriptorDimension}
                        mono
                      />
                      <Row
                        label="Clinical features"
                        value={summary.clinicalFeatureCount}
                        mono
                      />
                      <Row
                        label="Band set by"
                        value={summary.classicalReference}
                        mono
                      />
                      <Row
                        label="Thresholds clinically validated"
                        value={
                          summary.thresholdsValidated === null
                            ? null
                            : summary.thresholdsValidated
                              ? 'Yes'
                              : 'No — research thresholds'
                        }
                      />
                      {/* An operator needs to see a hardware request that silently
                          became a simulator run. Only shown when it happened. */}
                      {summary.fellBack === true && (
                        <Row
                          label="Fell back to simulator"
                          value={
                            summary.fallbackReason
                              ? `Yes — ${summary.fallbackReason}`
                              : 'Yes'
                          }
                        />
                      )}
                    </dl>
                  );
                })()
              ) : (
                <div className="flex items-start gap-2 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                  <Activity className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>
                    Model details are unavailable. The health endpoint answered, so the
                    service is up — this endpoint either failed or reported nothing.
                  </span>
                </div>
              )}
            </Card>
          </>
        )
      )}
    </>
  );
}
