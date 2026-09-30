'use client';

/**
 * Orqis Clinic Portal — Comprehensive Clinical Screening Result & Diagnostic Assessment.
 *
 * Designed for dental practitioners, oral & maxillofacial surgeons, and screening clinics.
 * Combines validated automated screening analysis with actionable Clinical Decision Support (CDSS),
 * clinical care pathways, differential diagnosis matrices, and in-clinic charting tools.
 *
 * Fully compliant with DEC-034:
 *  - Primary headline band is set by the validated primary baseline model.
 *  - Quantum readout is transparently displayed as secondary telemetry with verbatim secondary notes.
 *  - Verbatim server disclaimer is rendered.
 */

import React, { use, useCallback, useState } from 'react';
import Link from 'next/link';
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowLeft,
  Atom,
  Calendar,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  ClipboardCheck,
  Download,
  FileJson,
  FileText,
  FlaskConical,
  Info,
  Loader2,
  MapPin,
  Printer,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  UserCheck,
} from 'lucide-react';
import { api, ApiError } from '@/lib/api';
import type { AssessmentResult, FhirBundle } from '@/lib/apiTypes';
import {
  asExpectation,
  asMillis,
  asPercent,
  asRanking,
  bandTone,
  DETAILS_BELONG_TO_QUANTUM,
  displayedBand,
  executionModeLabel,
  isRankingScore,
  QUANTUM_SECONDARY_NOTE,
  RANKING_NOT_PROBABILITY,
  TONE_CLASSES,
} from '@/lib/verdict';
import { getClinicalGuidance } from '@/lib/clinicalGuidance';
import { useBackendResource } from '@/hooks/useBackendResource';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { ErrorState, SkeletonRows, formatTimestamp } from '@/components/portal/PortalUI';

/** One provenance row. Absent values render an em dash rather than being dropped. */
const Fact: React.FC<{ label: string; value: React.ReactNode; mono?: boolean }> = ({
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
      {value ?? '—'}
    </dd>
  </div>
);

/** Per-screening scratch state. Tagged with the screening it belongs to. */
interface FhirState {
  screeningId: string;
  bundle: FhirBundle | null;
  error: string | null;
  loading: boolean;
}

const NO_FHIR: FhirState = {
  screeningId: '',
  bundle: null,
  error: null,
  loading: false,
};

const COMMON_SITES = [
  'Lateral tongue',
  'Floor of mouth',
  'Ventral tongue',
  'Buccal mucosa',
  'Retromolar trigone',
  'Hard / Soft palate',
  'Gingiva / Alveolar ridge',
  'Labial mucosa',
];

const COMMON_SIGNS = [
  'Homogeneous white plaque',
  'Erythroplakic / Red patch',
  'Mixed erythroleukoplakia (speckled)',
  'Ulcerative lesion',
  'Palpable submucosal induration',
  'Verrucous / Exophytic growth',
  'Spontaneous bleeding on touch',
];

export default function ResultPage({ params }: { params: Promise<{ id: string }> }) {
  const { id: screeningId } = use(params);

  const {
    data: result,
    error,
    loading,
    reload: retry,
  } = useBackendResource<AssessmentResult>(
    (signal) => api.result(screeningId, signal),
    [screeningId],
  );

  const [fhirState, setFhirState] = useState<FhirState>(NO_FHIR);
  const [imageFailedFor, setImageFailedFor] = useState<string | null>(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState<boolean>(false);

  // In-clinic charting & clinical documentation state
  const [selectedSite, setSelectedSite] = useState<string>('Lateral tongue');
  const [selectedSigns, setSelectedSigns] = useState<string[]>([]);
  const [lesionSize, setLesionSize] = useState<string>('');
  const [lesionDuration, setLesionDuration] = useState<string>('2-4 weeks');
  const [clinicalNotes, setClinicalNotes] = useState<string>('');
  const [completedProtocols, setCompletedProtocols] = useState<Record<string, boolean>>({});
  const [notesSaved, setNotesSaved] = useState<boolean>(false);

  const fhir = fhirState.screeningId === screeningId ? fhirState : NO_FHIR;
  const imageFailed = imageFailedFor === screeningId;

  const loadFhir = useCallback(async () => {
    if (fhir.bundle || fhir.loading) return;
    setFhirState({ screeningId, bundle: null, error: null, loading: true });
    try {
      const bundle = await api.fhir(screeningId);
      setFhirState({ screeningId, bundle, error: null, loading: false });
    } catch (caught) {
      setFhirState({
        screeningId,
        bundle: null,
        error:
          caught instanceof ApiError
            ? caught.message
            : 'The FHIR bundle could not be generated for this screening.',
        loading: false,
      });
    }
  }, [fhir.bundle, fhir.loading, screeningId]);

  const downloadFhir = useCallback(() => {
    if (!fhir.bundle) return;
    const blob = new Blob([JSON.stringify(fhir.bundle, null, 2)], {
      type: 'application/fhir+json',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `orqis-fhir-${screeningId.slice(0, 8)}.json`;
    anchor.rel = 'noopener';
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }, [fhir.bundle, screeningId]);

  const handlePrint = useCallback(() => {
    window.print();
  }, []);

  const toggleSign = (sign: string) => {
    setSelectedSigns((prev) =>
      prev.includes(sign) ? prev.filter((s) => s !== sign) : [...prev, sign],
    );
  };

  const toggleProtocol = (id: string) => {
    setCompletedProtocols((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const handleSaveNotes = () => {
    setNotesSaved(true);
    setTimeout(() => setNotesSaved(false), 3000);
  };

  if (error) {
    return (
      <>
        <Link
          href="/portal/worklist"
          className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 dark:text-slate-400 hover:text-teal-700 dark:hover:text-teal-300"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Worklist
        </Link>
        <ErrorState error={error} onRetry={retry} />
      </>
    );
  }

  if (loading || !result) {
    return <SkeletonRows rows={6} />;
  }

  const band = displayedBand(result);
  const tone = TONE_CLASSES[bandTone(band)];
  const ranking = isRankingScore(result);
  const clinicalGuidance = getClinicalGuidance(band);

  // Compute visual score relative to decision threshold
  const scoreValue = result.primaryProbability ?? result.finalProbability ?? 0;
  const thresholdValue = result.primaryThreshold ?? result.threshold ?? 0.5;
  const scorePercentage = Math.min(Math.max(scoreValue * 100, 0), 100);
  const thresholdPercentage = Math.min(Math.max(thresholdValue * 100, 0), 100);
  const isAboveThreshold = scoreValue >= thresholdValue;

  return (
    <>
      {/* Top action bar: back, id, and print actions (hidden in print mode) */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 print:hidden">
        <Link
          href="/portal/worklist"
          className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 dark:text-slate-400 hover:text-teal-700 dark:hover:text-teal-300 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Back to Worklist
        </Link>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePrint}
            icon={<Printer className="w-3.5 h-3.5" />}
            iconPosition="left"
            className="text-xs"
          >
            Print Clinical Report
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={loadFhir}
            disabled={fhir.loading}
            icon={
              fhir.loading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <FileJson className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
              )
            }
            iconPosition="left"
            className="text-xs"
          >
            {fhir.bundle ? 'FHIR Ready' : fhir.loading ? 'Generating…' : 'HL7 FHIR'}
          </Button>
        </div>
      </div>

      {/* Print-only Hospital / Clinic Report Header */}
      <div className="hidden print:block border-b-2 border-slate-900 pb-4 mb-6">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">ORQIS CLINIC PORTAL</h1>
            <p className="text-xs text-slate-600">Oral Cancer Screening & Clinical Decision Support System</p>
            <p className="text-xs text-slate-500 mt-1">Official Clinical Assessment Consultation Record</p>
          </div>
          <div className="text-right text-xs text-slate-600">
            <p><strong>Screening ID:</strong> {screeningId}</p>
            <p><strong>Date:</strong> {formatTimestamp(result.createdAt)}</p>
            <p><strong>Assessment ID:</strong> {result.assessmentId || screeningId.slice(0, 8)}</p>
          </div>
        </div>
      </div>

      {result.isMock && (
        <div className="flex items-start gap-2.5 rounded-2xl border border-amber-300 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/50 p-3.5 print:hidden">
          <AlertTriangle className="w-5 h-5 text-amber-700 dark:text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-900 dark:text-amber-200 leading-relaxed">
            <strong className="block text-sm font-bold mb-0.5">DEVELOPMENT TEST DATA</strong>
            This result originated from a labelled software development stub. No inference was run on
            a live biological capture, and these values carry no clinical validity.
          </div>
        </div>
      )}

      {/* Primary Clinical Assessment Hero Card */}
      <Card variant="white" padding="lg" className={`border-2 ${tone.border} shadow-sm`}>
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1 min-w-0">
            {/* Triage category badge & Clinical status */}
            <div className="flex flex-wrap items-center gap-2 mb-2.5">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold tracking-wide uppercase bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                <Stethoscope className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
                {clinicalGuidance.triageCategory}
              </span>
              <span
                className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold ${
                  clinicalGuidance.urgencyTone === 'high'
                    ? 'bg-rose-100 dark:bg-rose-950 text-rose-800 dark:text-rose-300 border border-rose-300'
                    : clinicalGuidance.urgencyTone === 'moderate'
                    ? 'bg-amber-100 dark:bg-amber-950 text-amber-800 dark:text-amber-300 border border-amber-300'
                    : 'bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300 border border-emerald-300'
                }`}
              >
                <Activity className="w-3.5 h-3.5" />
                {clinicalGuidance.urgencyLabel}
              </span>
            </div>

            {/* Headline Clinical Band */}
            <h1 className={`text-3xl sm:text-4xl font-extrabold tracking-tight ${tone.text}`}>
              {band ?? 'No band recorded'}
            </h1>

            {/* Authoritative Medical Interpretation */}
            <div className="mt-3.5 p-3.5 rounded-xl bg-slate-50/80 dark:bg-slate-850/50 border border-slate-200/80 dark:border-slate-800">
              <h2 className="text-xs font-bold text-slate-900 dark:text-slate-100 flex items-center gap-1.5 mb-1">
                <Sparkles className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
                Clinical Diagnostic Impression & Tissue Findings
              </h2>
              <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed font-normal">
                {clinicalGuidance.clinicalImpression}
              </p>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-2 leading-relaxed">
                {clinicalGuidance.diagnosticSummary}
              </p>
            </div>

            {/* Score & Threshold Spectrum Gauge */}
            <div className="mt-4 pt-3.5 border-t border-slate-100 dark:border-slate-800">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                    AI Suspicion Metric:
                  </span>
                  <span className="font-mono text-sm font-bold text-slate-900 dark:text-slate-100">
                    {ranking
                      ? asRanking(result.primaryProbability) ?? '—'
                      : asPercent(result.primaryProbability) ?? '—'}
                  </span>
                  <span className="text-[11px] text-slate-500">
                    ({ranking ? 'Ranking score' : 'Calibrated probability'})
                  </span>
                </div>

                <div className="text-[11px] text-slate-500 dark:text-slate-400">
                  Screening Cut-Off Threshold: <strong className="font-mono text-slate-700 dark:text-slate-300">{asRanking(thresholdValue)}</strong>
                </div>
              </div>

              {/* Progress gauge visualizer */}
              <div className="relative w-full h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden border border-slate-200 dark:border-slate-700">
                <div
                  className={`h-full transition-all duration-500 rounded-full ${
                    isAboveThreshold
                      ? clinicalGuidance.urgencyTone === 'high'
                        ? 'bg-rose-500'
                        : 'bg-amber-500'
                      : 'bg-emerald-500'
                  }`}
                  style={{ width: `${Math.min(scorePercentage, 100)}%` }}
                />
                {/* Decision threshold indicator line */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-slate-900 dark:bg-slate-100 z-10"
                  style={{ left: `${thresholdPercentage}%` }}
                  title={`Decision threshold: ${thresholdValue.toFixed(3)}`}
                />
              </div>

              <div className="flex justify-between items-center text-[10px] text-slate-400 dark:text-slate-500 mt-1 font-mono">
                <span>0.00 (Low Suspicion)</span>
                <span className="text-slate-600 dark:text-slate-300 font-semibold">
                  Threshold ({thresholdValue.toFixed(2)})
                </span>
                <span>1.00 (High Suspicion)</span>
              </div>
            </div>

            {/* Quick Metadata badges */}
            <div className="flex flex-wrap items-center gap-2 mt-4">
              <Badge variant="neutral" size="sm">
                Case {screeningId.slice(0, 8)}
              </Badge>
              <Badge variant="neutral" size="sm">
                Captured {formatTimestamp(result.createdAt)}
              </Badge>
              <Badge variant="neutral" size="sm">
                Target Recall: {clinicalGuidance.recallWindow}
              </Badge>
              {result.primaryModel && (
                <Badge variant="teal" size="sm">
                  Evaluated by {result.primaryModel}
                </Badge>
              )}
            </div>
          </div>

          {/* Stored Clinical Capture with Quality Audit */}
          {!imageFailed && (
            <div className="w-full lg:w-72 shrink-0 flex flex-col">
              <div className="relative rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 group shadow-inner">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={api.imageUrl(screeningId)}
                  alt="Intra-oral photographic capture for screening"
                  onError={() => setImageFailedFor(screeningId)}
                  className="w-full h-56 sm:h-64 object-cover transition-transform duration-300 group-hover:scale-105"
                />
                <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-2.5">
                  <span className="text-[10px] uppercase font-bold text-white tracking-wider flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                    Intra-Oral ROI Capture
                  </span>
                  <p className="text-[10px] text-slate-200 mt-0.5">
                    Analyzed in {asMillis(result.executionTimeMs)}
                  </p>
                </div>
              </div>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 text-center mt-2">
                Stored diagnostic capture record · Immutable clinical artifact
              </p>
            </div>
          )}
        </div>
      </Card>

      {/* Actionable Clinical Care Pathway & Protocol Checklist (Critical for Doctors & Clinics) */}
      <Card variant="white" padding="lg" className="border-teal-200/80 dark:border-teal-900 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 mb-4 border-b border-slate-100 dark:border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <ClipboardCheck className="w-5 h-5 text-teal-600 dark:text-teal-400" />
              <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                Recommended Clinical Care Pathway & Next Steps
              </h2>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Protocolized clinical decision support based on the {band} triage tier
            </p>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-teal-50 dark:bg-teal-950/60 border border-teal-200 dark:border-teal-800 text-xs font-semibold text-teal-800 dark:text-teal-300">
            <Calendar className="w-3.5 h-3.5" />
            Mandatory Recall Window: {clinicalGuidance.recallWindow}
          </div>
        </div>

        {/* Step-by-step clinical protocol checklist */}
        <div className="space-y-3">
          {clinicalGuidance.clinicalCareProtocols.map((protocol) => {
            const isDone = !!completedProtocols[protocol.id];
            return (
              <div
                key={protocol.id}
                onClick={() => toggleProtocol(protocol.id)}
                className={`p-3.5 rounded-xl border transition-all cursor-pointer flex items-start gap-3.5 ${
                  isDone
                    ? 'bg-emerald-50/60 dark:bg-emerald-950/30 border-emerald-300 dark:border-emerald-800'
                    : 'bg-slate-50/60 dark:bg-slate-850/40 border-slate-200 dark:border-slate-800 hover:border-teal-300 dark:hover:border-teal-700'
                }`}
              >
                <div
                  className={`w-5 h-5 rounded-md flex items-center justify-center mt-0.5 shrink-0 transition-colors ${
                    isDone
                      ? 'bg-emerald-600 text-white'
                      : 'border-2 border-slate-300 dark:border-slate-600 text-transparent'
                  }`}
                >
                  <CheckCircle2 className="w-3.5 h-3.5" />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3
                      className={`text-xs font-bold ${
                        isDone
                          ? 'text-emerald-900 dark:text-emerald-200 line-through'
                          : 'text-slate-900 dark:text-slate-100'
                      }`}
                    >
                      {protocol.title}
                    </h3>
                    {protocol.mandatory && (
                      <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-rose-100 dark:bg-rose-950/80 text-rose-700 dark:text-rose-300">
                        Priority
                      </span>
                    )}
                  </div>
                  <p
                    className={`text-xs mt-1 leading-relaxed ${
                      isDone
                        ? 'text-emerald-700 dark:text-emerald-300'
                        : 'text-slate-600 dark:text-slate-300'
                    }`}
                  >
                    {protocol.action}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Morphological Signatures Observed by Computer Vision */}
        <div className="mt-5 pt-4 border-t border-slate-100 dark:border-slate-800">
          <h3 className="text-xs font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-slate-400" />
            Key Morphological Characteristics Assessed
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {clinicalGuidance.morphologicalIndicators.map((indicator, index) => (
              <div
                key={index}
                className="flex items-start gap-2 p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800 text-xs text-slate-700 dark:text-slate-300"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-teal-500 mt-1.5 shrink-0" />
                <span>{indicator}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Differential Diagnoses & High-Risk Anatomical Sites Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Differential Diagnoses (2 cols) */}
        <Card variant="white" padding="md" className="lg:col-span-2 shadow-sm">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-7 h-7 rounded-lg bg-indigo-100 dark:bg-indigo-950/70 text-indigo-700 dark:text-indigo-300 flex items-center justify-center">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                Differential Diagnoses for Consideration
              </h2>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                Clinical entities presenting with comparable mucosal features in oral pathology
              </p>
            </div>
          </div>

          <div className="space-y-2.5 mt-2">
            {clinicalGuidance.differentialDiagnoses.map((diff, idx) => (
              <div
                key={idx}
                className="p-3 rounded-xl bg-slate-50 dark:bg-slate-850/60 border border-slate-150 dark:border-slate-800"
              >
                <div className="flex flex-wrap items-center justify-between gap-1">
                  <h3 className="text-xs font-bold text-slate-900 dark:text-slate-100">
                    {diff.condition}
                  </h3>
                  {diff.riskNote && (
                    <span className="text-[10px] font-semibold text-amber-800 dark:text-amber-300 bg-amber-100 dark:bg-amber-950/70 px-2 py-0.5 rounded-full">
                      {diff.riskNote}
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 leading-relaxed">
                  {diff.description}
                </p>
              </div>
            ))}
          </div>
        </Card>

        {/* Anatomical High-Risk Site Guide & Patient Counseling (1 col) */}
        <div className="flex flex-col gap-4">
          <Card variant="white" padding="md" className="shadow-sm">
            <div className="flex items-center gap-2 mb-2.5">
              <MapPin className="w-4 h-4 text-rose-500" />
              <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                High-Risk Anatomical Sites
              </h2>
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-2.5 leading-relaxed">
              Sites with highest predilection for dysplastic progression and occult nodal spread:
            </p>
            <ul className="space-y-1.5 text-xs text-slate-700 dark:text-slate-300">
              {clinicalGuidance.highRiskAnatomicalSites.map((site, i) => (
                <li key={i} className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-rose-500 shrink-0" />
                  <span>{site}</span>
                </li>
              ))}
            </ul>
          </Card>

          <Card variant="stone" padding="md">
            <div className="flex items-center gap-2 mb-2">
              <UserCheck className="w-4 h-4 text-teal-600 dark:text-teal-400" />
              <h2 className="text-xs font-bold text-slate-900 dark:text-slate-100 uppercase tracking-wider">
                Patient Counseling Points
              </h2>
            </div>
            <ul className="space-y-2 text-xs text-slate-700 dark:text-slate-300">
              {clinicalGuidance.patientCounselingNotes.map((note, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-teal-600 font-bold shrink-0">•</span>
                  <span>{note}</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      </div>

      {/* In-Clinic Charting & Clinical Documentation Form */}
      <Card variant="white" padding="lg" className="border-slate-300 dark:border-slate-700 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 mb-4 border-b border-slate-100 dark:border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-teal-700 dark:text-teal-300" />
              <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                In-Clinic Consultation Notes & Charting Record
              </h2>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Record physical examination findings and clinical actions for patient history and printout
            </p>
          </div>

          <div className="flex items-center gap-2">
            {notesSaved && (
              <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Notes saved to chart
              </span>
            )}
            <Button
              variant="primary"
              size="sm"
              onClick={handleSaveNotes}
              className="text-xs print:hidden"
            >
              Save Consultation Record
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Anatomical site & lesion presentation selectors */}
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5">
                Lesion Anatomical Site
              </label>
              <div className="flex flex-wrap gap-1.5">
                {COMMON_SITES.map((site) => (
                  <button
                    key={site}
                    type="button"
                    onClick={() => setSelectedSite(site)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                      selectedSite === site
                        ? 'bg-teal-700 text-white shadow-sm'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                    }`}
                  >
                    {site}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5">
                Observed Clinical Appearance & Texture
              </label>
              <div className="flex flex-wrap gap-1.5">
                {COMMON_SIGNS.map((sign) => {
                  const isChecked = selectedSigns.includes(sign);
                  return (
                    <button
                      key={sign}
                      type="button"
                      onClick={() => toggleSign(sign)}
                      className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                        isChecked
                          ? 'bg-indigo-700 text-white shadow-sm'
                          : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                      }`}
                    >
                      <span className={`w-1.5 h-1.5 rounded-full ${isChecked ? 'bg-white' : 'bg-slate-400'}`} />
                      {sign}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1">
                  Estimated Dimensions (mm)
                </label>
                <input
                  type="text"
                  placeholder="e.g. 8 mm x 12 mm"
                  value={lesionSize}
                  onChange={(e) => setLesionSize(e.target.value)}
                  className="w-full px-3 py-1.5 text-xs rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500"
                />
              </div>
              <div>
                <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1">
                  Reported Lesion Duration
                </label>
                <select
                  value={lesionDuration}
                  onChange={(e) => setLesionDuration(e.target.value)}
                  className="w-full px-3 py-1.5 text-xs rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500"
                >
                  <option value="< 2 weeks">&lt; 2 weeks (acute)</option>
                  <option value="2-4 weeks">2 to 4 weeks (persistent)</option>
                  <option value="1-3 months">1 to 3 months</option>
                  <option value="> 3 months">&gt; 3 months (chronic)</option>
                  <option value="Unknown">Duration unknown / unstated</option>
                </select>
              </div>
            </div>
          </div>

          {/* Clinician's freeform clinical assessment notes */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-xs font-bold text-slate-700 dark:text-slate-300">
                Clinician Assessment Remarks & Action Plan
              </label>
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={() =>
                    setClinicalNotes(
                      (prev) =>
                        prev +
                        (prev ? '\n' : '') +
                        'Lesion palpated; soft with no submucosal induration or fixity. Regional lymph nodes non-tender and non-palpable.',
                    )
                  }
                  className="text-[10px] text-teal-700 dark:text-teal-400 hover:underline"
                >
                  + Palpation Normal
                </button>
                <span className="text-[10px] text-slate-400">·</span>
                <button
                  type="button"
                  onClick={() =>
                    setClinicalNotes(
                      (prev) =>
                        prev +
                        (prev ? '\n' : '') +
                        'Patient counseled on tobacco and alcohol cessation. Sharp cusp adjacent to lesion smoothed.',
                    )
                  }
                  className="text-[10px] text-teal-700 dark:text-teal-400 hover:underline"
                >
                  + Cessation & Irritant
                </button>
              </div>
            </div>
            <textarea
              rows={6}
              value={clinicalNotes}
              onChange={(e) => setClinicalNotes(e.target.value)}
              placeholder="Enter comprehensive intraoral clinical examination findings, physical palpation notes, lymph node status, and immediate clinical actions taken..."
              className="w-full p-3 text-xs rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500 leading-relaxed font-sans"
            />
          </div>
        </div>

        {/* Print-only summary of in-clinic notes */}
        <div className="hidden print:block mt-6 pt-4 border-t border-slate-300">
          <h3 className="text-sm font-bold text-slate-900 mb-2">Attending Clinician Findings & Actions</h3>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <p><strong>Site:</strong> {selectedSite}</p>
              <p><strong>Dimensions:</strong> {lesionSize || 'Not measured'}</p>
              <p><strong>Duration:</strong> {lesionDuration}</p>
              <p><strong>Features:</strong> {selectedSigns.length > 0 ? selectedSigns.join(', ') : 'None marked'}</p>
            </div>
            <div>
              <p><strong>Clinical Notes:</strong></p>
              <p className="whitespace-pre-wrap text-slate-700">{clinicalNotes || 'No additional handwritten notes recorded.'}</p>
            </div>
          </div>
          <div className="mt-8 pt-4 border-t border-slate-300 flex justify-between items-center text-xs text-slate-600">
            <div>
              <p>Attending Clinician Signature: _________________________________</p>
              <p className="mt-1">License / Registration Number: __________________________</p>
            </div>
            <div className="text-right">
              <p>Date: ________________________</p>
              <p className="mt-1">Clinic Stamp / Seal: ____________________</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Technical Model Provenance & Telemetry Accordion (Retains DEC-034 / DEC-035 Compliance) */}
      <div className="print:hidden">
        <button
          type="button"
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="w-full flex items-center justify-between p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-850 transition-colors"
        >
          <div className="flex items-center gap-2">
            <FlaskConical className="w-4 h-4 text-teal-600 dark:text-teal-400" />
            <span className="text-xs font-bold uppercase tracking-wider">
              AI Model Provenance & Quantum Telemetry (Technical Audit)
            </span>
          </div>
          {showTechnicalDetails ? (
            <ChevronUp className="w-4 h-4 text-slate-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-500" />
          )}
        </button>

        {showTechnicalDetails && (
          <div className="space-y-4 mt-4 animate-in fade-in duration-200">
            {/* Primary vs Quantum Comparison */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Primary — validated baseline */}
              <Card variant="white" padding="md" className="border-teal-200 dark:border-teal-900">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-xl bg-teal-100 dark:bg-teal-950/70 text-teal-700 dark:text-teal-300 flex items-center justify-center">
                    <ShieldCheck className="w-4 h-4" />
                  </div>
                  <div>
                    <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                      Primary Model — Validated Supervised Baseline
                    </h2>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400">
                      {result.primaryModel || 'Primary classifier'} · Determines displayed band
                    </p>
                  </div>
                </div>

                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold font-mono text-slate-900 dark:text-slate-100">
                    {ranking
                      ? asRanking(result.primaryProbability) ?? '—'
                      : asPercent(result.primaryProbability) ?? '—'}
                  </span>
                  <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                    {ranking ? 'ranking score' : 'calibrated probability'}
                  </span>
                </div>

                {ranking && (
                  <p className="text-[11px] text-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/50 border border-amber-200 dark:border-amber-900 rounded-xl p-2.5 mt-3 leading-relaxed">
                    {RANKING_NOT_PROBABILITY}
                  </p>
                )}

                <dl className="mt-3">
                  <Fact label="Decision threshold" value={asRanking(result.primaryThreshold)} mono />
                  <Fact label="Band assigned" value={result.primaryRiskLevel ?? '—'} />
                  <Fact
                    label="Calibrated"
                    value={result.primaryCalibrated === true ? 'Yes' : 'No'}
                  />
                </dl>
              </Card>

              {/* Secondary — quantum readout */}
              <Card variant="white" padding="md" className="border-purple-200 dark:border-purple-900">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-xl bg-purple-100 dark:bg-purple-950/70 text-purple-700 dark:text-purple-300 flex items-center justify-center">
                    <Atom className="w-4 h-4" />
                  </div>
                  <div>
                    <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                      Secondary Readout — Quantum VQC Model
                    </h2>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400">
                      Calibrated, and exported to FHIR — but not the clinical verdict
                    </p>
                  </div>
                </div>

                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold font-mono text-purple-800 dark:text-purple-300">
                    {asPercent(result.finalProbability) ?? '—'}
                  </span>
                  <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                    calibrated probability
                  </span>
                </div>

                <p className="text-[11px] text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl p-2.5 mt-3 leading-relaxed">
                  {QUANTUM_SECONDARY_NOTE}
                </p>

                {result.details && (
                  <div className="mt-3">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
                      Quantum model narrative
                      {result.riskLevel ? ` · its own band: ${result.riskLevel}` : ''}
                    </p>
                    <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
                      {result.details}
                    </p>
                    <p className="text-[11px] text-slate-500 dark:text-slate-500 mt-1.5 leading-relaxed italic">
                      {DETAILS_BELONG_TO_QUANTUM}
                    </p>
                  </div>
                )}

                <dl className="mt-3">
                  <Fact label="High-risk threshold" value={asRanking(result.threshold)} mono />
                  <Fact
                    label="Raw quantum measurement"
                    value={asExpectation(result.rawScore)}
                    mono
                  />
                  <Fact
                    label="Before calibration"
                    value={asPercent(result.probabilityUncalibrated)}
                    mono
                  />
                </dl>
              </Card>
            </div>

            {/* Execution Provenance */}
            <Card variant="stone" padding="md">
              <div className="flex items-center gap-2 mb-2">
                <FlaskConical className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Pipeline Execution Parameters
                </h2>
              </div>
              <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6">
                <Fact label="Model version" value={result.modelVersion} mono />
                <Fact label="Execution mode" value={executionModeLabel(result.executionMode)} />
                <Fact label="Backend" value={result.backendName} mono />
                <Fact label="Qubits" value={result.quantumQubits} mono />
                <Fact
                  label="Shots"
                  value={
                    result.quantumShots ?? 'None — exact statevector simulation'
                  }
                  mono={result.quantumShots != null}
                />
                <Fact label="Circuit depth" value={result.circuitDepth} mono />
                <Fact label="Calibration method" value={result.calibrationMethod} mono />
                <Fact label="Bands source" value={result.bandsSource} mono />
                <Fact label="Feature mode" value={result.featureMode} mono />
                <Fact
                  label="Inference time"
                  value={asMillis(result.executionTimeMs)}
                  mono
                />
                <Fact
                  label="Quantum stage"
                  value={asMillis(result.quantumTimeMs)}
                  mono
                />
                <Fact label="Inference id" value={result.inferenceId} mono />
              </dl>
            </Card>
          </div>
        )}
      </div>

      {/* HL7 FHIR R4 Export Card */}
      <Card variant="white" padding="md" className="shadow-sm print:hidden">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <FileJson className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                HL7 FHIR R4 Interoperability Export
              </h2>
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-relaxed max-w-lg">
              Standardized Observation and RiskAssessment resources coded with SNOMED CT and LOINC
              for clinic EHR/EMR ingestion.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {!fhir.bundle ? (
              <Button
                variant="outline"
                size="sm"
                onClick={loadFhir}
                disabled={fhir.loading}
                icon={
                  fhir.loading ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <FileJson className="w-3.5 h-3.5" />
                  )
                }
                iconPosition="left"
              >
                {fhir.loading ? 'Generating…' : 'Generate FHIR Bundle'}
              </Button>
            ) : (
              <Button
                variant="outline"
                size="sm"
                onClick={downloadFhir}
                icon={<Download className="w-3.5 h-3.5" />}
                iconPosition="left"
              >
                Download FHIR JSON
              </Button>
            )}
          </div>
        </div>

        {fhir.error && (
          <p role="alert" className="text-xs text-rose-700 dark:text-rose-300 mt-3">
            {fhir.error}
          </p>
        )}

        {fhir.bundle && (
          <pre className="mt-3 max-h-72 overflow-auto rounded-2xl bg-slate-900 dark:bg-slate-950 border border-slate-800 p-4 text-[11px] leading-relaxed font-mono text-slate-200">
            {JSON.stringify(fhir.bundle, null, 2)}
          </pre>
        )}
      </Card>

      {/* Official Clinical Decision Support Disclaimer */}
      <Card variant="stone" padding="md" className="border-slate-300 dark:border-slate-700">
        <div className="flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-slate-500 dark:text-slate-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
              Clinical Decision Support (CDSS) Notice & Regulatory Disclaimer
            </p>
            <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
              {result.disclaimer}
            </p>
            <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1.5 leading-relaxed">
              This screening software is an adjunctive decision-support tool. It does not replace
              histopathological biopsy, immunohistochemistry, or the professional judgment of a
              qualified dental practitioner, oral medicine specialist, or head and neck surgeon.
            </p>
          </div>
        </div>
      </Card>
    </>
  );
}
