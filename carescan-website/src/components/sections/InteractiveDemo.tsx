/* eslint-disable @next/next/no-img-element */
'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { SectionHeader } from '../ui/SectionHeader';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { TEST_CASE_PRESETS } from '@/lib/mockData';
import { OrganicDivider } from '../visual/OrganicDivider';
import { useCredentials } from '@/context/CredentialsContext';
import { useBackendStatus } from '@/context/BackendStatusContext';
import { BackendStatusPill } from '../ui/BackendStatusPill';
import { api, ApiError } from '@/lib/api';
import type { AssessmentResult, FhirBundle } from '@/lib/apiTypes';
import {
  asExpectation,
  asPercent,
  asRanking,
  badgeVariantForTone,
  bandTone,
  DETAILS_BELONG_TO_QUANTUM,
  displayedBand,
  executionModeLabel,
  isRankingScore,
  QUANTUM_SECONDARY_NOTE,
  RANKING_NOT_PROBABILITY,
} from '@/lib/verdict';
import {
  AlertTriangle,
  Camera,
  FlaskConical,
  HeartHandshake,
  KeyRound,
  Loader2,
  Play,
  Upload,
} from 'lucide-react';

/** Local pipeline state for the preset lane. The patient lane uses the context's. */
type PresetStatus = 'idle' | 'running' | 'ready' | 'error';

/** Turn any thrown value into a line that can sit in front of a clinician. */
function describeFailure(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error && error.message) return error.message;
  return 'The screening could not be completed. Please try again.';
}

export const InteractiveDemo: React.FC = () => {
  const {
    credentials,
    patientProfile,
    openModal,
    updatePatientProfile,
    analyzePatientCase,
    analysisStatus,
    analysisError,
  } = useCredentials();

  // Whether the backend is answering right now. The sandbox posts a real
  // screening, so "Run" against a dead service produces a network error with no
  // explanation — the state is worth showing *before* the button is pressed.
  const { status: backendStatus } = useBackendStatus();

  // Selected case state: a described clinical scenario, or the patient's own photo.
  const [caseSource, setCaseSource] = useState<'preset' | 'patient'>('preset');
  const [selectedPresetId, setSelectedPresetId] = useState<string>(TEST_CASE_PRESETS[0].id);

  const [smoking, setSmoking] = useState<boolean>(false);
  const [alcohol, setAlcohol] = useState<boolean>(false);
  const [betelQuid, setBetelQuid] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'triage' | 'fhir'>('triage');

  // The preset lane keeps its own result so switching tabs does not disturb the
  // patient's screening, which lives in the shared profile.
  const [presetResult, setPresetResult] = useState<AssessmentResult | null>(null);
  const [presetStatus, setPresetStatus] = useState<PresetStatus>('idle');
  const [presetError, setPresetError] = useState<string | null>(null);

  // One tagged record rather than three loose flags. The tag is the screening the
  // bundle belongs to, so switching screenings can never show the previous one's
  // export, and "loading" is derived from the tag not matching rather than set on
  // the way into an effect.
  const [fhirState, setFhirState] = useState<{
    key: string;
    bundle: FhirBundle | null;
    error: string | null;
  } | null>(null);

  const selectedPreset =
    TEST_CASE_PRESETS.find((p) => p.id === selectedPresetId) || TEST_CASE_PRESETS[0];

  // Sync default lifestyle factors when preset changes
  const handleSelectPreset = (presetId: string) => {
    setCaseSource('preset');
    setSelectedPresetId(presetId);
    setPresetResult(null);
    setPresetStatus('idle');
    setPresetError(null);
    const p = TEST_CASE_PRESETS.find((item) => item.id === presetId);
    if (p) {
      setSmoking(p.lifestyle.smoking);
      setAlcohol(p.lifestyle.alcohol);
      setBetelQuid(p.lifestyle.betelQuid);
    }
  };

  // Switch to Patient Uploaded Case
  const handleSelectPatientCase = () => {
    setCaseSource('patient');
    setSmoking(patientProfile.tobaccoUse);
    setAlcohol(patientProfile.alcoholUse);
    setBetelQuid(patientProfile.betelNutUse);
  };

  /**
   * Screen the selected scenario.
   *
   * The three built-in scenarios are written clinical descriptions, not
   * photographs, so they post `is_mock: true` and the *server* decides what a
   * labelled stub result looks like. That still exercises the real route, schema,
   * persistence and FHIR export rather than inventing a score in the browser --
   * and every number that comes back is stamped MOCK by the backend itself.
   */
  const runPresetScreening = useCallback(async () => {
    setPresetStatus('running');
    setPresetError(null);
    setPresetResult(null);

    try {
      const result = await api.analyze({
        patient_id: `DEMO-${selectedPreset.id.toUpperCase()}`,
        // Seed material for the server's deterministic stub, never opened as a file.
        image_path: `preset://${selectedPreset.id}/scenario`,
        scan_type: 'Intra-oral Scan',
        // The sandbox always asks all three, so `false` means "asked and answered
        // no" -- which the clinical encoder keeps distinct from "not collected".
        smoking_history: smoking,
        alcohol_consumption: alcohol,
        betel_quid: betelQuid,
        // The written scenarios carry no age or sex, and guessing one would be
        // inventing patient data.
        age: null,
        sex: null,
        is_mock: true,
      });
      setPresetResult(result);
      setPresetStatus('ready');
    } catch (error) {
      setPresetError(describeFailure(error));
      setPresetStatus('error');
    }
  }, [selectedPreset.id, smoking, alcohol, betelQuid]);

  /** Push the toggles into the shared profile, then re-screen the patient's case. */
  const runPatientScreening = useCallback(async () => {
    updatePatientProfile({
      tobaccoUse: smoking,
      alcoholUse: alcohol,
      betelNutUse: betelQuid,
    });
    await analyzePatientCase();
  }, [updatePatientProfile, analyzePatientCase, smoking, alcohol, betelQuid]);

  const handleRun = useCallback(() => {
    if (caseSource === 'patient') {
      void runPatientScreening();
    } else {
      void runPresetScreening();
    }
  }, [caseSource, runPatientScreening, runPresetScreening]);

  // ------------------------------------------------------------- active result

  const patientAnalysis = patientProfile.analysisResult;

  const result: AssessmentResult | null =
    caseSource === 'patient' ? (patientAnalysis?.raw ?? null) : presetResult;

  const isBusy =
    caseSource === 'patient'
      ? analysisStatus === 'uploading' || analysisStatus === 'analyzing'
      : presetStatus === 'running';

  const errorText =
    caseSource === 'patient'
      ? analysisStatus === 'error'
        ? analysisError
        : null
      : presetStatus === 'error'
        ? presetError
        : null;

  const screeningId = result ? result.assessmentId || result.id : null;

  // Every displayed figure below is copied from the server payload. Nothing is
  // derived, adjusted, or recomputed here -- a screening score is a clinical
  // statement and the only thing entitled to make one is the model.
  const band = result ? displayedBand(result) : null;
  const tone = bandTone(band);
  const ranking = result ? isRankingScore(result) : true;
  const primaryScore = result?.primaryProbability ?? null;
  const quantumProbability = result?.finalProbability ?? null;
  const classicalProbability = result?.classicalProbability ?? null;
  const expectation = result?.rawScore ?? null;

  // ------------------------------------------------------------- FHIR (server)

  // Fetched only while the tab is open: the bundle is generated server-side, and
  // there is no reason to ask for one nobody is looking at.
  const fhirKey = activeTab === 'fhir' && screeningId ? screeningId : null;

  useEffect(() => {
    if (!fhirKey) return;

    const controller = new AbortController();
    let superseded = false;

    api
      .fhir(fhirKey, controller.signal)
      .then((bundle) => {
        if (!superseded) setFhirState({ key: fhirKey, bundle, error: null });
      })
      .catch((error: unknown) => {
        if (superseded || controller.signal.aborted) return;
        setFhirState({ key: fhirKey, bundle: null, error: describeFailure(error) });
      });

    return () => {
      superseded = true;
      controller.abort();
    };
  }, [fhirKey]);

  const settledFhir = fhirKey && fhirState?.key === fhirKey ? fhirState : null;
  const fhir = settledFhir?.bundle ?? null;
  const fhirError = settledFhir?.error ?? null;
  const fhirStatus: 'idle' | 'loading' | 'ready' | 'error' = !fhirKey
    ? 'idle'
    : !settledFhir
      ? 'loading'
      : settledFhir.error
        ? 'error'
        : 'ready';

  const runLabel = isBusy
    ? analysisStatus === 'uploading'
      ? 'Uploading…'
      : 'Screening…'
    : result
      ? 'Re-run Screening'
      : 'Run Screening';

  return (
    <section id="sandbox" className="relative py-14 md:py-20 bg-stone-50 dark:bg-slate-950 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
        {/* Section Header */}
        <SectionHeader
          badge="Interactive Point-of-Care Simulator"
          badgeVariant="teal"
          title="Try the Guided"
          highlightText="Orqis Triage Sandbox"
          subtitle="Follow the 3 steps below to submit a case to the live Orqis screening backend. Every score on this page is returned by the server — the classical baseline sets the headline band, and the 8-qubit angle-encoded quantum feature map is reported alongside it for transparency."
        />

        {/* Guided 3-Step Progress Header Banner */}
        <div className="max-w-4xl mx-auto bg-white dark:bg-slate-900 rounded-2xl p-4 sm:p-5 border border-slate-200 dark:border-slate-800 shadow-2xs">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-center sm:text-left">
            <div className="flex items-center gap-3 p-2 rounded-xl bg-teal-50/60 dark:bg-teal-950/40 border border-teal-100 dark:border-teal-800/60">
              <span className="w-7 h-7 rounded-full bg-teal-700 text-white font-bold text-xs flex items-center justify-center shrink-0">
                1
              </span>
              <div>
                <p className="text-xs font-bold text-slate-900 dark:text-white">Choose Patient Case</p>
                <p className="text-[11px] text-slate-600 dark:text-slate-400">Preset scenario or live photo</p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-2 rounded-xl bg-indigo-50/60 dark:bg-indigo-950/40 border border-indigo-100 dark:border-indigo-800/60">
              <span className="w-7 h-7 rounded-full bg-indigo-600 text-white font-bold text-xs flex items-center justify-center shrink-0">
                2
              </span>
              <div>
                <p className="text-xs font-bold text-slate-900 dark:text-white">Toggle Risk Factors</p>
                <p className="text-[11px] text-slate-600 dark:text-slate-400">Tobacco, alcohol, betel quid</p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-2 rounded-xl bg-purple-50/60 dark:bg-purple-950/40 border border-purple-100 dark:border-purple-800/60">
              <span className="w-7 h-7 rounded-full bg-purple-600 text-white font-bold text-xs flex items-center justify-center shrink-0">
                3
              </span>
              <div>
                <p className="text-xs font-bold text-slate-900 dark:text-white">Review AI &amp; Quantum Score</p>
                <p className="text-[11px] text-slate-600 dark:text-slate-400">See clinical referral output</p>
              </div>
            </div>
          </div>
        </div>

        {/* Sandbox Main Container */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Preset Selection, Patient Photo Case & Lifestyle Toggles */}
          <div className="lg:col-span-5 space-y-5">
            {/* Step 1: Case Selector */}
            <Card variant="white" padding="md" organic="subtle" className="border-slate-200 dark:border-slate-800 shadow-2xs">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                  Step 1 • Select Screening Case
                </span>
                <span className="text-[11px] text-teal-700 dark:text-teal-400 font-bold">
                  {caseSource === 'patient' ? 'Patient Photo Active' : 'Written Scenarios'}
                </span>
              </div>

              {/* Case Source Switcher Tabs */}
              <div className="grid grid-cols-2 gap-1.5 p-1 bg-slate-100 dark:bg-slate-800 rounded-xl mb-3">
                <button
                  type="button"
                  onClick={() => setCaseSource('preset')}
                  className={`text-xs font-bold py-1.5 px-2 rounded-lg transition-all cursor-pointer ${caseSource === 'preset'
                      ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-2xs'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }`}
                >
                  Clinical Presets
                </button>
                <button
                  type="button"
                  onClick={handleSelectPatientCase}
                  className={`text-xs font-bold py-1.5 px-2 rounded-lg transition-all cursor-pointer flex items-center justify-center gap-1.5 ${caseSource === 'patient'
                      ? 'bg-teal-700 text-white shadow-2xs'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }`}
                >
                  <Camera className="w-3.5 h-3.5" />
                  <span>Patient Photo Case</span>
                </button>
              </div>

              {/* View 1: Patient Photo Case Display */}
              {caseSource === 'patient' && (
                <div className="space-y-3 p-3 rounded-2xl bg-teal-50/70 dark:bg-teal-950/40 border border-teal-200 dark:border-teal-800">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-xs font-bold text-slate-900 dark:text-white">{patientProfile.name}</h4>
                      <p className="text-[11px] text-slate-600 dark:text-slate-400">
                        {patientProfile.age} yrs • {patientProfile.gender} • {patientProfile.symptomRegion}
                      </p>
                    </div>
                    {patientAnalysis ? (
                      <Badge variant={badgeVariantForTone(patientAnalysis.tone)} size="sm">
                        {patientAnalysis.band}
                      </Badge>
                    ) : (
                      <Badge variant="neutral" size="sm">
                        Not yet screened
                      </Badge>
                    )}
                  </div>

                  {/* Photo Preview with Scanner */}
                  {patientProfile.uploadedImage && (
                    <div className="relative w-full h-36 rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center border border-slate-800">
                      <img
                        src={patientProfile.uploadedImage}
                        alt="Patient oral photo"
                        className="w-full h-full object-contain"
                      />
                      {isBusy && (
                        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-teal-400/20 to-transparent animate-scan-line pointer-events-none" />
                      )}
                      <div className="absolute bottom-1.5 left-2 right-2 bg-slate-900/80 px-2 py-0.5 rounded text-[10px] text-teal-300 font-mono truncate flex items-center justify-between gap-2">
                        <span className="truncate">{patientProfile.uploadedImageName || 'patient_lesion.png'}</span>
                        {patientAnalysis ? (
                          patientAnalysis.isMock ? (
                            <span className="text-amber-300 font-bold shrink-0">TEST DATA</span>
                          ) : (
                            <span className="text-emerald-400 font-bold shrink-0">Screened by backend</span>
                          )
                        ) : (
                          <span className="text-slate-400 font-bold shrink-0">Not yet screened</span>
                        )}
                      </div>
                    </div>
                  )}

                  <div className="flex items-center justify-between pt-1">
                    <span className="text-[11px] text-slate-500 dark:text-slate-400 font-mono">
                      ID: {patientProfile.patientId}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => openModal('patient')}
                      icon={<Upload className="w-3 h-3 text-teal-700 dark:text-teal-300" />}
                      className="text-xs font-bold text-teal-800 dark:text-teal-300 bg-white dark:bg-slate-900 hover:bg-teal-50 dark:hover:bg-slate-800 border border-teal-200 dark:border-teal-800"
                    >
                      Change Photo / Patient
                    </Button>
                  </div>
                </div>
              )}

              {/* View 2: Preset Scenarios List */}
              {caseSource === 'preset' && (
                <div className="space-y-2">
                  {TEST_CASE_PRESETS.map((preset) => {
                    const isSelected = preset.id === selectedPresetId;
                    return (
                      <button
                        key={preset.id}
                        onClick={() => handleSelectPreset(preset.id)}
                        className={`w-full text-left p-3 rounded-2xl border transition-all cursor-pointer ${isSelected
                            ? 'bg-teal-50 dark:bg-teal-950/60 border-teal-600 dark:border-teal-500 ring-2 ring-teal-500/20 shadow-xs'
                            : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800'
                          }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-bold text-slate-900 dark:text-white">{preset.name}</span>
                          <Badge
                            variant={
                              preset.clinicalSeverity === 'High'
                                ? 'danger'
                                : preset.clinicalSeverity === 'Moderate'
                                  ? 'warning'
                                  : 'neutral'
                            }
                            size="sm"
                          >
                            {preset.category}
                          </Badge>
                        </div>
                        <p className="text-xs text-slate-600 dark:text-slate-400 line-clamp-2">
                          {preset.clinicalDescription}
                        </p>
                      </button>
                    );
                  })}
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 pt-1 leading-relaxed">
                    These are written presentations, not photographs. They are submitted to the
                    backend as labelled test data, so the server returns a result stamped{' '}
                    <span className="font-bold">MOCK</span>.
                  </p>
                </div>
              )}
            </Card>

            {/* Step 2: Lifestyle Auxiliary Risk Toggles */}
            <Card variant="white" padding="md" organic="subtle" className="border-slate-200 dark:border-slate-800 shadow-2xs">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                  Step 2 • Add Patient Risk Exposures
                </span>
                <span className="text-[11px] text-indigo-700 dark:text-indigo-400 font-bold">Lifestyle Inputs</span>
              </div>

              <div className="space-y-2.5">
                {[
                  {
                    label: 'Tobacco / Cigarette Smoking',
                    desc: 'Regular consumption (>5 pack-years)',
                    checked: smoking,
                    setter: setSmoking,
                  },
                  {
                    label: 'Regular Alcohol Consumption',
                    desc: 'Frequent or heavy alcohol intake',
                    checked: alcohol,
                    setter: setAlcohol,
                  },
                  {
                    label: 'Betel Quid / Areca Nut Chewing',
                    desc: 'High synergistic oral mucosal carcinogen',
                    checked: betelQuid,
                    setter: setBetelQuid,
                  },
                ].map((item, idx) => (
                  <label
                    key={idx}
                    className="flex items-center justify-between p-2.5 rounded-2xl border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/60 cursor-pointer transition-colors"
                  >
                    <div>
                      <p className="text-xs sm:text-sm font-bold text-slate-800 dark:text-slate-200">{item.label}</p>
                      <p className="text-[11px] text-slate-500 dark:text-slate-400">{item.desc}</p>
                    </div>
                    <input
                      type="checkbox"
                      checked={item.checked}
                      onChange={(e) => item.setter(e.target.checked)}
                      className="w-4 h-4 rounded text-teal-600 focus:ring-teal-500 border-slate-300 dark:border-slate-700 cursor-pointer"
                    />
                  </label>
                ))}
              </div>

              {/* A score only changes when the model runs again, so submitting is an
                  explicit act rather than a side effect of ticking a box. */}
              <Button
                variant="primary"
                size="md"
                onClick={handleRun}
                disabled={isBusy}
                icon={
                  isBusy ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )
                }
                iconPosition="left"
                className="w-full mt-3.5"
              >
                {runLabel}
              </Button>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-2 text-center">
                Submits to the Orqis backend. Editing a risk factor does not change a
                result until the screening is re-run.
              </p>
              {/* Liveness of the thing the button posts to. The button stays enabled
                  even while this reads "down": the poll is up to 5 s stale, and
                  refusing a run the user could have completed is worse than letting
                  them see the real error. */}
              <div className="flex flex-wrap items-center justify-center gap-2 mt-2.5">
                <BackendStatusPill />
                {backendStatus === 'offline' && (
                  <span className="text-[11px] text-rose-700 dark:text-rose-300 font-medium">
                    A run will fail until it is back.
                  </span>
                )}
              </div>
            </Card>
          </div>

          {/* Right Column: Live Triage & FHIR Output */}
          <div className="lg:col-span-7 space-y-5">
            {/* Active Practitioner & Patient Session Banner */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-2xl bg-teal-50/80 dark:bg-teal-950/40 border border-teal-200/90 dark:border-teal-800/80 shadow-2xs">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-teal-600 text-white flex items-center justify-center text-xs font-bold shadow-xs shrink-0">
                  <KeyRound className="w-4 h-4" />
                </div>
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs font-bold text-slate-900 dark:text-white">{credentials.name}</span>
                    <span className="text-[10px] font-bold text-teal-800 dark:text-teal-300 bg-teal-100/80 dark:bg-teal-900/60 px-2 py-0.5 rounded-full border border-teal-200 dark:border-teal-800">
                      {credentials.role}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-600 dark:text-slate-400 font-mono">
                    Facility: {credentials.facilityId} • Patient:{' '}
                    {caseSource === 'patient' ? patientProfile.name : selectedPreset.name}
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => openModal('clinician')}
                icon={<KeyRound className="w-3.5 h-3.5 text-teal-700 dark:text-teal-300" />}
                className="text-xs font-bold text-teal-800 dark:text-teal-300 hover:bg-white dark:hover:bg-slate-800 border border-teal-200/80 dark:border-teal-800 bg-white/80 dark:bg-slate-900 shadow-2xs"
              >
                Change Credentials
              </Button>
            </div>

            <Card variant="white" padding="lg" organic="subtle" className="border-teal-200 dark:border-slate-800 shadow-sm">
              {/* Tab Switcher */}
              <div className="flex items-center justify-between pb-3 mb-5 border-b border-slate-200 dark:border-slate-800">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setActiveTab('triage')}
                    className={`text-xs sm:text-sm font-bold px-3.5 py-1.5 rounded-full transition-colors cursor-pointer ${activeTab === 'triage'
                        ? 'bg-teal-700 text-white shadow-xs'
                        : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                      }`}
                  >
                    Clinical Triage Result
                  </button>
                  <button
                    onClick={() => setActiveTab('fhir')}
                    className={`text-xs sm:text-sm font-bold px-3.5 py-1.5 rounded-full transition-colors cursor-pointer ${activeTab === 'fhir'
                        ? 'bg-teal-700 text-white shadow-xs'
                        : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                      }`}
                  >
                    HL7 FHIR R4 Record
                  </button>
                </div>

                <span className="text-[11px] font-mono text-emerald-800 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-950/60 px-2.5 py-0.5 rounded-full border border-emerald-200 dark:border-emerald-800 font-bold hidden sm:inline">
                  {result ? executionModeLabel(result.executionMode) : 'Awaiting screening'}
                </span>
              </div>

              {/* Busy */}
              {isBusy && (
                <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
                  <Loader2 className="w-7 h-7 text-teal-600 dark:text-teal-400 animate-spin" />
                  <p className="text-sm font-bold text-slate-900 dark:text-white">
                    {analysisStatus === 'uploading'
                      ? 'Uploading the capture…'
                      : 'Running the screening pipeline…'}
                  </p>
                  <p className="text-xs text-slate-600 dark:text-slate-400 max-w-sm">
                    The backend is scoring this case. Results appear here as soon as the
                    server returns them.
                  </p>
                </div>
              )}

              {/* Error */}
              {!isBusy && errorText && (
                <div className="flex flex-col items-center justify-center gap-3 py-14 text-center">
                  <AlertTriangle className="w-7 h-7 text-rose-600 dark:text-rose-400" />
                  <p className="text-sm font-bold text-slate-900 dark:text-white">No screening result</p>
                  <p className="text-xs text-rose-700 dark:text-rose-300 max-w-md leading-relaxed">{errorText}</p>
                </div>
              )}

              {/* Idle */}
              {!isBusy && !errorText && !result && (
                <div className="flex flex-col items-center justify-center gap-3 py-14 text-center">
                  <Play className="w-7 h-7 text-slate-400 dark:text-slate-500" />
                  <p className="text-sm font-bold text-slate-900 dark:text-white">Nothing screened yet</p>
                  <p className="text-xs text-slate-600 dark:text-slate-400 max-w-sm leading-relaxed">
                    Pick a case in Step 1, set the risk exposures in Step 2, then run the
                    screening. No verdict is shown until the backend produces one.
                  </p>
                </div>
              )}

              {/* Tab 1: Clinical Triage Presentation */}
              {!isBusy && !errorText && result && activeTab === 'triage' && (
                <div className="space-y-5">
                  {/* Labelled test data banner */}
                  {result.isMock && (
                    <div className="flex items-start gap-2.5 p-3 rounded-2xl bg-amber-50 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800">
                      <FlaskConical className="w-4 h-4 text-amber-700 dark:text-amber-400 mt-0.5 shrink-0" />
                      <p className="text-[11px] text-amber-900 dark:text-amber-200 leading-relaxed">
                        <span className="font-bold">TEST DATA.</span> The backend served this
                        from its labelled development stub. The numbers below are not a
                        clinical assessment of anything.
                      </p>
                    </div>
                  )}

                  {/* Risk Level Banner */}
                  <div className="p-4 sm:p-5 rounded-3xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        Step 3 • Clinical Triage Band
                      </p>
                      <h4 className="text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-white mt-1">
                        {band ?? 'Unknown'}
                      </h4>
                      <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                        {caseSource === 'patient'
                          ? `Evaluated for ${patientProfile.name} on ${patientProfile.symptomRegion}`
                          : `Clinical presentation for ${selectedPreset.name}`}
                        {result.primaryModel ? ` • Band set by ${result.primaryModel}` : ''}
                      </p>
                    </div>
                    <div>
                      <Badge variant={badgeVariantForTone(tone)} pulse={tone === 'high'} size="md">
                        {band ?? 'Unknown'}
                      </Badge>
                    </div>
                  </div>

                  {/* Diagnostic Gauges Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
                    {/* Headline score from the primary (validated) model */}
                    <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-2xs">
                      <p className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                        {ranking ? 'Primary Ranking Score' : 'Primary Risk Probability'}
                      </p>
                      <p className="text-lg font-extrabold text-slate-900 dark:text-white font-mono mt-0.5">
                        {(ranking ? asRanking(primaryScore) : asPercent(primaryScore)) ?? '—'}
                      </p>
                      <div className="w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-full mt-1.5 overflow-hidden">
                        <div
                          className="bg-indigo-600 h-full transition-all duration-300"
                          style={{ width: `${Math.min(100, Math.max(0, (primaryScore ?? 0) * 100))}%` }}
                        />
                      </div>
                      <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">
                        {result.primaryModel ?? 'Primary model'}
                        {result.primaryThreshold != null
                          ? ` • threshold ${result.primaryThreshold.toFixed(3)}`
                          : ''}
                      </p>
                    </div>

                    {/* Quantum Pauli-Z expectation, pre-calibration */}
                    <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-2xs">
                      <p className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                        {result.quantumQubits ? `${result.quantumQubits}-Qubit VQC ⟨Z⟩` : 'VQC ⟨Z⟩'}
                      </p>
                      <p className="text-lg font-extrabold text-purple-700 dark:text-purple-400 font-mono mt-0.5">
                        {asExpectation(expectation) ?? '—'}
                      </p>
                      <div className="w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-full mt-1.5 overflow-hidden">
                        <div
                          className="bg-purple-600 h-full transition-all duration-300"
                          style={{
                            width: `${Math.min(100, Math.max(0, (1 - ((expectation ?? 0) + 1) / 2) * 100))}%`,
                          }}
                        />
                      </div>
                      <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">
                        Raw measurement before calibration — not a probability
                      </p>
                    </div>

                    {/* Calibrated quantum probability: secondary readout, FHIR value */}
                    <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-2xs">
                      <p className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                        Calibrated Quantum Probability
                      </p>
                      <p className="text-lg font-extrabold text-teal-800 dark:text-teal-400 font-mono mt-0.5">
                        {asPercent(quantumProbability) ?? '—'}
                      </p>
                      <div className="w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-full mt-1.5 overflow-hidden">
                        <div
                          className={`h-full transition-all duration-300 ${tone === 'high'
                              ? 'bg-rose-500'
                              : tone === 'moderate'
                                ? 'bg-amber-500'
                                : 'bg-emerald-500'
                            }`}
                          style={{
                            width: `${Math.min(100, Math.max(0, (quantumProbability ?? 0) * 100))}%`,
                          }}
                        />
                      </div>
                      <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">
                        Secondary readout
                        {result.calibrationMethod ? ` • ${result.calibrationMethod}` : ''}
                      </p>
                    </div>
                  </div>

                  {/* DEC-034: an uncalibrated headline score carries no percentage meaning */}
                  {ranking && primaryScore != null && (
                    <div className="p-3 rounded-2xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                      <p className="text-[11px] text-slate-700 dark:text-slate-300 leading-relaxed">
                        {RANKING_NOT_PROBABILITY}
                      </p>
                    </div>
                  )}

                  {/* Honest standing of the quantum model on this dataset */}
                  <div className="p-3 rounded-2xl bg-indigo-50/70 dark:bg-indigo-950/40 border border-indigo-200 dark:border-indigo-800">
                    <p className="text-[11px] text-indigo-950 dark:text-indigo-200 leading-relaxed">
                      {QUANTUM_SECONDARY_NOTE}
                    </p>
                  </div>

                  {/* The backend's narrative. It is written from the *quantum*
                      probability, in the same call that sets the quantum band, so
                      it is attributed rather than presented as the headline's own
                      explanation — the two bands disagree often on this dataset. */}
                  <div className="p-4 rounded-2xl bg-teal-50/60 dark:bg-teal-950/40 border border-teal-200 dark:border-teal-800 space-y-1.5">
                    <div className="flex items-center gap-2 text-teal-950 dark:text-teal-300 font-bold text-xs uppercase tracking-wider">
                      <HeartHandshake className="w-4 h-4 text-teal-700 dark:text-teal-400" />
                      Clinical Action Plan for Health Worker &amp; Patient
                    </div>
                    <p className="text-xs sm:text-sm text-slate-800 dark:text-slate-200 leading-relaxed font-medium">
                      {result.details}
                    </p>
                    <p className="text-[10px] text-teal-900/80 dark:text-teal-300/80 leading-relaxed italic border-t border-teal-200/70 dark:border-teal-800/60 pt-1.5">
                      {DETAILS_BELONG_TO_QUANTUM}
                    </p>
                  </div>

                  {/* Provenance + the server's verbatim disclaimer */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-[10px]">
                    {[
                      { label: 'Model', value: result.modelVersion },
                      { label: 'Engine', value: executionModeLabel(result.executionMode) },
                      { label: 'Backend', value: result.backendName },
                      {
                        label: 'Circuit depth',
                        value: result.circuitDepth != null ? String(result.circuitDepth) : null,
                      },
                      {
                        label: 'Classical baseline',
                        value: asPercent(classicalProbability),
                      },
                      {
                        label: 'Latency',
                        value:
                          result.executionTimeMs != null
                            ? `${Math.round(result.executionTimeMs)} ms`
                            : null,
                      },
                      { label: 'Feature mode', value: result.featureMode },
                      { label: 'Screening', value: screeningId ? screeningId.slice(0, 8) : null },
                    ].map((item) => (
                      <div
                        key={item.label}
                        className="p-2 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800"
                      >
                        <p className="font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                          {item.label}
                        </p>
                        <p className="font-mono text-slate-800 dark:text-slate-200 truncate mt-0.5">
                          {item.value ?? '—'}
                        </p>
                      </div>
                    ))}
                  </div>

                  <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed border-t border-slate-200 dark:border-slate-800 pt-3">
                    {result.disclaimer}
                  </p>
                </div>
              )}

              {/* Tab 2: HL7 FHIR R4 JSON Record */}
              {!isBusy && !errorText && result && activeTab === 'fhir' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-400">
                    <span className="font-bold">HL7 FHIR R4 Bundle — served by the backend</span>
                    <span className="font-mono text-slate-500 dark:text-slate-400">SNOMED CT: 363349007</span>
                  </div>

                  {fhirStatus === 'loading' && (
                    <div className="flex items-center gap-2 p-4 text-xs text-slate-600 dark:text-slate-400">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Fetching the FHIR record for screening {screeningId?.slice(0, 8)}…
                    </div>
                  )}

                  {fhirStatus === 'error' && (
                    <div className="flex items-start gap-2 p-4 rounded-2xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800">
                      <AlertTriangle className="w-4 h-4 text-rose-600 dark:text-rose-400 mt-0.5 shrink-0" />
                      <p className="text-xs text-rose-800 dark:text-rose-300 leading-relaxed">{fhirError}</p>
                    </div>
                  )}

                  {fhirStatus === 'ready' && fhir && (
                    <pre className="p-3.5 rounded-2xl bg-slate-900 text-teal-300 font-mono text-xs overflow-x-auto max-h-80 border border-slate-800 leading-relaxed">
                      {JSON.stringify(fhir, null, 2)}
                    </pre>
                  )}

                  <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
                    Exported by <span className="font-mono">GET /api/screening/{'{id}'}/fhir</span>. The
                    probability it carries is the calibrated quantum value, matching the
                    secondary readout on the triage tab.
                  </p>
                </div>
              )}
            </Card>
          </div>
        </div>
      </div>

      <div className="mt-12">
        <OrganicDivider position="bottom" fillColor="var(--warm-surface)" variant="curve-1" />
      </div>
    </section>
  );
};
