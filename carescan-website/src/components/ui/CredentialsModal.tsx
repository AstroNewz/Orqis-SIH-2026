/* eslint-disable @next/next/no-img-element */
'use client';

import React, { useState, useRef } from 'react';
import {
  useCredentials,
  CREDENTIAL_PRESETS,
  PATIENT_PRESETS,
  DEFAULT_CREDENTIALS,
  DEFAULT_PATIENT,
} from '@/context/CredentialsContext';
import {
  asExpectation,
  asPercent,
  asRanking,
  executionModeLabel,
  isRankingScore,
  QUANTUM_SECONDARY_NOTE,
  RANKING_NOT_PROBABILITY,
} from '@/lib/verdict';
import { Button } from './Button';
import {
  X,
  KeyRound,
  User,
  Building2,
  ShieldCheck,
  RefreshCw,
  Copy,
  Check,
  Server,
  Sparkles,
  Lock,
  Stethoscope,
  Upload,
  Camera,
  Image as ImageIcon,
  Activity,
  AlertTriangle,
  FlaskConical,
  Loader2,
  CheckCircle2,
} from 'lucide-react';

const CredentialsModalDialog: React.FC = () => {
  const {
    activeAuthTab,
    setActiveAuthTab,
    credentials,
    patientProfile,
    closeModal,
    updateCredentials,
    updatePatientProfile,
    applyPreset,
    applyPatientPreset,
    resetDefaults,
    resetPatientDefaults,
    generateNewKey,
    uploadPatientImage,
    analyzePatientCase,
    analysisStatus,
    analysisError,
  } = useCredentials();

  // Clinician Form State
  const [clinicianForm, setClinicianForm] = useState(credentials);
  const [copiedKey, setCopiedKey] = useState(false);
  const [clinicianSaveSuccess, setClinicianSaveSuccess] = useState(false);

  // Patient Form State
  const [patientForm, setPatientForm] = useState(patientProfile);
  const [patientSaveSuccess, setPatientSaveSuccess] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // The verdict is whatever the backend last returned, so it is read from the
  // context rather than from the local form copy: an edit to a risk factor must not
  // look as though it changed a score that no model has re-computed.
  const analysis = patientProfile.analysisResult;
  const isBusy = analysisStatus === 'uploading' || analysisStatus === 'analyzing';

  // Handle Clinician Save
  const handleSaveClinician = (e: React.FormEvent) => {
    e.preventDefault();
    updateCredentials(clinicianForm);
    setClinicianSaveSuccess(true);
    setTimeout(() => {
      setClinicianSaveSuccess(false);
      closeModal();
    }, 1000);
  };

  // Handle Patient Save & Launch Sandbox
  const handleSavePatient = (e?: React.FormEvent, navigateToSandbox = false) => {
    if (e) e.preventDefault();
    // Commit the edited demographics and risk factors first, then ask the backend to
    // score them. Nothing is computed here -- a screening verdict comes from the
    // model or it does not exist.
    updatePatientProfile(patientForm);
    void analyzePatientCase();
    setPatientSaveSuccess(true);
    setTimeout(() => {
      setPatientSaveSuccess(false);
      closeModal();
      if (navigateToSandbox) {
        const el = document.getElementById('sandbox');
        if (el) {
          el.scrollIntoView({ behavior: 'smooth' });
        }
      }
    }, 900);
  };

  // Handle File Upload from Input or Drag-Drop
  const handleFileChange = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPG, PNG, or WebP)');
      return;
    }

    // Show the capture straight away so the preview is not waiting on the network,
    // then commit the current form so the screening carries the risk factors on
    // screen rather than the ones last saved.
    const preview = URL.createObjectURL(file);
    setPatientForm((prev) => ({
      ...prev,
      uploadedImage: preview,
      uploadedImageName: file.name,
      uploadedImagePath: null,
    }));
    updatePatientProfile({ ...patientForm });

    await uploadPatientImage(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFileChange(e.dataTransfer.files);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 overflow-y-auto animate-in fade-in duration-200 font-body">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-slate-900/65 backdrop-blur-sm transition-opacity"
        onClick={closeModal}
      />

      {/* Modal Container */}
      <div className="relative w-full max-w-3xl bg-white dark:bg-slate-900 rounded-3xl shadow-2xl border border-slate-200/90 dark:border-slate-800 overflow-hidden z-10 my-6 transition-colors">
        {/* Header with Dual Role Tab Switcher */}
        <div className="bg-gradient-to-r from-teal-950 via-slate-900 to-indigo-950 p-5 sm:px-7 sm:py-5 text-white border-b border-white/10">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-teal-500/20 border border-teal-400/30 flex items-center justify-center text-teal-300 shrink-0">
                {activeAuthTab === 'patient' ? (
                  <User className="w-5 h-5" />
                ) : (
                  <KeyRound className="w-5 h-5" />
                )}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-heading text-lg font-bold tracking-tight text-white">
                    {activeAuthTab === 'patient'
                      ? 'Patient Self-Screening Portal'
                      : 'Healthcare Clinician & Researcher Portal'}
                  </h3>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-teal-300 bg-teal-900/60 px-2 py-0.5 rounded-full border border-teal-700/50">
                    Local Session
                  </span>
                </div>
                <p className="text-xs text-slate-300 font-body">
                  {activeAuthTab === 'patient'
                    ? 'Direct image upload, backend screening & risk assessment'
                    : 'Configure practitioner credentials for clinical screening & FHIR telemetry'}
                </p>
              </div>
            </div>
            <button
              onClick={closeModal}
              className="p-2 rounded-full text-slate-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
              aria-label="Close modal"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Mode Tabs Switcher */}
          <div className="grid grid-cols-2 p-1 bg-slate-800/80 rounded-2xl border border-slate-700/80">
            <button
              type="button"
              onClick={() => setActiveAuthTab('patient')}
              className={`flex items-center justify-center gap-2 py-2 px-3 rounded-xl text-xs sm:text-sm font-bold transition-all cursor-pointer ${
                activeAuthTab === 'patient'
                  ? 'bg-teal-600 text-white shadow-md'
                  : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              <Camera className="w-4 h-4" />
              <span>Patient Self-Screening & Image Check</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveAuthTab('clinician')}
              className={`flex items-center justify-center gap-2 py-2 px-3 rounded-xl text-xs sm:text-sm font-bold transition-all cursor-pointer ${
                activeAuthTab === 'clinician'
                  ? 'bg-teal-600 text-white shadow-md'
                  : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              <Stethoscope className="w-4 h-4" />
              <span>Clinician & Researcher Login</span>
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="p-5 sm:p-7 space-y-6 max-h-[75vh] overflow-y-auto">
          {/* ========================================================================= */}
          {/* PATIENT SELF-SCREENING & DIRECT IMAGE ANOMALY CHECK TAB */}
          {/* ========================================================================= */}
          {activeAuthTab === 'patient' && (
            <div className="space-y-6">
              {/* Patient Quick Scenario Presets */}
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
                    Quick Sample Patient Scenarios
                  </label>
                  <span className="text-[11px] text-slate-500 dark:text-slate-400">1-click test cases with clinical photos</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                  {PATIENT_PRESETS.map((preset) => {
                    const isSelected = patientForm.patientId === preset.patientId;
                    return (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => {
                          // Applying the preset also submits it to the backend as a
                          // labelled mock; the verdict arrives from the server.
                          applyPatientPreset(preset.id);
                          setPatientForm({
                            name: preset.name,
                            patientId: preset.patientId,
                            age: preset.age,
                            gender: preset.gender,
                            symptomRegion: preset.symptomRegion,
                            duration: preset.duration,
                            tobaccoUse: preset.tobaccoUse,
                            betelNutUse: preset.betelNutUse,
                            alcoholUse: preset.alcoholUse,
                            uploadedImage: preset.sampleImage,
                            uploadedImageName: preset.sampleImageName,
                            uploadedImagePath: null,
                            analysisResult: null,
                          });
                        }}
                        className={`p-3 rounded-2xl border text-left transition-all cursor-pointer ${
                          isSelected
                            ? 'border-teal-600 bg-teal-50/80 dark:bg-teal-950/40 ring-2 ring-teal-500/20 shadow-xs'
                            : 'border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 bg-slate-50/60 dark:bg-slate-800/40 hover:bg-slate-50 dark:hover:bg-slate-800'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span
                            className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded-full ${
                              preset.badgeVariant === 'danger'
                                ? 'bg-rose-100 dark:bg-rose-950/60 text-rose-800 dark:text-rose-300 border border-rose-200 dark:border-rose-800'
                                : preset.badgeVariant === 'warning'
                                ? 'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-800'
                                : 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800'
                            }`}
                          >
                            {preset.badge}
                          </span>
                        </div>
                        <span className="block text-xs font-bold text-slate-900 dark:text-slate-100 truncate">
                          {preset.name}
                        </span>
                        <span className="block text-[11px] text-slate-500 dark:text-slate-400 line-clamp-1 mt-0.5">
                          {preset.description}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Direct Image Upload / Camera Input Area */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                    <Camera className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
                    Direct Oral Image Input for Anomaly Detection
                  </label>
                  <span className="text-[11px] text-slate-500 dark:text-slate-400">
                    JPG, PNG, BMP or WebP · up to 25 MB
                  </span>
                </div>

                {/* Upload Box & Image Preview */}
                <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                  {/* Dropzone */}
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`md:col-span-6 p-5 rounded-2xl border-2 border-dashed flex flex-col items-center justify-center text-center cursor-pointer transition-all ${
                      dragActive
                        ? 'border-teal-600 bg-teal-50 dark:bg-teal-950/30'
                        : 'border-slate-300 dark:border-slate-700 hover:border-teal-500 bg-slate-50/80 dark:bg-slate-800/40 hover:bg-teal-50/30 dark:hover:bg-teal-950/20'
                    }`}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => handleFileChange(e.target.files)}
                    />
                    <div className="w-11 h-11 rounded-2xl bg-teal-100 dark:bg-teal-900/60 text-teal-700 dark:text-teal-300 flex items-center justify-center mb-2 shadow-xs">
                      <Upload className="w-5 h-5" />
                    </div>
                    <p className="text-xs sm:text-sm font-bold text-slate-800 dark:text-slate-200">
                      Upload Oral Photo or Drag Here
                    </p>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 max-w-xs">
                      Click to browse from your device or camera capture. Sent to the Orqis
                      screening backend for analysis.
                    </p>
                  </div>

                  {/* Image Preview & Anomaly Trigger Card */}
                  <div className="md:col-span-6 bg-slate-900 dark:bg-slate-950 rounded-2xl p-3.5 border border-slate-800 dark:border-slate-800/80 flex flex-col items-center justify-center text-white relative overflow-hidden min-h-[170px]">
                    {patientForm.uploadedImage ? (
                      <div className="relative w-full h-36 rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center group">
                        <img
                          src={patientForm.uploadedImage}
                          alt="Oral scan preview"
                          className="w-full h-full object-contain"
                        />
                        {/* Scanning beam animation */}
                        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-teal-400/20 to-transparent animate-scan-line pointer-events-none" />
                        <div className="absolute bottom-2 left-2 right-2 bg-slate-900/80 backdrop-blur-xs px-2.5 py-1 rounded-lg flex items-center justify-between text-[11px]">
                          <span className="font-mono text-teal-300 truncate max-w-[160px]">
                            {patientForm.uploadedImageName || 'oral_photo.png'}
                          </span>
                          <span className="text-emerald-400 font-bold flex items-center gap-1 text-[10px]">
                            <CheckCircle2 className="w-3 h-3" /> Ready
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center p-4 text-slate-400">
                        <ImageIcon className="w-8 h-8 mx-auto mb-2 opacity-50" />
                        <p className="text-xs">No image loaded yet</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Patient Demographics & Risk Inputs */}
              <form onSubmit={handleSavePatient} className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {/* Patient Name / Pseudonym */}
                  <div className="space-y-1 sm:col-span-2">
                    <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                      <User className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500" />
                      Patient Name / Pseudonymous Label
                    </label>
                    <input
                      type="text"
                      required
                      value={patientForm.name}
                      onChange={(e) => setPatientForm({ ...patientForm, name: e.target.value })}
                      placeholder="e.g. Rohan Verma (Self-Check)"
                      className="w-full px-3.5 py-2 text-xs sm:text-sm bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600 dark:text-slate-100 transition-colors"
                    />
                  </div>

                  {/* Patient ID with Generator */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">Patient ID</label>
                      <button
                        type="button"
                        onClick={() => {
                          const randomHex = Math.random().toString(36).substring(2, 7).toUpperCase();
                          setPatientForm({ ...patientForm, patientId: `PAT-SELF-${randomHex}` });
                        }}
                        className="text-[10px] text-teal-600 dark:text-teal-400 hover:text-teal-700 font-bold"
                      >
                        New ID
                      </button>
                    </div>
                    <input
                      type="text"
                      required
                      value={patientForm.patientId}
                      onChange={(e) => setPatientForm({ ...patientForm, patientId: e.target.value })}
                      className="w-full px-3 py-2 text-xs bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 font-mono text-slate-800 dark:text-slate-100"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {/* Age */}
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">Age</label>
                    <input
                      type="number"
                      value={patientForm.age}
                      onChange={(e) => setPatientForm({ ...patientForm, age: e.target.value })}
                      className="w-full px-3 py-2 text-xs bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 dark:text-slate-100"
                    />
                  </div>

                  {/* Gender */}
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">Gender</label>
                    <select
                      value={patientForm.gender}
                      onChange={(e) => setPatientForm({ ...patientForm, gender: e.target.value })}
                      className="w-full px-3 py-2 text-xs bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 text-slate-800 dark:text-slate-100"
                    >
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other / Non-Binary</option>
                      <option value="Prefer not to say">Prefer not to say</option>
                    </select>
                  </div>

                  {/* Oral Region */}
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">Symptom Location</label>
                    <select
                      value={patientForm.symptomRegion}
                      onChange={(e) =>
                        setPatientForm({ ...patientForm, symptomRegion: e.target.value })
                      }
                      className="w-full px-3 py-2 text-xs bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 text-slate-800 dark:text-slate-100"
                    >
                      <option value="Buccal Mucosa (Inner Cheek)">Buccal Mucosa (Inner Cheek)</option>
                      <option value="Left Lateral Border of Tongue">Left Lateral Tongue</option>
                      <option value="Right Lateral Border of Tongue">Right Lateral Tongue</option>
                      <option value="Dorsal Tongue">Dorsal Tongue</option>
                      <option value="Floor of Mouth">Floor of Mouth</option>
                      <option value="Gingiva / Gums">Gingiva / Gums</option>
                      <option value="Palate (Roof of Mouth)">Palate (Roof of Mouth)</option>
                      <option value="Lips / Labial Mucosa">Lips / Labial Mucosa</option>
                    </select>
                  </div>
                </div>

                {/* Risk Exposures */}
                <div className="p-3.5 bg-slate-50 dark:bg-slate-800/40 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-2">
                  <span className="text-xs font-bold text-slate-700 dark:text-slate-300 block">
                    Patient Lifestyle & Habit Exposures
                  </span>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    <label className="flex items-center gap-2 p-2 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs text-slate-800 dark:text-slate-200 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={patientForm.tobaccoUse}
                        onChange={(e) =>
                          setPatientForm({ ...patientForm, tobaccoUse: e.target.checked })
                        }
                        className="rounded text-teal-600 focus:ring-teal-500"
                      />
                      <span>Tobacco / Smoking</span>
                    </label>

                    <label className="flex items-center gap-2 p-2 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs text-slate-800 dark:text-slate-200 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={patientForm.betelNutUse}
                        onChange={(e) =>
                          setPatientForm({ ...patientForm, betelNutUse: e.target.checked })
                        }
                        className="rounded text-teal-600 focus:ring-teal-500"
                      />
                      <span>Betel Nut / Quid</span>
                    </label>

                    <label className="flex items-center gap-2 p-2 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs text-slate-800 dark:text-slate-200 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={patientForm.alcoholUse}
                        onChange={(e) =>
                          setPatientForm({ ...patientForm, alcoholUse: e.target.checked })
                        }
                        className="rounded text-teal-600 focus:ring-teal-500"
                      />
                      <span>Alcohol Intake</span>
                    </label>
                  </div>
                </div>

                {/* Screening pipeline: in-flight, failed, or the backend's verdict. */}
                {isBusy && (
                  <div className="p-4 rounded-2xl bg-slate-900 dark:bg-slate-950 text-white border border-teal-500/30 flex items-center gap-3">
                    <Loader2 className="w-4 h-4 text-teal-400 animate-spin shrink-0" />
                    <div>
                      <span className="block text-xs font-bold text-teal-300">
                        {analysisStatus === 'uploading'
                          ? 'Uploading the capture to the screening backend…'
                          : 'Running the screening pipeline…'}
                      </span>
                      <span className="block text-[11px] text-slate-400 mt-0.5">
                        Quality control, ROI localization, classical scoring, the quantum feature
                        map and calibration all run server-side.
                      </span>
                    </div>
                  </div>
                )}

                {analysisStatus === 'error' && analysisError && (
                  <div className="p-4 rounded-2xl bg-rose-950/80 text-white border border-rose-500/40 flex items-start gap-3">
                    <AlertTriangle className="w-4 h-4 text-rose-300 shrink-0 mt-0.5" />
                    <div className="space-y-1">
                      <span className="block text-xs font-bold text-rose-200">
                        No screening result
                      </span>
                      <p className="text-[11px] text-rose-100/90 leading-relaxed">{analysisError}</p>
                    </div>
                  </div>
                )}

                {!isBusy && analysisStatus !== 'error' && analysis && (
                  <div className="p-4 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-teal-950 text-white border border-teal-500/30 space-y-3">
                    {/* The server stamps this, not the browser. A mock verdict must never
                        be mistakable for a screening. */}
                    {analysis.isMock && (
                      <div className="flex items-start gap-2 p-2.5 rounded-xl bg-amber-500/15 border border-amber-400/40">
                        <FlaskConical className="w-3.5 h-3.5 text-amber-300 shrink-0 mt-0.5" />
                        <p className="text-[11px] text-amber-100 leading-relaxed">
                          <span className="font-bold">TEST DATA — no inference was run.</span>{' '}
                          This demonstration scenario ships as an illustration rather than a
                          photograph, so the backend returned its labelled development stub. The
                          numbers below carry no clinical meaning.
                        </p>
                      </div>
                    )}

                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4 text-teal-400" />
                        <span className="font-heading text-xs font-bold uppercase tracking-wider text-teal-300">
                          Screening Verdict
                        </span>
                      </div>
                      <span
                        className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${
                          analysis.tone === 'high'
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                            : analysis.tone === 'moderate'
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                            : analysis.tone === 'low'
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                            : 'bg-slate-500/20 text-slate-300 border border-slate-500/40'
                        }`}
                      >
                        {analysis.band}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                      {/* Headline score. When the primary model is uncalibrated this is a
                          ranking, so it is never rendered with a percent sign. */}
                      <div className="bg-white/5 p-2.5 rounded-xl border border-white/10">
                        <span className="text-[10px] text-slate-400 block">
                          {isRankingScore(analysis.raw) ? 'Ranking Score' : 'Risk Probability'}
                        </span>
                        <span className="text-base font-bold font-mono text-teal-300">
                          {(isRankingScore(analysis.raw)
                            ? asRanking(analysis.primaryScore)
                            : asPercent(analysis.primaryScore)) ?? '—'}
                        </span>
                        <span className="text-[9px] text-slate-500 block truncate">
                          {analysis.primaryModel ?? 'primary model'}
                        </span>
                      </div>

                      <div className="bg-white/5 p-2.5 rounded-xl border border-white/10">
                        <span className="text-[10px] text-slate-400 block">Quantum VQC ⟨Z⟩</span>
                        <span className="text-base font-bold font-mono text-indigo-300">
                          {asExpectation(analysis.rawScore) ?? '—'}
                        </span>
                        <span className="text-[9px] text-slate-500 block">
                          pre-calibration measurement
                        </span>
                      </div>

                      <div className="bg-white/5 p-2.5 rounded-xl border border-white/10 col-span-2 sm:col-span-1">
                        <span className="text-[10px] text-slate-400 block">Analysis Engine</span>
                        <span className="text-[11px] font-semibold text-slate-200 block leading-snug">
                          {executionModeLabel(analysis.executionMode)}
                        </span>
                        <span className="text-[9px] text-slate-500 block">
                          {analysis.quantumQubits ? `${analysis.quantumQubits} qubits` : ''}
                          {analysis.modelVersion ? ` · ${analysis.modelVersion}` : ''}
                        </span>
                      </div>
                    </div>

                    {isRankingScore(analysis.raw) && (
                      <p className="text-[10px] text-amber-200/90 leading-relaxed">
                        {RANKING_NOT_PROBABILITY}
                      </p>
                    )}

                    {/* Secondary quantum readout: the calibrated probability that goes into
                        FHIR, shown for transparency and never as the verdict. */}
                    <div className="bg-indigo-950/50 border border-indigo-700/40 p-3 rounded-xl space-y-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[11px] font-bold text-indigo-200">
                          Secondary: calibrated quantum probability
                        </span>
                        <span className="text-sm font-bold font-mono text-indigo-200">
                          {asPercent(analysis.quantumProbability) ?? '—'}
                        </span>
                      </div>
                      <p className="text-[10px] text-indigo-100/70 leading-relaxed">
                        {QUANTUM_SECONDARY_NOTE}
                      </p>
                    </div>

                    {/* The backend's narrative, and its disclaimer. Both are rendered
                        verbatim -- rewording a clinical statement in the browser is
                        exactly the failure this whole layer exists to prevent. The
                        narrative is written from the *quantum* probability, so it sits
                        under the quantum panel above and says whose band it describes;
                        the headline band is the primary model's (DEC-034). */}
                    <div className="bg-teal-950/70 border border-teal-700/50 p-3 rounded-xl text-xs space-y-1.5">
                      <span className="text-[9px] font-bold uppercase tracking-wider text-teal-300/80 block">
                        Quantum model narrative
                        {analysis.raw.riskLevel ? ` · its own band: ${analysis.raw.riskLevel}` : ''}
                      </span>
                      <p className="text-slate-200 leading-relaxed text-[11px]">
                        {analysis.details}
                      </p>
                      <p className="text-teal-200/80 leading-relaxed text-[10px] border-t border-teal-800/60 pt-1.5">
                        {analysis.disclaimer}
                      </p>
                    </div>

                    <p className="text-[9px] text-slate-500 font-mono truncate">
                      screening {analysis.screeningId}
                      {analysis.calibrationMethod ? ` · ${analysis.calibrationMethod}` : ''}
                      {analysis.circuitDepth ? ` · depth ${analysis.circuitDepth}` : ''}
                    </p>
                  </div>
                )}

                {/* Footer Buttons */}
                <div className="pt-3 border-t border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      resetPatientDefaults();
                      setPatientForm(DEFAULT_PATIENT);
                    }}
                    className="text-xs text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 font-medium transition-colors cursor-pointer"
                  >
                    Reset Patient Form
                  </button>

                  <div className="flex items-center gap-2 w-full sm:w-auto">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={closeModal}
                      className="flex-1 sm:flex-initial text-xs"
                    >
                      Close
                    </Button>
                    <Button
                      type="button"
                      variant="primary"
                      size="sm"
                      disabled={isBusy}
                      onClick={() => handleSavePatient(undefined, true)}
                      icon={<Sparkles className="w-3.5 h-3.5" />}
                      className="flex-1 sm:flex-initial text-xs"
                    >
                      {isBusy ? 'Screening…' : patientSaveSuccess ? 'Applied!' : 'Open in Triage Sandbox'}
                    </Button>
                  </div>
                </div>
              </form>
            </div>
          )}

          {/* ========================================================================= */}
          {/* CLINICIAN & RESEARCHER LOGIN TAB */}
          {/* ========================================================================= */}
          {activeAuthTab === 'clinician' && (
            <div className="space-y-6">
              {/* Preset Roles Quick Selector */}
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-300 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
                    Quick-Select Clinician Profiles
                  </label>
                  <span className="text-[11px] text-slate-400 dark:text-slate-500">Click to auto-populate</span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {CREDENTIAL_PRESETS.map((preset) => {
                    const isSelected =
                      clinicianForm.role === preset.role && clinicianForm.name === preset.name;
                    return (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => {
                          applyPreset(preset.id);
                          const p = CREDENTIAL_PRESETS.find((x) => x.id === preset.id);
                          if (p) {
                            setClinicianForm({
                              name: p.name,
                              role: p.role,
                              facilityId: p.facilityId,
                              accessKey: p.accessKey,
                              fhirEndpoint: p.fhirEndpoint,
                              notes: p.description,
                            });
                          }
                        }}
                        className={`p-2.5 rounded-2xl border text-left transition-all cursor-pointer ${
                          isSelected
                            ? 'border-teal-600 bg-teal-50/80 dark:bg-teal-950/40 ring-2 ring-teal-600/20 shadow-xs'
                            : 'border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 bg-slate-50/60 dark:bg-slate-800/40 hover:bg-slate-50 dark:hover:bg-slate-800'
                        }`}
                      >
                        <span className="block text-[10px] font-bold uppercase text-teal-700 dark:text-teal-400">
                          {preset.badge}
                        </span>
                        <span className="block text-xs font-semibold text-slate-900 dark:text-slate-100 truncate">
                          {preset.name}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Clinician Form Inputs */}
              <form onSubmit={handleSaveClinician} className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Clinician Name */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                      <User className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500" />
                      Clinician / User Name
                    </label>
                    <input
                      type="text"
                      required
                      value={clinicianForm.name}
                      onChange={(e) =>
                        setClinicianForm({ ...clinicianForm, name: e.target.value })
                      }
                      placeholder="e.g. Dr. Sunita Devi"
                      className="w-full px-3.5 py-2 text-sm bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600 dark:text-slate-100 transition-colors"
                    />
                  </div>

                  {/* Facility ID */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                      <Building2 className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500" />
                      Facility / Clinic ID
                    </label>
                    <input
                      type="text"
                      required
                      value={clinicianForm.facilityId}
                      onChange={(e) =>
                        setClinicianForm({ ...clinicianForm, facilityId: e.target.value })
                      }
                      placeholder="e.g. PHC-PURULIA-NODE-04"
                      className="w-full px-3.5 py-2 text-sm bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600 font-mono dark:text-slate-100 transition-colors"
                    />
                  </div>
                </div>

                {/* Clinical Role */}
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                    <Stethoscope className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500" />
                    Designation / Clinical Role
                  </label>
                  <input
                    type="text"
                    required
                    value={clinicianForm.role}
                    onChange={(e) =>
                      setClinicianForm({ ...clinicianForm, role: e.target.value })
                    }
                    placeholder="e.g. Frontline Health Worker (ASHA / PHC)"
                    className="w-full px-3.5 py-2 text-sm bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-600 dark:text-slate-100 transition-colors"
                  />
                </div>

                {/* Session Access Key */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                      <Lock className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500" />
                      Session Access Key / API Token
                    </label>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          generateNewKey();
                          const randomHex = Math.random()
                            .toString(36)
                            .substring(2, 8)
                            .toUpperCase();
                          const newK = `ORQIS-CLINIC-${randomHex}-${Date.now().toString().slice(-4)}`;
                          setClinicianForm({ ...clinicianForm, accessKey: newK });
                        }}
                        className="text-[11px] text-teal-600 dark:text-teal-400 hover:text-teal-700 font-medium flex items-center gap-1 cursor-pointer"
                      >
                        <RefreshCw className="w-3 h-3" /> Generate New
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          navigator.clipboard.writeText(clinicianForm.accessKey);
                          setCopiedKey(true);
                          setTimeout(() => setCopiedKey(false), 2000);
                        }}
                        className="text-[11px] text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 font-medium flex items-center gap-1 cursor-pointer"
                      >
                        {copiedKey ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                            <span className="text-emerald-600 dark:text-emerald-400">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" /> Copy
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                  <input
                    type="text"
                    required
                    value={clinicianForm.accessKey}
                    onChange={(e) =>
                      setClinicianForm({ ...clinicianForm, accessKey: e.target.value })
                    }
                    className="w-full px-3.5 py-2 text-xs bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 font-mono text-slate-800 dark:text-slate-100"
                  />
                </div>

                {/* FHIR Server Target Endpoint */}
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                    <Server className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500" />
                    HL7 FHIR R4 Target Server
                  </label>
                  <input
                    type="text"
                    required
                    value={clinicianForm.fhirEndpoint}
                    onChange={(e) =>
                      setClinicianForm({ ...clinicianForm, fhirEndpoint: e.target.value })
                    }
                    placeholder="https://rural-care.health.gov.in/fhir/r4"
                    className="w-full px-3.5 py-2 text-xs bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 font-mono text-slate-800 dark:text-slate-100"
                  />
                </div>

                {/* Privacy & Zero-Cloud Notice */}
                <div className="bg-teal-50/60 dark:bg-teal-950/40 border border-teal-200/70 dark:border-teal-800/60 rounded-2xl p-3.5 flex items-start gap-3 text-xs text-teal-900 dark:text-teal-200">
                  <ShieldCheck className="w-4 h-4 text-teal-600 dark:text-teal-400 shrink-0 mt-0.5" />
                  <div className="space-y-0.5">
                    <span className="font-bold">Zero-Cloud PHI Guarantee:</span>
                    <p className="text-teal-800/90 dark:text-teal-300/90 text-[11px] leading-relaxed">
                      Entered credentials and session tokens are strictly stored in your local browser session and utilized solely to dynamically stamp client-side FHIR telemetry in the Sandbox.
                    </p>
                  </div>
                </div>

                {/* Footer Buttons */}
                <div className="pt-3 border-t border-slate-100 dark:border-slate-800 flex flex-col-reverse sm:flex-row items-center justify-between gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      resetDefaults();
                      setClinicianForm(DEFAULT_CREDENTIALS);
                    }}
                    className="text-xs text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 font-medium transition-colors cursor-pointer"
                  >
                    Reset to Default Clinician
                  </button>

                  <div className="flex items-center gap-2 w-full sm:w-auto">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={closeModal}
                      className="flex-1 sm:flex-initial"
                    >
                      Cancel
                    </Button>
                    <Button
                      type="submit"
                      variant="secondary"
                      size="sm"
                      icon={
                        clinicianSaveSuccess ? (
                          <Check className="w-3.5 h-3.5" />
                        ) : (
                          <ShieldCheck className="w-3.5 h-3.5" />
                        )
                      }
                      className="flex-1 sm:flex-initial bg-teal-600 hover:bg-teal-700"
                    >
                      {clinicianSaveSuccess ? 'Saved & Applied!' : 'Save & Apply Credentials'}
                    </Button>
                  </div>
                </div>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export const CredentialsModal: React.FC = () => {
  const { isModalOpen } = useCredentials();
  if (!isModalOpen) return null;
  return <CredentialsModalDialog />;
};
