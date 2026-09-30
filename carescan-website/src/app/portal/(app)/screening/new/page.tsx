'use client';

/**
 * Capture and analyse a new screening.
 *
 * The three risk factors are tri-state on purpose. `false` means the patient was
 * asked and said no; `null` means it was never collected. The backend's clinical
 * encoder keeps those apart, so the form must too — "Not asked" is a real answer
 * here, not a placeholder, and it is the default.
 *
 * Client-side file validation mirrors `ALLOWED_IMAGE_EXTENSIONS` and
 * `MAX_UPLOAD_BYTES` in `backend/routes/screening_routes.py`. It is a courtesy to
 * save a round trip; the server still enforces both.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  AlertCircle,
  ArrowRight,
  ImageUp,
  Loader2,
  RefreshCw,
  Sparkles,
  Upload,
  X,
} from 'lucide-react';
import { api, ApiError } from '@/lib/api';
import type { AnalyzeRequest } from '@/lib/apiTypes';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { PageHeading } from '@/components/portal/PortalUI';

const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

type TriValue = 'yes' | 'no' | 'unknown';

/** `unknown` maps to null — not collected, which is not the same as "no". */
function toTriState(value: TriValue): boolean | null {
  if (value === 'yes') return true;
  if (value === 'no') return false;
  return null;
}

const TriStateField: React.FC<{
  label: string;
  hint: string;
  value: TriValue;
  onChange: (next: TriValue) => void;
  disabled?: boolean;
}> = ({ label, hint, value, onChange, disabled }) => (
  <fieldset disabled={disabled} className="min-w-0">
    <legend className="text-xs font-bold text-slate-800 dark:text-slate-200">{label}</legend>
    <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-2 leading-snug">{hint}</p>
    <div className="inline-flex rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      {(
        [
          ['yes', 'Yes'],
          ['no', 'No'],
          ['unknown', 'Not asked'],
        ] as [TriValue, string][]
      ).map(([option, text]) => (
        <button
          key={option}
          type="button"
          aria-pressed={value === option}
          onClick={() => onChange(option)}
          className={`text-xs font-semibold px-3 py-1.5 transition-colors cursor-pointer disabled:cursor-not-allowed ${
            value === option
              ? 'bg-teal-600 text-white'
              : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
          }`}
        >
          {text}
        </button>
      ))}
    </div>
  </fieldset>
);

export default function NewScreeningPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const [patientId, setPatientId] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('');
  const [smoking, setSmoking] = useState<TriValue>('unknown');
  const [alcohol, setAlcohol] = useState<TriValue>('unknown');
  const [betel, setBetel] = useState<TriValue>('unknown');

  const [phase, setPhase] = useState<'idle' | 'uploading' | 'analyzing'>('idle');
  const [error, setError] = useState<string | null>(null);

  const busy = phase !== 'idle';

  // Revoke the object URL when the preview changes or the page unmounts.
  useEffect(() => {
    if (!previewUrl) return;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const acceptFile = useCallback((next: File | null) => {
    if (!next) return;

    const extension = next.name.slice(next.name.lastIndexOf('.')).toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setError(
        `That file type is not supported. Use one of ${ALLOWED_EXTENSIONS.join(', ')}.`,
      );
      return;
    }
    if (next.size > MAX_UPLOAD_BYTES) {
      setError(
        `That image is ${(next.size / 1024 / 1024).toFixed(1)} MB. The limit is 25 MB.`,
      );
      return;
    }

    setError(null);
    setFile(next);
    setPreviewUrl(URL.createObjectURL(next));
  }, []);

  const clearFile = useCallback(() => {
    setFile(null);
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = '';
  }, []);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!file || busy) return;

    setError(null);
    setPhase('uploading');

    try {
      const uploaded = await api.upload(file, patientId.trim() || null);

      setPhase('analyzing');

      const parsedAge = age.trim() ? Number.parseInt(age, 10) : null;
      const payload: AnalyzeRequest = {
        patient_id: uploaded.patient_id,
        image_path: uploaded.image_path,
        scan_type: 'oral_cavity',
        smoking_history: toTriState(smoking),
        alcohol_consumption: toTriState(alcohol),
        betel_quid: toTriState(betel),
        age: parsedAge != null && Number.isFinite(parsedAge) ? parsedAge : null,
        sex: sex || null,
      };

      const result = await api.analyze(payload);
      router.push(`/portal/result/${encodeURIComponent(result.assessmentId)}`);
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : 'The screening could not be completed. Please try again.',
      );
      setPhase('idle');
    }
  }

  return (
    <>
      <PageHeading
        title="New screening"
        subtitle="Upload an intra-oral photograph and record what was actually asked at the chairside. Quality control, lesion localization, scoring and calibration all run on the backend."
      />

      <form onSubmit={handleSubmit} className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        {/* Capture */}
        <Card variant="white" padding="md" className="lg:col-span-3">
          <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100 mb-1">
            Intra-oral photograph
          </h2>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-3 leading-relaxed">
            JPEG, PNG, BMP or WebP, up to 25 MB. The image is sent to the Orqis backend;
            nothing about it is analysed in this browser.
          </p>

          {!previewUrl ? (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                acceptFile(e.dataTransfer.files?.[0] ?? null);
              }}
              onClick={() => inputRef.current?.click()}
              className={`rounded-3xl border-2 border-dashed p-10 text-center cursor-pointer transition-colors ${
                dragging
                  ? 'border-teal-500 bg-teal-50/60 dark:bg-teal-950/40'
                  : 'border-slate-300 dark:border-slate-700 hover:border-teal-400 hover:bg-slate-50/70 dark:hover:bg-slate-800/40'
              }`}
            >
              <div className="w-12 h-12 rounded-3xl bg-slate-100 dark:bg-slate-800 text-teal-700 dark:text-teal-300 flex items-center justify-center mx-auto mb-3">
                <ImageUp className="w-6 h-6" />
              </div>
              <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                Drop a photograph here, or click to choose one
              </p>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                {ALLOWED_EXTENSIONS.join(' · ')}
              </p>
              <input
                ref={inputRef}
                type="file"
                accept={ALLOWED_EXTENSIONS.join(',')}
                className="hidden"
                onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
              />
            </div>
          ) : (
            <div className="space-y-3">
              <div className="relative">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt="Selected intra-oral photograph"
                  className="w-full max-h-80 object-contain rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800"
                />
                {!busy && (
                  <button
                    type="button"
                    onClick={clearFile}
                    aria-label="Remove this photograph"
                    className="absolute top-2 right-2 w-8 h-8 rounded-full bg-slate-900/80 text-white flex items-center justify-center hover:bg-slate-900 transition-colors cursor-pointer"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="neutral" size="sm">
                  {file?.name}
                </Badge>
                <Badge variant="neutral" size="sm">
                  {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : ''}
                </Badge>
                {!busy && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => inputRef.current?.click()}
                    icon={<RefreshCw className="w-3.5 h-3.5" />}
                    iconPosition="left"
                  >
                    Retake
                  </Button>
                )}
                <input
                  ref={inputRef}
                  type="file"
                  accept={ALLOWED_EXTENSIONS.join(',')}
                  className="hidden"
                  onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
                />
              </div>
            </div>
          )}
        </Card>

        {/* Clinical context */}
        <Card variant="white" padding="md" className="lg:col-span-2 space-y-5">
          <div>
            <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100 mb-1">
              Clinical context
            </h2>
            <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
              Leave anything you did not ask about as <strong>Not asked</strong>. The
              model treats that differently from a recorded &ldquo;no&rdquo;, so guessing
              would change the input.
            </p>
          </div>

          <div className="space-y-4">
            <TriStateField
              label="Tobacco use"
              hint="Smoking or smokeless tobacco"
              value={smoking}
              onChange={setSmoking}
              disabled={busy}
            />
            <TriStateField
              label="Alcohol consumption"
              hint="Regular consumption"
              value={alcohol}
              onChange={setAlcohol}
              disabled={busy}
            />
            <TriStateField
              label="Betel quid / areca nut"
              hint="Chewing, with or without tobacco"
              value={betel}
              onChange={setBetel}
              disabled={busy}
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label
                htmlFor="screening-age"
                className="block text-xs font-bold text-slate-800 dark:text-slate-200 mb-1.5"
              >
                Age <span className="font-normal text-slate-400">(optional)</span>
              </label>
              <input
                id="screening-age"
                type="number"
                min={0}
                max={120}
                inputMode="numeric"
                value={age}
                disabled={busy}
                onChange={(e) => setAge(e.target.value)}
                className="w-full px-3 py-2 text-sm rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/40 focus:border-teal-500"
              />
            </div>
            <div>
              <label
                htmlFor="screening-sex"
                className="block text-xs font-bold text-slate-800 dark:text-slate-200 mb-1.5"
              >
                Sex <span className="font-normal text-slate-400">(optional)</span>
              </label>
              <select
                id="screening-sex"
                value={sex}
                disabled={busy}
                onChange={(e) => setSex(e.target.value)}
                className="w-full px-3 py-2 text-sm rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/40 focus:border-teal-500"
              >
                <option value="">Not recorded</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>

          <div>
            <label
              htmlFor="screening-patient"
              className="block text-xs font-bold text-slate-800 dark:text-slate-200 mb-1.5"
            >
              Existing patient identifier{' '}
              <span className="font-normal text-slate-400">(optional)</span>
            </label>
            <input
              id="screening-patient"
              type="text"
              value={patientId}
              disabled={busy}
              placeholder="Leave blank to create a new patient record"
              onChange={(e) => setPatientId(e.target.value)}
              className="w-full px-3 py-2 text-xs font-mono rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-slate-100 placeholder:font-sans placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/40 focus:border-teal-500"
            />
            <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1.5 leading-snug">
              Paste an identifier from the{' '}
              <Link
                href="/portal/patients"
                className="text-teal-700 dark:text-teal-300 hover:underline"
              >
                patient roster
              </Link>{' '}
              to add this screening to an existing history.
            </p>
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
            disabled={!file || busy}
            className="w-full"
            icon={
              busy ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <ArrowRight className="w-4 h-4" />
              )
            }
          >
            {phase === 'uploading'
              ? 'Uploading the photograph…'
              : phase === 'analyzing'
                ? 'Running the screening…'
                : 'Run screening'}
          </Button>

          {busy && (
            <div className="flex items-start gap-2 text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
              {phase === 'uploading' ? (
                <Upload className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              ) : (
                <Sparkles className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              )}
              <span>
                {phase === 'uploading'
                  ? 'Sending the image to the backend.'
                  : 'Quality control, ROI localization, classical scoring, the quantum feature map and calibration are running server-side.'}
              </span>
            </div>
          )}
        </Card>
      </form>
    </>
  );
}
