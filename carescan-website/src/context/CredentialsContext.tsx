'use client';

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import {
  SAMPLE_IMAGE_HEALTHY,
  SAMPLE_IMAGE_LICHENOID,
  SAMPLE_IMAGE_LEUKOPLAKIA,
} from '@/lib/patientImageSamples';
import { api, ApiError } from '@/lib/api';
import type { AssessmentResult } from '@/lib/apiTypes';
import { bandTone, displayedBand, type BandTone } from '@/lib/verdict';

export interface UserCredentials {
  name: string;
  role: string;
  facilityId: string;
  accessKey: string;
  fhirEndpoint: string;
  notes?: string;
}

export interface CredentialPreset {
  id: string;
  name: string;
  role: string;
  facilityId: string;
  accessKey: string;
  fhirEndpoint: string;
  badge: string;
  description: string;
}

export interface PatientProfile {
  name: string;
  patientId: string;
  age: string;
  gender: string;
  symptomRegion: string;
  duration: string;
  tobaccoUse: boolean;
  betelNutUse: boolean;
  alcoholUse: boolean;
  uploadedImage: string | null;
  uploadedImageName: string | null;
  /**
   * Where the *server* put the capture, as returned by `POST /screening/upload`.
   * Held so the case can be re-screened after the risk factors are edited without
   * asking for the photograph again. Chosen by the backend, never composed here.
   * Null for a demonstration preset, which has no uploaded file.
   */
  uploadedImagePath: string | null;
  analysisResult: PatientAnalysisResult | null;
}

/**
 * A screening verdict as the backend produced it.
 *
 * Every field here is copied from a server response. Nothing is derived from a
 * filename, a patient name, or arithmetic in the browser -- a screening result is
 * a clinical statement and the only thing entitled to make one is the model.
 *
 * The DEC-034 split is preserved deliberately:
 *   - `band` / `primaryScore` come from the strongest *validated* model, and
 *     `primaryCalibrated === false` means `primaryScore` is a RANKING, not a
 *     percentage. Render it through `asRanking()`, never with a `%`.
 *   - `quantumProbability` is the calibrated quantum probability. It is the value
 *     that goes into FHIR, and it is shown as a labelled secondary readout.
 */
export interface PatientAnalysisResult {
  /** The screening row the backend created. Needed for FHIR export and re-reads. */
  screeningId: string;
  /** Displayed band, verbatim from the server -- including any `MOCK` prefix. */
  band: string;
  /** Visual tone for the band. */
  tone: BandTone;
  /** The server's own explanatory text for this band. */
  details: string;

  /** Which model set the headline band. */
  primaryModel: string | null;
  /** False => `primaryScore` is a ranking score and carries no percentage meaning. */
  primaryCalibrated: boolean;
  /** Headline score. A probability only when `primaryCalibrated` is true. */
  primaryScore: number | null;
  primaryThreshold: number | null;

  /** Calibrated quantum probability: the secondary readout and the FHIR value. */
  quantumProbability: number | null;
  /** Classical baseline probability, when the served model reports one. */
  classicalProbability: number | null;
  /** Pre-calibration quantum measurement. Not a probability. */
  rawScore: number | null;

  /** Server-stamped. True when the labelled development stub produced this. */
  isMock: boolean;
  executionMode: string | null;
  backendName: string | null;
  modelVersion: string | null;
  quantumQubits: number | null;
  circuitDepth: number | null;
  calibrationMethod: string | null;
  executionTimeMs: number | null;

  /** Patient-facing wording supplied by the server. Rendered verbatim, always. */
  disclaimer: string;
  timestamp: string;

  /** The untouched server payload, for screens needing a field not surfaced above. */
  raw: AssessmentResult;
}

/** Where the analysis pipeline currently is. Drives spinners and error panels. */
export type AnalysisStatus = 'idle' | 'uploading' | 'analyzing' | 'ready' | 'error';

export interface PatientPreset {
  id: string;
  name: string;
  patientId: string;
  age: string;
  gender: string;
  symptomRegion: string;
  duration: string;
  tobaccoUse: boolean;
  betelNutUse: boolean;
  alcoholUse: boolean;
  badge: string;
  badgeVariant: 'success' | 'warning' | 'danger';
  description: string;
  sampleImage: string;
  sampleImageName: string;
}

export const CREDENTIAL_PRESETS: CredentialPreset[] = [
  {
    id: 'asha',
    name: 'Sunita Devi',
    role: 'Frontline Health Worker (ASHA / PHC)',
    facilityId: 'PHC-PURULIA-NODE-04',
    accessKey: 'ASHA-RURAL-2026-9812',
    fhirEndpoint: 'https://rural-care.health.gov.in/fhir/r4',
    badge: 'Frontline Clinic',
    description: 'Optimized for rapid mobile image capture and offline triage sync in rural community clinics.',
  },
  {
    id: 'oncologist',
    name: 'Dr. Rajesh Varma, MD',
    role: 'Consultant Head & Neck Surgical Oncologist',
    facilityId: 'AIIMS-ORAL-ONCO-DEPT',
    accessKey: 'ONC-SPECIALIST-SEC-7712',
    fhirEndpoint: 'https://aiims.edu/fhir/r4/oncology',
    badge: 'Tertiary Hospital',
    description: 'Full clinical permissions for histopathology referral and SNOMED-CT 363349007 telemetry.',
  },
  {
    id: 'researcher',
    name: 'Prof. Elena Rostova',
    role: 'Lead Quantum Computing & VQC Architect',
    facilityId: 'Q-LAB-AER-SIMULATOR',
    accessKey: 'QISKIT-AER-8Q-VQC-8890',
    fhirEndpoint: 'https://quantum-ml.orqis.health/v1/fhir',
    badge: 'Quantum ML Lab',
    // 8 qubits / 256 amplitudes and Qiskit Aer, matching QUANTUM_QUBITS and the
    // default QUANTUM_EXECUTION_MODE in the backend. The deployed system runs on a
    // simulator; naming hardware it has not run on would be a false provenance claim.
    description: 'Direct parameter access to 8-qubit (256 amplitude) ansatz rotations and Pauli-Z expectation value debugging on the Qiskit Aer statevector simulator.',
  },
  {
    id: 'dental',
    name: 'Dr. Ananya Iyer, BDS',
    role: 'Primary Oral & Dental Health Practitioner',
    facilityId: 'DENTAL-CARE-METRO-09',
    accessKey: 'DENT-CLINIC-KEY-4421',
    fhirEndpoint: 'https://dental-network.org/fhir/r4',
    badge: 'Dental Practice',
    description: 'Routine mucosal screening and automated patient follow-up scheduling.',
  },
];

export const PATIENT_PRESETS: PatientPreset[] = [
  {
    id: 'patient-healthy',
    name: 'Rohan Verma (Self-Check)',
    patientId: 'PAT-ROUTINE-4819',
    age: '34',
    gender: 'Male',
    symptomRegion: 'Buccal Mucosa (Inner Cheek)',
    duration: '< 1 week (Mild irritation)',
    tobaccoUse: false,
    betelNutUse: false,
    alcoholUse: false,
    badge: 'Routine Check (Benign)',
    badgeVariant: 'success',
    description: 'Mild non-tender pink mucosal surface after rough food scraping. No smoking or tobacco history.',
    sampleImage: SAMPLE_IMAGE_HEALTHY,
    sampleImageName: 'normal_buccal_mucosa.png',
  },
  {
    id: 'patient-lichenoid',
    name: 'Pooja Sharma (Self-Check)',
    patientId: 'PAT-LICHEN-7731',
    age: '48',
    gender: 'Female',
    symptomRegion: 'Bilateral Buccal Mucosa',
    duration: '4 weeks (Spicy food sensitivity)',
    tobaccoUse: false,
    betelNutUse: false,
    alcoholUse: true,
    badge: 'Lichenoid Striae (Moderate)',
    badgeVariant: 'warning',
    description: 'Reticular lace-like white striae on inner cheeks with episodic burning upon eating spices.',
    sampleImage: SAMPLE_IMAGE_LICHENOID,
    sampleImageName: 'reticular_striae_lesion.png',
  },
  {
    id: 'patient-leukoplakia',
    name: 'Jagdish Kumar (Self-Check)',
    patientId: 'PAT-LEUKO-9942',
    age: '56',
    gender: 'Male',
    symptomRegion: 'Left Lateral Border of Tongue',
    duration: '3+ months (Non-healing firm patch)',
    tobaccoUse: true,
    betelNutUse: true,
    alcoholUse: false,
    badge: 'Leukoplakia Plaque (High Risk)',
    badgeVariant: 'danger',
    description: 'Elevated, non-scrapable keratotic white patch with irregular margins and 12-year betel quid history.',
    sampleImage: SAMPLE_IMAGE_LEUKOPLAKIA,
    sampleImageName: 'keratotic_leukoplakia_patch.png',
  },
];

export const DEFAULT_CREDENTIALS: UserCredentials = {
  name: 'Sunita Devi',
  role: 'Frontline Health Worker (ASHA / PHC)',
  facilityId: 'PHC-PURULIA-NODE-04',
  accessKey: 'ASHA-RURAL-2026-9812',
  fhirEndpoint: 'https://rural-care.health.gov.in/fhir/r4',
  notes: 'Authenticated local testing profile for preliminary oral screening.',
};

/**
 * `analysisResult` starts null on purpose. A verdict exists only after the backend
 * has produced one; shipping a pre-filled "Low risk" would be a clinical claim about
 * a screening that never ran.
 */
export const DEFAULT_PATIENT: PatientProfile = {
  name: 'Rohan Verma (Self-Check)',
  patientId: 'PAT-SELF-8821',
  age: '34',
  gender: 'Male',
  symptomRegion: 'Buccal Mucosa (Inner Cheek)',
  duration: '3 days',
  tobaccoUse: false,
  betelNutUse: false,
  alcoholUse: false,
  uploadedImage: SAMPLE_IMAGE_HEALTHY,
  uploadedImageName: 'sample_healthy_mucosa.png',
  uploadedImagePath: null,
  analysisResult: null,
};

// The stored profile shape changed when the fabricated analysis was removed, so the
// key is versioned: a browser holding the old object would otherwise rehydrate a
// result with fields this build no longer understands.
const PROFILE_KEY = 'orqis_patient_profile_v2';
const CREDENTIALS_KEY = 'orqis_user_credentials';
const AUTH_TAB_KEY = 'orqis_active_auth_tab';

/**
 * Synthetic `image_path` for a demonstration case.
 *
 * The three built-in scenarios ship as SVG illustrations, not photographs, so they
 * are never submitted for real inference -- a classifier run over a cartoon would
 * produce a number with no meaning. They round-trip through the backend with
 * `is_mock: true` instead, which exercises the whole stack (persistence, schemas,
 * FHIR export) while the *server* stamps the result MOCK. The path is seed material
 * for the deterministic stub and is never opened.
 */
function presetImagePath(preset: PatientPreset): string {
  return `preset://${preset.id}/${preset.sampleImageName}`;
}

/** Backend wants 'male' / 'female' or nothing. Anything else is not collected. */
function normaliseSex(gender: string): string | null {
  const value = gender.trim().toLowerCase();
  return value === 'male' || value === 'female' ? value : null;
}

/** Backend wants years in [0, 120]. A blank or unparseable age is "not collected". */
function normaliseAge(age: string): number | null {
  const value = Number.parseFloat(age);
  if (!Number.isFinite(value) || value < 0 || value > 120) return null;
  return value;
}

/** Project a server response into the shape the screens render. */
function toAnalysisResult(result: AssessmentResult): PatientAnalysisResult {
  const band = displayedBand(result) ?? 'UNKNOWN';
  return {
    screeningId: result.assessmentId || result.id,
    band,
    tone: bandTone(band),
    details: result.details,

    primaryModel: result.primaryModel ?? null,
    // Absent is treated as uncalibrated: suppressing a percentage that could have
    // been shown loses detail, showing one that should have been suppressed is a
    // clinical misstatement.
    primaryCalibrated: result.primaryCalibrated === true,
    primaryScore: result.primaryProbability ?? null,
    primaryThreshold: result.primaryThreshold ?? null,

    quantumProbability: result.finalProbability ?? null,
    classicalProbability: result.classicalProbability ?? null,
    rawScore: result.rawScore ?? null,

    isMock: result.isMock,
    executionMode: result.executionMode ?? null,
    backendName: result.backendName ?? null,
    modelVersion: result.modelVersion ?? null,
    quantumQubits: result.quantumQubits ?? null,
    circuitDepth: result.circuitDepth ?? null,
    calibrationMethod: result.calibrationMethod ?? null,
    executionTimeMs: result.executionTimeMs ?? null,

    disclaimer: result.disclaimer,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    raw: result,
  };
}

/** Turn any thrown value into a line that can sit in front of a clinician. */
function describeFailure(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error && error.message) return error.message;
  return 'The screening could not be completed. Please try again.';
}

interface CredentialsContextType {
  activeAuthTab: 'clinician' | 'patient';
  setActiveAuthTab: (tab: 'clinician' | 'patient') => void;
  credentials: UserCredentials;
  patientProfile: PatientProfile;
  isModalOpen: boolean;
  openModal: (initialTab?: 'clinician' | 'patient') => void;
  closeModal: () => void;
  setCredentials: React.Dispatch<React.SetStateAction<UserCredentials>>;
  updateCredentials: (updates: Partial<UserCredentials>) => void;
  updatePatientProfile: (updates: Partial<PatientProfile>) => void;
  applyPreset: (presetId: string) => void;
  applyPatientPreset: (presetId: string) => void;
  resetDefaults: () => void;
  resetPatientDefaults: () => void;
  generateNewKey: () => void;
  generateNewPatientId: () => void;
  uploadPatientImage: (file: File) => Promise<void>;
  analyzePatientCase: () => Promise<void>;

  /** Where the current screening is in the upload → analyze pipeline. */
  analysisStatus: AnalysisStatus;
  /** Server-supplied failure text, or null. Only meaningful while status is 'error'. */
  analysisError: string | null;
  /** Discard the current verdict without touching the rest of the profile. */
  clearAnalysis: () => void;
}

const CredentialsContext = createContext<CredentialsContextType | undefined>(undefined);

export const CredentialsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activeAuthTab, setActiveAuthTab] = useState<'clinician' | 'patient'>(() => {
    if (typeof window === 'undefined') return 'clinician';
    try {
      const savedTab = localStorage.getItem(AUTH_TAB_KEY);
      if (savedTab === 'patient' || savedTab === 'clinician') {
        return savedTab;
      }
    } catch {
      // ignore
    }
    return 'clinician';
  });

  const [credentials, setCredentials] = useState<UserCredentials>(() => {
    if (typeof window === 'undefined') return DEFAULT_CREDENTIALS;
    try {
      const savedCreds = localStorage.getItem(CREDENTIALS_KEY);
      if (savedCreds) {
        return JSON.parse(savedCreds);
      }
    } catch {
      // ignore
    }
    return DEFAULT_CREDENTIALS;
  });

  const [patientProfile, setPatientProfile] = useState<PatientProfile>(() => {
    if (typeof window === 'undefined') return DEFAULT_PATIENT;
    try {
      const savedPatient = localStorage.getItem(PROFILE_KEY);
      if (savedPatient) {
        return JSON.parse(savedPatient);
      }
    } catch {
      // ignore
    }
    return DEFAULT_PATIENT;
  });

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus>(() =>
    patientProfile.analysisResult ? 'ready' : 'idle',
  );
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // The profile lives in state, but an in-flight screening needs the values as they
  // are *now*, not as they were when the callback was created. A ref keeps the async
  // path reading current data without making every handler depend on the profile.
  // Written in an effect rather than during render: a ref mutated while rendering is
  // not safe under concurrent rendering, and `persistProfile` below keeps it exact
  // for the one path that reads it immediately after a write.
  const profileRef = useRef(patientProfile);
  useEffect(() => {
    profileRef.current = patientProfile;
  }, [patientProfile]);

  // Only the newest submission may write a result. Without this, a slow first
  // analysis landing after a second one would overwrite the newer verdict.
  const runIdRef = useRef(0);

  const persistProfile = useCallback((profile: PatientProfile) => {
    setPatientProfile(profile);
    profileRef.current = profile;
    try {
      localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    } catch {
      // ignore
    }
  }, []);

  const openModal = (initialTab?: 'clinician' | 'patient') => {
    if (initialTab) {
      setActiveAuthTab(initialTab);
    }
    setIsModalOpen(true);
  };

  const closeModal = () => setIsModalOpen(false);

  const handleSetActiveTab = (tab: 'clinician' | 'patient') => {
    setActiveAuthTab(tab);
    try {
      localStorage.setItem(AUTH_TAB_KEY, tab);
    } catch {
      // ignore
    }
  };

  const updateCredentials = (updates: Partial<UserCredentials>) => {
    setCredentials((prev) => {
      const updated = { ...prev, ...updates };
      try {
        localStorage.setItem(CREDENTIALS_KEY, JSON.stringify(updated));
      } catch {
        // ignore
      }
      return updated;
    });
  };

  const updatePatientProfile = useCallback(
    (updates: Partial<PatientProfile>) => {
      persistProfile({ ...profileRef.current, ...updates });
    },
    [persistProfile],
  );

  const clearAnalysis = useCallback(() => {
    // Invalidate any in-flight run so its result cannot land after this clear.
    runIdRef.current += 1;
    setAnalysisStatus('idle');
    setAnalysisError(null);
    persistProfile({ ...profileRef.current, analysisResult: null });
  }, [persistProfile]);

  const applyPreset = (presetId: string) => {
    const preset = CREDENTIAL_PRESETS.find((p) => p.id === presetId);
    if (preset) {
      const updated: UserCredentials = {
        name: preset.name,
        role: preset.role,
        facilityId: preset.facilityId,
        accessKey: preset.accessKey,
        fhirEndpoint: preset.fhirEndpoint,
        notes: preset.description,
      };
      setCredentials(updated);
      try {
        localStorage.setItem(CREDENTIALS_KEY, JSON.stringify(updated));
      } catch {
        // ignore
      }
    }
  };

  /**
   * Submit the current profile for analysis.
   *
   * `imagePath` comes either from a real upload or from {@link presetImagePath};
   * `isMock` follows from which. Both paths post the same payload to the same route,
   * so the demonstration exercises the production code path rather than bypassing it.
   */
  const runAnalysis = useCallback(
    async (profile: PatientProfile, imagePath: string, isMock: boolean) => {
      const runId = (runIdRef.current += 1);
      setAnalysisStatus('analyzing');
      setAnalysisError(null);

      try {
        const result = await api.analyze({
          patient_id: profile.patientId || null,
          image_path: imagePath,
          scan_type: 'Intra-oral Scan',
          // The form always asks these three, so `false` genuinely means "asked and
          // answered no" -- which the clinical encoder keeps distinct from "not
          // collected". Never send null here just because the box is unticked.
          smoking_history: profile.tobaccoUse,
          alcohol_consumption: profile.alcoholUse,
          betel_quid: profile.betelNutUse,
          age: normaliseAge(profile.age),
          sex: normaliseSex(profile.gender),
          is_mock: isMock,
        });

        if (runId !== runIdRef.current) return;
        persistProfile({ ...profileRef.current, analysisResult: toAnalysisResult(result) });
        setAnalysisStatus('ready');
      } catch (error) {
        if (runId !== runIdRef.current) return;
        setAnalysisError(describeFailure(error));
        setAnalysisStatus('error');
        // Drop any stale verdict: leaving the previous result on screen next to a
        // failure message reads as though the new case was scored.
        persistProfile({ ...profileRef.current, analysisResult: null });
      }
    },
    [persistProfile],
  );

  const applyPatientPreset = useCallback(
    (presetId: string) => {
      const preset = PATIENT_PRESETS.find((p) => p.id === presetId);
      if (!preset) return;

      const updated: PatientProfile = {
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
      };
      persistProfile(updated);
      void runAnalysis(updated, presetImagePath(preset), true);
    },
    [persistProfile, runAnalysis],
  );

  const resetDefaults = () => {
    setCredentials(DEFAULT_CREDENTIALS);
    try {
      localStorage.setItem(CREDENTIALS_KEY, JSON.stringify(DEFAULT_CREDENTIALS));
    } catch {
      // ignore
    }
  };

  const resetPatientDefaults = useCallback(() => {
    runIdRef.current += 1;
    setAnalysisStatus('idle');
    setAnalysisError(null);
    persistProfile(DEFAULT_PATIENT);
  }, [persistProfile]);

  const generateNewKey = () => {
    const randomHex = Math.random().toString(36).substring(2, 8).toUpperCase();
    const newKey = `ORQIS-${credentials.role.includes('Quantum') ? 'VQC' : 'CLINIC'}-${randomHex}-${Date.now().toString().slice(-4)}`;
    updateCredentials({ accessKey: newKey });
  };

  const generateNewPatientId = () => {
    const randomHex = Math.random().toString(36).substring(2, 7).toUpperCase();
    const newId = `PAT-ORQIS-${randomHex}`;
    updatePatientProfile({ patientId: newId });
  };

  /**
   * Upload a real photograph and screen it.
   *
   * The data URL is kept only so the page can show the clinician what they submitted.
   * The bytes that get scored are the ones the server stored, addressed by the path
   * the server chose -- the browser never tells the backend where to read from.
   */
  const uploadPatientImage = useCallback(
    async (file: File): Promise<void> => {
      const runId = (runIdRef.current += 1);
      setAnalysisStatus('uploading');
      setAnalysisError(null);

      const preview = await new Promise<string | null>((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve((e.target?.result as string) ?? null);
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(file);
      });

      if (runId !== runIdRef.current) return;

      const staged: PatientProfile = {
        ...profileRef.current,
        uploadedImage: preview,
        uploadedImageName: file.name,
        uploadedImagePath: null,
        analysisResult: null,
      };
      persistProfile(staged);

      let imagePath: string;
      try {
        const uploaded = await api.upload(file, staged.patientId || null);
        imagePath = uploaded.image_path;
      } catch (error) {
        if (runId !== runIdRef.current) return;
        setAnalysisError(describeFailure(error));
        setAnalysisStatus('error');
        return;
      }

      if (runId !== runIdRef.current) return;
      const stored: PatientProfile = { ...staged, uploadedImagePath: imagePath };
      persistProfile(stored);
      await runAnalysis(stored, imagePath, false);
    },
    [persistProfile, runAnalysis],
  );

  /**
   * Re-screen the current case, e.g. after the risk factors were edited.
   *
   * A demonstration preset re-runs as a labelled mock; a real upload re-runs against
   * the file the server already stored, addressed by the path the server itself
   * returned. If neither is available there is nothing to score, and saying so beats
   * leaving a verdict on screen that no longer matches the inputs.
   */
  const analyzePatientCase = useCallback(async (): Promise<void> => {
    const profile = profileRef.current;

    if (profile.uploadedImagePath) {
      await runAnalysis(profile, profile.uploadedImagePath, false);
      return;
    }

    const preset = PATIENT_PRESETS.find((p) => p.sampleImageName === profile.uploadedImageName);
    if (preset) {
      await runAnalysis(profile, presetImagePath(preset), true);
      return;
    }

    runIdRef.current += 1;
    setAnalysisError(
      'Select a photograph or a demonstration scenario first — there is nothing to screen yet.',
    );
    setAnalysisStatus('error');
  }, [runAnalysis]);

  return (
    <CredentialsContext.Provider
      value={{
        activeAuthTab,
        setActiveAuthTab: handleSetActiveTab,
        credentials,
        patientProfile,
        isModalOpen,
        openModal,
        closeModal,
        setCredentials,
        updateCredentials,
        updatePatientProfile,
        applyPreset,
        applyPatientPreset,
        resetDefaults,
        resetPatientDefaults,
        generateNewKey,
        generateNewPatientId,
        uploadPatientImage,
        analyzePatientCase,
        analysisStatus,
        analysisError,
        clearAnalysis,
      }}
    >
      {children}
    </CredentialsContext.Provider>
  );
};

export const useCredentials = (): CredentialsContextType => {
  const context = useContext(CredentialsContext);
  if (!context) {
    throw new Error('useCredentials must be used within a CredentialsProvider');
  }
  return context;
};
