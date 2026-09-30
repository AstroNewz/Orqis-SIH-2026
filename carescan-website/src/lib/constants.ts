import {
  NavItem,
  MetricItem,
  WorkflowStage,
  TechnologyTier,
  QuantumConcept,
  PhilosophyPillar,
  TimelineMilestone,
  TeamMember,
} from '@/types';

export const NAV_ITEMS: NavItem[] = [
  { label: 'Mission', href: '#project' },
  { label: 'Workflow', href: '#how-it-works' },
  { label: 'Technology', href: '#technology' },
  { label: 'Sandbox', href: '#sandbox' },
  { label: 'Ethics', href: '#philosophy' },
  { label: 'Roadmap', href: '#journey' },
  { label: 'Team', href: '#team' },
];

export const HERO_METRICS: MetricItem[] = [
  {
    value: '0.913 PR-AUC',
    label: 'The Model That Sets the Band',
    description:
      'Classical baseline C7 — 1,195-d MobileNetV3-Small + descriptor features, logistic head — on the leak-free lesion-polygon validation condition',
  },
  {
    value: '2,436 / 328',
    label: 'Images / Patients',
    description: 'Strict patient-level split (P_train ∩ P_val ∩ P_test = ∅) — 230 train / 50 val / 48 test patients',
  },
  {
    value: '98.16%',
    label: 'ROI Localization Rate',
    description: 'Frozen MobileNet localizer: 374/381 test images localized (0.5279 mean IoU)',
  },
  {
    value: '4 / 4 Null',
    label: 'Quantum Attempts, Published',
    description:
      'Hybrid fusion, fidelity kernel, trainable VQC and ZZ feature map each lost to a matched classical control — every one recorded, none buried',
  },
];

export const PURPOSE_STATS = [
  {
    stat: '377,000+',
    label: 'Lives Impacted Globally Each Year',
    detail: 'Oral cancer is one of the most prevalent and lethal malignancies in underserved regions. Early detection is critical — Orqis brings preliminary screening to frontline health workers.',
  },
  {
    stat: '0.5279 IoU',
    label: 'Validated MobileNet Test Performance',
    detail: 'Frozen MobileNet lesion localizer achieves 0.5285 validation vs 0.5279 test mean IoU on held-out annotated images, trained exclusively on lesion boxes per DEC-020.',
  },
  {
    stat: '328 Patients',
    label: 'Strict Patient-Level Partitioning',
    detail: 'University of Peradeniya Oral Cancer Dataset v1: 2,436 usable images from 328 patients. Split strictly at patient level (230 train / 50 val / 48 test) — no cross-partition patient leakage.',
  },
];

export const WORKFLOW_STAGES: WorkflowStage[] = [
  {
    step: '01',
    title: 'Guided Mobile Capture & QC',
    subtitle: 'On-Device Acquisition (com.orqis.patient)',
    description:
      'A frontline healthcare worker captures intra-oral photographs using the Flutter mobile application. The on-device engine performs instant quality checks for illumination, blur, and visibility while stripping all EXIF metadata.',
    icon: 'Camera',
    badge: 'Step 1 • Capture & QC',
    details: [
      'On-device image quality validation flags inadequate lighting or excessive blur',
      'Local preprocessing removes unnecessary EXIF metadata before feature generation',
      'Operates natively on standard budget smartphones in community field clinics',
    ],
  },
  {
    step: '02',
    title: 'Lesion ROI Localization',
    subtitle: 'Frozen MobileNet Localizer (DEC-020)',
    description:
      'The image passes to a frozen MobileNet localizer trained exclusively on supervised lesion boxes. It isolates the suspicious oral region and turns it into the descriptor both the classical baseline and the quantum feature map read.',
    icon: 'Cpu',
    badge: 'Step 2 • Localize & Crop',
    details: [
      'Excludes broad region boxes to prevent geometric classification shortcuts (AUC 0.983)',
      'Achieves 0.5279 mean IoU on held-out test data with 98.16% localization rate',
      'Emits the ROI descriptor; a TRAIN-fitted PCA reduces it to the 8 dimensions the quantum map encodes',
    ],
  },
  {
    step: '03',
    title: 'Classical Scoring, Quantum Alongside',
    subtitle: 'Baseline C7 Sets the Band · 8-Qubit Map Reports Beside It',
    description:
      'The ROI descriptor is scored by the validated classical baseline, which alone determines the headline risk band. In parallel a fixed 8-qubit quantum feature map — Ry angle encoding over TRAIN-fitted PCA-8, a nearest-neighbour CZ ring, no trainable quantum gates — reports 16 local Z observables as clearly-labelled telemetry.',
    icon: 'Sparkles',
    badge: 'Step 3 • Score & Measure',
    details: [
      'Classical baseline C7 (1,195-d features, logistic head) reaches 0.913038 validation PR-AUC',
      'Quantum map: 8 qubits, angles bounded to [0, π/2], ≤2 re-uploading blocks, 0 trainable quantum gates',
      'Exact Aer statevector matched to 8.9e-16; 1.47 ms mean per image — telemetry, never the verdict',
    ],
  },
  {
    step: '04',
    title: 'Calibrated Triage & FHIR Reporting',
    subtitle: 'Actionable Clinical Decision Support',
    description:
      'The score is turned into a probability by a calibrator chosen on out-of-fold cross-validation, stratified into a triage band, and formatted into HL7 FHIR R4 resources and a local clinical referral summary.',
    icon: 'FileCheck',
    badge: 'Step 4 • Triage & FHIR',
    details: [
      'Calibrator (Platt or isotonic) selected by out-of-fold CV with a paired-SE tie rule — never hand-picked',
      'Direct generation of HL7 FHIR R4 Observation and RiskAssessment resources',
      'Includes clear clinical disclaimers emphasizing AI-assisted screening decision support',
    ],
  },
];

export const TECH_TIERS: TechnologyTier[] = [
  {
    id: 'mobile',
    title: 'Frontline Mobile Application',
    tagline: 'Flutter & Dart (com.orqis.patient)',
    description:
      'Cross-platform mobile application engineered for frontline health workers and community nurses. Provides guided camera acquisition, local OpenCV image quality validation, EXIF scrubbing, and offline session resilience.',
    specs: [
      { label: 'Package Identifier', value: 'com.orqis.patient' },
      { label: 'Framework', value: 'Flutter / Dart' },
      { label: 'Quality Controls', value: 'Illumination, Blur & ROI Size Checks' },
      { label: 'Routing & Navigation', value: 'Centralized go_router' },
    ],
    highlights: [
      'Real-time viewfinder guidance ensuring sharp focus and clinical-grade exposure',
      'Local stripping of all personal EXIF metadata prior to payload preparation',
      'Standardized design tokens for high-contrast accessibility in bright field clinics',
      'Defensive error handling for camera permission denial and offline operations',
    ],
    icon: 'Smartphone',
    color: 'iris',
  },
  {
    id: 'classical',
    title: 'MobileNet ROI Localizer',
    tagline: 'Lesion-Only Supervised Detection',
    description:
      'A frozen MobileNet convolutional network that isolates the lesion region of interest (ROI). Trained strictly on lesion boxes (DEC-020) to eliminate geometric crop shortcuts, outputting a deterministic 256×256 grayscale matrix.',
    specs: [
      { label: 'Model Checkpoint', value: 'localizer.pt (Epoch 17)' },
      { label: 'Input Dimensions', value: '224×224 Bilinear ImageNet Normalization' },
      { label: 'Decision Threshold', value: '0.30 (Selected on Validation F1)' },
      { label: 'Held-Out Test IoU', value: '0.5279 Mean IoU (98.16% Localized)' },
    ],
    highlights: [
      'Eliminated broad region boxes that leaked diagnostic class through geometry (AUC ≈ 0.983)',
      'Validation mean IoU (0.5285) matches test mean IoU (0.5279) with zero overfitting collapse',
      'Outputs the ROI descriptor that feeds both the classical baseline and the 8-qubit feature map',
      'Explicit separation of A (Oracle ROI), B (Predicted ROI), and C (Fallback) conditions',
    ],
    icon: 'Layers',
    color: 'teal',
  },
  {
    id: 'quantum',
    title: '8-Qubit Quantum Feature Map',
    tagline: 'Zero Trainable Gates · Always Beside Its Classical Control',
    description:
      'A fixed 8-qubit circuit reading the ROI descriptor after TRAIN-fitted PCA-8: Ry angle encoding bounded to [0, π/2], a nearest-neighbour CZ ring, at most two data re-uploading blocks, and 16 local Z observables feeding a small classical head. It is a demonstrator, not an advantage claim — a matched classical control sits beside it in every payload.',
    specs: [
      { label: 'Register', value: '8 Qubits · ~16 Two-Qubit Gates' },
      { label: 'Encoding', value: 'Ry Angle, ANGLE_MAX = π/2 (Selected on TRAIN CV)' },
      { label: 'Trainable Quantum Gates', value: '0 — Only the Classical Head Learns' },
      { label: 'Simulation', value: 'Exact Aer Statevector (8.9e-16 Max Deviation)' },
    ],
    highlights: [
      'Validation PR-AUC 0.8786 against its matched RFF-16 control at 0.8712 — parity, inside sampling noise',
      'Behind that same control on ROC-AUC (0.8802 vs 0.9078); no superiority claim is made or authorised',
      'Clears its own patient-blocked permutation null (p = 0.004975, z = 5.76) — the computation is real',
      '1.47 ms mean per image, effective rank 16/16, zero constant observables',
    ],
    icon: 'Atom',
    color: 'quantum',
  },
  {
    id: 'clinical',
    title: 'Clinical Interoperability & Security',
    tagline: 'FastAPI, PostgreSQL & HL7 FHIR R4',
    description:
      'Backend infrastructure orchestrating typed inference payloads, model artifact management, and healthcare system integration. Connects field screenings to hospital EHRs using standardized HL7 FHIR R4 resources.',
    specs: [
      { label: 'Backend Architecture', value: 'FastAPI / Python (Pydantic Schemas)' },
      { label: 'Database Security', value: 'PostgreSQL Row-Level Security (RLS)' },
      { label: 'Clinical Interoperability', value: 'HL7 FHIR R4 (Observation & RiskAssessment)' },
      { label: 'Pseudonymization', value: 'UUIDv4 (clinic_record = clinic_session)' },
    ],
    highlights: [
      'Calibrator (Platt or isotonic) chosen by out-of-fold CV with a paired-SE tie rule, never hand-picked',
      'SNOMED CT coding (363349007) for malignant tumor of oral cavity screening assessment',
      'Zero default retention of raw patient photos or high-dimensional pixel matrices',
      'Row-aligned classical baseline comparisons ensuring fair evaluation on identical populations',
    ],
    icon: 'ShieldCheck',
    color: 'slate',
  },
];

export const QUANTUM_CONCEPTS: QuantumConcept[] = [
  {
    title: 'Angle Encoding into 8 Qubits',
    description:
      'The ROI descriptor is reduced by a TRAIN-fitted PCA to 8 dimensions, each scaled into a single Ry rotation. The angle ceiling π/2 was chosen on nested patient-grouped cross-validation over TRAIN only — the specced [0, π] range folded like cos(2θ) and scored 0.534 against π/2\'s 0.706.',
    badge: 'Ry Angle Encoding',
    formula: '|ψ(x)⟩ = ⨂_{j=1}^{8} R_y(θ_j)|0⟩,  θ_j ∈ [0, π/2]',
    analogy: 'Eight dials, each turned by one feature — turned too far, two different features land on the same reading.',
  },
  {
    title: 'Entanglement, Measured Not Assumed',
    description:
      'A nearest-neighbour CZ ring sits between two re-uploading blocks. Whether it actually does anything is measured, not asserted: the connected correlation is the witness. A single block\'s ring commutes through every diagonal Z-string and contributes identically zero, so it would not count as quantum content at all.',
    badge: 'Entanglement Witness',
    formula: 'C_jk = ⟨Z_j Z_k⟩ − ⟨Z_j⟩⟨Z_k⟩,  mean |C| = 0.184',
    analogy: 'Proving the wires between the dials are connected by checking that turning one moves another.',
  },
  {
    title: 'Zero Trainable Quantum Parameters',
    description:
      'Nothing inside the circuit learns. The feature map is fixed; only a small classical logistic head is fitted on the 16 observables it emits. That keeps the generalization budget honest at this sample size and removes any question of the quantum stage having been tuned toward a score.',
    badge: 'Fixed Map, Learned Head',
    formula: 'T = 0 trainable quantum gates;  16 observables → logistic head',
    analogy: 'A fixed lens, not an adjustable one — you cannot bend it until the picture flatters you.',
  },
  {
    title: 'The 16-Qubit Amplitude Encoding Is Our Negative Control',
    description:
      'Mapping a whole 256×256 image onto 2¹⁶ amplitudes was the project\'s original thesis. It was measured and it failed: none of 15 variant × weight settings beat its own patient-blocked null, best PR-AUC 0.545337 against 0.403846 chance, p = 1.0. An RBF-SVM on the identical amplitude vectors also landed near chance. It is now kept deliberately, as the control that says what a dead representation looks like.',
    badge: 'Retained Negative Control',
    formula: '|ψ⟩ = ∑_{i=0}^{65,535} a_i |i⟩  →  no exposable class signal',
    analogy: 'The experiment we did not delete when it said no.',
  },
];

export const PHILOSOPHY_PILLARS: PhilosophyPillar[] = [
  {
    title: 'Patient-Level Data Isolation',
    tagline: 'Eliminating Cross-Patient Data Leakage',
    description:
      'Multiple images from the same patient share correlated visual patterns. Orqis strictly enforces patient-level dataset partitioning (P_train ∩ P_val ∩ P_test = ∅) across all 328 patients.',
    icon: 'Shield',
    points: [
      'Guarantees no patient images appear across training and test partitions simultaneously',
      'Dataset reconstruction: 2,436 usable images from 328 patients (University of Peradeniya Oral Cancer Dataset v1, SMART-OM)',
      'Structural tripwires in test suites prevent accidental condition merging or leakage regressions',
    ],
  },
  {
    title: 'Methodological Rigor over Shortcuts',
    tagline: 'Lesion-Only Supervised Targets (DEC-020)',
    description:
      'Analysis revealed that broad region-box area alone separated diagnostic classes at AUC ≈ 0.983. Orqis decisively rejected broad region annotations to prevent models from learning geometric shortcuts.',
    icon: 'Eye',
    points: [
      'Trained exclusively on 318 lesion-annotated images with verified ground-truth boundaries',
      'Separates experimental conditions: A (Oracle ROI), B (Predicted ROI), and C (Fallback)',
      'Unannotated test images are transparently reported as unverified rather than false successes',
    ],
  },
  {
    title: 'Privacy-Conscious Data Minimization',
    tagline: 'Zero Default Storage of Raw Patient Photos',
    description:
      'The architecture follows strict data minimization. High-dimensional raw images are processed locally on-device and cleared from memory immediately following feature extraction.',
    icon: 'Lock',
    points: [
      'Pseudonymous UUIDv4 tokens separate clinical identifiers from inference records',
      'PostgreSQL Row-Level Security (RLS) ensures strict clinic tenant isolation',
      'Local inference mode and privacy-conscious minimized payload quantum-cloud mode',
    ],
  },
  {
    title: 'Defensible Science & Clinical Humility',
    tagline: 'Matched Controls & Published Null Results',
    description:
      'Orqis does not assume quantum superiority in advance. Every quantum configuration is benchmarked against a matched classical control evaluated on the exact same patient rows — and when it loses, the loss is what gets published.',
    icon: 'HeartHandshake',
    points: [
      'The current dataset has not demonstrated a classical bottleneck that justifies quantum processing',
      'Simulator conveniences (Aer set_statevector) are clearly distinguished from hardware constraints',
      'Explicitly framed as AI-assisted preliminary decision support, not a definitive diagnosis',
    ],
  },
];

export const TIMELINE_MILESTONES: TimelineMilestone[] = [
  {
    phase: 'Phase A & B',
    quarter: 'Complete',
    title: 'Dataset Reconstruction & Leakage Controls',
    status: 'completed',
    description:
      'Curated the University of Peradeniya Oral Cancer Dataset v1 (2,436 usable images from 328 patients after QC rejection of 33 images). Enforced strict patient-level partitioning and resolved a Ca 2.png filename collision.',
    deliverables: [
      '2,436 usable images from 328 patients (original 2,469 − 33 QC rejections)',
      'Patient-level split: 230 train / 50 val / 48 test — zero cross-partition leakage',
      'Reproducible cache generation and verified dataset manifests',
    ],
  },
  {
    phase: 'Phase C',
    quarter: 'Complete',
    title: 'Frozen MobileNet Lesion Localizer (DEC-020)',
    status: 'completed',
    description:
      'Trained MobileNet exclusively on lesion boxes (DEC-020) to eliminate crop-geometry shortcuts (region-box AUC ≈ 0.983). Frozen checkpoint evaluated on held-out test set: 0.5279 mean IoU.',
    deliverables: [
      'Frozen localizer.pt (Epoch 17, threshold 0.30, SHA-256: 27d6036e…)',
      '98.16% test localization rate (374/381 localized; 7 fallback; 0 rejected)',
      'Validation: 0.5285 mean IoU | Test: 0.5279 mean IoU (96% IoU≥0.25 val)',
    ],
  },
  {
    phase: 'Phase C',
    quarter: 'Complete',
    title: 'Deterministic Pixel Pipeline & Quantum Cost Accounting',
    status: 'completed',
    description:
      'Engineered the exact 256×256 grayscale representation (65,536 pixels = 2¹⁶ amplitudes) and measured what it would actually cost on hardware. State preparation, not the ansatz, turned out to be the entire bill — and it is reported separately from every simulator number.',
    deliverables: [
      'State diagnostics: mean uniform-state overlap 0.946, random ⟨Z₀⟩ spread 0.0186, 1024-shot SE 0.031',
      'State-prep cost measured 57 CX at 6 qubits → 4,083 at 12; extrapolated ~62,940 at 16',
      'Extrapolation is a cost estimate only — never presented as progress toward hardware viability',
    ],
  },
  {
    phase: 'Phase D',
    quarter: 'Complete',
    title: 'Pipeline Frozen, Then the Test Partition Read Exactly Once',
    status: 'completed',
    description:
      'The pipeline was declared frozen at three artifacts, and a single module — refusing to run without --confirm-frozen, containing no sweep, argmax or calibration fit — read the held-out test partition once. The quantum model lost, and the loss was reported as measured.',
    deliverables: [
      'Honest oracle condition (49 images, 18 positives): quantum PR-AUC 0.5443 vs classical 0.7095',
      'Thresholds read out of frozen artifacts; no re-run, no re-sweep, no second test read',
      'Result published rather than withheld pending a stronger ansatz',
    ],
  },
  {
    phase: 'Phase E0 & E1',
    quarter: 'Complete',
    title: 'The Negative Control and the Failure Map',
    status: 'completed',
    description:
      'Before repairing anything, E0 asked whether the amplitude state carried exposable class signal at all. It answered no. E1 then required a measured classical bottleneck before any second quantum model was authorised — and measured none.',
    deliverables: [
      'E0.1: 0 of 15 variant × weight settings beat their own patient-blocked null (best 0.545337 vs 0.403846 chance, p = 1.0)',
      'An RBF-SVM on the identical 65,536-amplitude vectors also landed near chance — the representation, not the optimizer',
      'E1.1 failure map returned a NULL result: classical_bottleneck_measured = false, no quantum model authorised',
    ],
  },
  {
    phase: 'Phase E2',
    quarter: 'Complete',
    title: 'An Honest 8-Qubit Demonstrator, Held to a Demonstrator Bar',
    status: 'completed',
    description:
      'With no advantage authorised, E2 built a demonstrator and said so up front. Its KEEP criteria are speed, non-degeneracy and beating its own permutation null — explicitly not beating the classical baseline. It passed all three, and ties-to-loses against its matched control exactly as pre-disclosed.',
    deliverables: [
      'Validation PR-AUC 0.8786 vs matched RFF-16 control 0.8712; ROC-AUC 0.8802 vs 0.9078 — parity to deficit',
      'Clears its own patient-blocked null at p = 0.004975, z = 5.76; exact Aer agreement to 8.9e-16',
      'Angle ceiling π/2 selected on TRAIN nested grouped CV after [0, π] was shown to fold like cos(2θ)',
    ],
  },
  {
    phase: 'Phase E3',
    quarter: 'Complete',
    title: 'Four Quantum Attempts, Four Null Results',
    status: 'completed',
    description:
      'Four pre-registered rounds descended the research ladder, each against a matched classical control on identical rows, each gated by the same pre-registered increment test. None beat the classical baseline C7. All four are in the decision log.',
    deliverables: [
      'Round 1 hybrid fusion · Round 2 fidelity quantum kernel · Round 3 trainable shallow VQC · Round 4 Havlíček ZZ map',
      'Diagnosed, not hand-waved: the kernel was healthy not concentrated; the VQC trained without a barren plateau',
      'C7 rebuilt and reproduced to 1e-6 (0.913038) as a correctness gate before any comparison was read',
    ],
  },
  {
    phase: 'Phase F – I',
    quarter: 'Complete',
    title: 'Backend, FHIR Interoperability & Clinical Delivery',
    status: 'completed',
    description:
      'Production FastAPI inference service with typed Pydantic schemas and model versioning, HL7 FHIR R4 export, and the Flutter client connected end to end — Camera/Gallery → Preview → Analyzing → API → Result → History.',
    deliverables: [
      'FastAPI backend: image upload → ROI → features → scoring → calibration → typed result',
      'HL7 FHIR R4 Observation & RiskAssessment generation with SNOMED CT 363349007',
      'Flutter client connected, plus security testing for camera denial, invalid images and timeouts',
    ],
  },
  {
    phase: 'Phase E4',
    quarter: 'Active Phase',
    title: 'PTB-XL: A New Arena, the Quantum Gate Still Shut',
    status: 'in-progress',
    description:
      'Four nulls on oral imaging made the arena itself the suspect. E4 moved the quantum question to PTB-XL — 21,799 open 12-lead ECG records from 18,869 patients — chosen on pre-registered criteria, audited with folds confirmed patient-disjoint before a single model was fitted.',
    deliverables: [
      'Classical ceiling measured first: ROC-AUC 0.940234, with the 8-dimensional near-term regime at 0.915006 (97.3%)',
      'Permutation p = 0.004975, its one assumption violation quantified rather than glossed',
      'Deep rung still owed — until it is run, no quantum model on PTB-XL is authorised',
    ],
  },
];

export const TEAM_MEMBERS: TeamMember[] = [
  {
    role: 'Clinical Informatics & Healthcare',
    discipline: 'Medical Systems & FHIR Standards',
    department: 'Healthcare Systems & Ethics',
    focus: 'HL7 FHIR R4 interoperability, SNOMED CT (363349007) clinical coding, patient-safe triage strata, and ethical compliance.',
    avatarPlaceholder: 'MD',
  },
  {
    role: 'Computer Vision & Deep Learning',
    discipline: 'Medical Image Processing',
    department: 'Classical ML & ROI Architecture',
    focus: 'MobileNet lesion localizer (DEC-020), deterministic 256×256 grayscale pixel pipeline, and geometric leakage prevention.',
    avatarPlaceholder: 'CV',
  },
  {
    role: 'Quantum Machine Learning',
    discipline: 'Quantum Algorithms & Optimization',
    department: 'Quantum Computing Research',
    focus: '8-qubit angle encoding and entanglement witnesses, matched classical controls for every quantum claim, the pre-registered null-result protocol, and the E4 PTB-XL research arena.',
    avatarPlaceholder: 'QM',
  },
  {
    role: 'Mobile Architecture & UX',
    discipline: 'Patient-Centered Design',
    department: 'Mobile Engineering (Flutter)',
    focus: 'com.orqis.patient client architecture, on-device image quality checks, EXIF scrubbing, and WCAG AA accessibility.',
    avatarPlaceholder: 'UX',
  },
];

export const CLINICAL_DISCLAIMER =
  'Orqis is an investigational research initiative and clinical decision support system designed to assist frontline healthcare professionals. It provides an AI-assisted preliminary screening risk estimate and does not provide a definitive pathological diagnosis or replace histopathological biopsy by certified specialists.';
