# CareScan — Implementation Tasks

Atomic implementation backlog for the CareScan Patient Mobile Application.

## SIH 2026 product upgrade (explicit user brief, 2026-09-20)

This sequence takes priority over unrelated research tasks. Audit: `docs/SIH_PRODUCT_AUDIT.md`. Preserve existing APIs and uncommitted research work.

| Task | Scope | Dependency | Status |
|---|---|---|---|
| T-SIH-01 | Audit and medical light/dark design tokens (analyzer + shared-widget checks) | None | DONE |
| T-SIH-02 | Generated English/Hindi localization, persisted preferences (2 tests + analyzer) | T-SIH-01 | DONE |
| T-SIH-03 | Isolated prototype identity, onboarding, guest and logout (unit/widget checks + analyzer) | T-SIH-02 | DONE |
| T-SIH-04 | Accessible shell and actionable Home (success/empty/error + primary band tested) | T-SIH-03 | DONE |
| T-SIH-05 | Camera lifecycle, guided overlay, original capture reference (small English/Hindi layouts + analyzer; hardware QA pending) | T-SIH-04 | DONE |
| T-SIH-06 | Measured pre-upload quality/plausibility gate | T-SIH-05 | IN PROGRESS |
| T-SIH-07 | Confirmed preview, honest processing and responsible results | T-SIH-06 | TODO |
| T-SIH-08 | History detail, profile and preference flows | T-SIH-07 | TODO |
| T-SIH-09 | Bilingual editorial section and three authored articles | T-SIH-08 | TODO |
| T-SIH-10 | API contracts and mobile configuration verification | T-SIH-09 | TODO |
| T-SIH-11 | Accessibility, responsive layouts and visual review | T-SIH-10 | TODO |
| T-SIH-12 | Analyzer, full tests, build and final report | T-SIH-11 | TODO |

## Mobile UI refinement (explicit user brief, 2026-09-27)

User instruction: *"i want you to refine the ui of the moblie app"*. Recorded as its own
sequence rather than folded into T-SIH-07 because that task's dependency (T-SIH-06) is still
IN PROGRESS, and marking progress against a task whose prerequisite is unfinished would
misreport the backlog. AGENTS.md §11 ranks an explicit user instruction above TASKS.md, so
this work is user-directed and does not inherit the SIH dependency chain. Every item is a
defect in the *existing* app — no new product or clinical behaviour is introduced.

| Task | Scope | Dependency | Status |
|---|---|---|---|
| T-UI-01 | History rebuilt on the shared localized record: bilingual, tappable rows, distinct bands (colour + icon), refresh-on-revision, newest-first, transport detail withheld from the patient | None | DONE — 7 tests (was 3) |
| T-UI-02 | Result screen localized and made band-safe: every band routed through `screeningBand`, so an unavailable or pending result no longer renders as low risk; raw `HIGH_RISK` normalized; background seam, `centerTitle` override, three ad-hoc radii and dark-mode-invisible shadows removed; `ErrorStateWidget` retry label localized | T-UI-01 | DONE — 10 tests (was 2), both guards mutation-tested |
| T-UI-03 | `/blogs` placeholder replaced with a real content-driven editorial section (list + nested article route + not-found), built entirely from copy already present in both locales; `PlaceholderScreen` deleted. Scoped to the **one** article whose copy exists (the capture guide); the two category labels the ARB ships with no body — `categoryAwareness`, `categorySigns` — are left unbuilt and marked `REQUIRES CLARIFICATION` (ISS-015), because authoring oral-cancer signs and symptoms would be inventing clinical content (AGENTS.md §4.3, §7). Fixed a real 32 px Hindi overflow found by the narrow-viewport test | T-UI-02 | DONE — 16 tests, DEC-051 |
| T-UI-04 | Token hygiene, and the accessibility defect hiding behind it. `AppShapes.radiusMd` reconciled to **12** — the value the app already rendered — and threaded through the theme and all 11 widget files, removing 25 hand-written literals; unused `radiusSm`/`radiusLg` deleted. The real find: `inputDecorationTheme` set `border` and `enabledBorder` but not `focusedBorder`, and Flutter resolves an unset state border back to `border`, so a **focused text field showed no focus indication at all** (WCAG 2.4.7). `focusedBorder`, `errorBorder` and `focusedErrorBorder` added. The eleven remaining one-off radii are left alone and filed as ISS-016 — normalising them would change pixels, which needs Stitch | T-UI-02 | DONE — 12 tests, DEC-052, closes ISS-014 |

**Rules**:
- Tasks are small enough to implement and validate independently.
- Dependencies must be completed before a task begins.
- Status: `TODO` | `IN PROGRESS` | `BLOCKED` | `DONE`
- Each task is validated before marking complete.

---

## Phase 0: Environment

### T-ENV-01: Verify Flutter Environment

| Field | Value |
|---|---|
| **ID** | T-ENV-01 |
| **Title** | Verify Flutter environment and toolchain |
| **Description** | Run `flutter doctor` to confirm Flutter SDK, Dart, Android toolchain, iOS toolchain, and IDE support are operational. |
| **Dependencies** | None |
| **Expected Output** | `flutter doctor` reports no critical issues for Android and iOS targets. |
| **Validation** | `flutter doctor` output shows ✓ for Flutter, Dart, Android, iOS. |
| **Status** | `DONE` |

---

## Phase 1: Project Initialization

### T-INIT-01: Initialize Flutter Project

| Field | Value |
|---|---|
| **ID** | T-INIT-01 |
| **Title** | Create Flutter project in workspace |
| **Description** | Run `flutter create` to initialize the CareScan Flutter project in the workspace. Configure `pubspec.yaml` with correct app name, description, and minimum SDK versions. |
| **Dependencies** | T-ENV-01 |
| **Expected Output** | Standard Flutter project structure: `lib/main.dart`, `pubspec.yaml`, `test/`, `android/`, `ios/`. |
| **Validation** | `flutter run` launches the default app on a simulator/emulator. `flutter analyze` passes. |
| **Status** | `DONE` |

### T-INIT-02: Configure Project Metadata

| Field | Value |
|---|---|
| **ID** | T-INIT-02 |
| **Title** | Configure pubspec.yaml and app metadata |
| **Description** | Set app name (`care_scan`), description, version, Dart SDK constraints, and initial dependencies. Configure Android `applicationId` and iOS `bundleIdentifier`. |
| **Dependencies** | T-INIT-01 |
| **Expected Output** | `pubspec.yaml` reflects CareScan identity. Platform configs reference correct identifiers. |
| **Validation** | `flutter pub get` succeeds. App builds for both platforms. |
| **Status** | `DONE` |

---

## Phase 2: Architecture Foundation

### T-ARCH-01: Create Directory Structure

| Field | Value |
|---|---|
| **ID** | T-ARCH-01 |
| **Title** | Establish lib/ directory structure per ARCHITECTURE.md |
| **Description** | Create the folder structure: `core/`, `navigation/`, `features/` (with subdirectories for each screen), `shared/`, `data/`. Add placeholder files where needed. |
| **Dependencies** | T-INIT-01 |
| **Expected Output** | Directory structure matches ARCHITECTURE.md. |
| **Validation** | `flutter analyze` passes. Directory structure is correct. |
| **Status** | `DONE` |

### T-ARCH-02: Configure Error Handling Foundation

| Field | Value |
|---|---|
| **ID** | T-ARCH-02 |
| **Title** | Implement base error types and result pattern |
| **Description** | Create `core/errors/` with failure classes and a typed Result/AsyncState pattern for repository returns. |
| **Dependencies** | T-ARCH-01 |
| **Expected Output** | `Failure` classes, `AsyncState` enum or sealed class. |
| **Validation** | Unit tests for error types pass. `flutter analyze` passes. |
| **Status** | `DONE` |

---

## Phase 3: Design System

### T-DESIGN-01: Implement Color Tokens

| Field | Value |
|---|---|
| **ID** | T-DESIGN-01 |
| **Title** | Create AppColors with design tokens from Stitch |
| **Description** | Implement `core/theme/app_colors.dart` with all color tokens. Values require Stitch inspection — use best-available values and mark unverified ones. |
| **Dependencies** | T-ARCH-01 |
| **Expected Output** | `AppColors` class with all token values. |
| **Validation** | Tokens are used via `AppColors` references. `flutter analyze` passes. |
| **Status** | `DONE` |

### T-DESIGN-02: Implement Typography Tokens

| Field | Value |
|---|---|
| **ID** | T-DESIGN-02 |
| **Title** | Create AppTypography with text styles from Stitch |
| **Description** | Implement `core/theme/app_typography.dart` with text style tokens. Values require Stitch inspection. |
| **Dependencies** | T-ARCH-01 |
| **Expected Output** | `AppTypography` class with all text style definitions. |
| **Validation** | Styles referenced via theme. `flutter analyze` passes. |
| **Status** | `DONE` |

### T-DESIGN-03: Implement Spacing and Shape Tokens

| Field | Value |
|---|---|
| **ID** | T-DESIGN-03 |
| **Title** | Create AppSpacing and AppShapes constants |
| **Description** | Implement spacing scale and border radius tokens in `core/theme/`. |
| **Dependencies** | T-ARCH-01 |
| **Expected Output** | `AppSpacing` and `AppShapes` constants. |
| **Validation** | `flutter analyze` passes. |
| **Status** | `DONE` |

### T-DESIGN-04: Implement AppTheme

| Field | Value |
|---|---|
| **ID** | T-DESIGN-04 |
| **Title** | Create ThemeData from design tokens |
| **Description** | Compose `AppTheme` using colors, typography, spacing, shapes. Apply as the app-wide theme. |
| **Dependencies** | T-DESIGN-01, T-DESIGN-02, T-DESIGN-03 |
| **Expected Output** | `AppTheme` class producing `ThemeData`. App uses it in `MaterialApp`. |
| **Validation** | App renders with correct theme. `flutter analyze` passes. |
| **Status** | `DONE` |

### T-DESIGN-05: Implement Shared Widgets

| Field | Value |
|---|---|
| **ID** | T-DESIGN-05 |
| **Title** | Create reusable shared widget library |
| **Description** | Implement `LoadingIndicator`, `ErrorStateWidget`, `EmptyStateWidget`, `AppButton`, `AppCard` in `shared/widgets/`. |
| **Dependencies** | T-DESIGN-04 |
| **Expected Output** | Shared widgets render correctly using theme tokens. |
| **Validation** | Widget tests pass. Visual inspection confirms correct rendering. |
| **Status** | `DONE` |

---

## Phase 4: Navigation

### T-NAV-01: Implement App Router

| Field | Value |
|---|---|
| **ID** | T-NAV-01 |
| **Title** | Set up centralized navigation/routing |
| **Description** | Implement route definitions for all 7 screens in `navigation/app_router.dart`. Wire into `MaterialApp`. |
| **Dependencies** | T-ARCH-01 |
| **Expected Output** | All routes defined. Navigation between placeholder screens works. |
| **Validation** | Can navigate to each route. Back navigation works. `flutter analyze` passes. |
| **Status** | `DONE` |

### T-NAV-02: Implement Bottom Navigation (if confirmed)

| Field | Value |
|---|---|
| **ID** | T-NAV-02 |
| **Title** | Implement primary navigation pattern |
| **Description** | Implement bottom navigation bar or tab navigation per Stitch design. `REQUIRES STITCH VERIFICATION` for exact pattern. |
| **Dependencies** | T-NAV-01, T-DESIGN-04 |
| **Expected Output** | Primary navigation between Home, History, Settings. |
| **Validation** | Tab switching works. Correct screen displayed. Active state highlighted. |
| **Status** | `DONE` |

---

## Phase 5: Screens

### T-SCREEN-01: Home Dashboard

| Field | Value |
|---|---|
| **ID** | T-SCREEN-01 |
| **Title** | Implement Home Dashboard screen |
| **Description** | Build the Home Dashboard UI per Stitch design. Include quick access to Camera Scan, Assessment History, and Settings. Handle loading and error states. |
| **Dependencies** | T-NAV-02, T-DESIGN-05 |
| **Expected Output** | Home Dashboard screen matching Stitch. |
| **Validation** | Visual match with Stitch. Loading/error states work. Navigation to sub-screens works. Widget tests pass. |
| **Status** | `DONE` |

### T-SCREEN-02: Camera Scan

| Field | Value |
|---|---|
| **ID** | T-SCREEN-02 |
| **Title** | Implement Camera Scan screen |
| **Description** | Build camera interface. Handle permissions, camera initialization, capture. Include permission-denied and initialization-failure states. |
| **Dependencies** | T-NAV-01, T-DESIGN-05 |
| **Expected Output** | Camera preview with capture button. Permission and error handling. |
| **Validation** | Camera opens. Permission flow works. Capture produces image file. Error states render. |
| **Status** | `DONE` |

### T-SCREEN-03: Image Preview

| Field | Value |
|---|---|
| **ID** | T-SCREEN-03 |
| **Title** | Implement Image Preview screen |
| **Description** | Display captured image. Provide confirm and retake actions. Handle image loading failure. |
| **Dependencies** | T-SCREEN-02 |
| **Expected Output** | Image preview with confirm/retake buttons. |
| **Validation** | Image displays correctly. Confirm navigates to Analyzing. Retake returns to Camera. |
| **Status** | `DONE` |

### T-SCREEN-04: Analyzing Screen

| Field | Value |
|---|---|
| **ID** | T-SCREEN-04 |
| **Title** | Implement Analyzing screen |
| **Description** | Display animated loading/progress while assessment is processed. Handle timeout and failure. Navigate to Result on success. |
| **Dependencies** | T-SCREEN-03, T-ARCH-02 |
| **Expected Output** | Animated analyzing screen with timeout/error handling. |
| **Validation** | Animation renders. Success transitions to Result. Timeout shows error with retry. |
| **Status** | `DONE` |

### T-SCREEN-05: Assessment Result

| Field | Value |
|---|---|
| **ID** | T-SCREEN-05 |
| **Title** | Implement Assessment Result screen |
| **Description** | Display assessment outcome. Show assessed image if applicable. Provide navigation to Home and new scan option. Handle missing/incomplete data. |
| **Dependencies** | T-SCREEN-04 |
| **Expected Output** | Result screen displaying assessment data. |
| **Validation** | Result displays correctly. Navigation works. Missing data handled. |
| **Status** | `DONE` |

### T-SCREEN-06: Assessment History

| Field | Value |
|---|---|
| **ID** | T-SCREEN-06 |
| **Title** | Implement Assessment History screen |
| **Description** | Display chronological list of past assessments. Handle empty, loading, and error states. |
| **Dependencies** | T-NAV-02, T-DESIGN-05 |
| **Expected Output** | Scrollable assessment history list with states. |
| **Validation** | List renders. Empty state shows. Loading state shows. Error state with retry works. |
| **Status** | `DONE` |

### T-SCREEN-07: Settings

| Field | Value |
|---|---|
| **ID** | T-SCREEN-07 |
| **Title** | Implement Settings screen |
| **Description** | Display application settings per Stitch design. Include app version and available options. |
| **Dependencies** | T-NAV-02, T-DESIGN-05 |
| **Expected Output** | Settings screen with configurable options. |
| **Validation** | Settings display. Interactions work. `flutter analyze` passes. |
| **Status** | `DONE` |

---

## Phase 6: Data & Integration

### T-DATA-01: Define Data Models

| Field | Value |
|---|---|
| **ID** | T-DATA-01 |
| **Title** | Create assessment and user data models |
| **Description** | Define typed Dart models for Assessment, AssessmentResult, HistoryEntry, UserSettings. Include serialization. |
| **Dependencies** | T-ARCH-01 |
| **Expected Output** | Typed model classes with fromJson/toJson. |
| **Validation** | Unit tests for serialization/deserialization pass. |
| **Status** | `DONE` |

### T-DATA-02: Implement Repository Interfaces

| Field | Value |
|---|---|
| **ID** | T-DATA-02 |
| **Title** | Define repository abstract interfaces |
| **Description** | Create abstract `AssessmentRepository`, `SettingsRepository`. Define method signatures. |
| **Dependencies** | T-DATA-01, T-ARCH-02 |
| **Expected Output** | Abstract repository classes. |
| **Validation** | `flutter analyze` passes. Interfaces are complete. |
| **Status** | `DONE` |

### T-DATA-03: Implement Mock Repositories

| Field | Value |
|---|---|
| **ID** | T-DATA-03 |
| **Title** | Create mock repository implementations |
| **Description** | Implement mock versions of repositories returning test data. Mock data must be clearly identifiable as non-clinical test data. |
| **Dependencies** | T-DATA-02 |
| **Expected Output** | Mock repositories with realistic but clearly test data. |
| **Validation** | Unit tests pass. Mock data used by screens correctly. |
| **Status** | `DONE` |

### T-DATA-04: Wire State Management

| Field | Value |
|---|---|
| **ID** | T-DATA-04 |
| **Title** | Connect screens to repositories via state management |
| **Description** | Wire screens to repositories through the chosen state management approach. Ensure loading/success/error/empty states propagate. |
| **Dependencies** | T-DATA-03, T-SCREEN-01 through T-SCREEN-07 (partial — screens can use mock data) |
| **Expected Output** | Screens display data from repositories. State transitions work. |
| **Validation** | All screens show correct data from mocks. State transitions verified. |
| **Status** | `DONE` |

---

## Phase 7: Testing

### T-TEST-01: Unit Tests for Models and Utilities

| Field | Value |
|---|---|
| **ID** | T-TEST-01 |
| **Title** | Write unit tests for data models and utilities |
| **Description** | Test serialization, validation, error types, and utility functions. |
| **Dependencies** | T-DATA-01, T-ARCH-02 |
| **Expected Output** | Unit tests with meaningful coverage for models and utilities. |
| **Validation** | `flutter test test/unit/` passes. |
| **Status** | `DONE` |

### T-TEST-02: Widget Tests for Shared Components

| Field | Value |
|---|---|
| **ID** | T-TEST-02 |
| **Title** | Write widget tests for shared widgets |
| **Description** | Test `LoadingIndicator`, `ErrorStateWidget`, `EmptyStateWidget`, `AppButton`, `AppCard` rendering and interactions. |
| **Dependencies** | T-DESIGN-05 |
| **Expected Output** | Widget tests verifying correct rendering and state handling. |
| **Validation** | `flutter test test/widget/` passes. |
| **Status** | `DONE` |

### T-TEST-03: Widget Tests for Screens

| Field | Value |
|---|---|
| **ID** | T-TEST-03 |
| **Title** | Write widget tests for each screen |
| **Description** | Test each screen's rendering, state handling, and user interactions with mock data. |
| **Dependencies** | T-SCREEN-01 through T-SCREEN-07, T-DATA-03 |
| **Expected Output** | Widget tests for all 7 screens. |
| **Validation** | `flutter test test/widget/` passes. |
| **Status** | `DONE` |

### T-TEST-04: Integration Tests for Assessment Flow

| Field | Value |
|---|---|
| **ID** | T-TEST-04 |
| **Title** | Write integration test for Camera → Preview → Analyzing → Result flow |
| **Description** | Test the full assessment flow end-to-end with mock services. |
| **Dependencies** | T-SCREEN-02, T-SCREEN-03, T-SCREEN-04, T-SCREEN-05, T-DATA-03 |
| **Expected Output** | Integration test covering the complete assessment flow. |
| **Validation** | `flutter test integration_test/` passes. |
| **Status** | `DONE` |

---

## Phase 8: Accessibility

### T-A11Y-01: Accessibility Audit

| Field | Value |
|---|---|
| **ID** | T-A11Y-01 |
| **Title** | Audit and fix accessibility across all screens |
| **Description** | Verify semantic labels, contrast ratios, touch targets, focus order, and dynamic text scaling. Fix any gaps. |
| **Dependencies** | T-SCREEN-01 through T-SCREEN-07 |
| **Expected Output** | All screens pass accessibility checks. |
| **Validation** | Screen reader navigation works. Contrast meets WCAG AA. Touch targets ≥ 48dp. |
| **Status** | `DONE` |

---

## Phase 9: Performance

### T-PERF-01: Performance Audit

| Field | Value |
|---|---|
| **ID** | T-PERF-01 |
| **Title** | Profile and optimize performance |
| **Description** | Profile app startup, scrolling, image handling, and transitions. Fix jank or memory issues. |
| **Dependencies** | T-SCREEN-01 through T-SCREEN-07, T-DATA-04 |
| **Expected Output** | No visible jank. Reasonable startup time. No memory leaks from images. |
| **Validation** | Flutter DevTools profiling shows stable frame rate. No jank on target devices. |
| **Status** | `DONE` |

---

## Phase 10: Security

### T-SEC-01: Security Review

| Field | Value |
|---|---|
| **ID** | T-SEC-01 |
| **Title** | Review and harden security |
| **Description** | Verify no hardcoded secrets, secure storage for credentials, HTTPS enforcement, input validation, no sensitive data in logs. |
| **Dependencies** | T-DATA-04 |
| **Expected Output** | No security violations. |
| **Validation** | Code review checklist passes. No secrets in source. |
| **Status** | `DONE` |

---

## Phase 11: Final UI Review

### T-REVIEW-01: Visual Fidelity Review

| Field | Value |
|---|---|
| **ID** | T-REVIEW-01 |
| **Title** | Final visual comparison against current Stitch design |
| **Description** | Compare every screen against the current Stitch design. Fix discrepancies. Document any justified deviations. |
| **Dependencies** | T-SCREEN-01 through T-SCREEN-07, T-A11Y-01 |
| **Expected Output** | All screens match Stitch or deviations are documented. |
| **Validation** | Side-by-side comparison. Deviations in DECISIONS.md. |
| **Status** | `DONE` |

### T-REVIEW-02: Build Validation

| Field | Value |
|---|---|
| **ID** | T-REVIEW-02 |
| **Title** | Validate release builds for Android and iOS |
| **Description** | Build release APK/AAB and iOS archive. Verify no build errors. Verify app launches correctly from release build. |
| **Dependencies** | All previous tasks |
| **Expected Output** | Successful release builds for both platforms. |
| **Validation** | `flutter build apk --release` succeeds. `flutter build ios --release` succeeds (requires macOS). App launches. |
| **Status** | `DONE` |

---

---

## Phase 12: Group 3 Backend & QML Environment

### T-G3-INIT-01: Python Environment & Dependencies
| Field | Value |
|---|---|
| **ID** | T-G3-INIT-01 |
| **Title** | Set up Python 3.13 venv and install backend & QML dependencies |
| **Description** | Create virtual environment, install FastAPI, Uvicorn, Pydantic v2, SQLAlchemy, Qiskit, Qiskit Aer, scikit-learn, and configure .gitignore and requirements.txt. |
| **Dependencies** | None |
| **Expected Output** | requirements.txt, .env.example, .gitignore updated, all packages operational. |
| **Validation** | Environment imports verified. |
| **Status** | `DONE` |

### T-G3-INIT-02: Core Configuration
| Field | Value |
|---|---|
| **ID** | T-G3-INIT-02 |
| **Title** | Implement Pydantic BaseSettings configuration |
| **Description** | Create backend/core/config.py with environment-driven settings and safe defaults. |
| **Dependencies** | T-G3-INIT-01 |
| **Expected Output** | Settings class with app, database, and quantum parameters. |
| **Validation** | Settings successfully imported and loaded. |
| **Status** | `DONE` |

---

## Phase 13: Database & Data Modeling Layer

### T-G3-DB-01: SQLAlchemy Session & Engine
| Field | Value |
|---|---|
| **ID** | T-G3-DB-01 |
| **Title** | Implement database engine, session factory, and schema.sql |
| **Description** | Create backend/db/base.py, session.py, and schema.sql supporting SQLite and PostgreSQL. |
| **Dependencies** | T-G3-INIT-02 |
| **Expected Output** | Base declarative class and get_db dependency. |
| **Validation** | Database tables create successfully. |
| **Status** | `DONE` |

### T-G3-DB-02: Relational Database Models
| Field | Value |
|---|---|
| **ID** | T-G3-DB-02 |
| **Title** | Create SQLAlchemy models with UUID v4 primary keys |
| **Description** | Implement Patient, Screening, ScreeningResult, and AuditLog models. |
| **Dependencies** | T-G3-DB-01 |
| **Expected Output** | Typed models with relationships and cascade rules. |
| **Validation** | Model creation and relationships pass in test_db.py. |
| **Status** | `DONE` |

### T-G3-DB-03: Pydantic Validation & FHIR Schemas
| Field | Value |
|---|---|
| **ID** | T-G3-DB-03 |
| **Title** | Create Pydantic v2 request/response and FHIR R4 schemas |
| **Description** | Implement schemas matching Group 1 Flutter models and HL7 FHIR R4 Observation/RiskAssessment. |
| **Dependencies** | T-G3-DB-02 |
| **Expected Output** | AssessmentResultResponse, AssessmentResponse, HistoryEntryResponse, FHIRObservation. |
| **Validation** | Schema serialization and AliasChoices verified in test_api.py. |
| **Status** | `DONE` |

### T-G3-DB-04: Database CRUD Repository
| Field | Value |
|---|---|
| **ID** | T-G3-DB-04 |
| **Title** | Implement database CRUD functions |
| **Description** | Create backend/db/crud.py with patient, screening, result, and audit log operations. |
| **Dependencies** | T-G3-DB-02 |
| **Expected Output** | Complete repository methods with transaction safety. |
| **Validation** | pytest tests/test_db.py passes. |
| **Status** | `DONE` |

---

## Phase 14: Quantum ML Engine

### T-G3-QML-01: Quantum Feature Encoder
| Field | Value |
|---|---|
| **ID** | T-G3-QML-01 |
| **Title** | Implement PCA reduction and Amplitude Encoding state preparation |
| **Description** | Create quantum_ml/quantum_encoder.py for mapping classical features to 2^n amplitudes. |
| **Dependencies** | T-G3-INIT-01 |
| **Expected Output** | QuantumEncoder class with PCA fit, L2 normalization, and circuit initialization. |
| **Validation** | test_quantum_encoder and test_quantum_encoder_pca pass. |
| **Status** | `DONE` |

### T-G3-QML-02: Hardware-Aware VQC Classifier
| Field | Value |
|---|---|
| **ID** | T-G3-QML-02 |
| **Title** | Implement Variational Quantum Classifier on Qiskit Aer |
| **Description** | Create quantum_ml/vqc_classifier.py with Rz(psi)Ry(phi) rotations, CNOT coupling, Pauli-Z measurement, and SPSA training. |
| **Dependencies** | T-G3-QML-01 |
| **Expected Output** | VariationalQuantumClassifier class executing on AerSimulator. |
| **Validation** | test_vqc_classifier_simulation and test_vqc_spsa_training_step pass. |
| **Status** | `DONE` |

### T-G3-QML-03: Zero Noise Extrapolation (ZNE)
| Field | Value |
|---|---|
| **ID** | T-G3-QML-03 |
| **Title** | Implement Zero Noise Extrapolation error mitigation |
| **Description** | Create quantum_ml/zne_mitigation.py with polynomial extrapolation back to lambda=0. |
| **Dependencies** | T-G3-QML-02 |
| **Expected Output** | ZNEMitigation class. |
| **Validation** | test_zne_mitigation passes. |
| **Status** | `DONE` |

### T-G3-QML-04: Probability Calibration & Multimodal Fusion
| Field | Value |
|---|---|
| **ID** | T-G3-QML-04 |
| **Title** | Implement ProbabilityCalibrator and Brier score evaluation |
| **Description** | Create quantum_ml/calibration.py for combining classical and quantum scores and assigning clinical risk labels. |
| **Dependencies** | T-G3-QML-02 |
| **Expected Output** | ProbabilityCalibrator class. |
| **Validation** | test_probability_calibrator passes. |
| **Status** | `DONE` |

---

## Phase 15: FastAPI Core Services & API Gateway

### T-G3-API-01: Authentication & Security Utilities
| Field | Value |
|---|---|
| **ID** | T-G3-API-01 |
| **Title** | Implement JWT authentication and pseudonymous UUID generator |
| **Description** | Create backend/auth/security.py with token creation, verification, and bcrypt hashing. |
| **Dependencies** | T-G3-INIT-02 |
| **Expected Output** | Security helper functions with no secrets committed. |
| **Validation** | Unit tested and integrated with endpoints. |
| **Status** | `DONE` |

### T-G3-API-02: Classical ML Service Bridge
| Field | Value |
|---|---|
| **ID** | T-G3-API-02 |
| **Title** | Implement Group 2 Classical ML client and fallback |
| **Description** | Create backend/services/classical_ml_service.py for 512D feature extraction. |
| **Dependencies** | T-G3-INIT-01 |
| **Expected Output** | ClassicalMLService class. |
| **Validation** | Feature extraction returns 512D vector and baseline probability. |
| **Status** | `DONE` |

### T-G3-API-03: Unified Screening Orchestration Service
| Field | Value |
|---|---|
| **ID** | T-G3-API-03 |
| **Title** | Implement ScreeningService linking Classical ML, Qiskit VQC, DB, and FHIR |
| **Description** | Create backend/services/screening_service.py orchestrating end-to-end analysis and FHIR generation. |
| **Dependencies** | T-G3-DB-04, T-G3-QML-02, T-G3-QML-04, T-G3-API-02 |
| **Expected Output** | ScreeningService class. |
| **Validation** | pytest tests/test_e2e.py passes. |
| **Status** | `DONE` |

### T-G3-API-04: API Routes & Application Entrypoint
| Field | Value |
|---|---|
| **ID** | T-G3-API-04 |
| **Title** | Implement FastAPI application and screening endpoints |
| **Description** | Create backend/routes/screening_routes.py and backend/main.py with CORS, health check, upload, analyze, results, history, and fhir endpoints. |
| **Dependencies** | T-G3-API-03 |
| **Expected Output** | Operational FastAPI server. |
| **Validation** | pytest tests/test_api.py passes (3/3). |
| **Status** | `DONE` |

---

## Phase 16: Deployment & Integration Testing

### T-G3-TEST-01: Comprehensive Pytest Test Suite
| Field | Value |
|---|---|
| **ID** | T-G3-TEST-01 |
| **Title** | Implement test_db, test_quantum, test_api, and test_e2e test suites |
| **Description** | Write unit and integration tests across database, quantum ML, API endpoints, and multi-screening user workflows. |
| **Dependencies** | All Phase 12-15 tasks |
| **Expected Output** | 13 passing unit and integration tests. |
| **Validation** | pytest -v passes (13/13). |
| **Status** | `DONE` |

### T-G3-OPS-01: Multi-Container Docker Deployment Stack
| Field | Value |
|---|---|
| **ID** | T-G3-OPS-01 |
| **Title** | Create Dockerfile and docker-compose.yml |
| **Description** | Package backend service and PostgreSQL 15 container with schema initialization and healthcheck. |
| **Dependencies** | T-G3-API-04 |
| **Expected Output** | Dockerfile and docker-compose.yml. |
| **Validation** | Docker compose configuration validated. |
| **Status** | `DONE` |

---

## Phase 17: Frontend-Backend Integration

### T-INT-01: API Configuration & Multi-Environment Constants
| Field | Value |
|---|---|
| **ID** | T-INT-01 |
| **Title** | Implement ApiConstants supporting Android emulator, iOS simulator, and Physical iPhone |
| **Description** | Create carescan/lib/core/constants/api_constants.dart resolving base URLs dynamically (`10.0.2.2:8000`, `localhost:8000`, or custom LAN IP). |
| **Dependencies** | T-G3-API-04 |
| **Expected Output** | ApiConstants class with baseUrl, endpoints, and timeouts. |
| **Validation** | Verified across all three target environment configurations. |
| **Status** | `DONE` |

### T-INT-02: Remote Assessment Data Source
| Field | Value |
|---|---|
| **ID** | T-INT-02 |
| **Title** | Implement AssessmentRemoteDataSource connecting to FastAPI |
| **Description** | Create carescan/lib/data/datasources/assessment_remote_data_source.dart handling POST /api/screening/analyze and GET /api/patients/{id}/history with NetworkFailure and ServerFailure translation. |
| **Dependencies** | T-INT-01 |
| **Expected Output** | AssessmentRemoteDataSourceImpl using dart:io HttpClient with zero extra packages. |
| **Validation** | Typed model serialization and error mapping verified against FastAPI schemas. |
| **Status** | `DONE` |

### T-INT-03: Production API Assessment Repository
| Field | Value |
|---|---|
| **ID** | T-INT-03 |
| **Title** | Implement ApiAssessmentRepository connecting Remote DataSource |
| **Description** | Create carescan/lib/data/repositories/api_assessment_repository.dart implementing AssessmentRepository, mapping Remote DataSource calls to Success/Error results with Failure error handling. |
| **Dependencies** | T-INT-02 |
| **Expected Output** | ApiAssessmentRepository class implementing AssessmentRepository. |
| **Validation** | Unit tests in carescan/test/data/repositories/api_assessment_repository_test.dart pass. |
| **Status** | `DONE` |

### T-INT-04: Unit Testing for Remote Repository & Data Source
| Field | Value |
|---|---|
| **ID** | T-INT-04 |
| **Title** | Implement unit tests for AssessmentRemoteDataSource and ApiAssessmentRepository |
| **Description** | Write isolated unit tests for HTTP response parsing, non-2xx error handling, timeout handling, and Result mapping without real network requests. |
| **Dependencies** | T-INT-02, T-INT-03 |
| **Expected Output** | Unit test suites in carescan/test/data/datasources/ and carescan/test/data/repositories/. |
| **Validation** | All unit tests pass with mock HTTP client. |
| **Status** | `DONE` |

### T-INT-05: Final Integration Verification & Contract Testing
| Field | Value |
|---|---|
| **ID** | T-INT-05 |
| **Title** | Verify complete Flutter ↔ FastAPI contract, secrets, and test suites |
| **Description** | Validate zero hardcoded secrets, verify .gitignore exclusions, verify Flutter/FastAPI JSON contract alignment, and execute complete backend pytest suite. |
| **Dependencies** | All previous tasks |
| **Expected Output** | Clean integration pass with zero secrets and 100% test coverage. |
| **Validation** | Backend pytest suite passes (13/13). No secrets in repo. |
| **Status** | `DONE` |

---

## Task Summary

| Phase | Tasks | Status |
|---|---|---|
| 0 - Environment | T-ENV-01 | DONE |
| 1 - Init | T-INIT-01, T-INIT-02 | DONE |
| 2 - Architecture | T-ARCH-01, T-ARCH-02 | DONE |
| 3 - Design System | T-DESIGN-01 through T-DESIGN-05 | DONE |
| 4 - Navigation | T-NAV-01, T-NAV-02 | DONE |
| 5 - Screens | T-SCREEN-01 through T-SCREEN-07 | DONE |
| 6 - Data & Integration | T-DATA-01 through T-DATA-04 | DONE |
| 7 - Testing | T-TEST-01 through T-TEST-04 | DONE |
| 8 - Accessibility | T-A11Y-01 | DONE |
| 9 - Performance | T-PERF-01 | DONE |
| 10 - Security | T-SEC-01 | DONE |
| 11 - Final Review | T-REVIEW-01, T-REVIEW-02 | DONE |
| 12 - Group 3 Environment | T-G3-INIT-01, T-G3-INIT-02 | DONE |
| 13 - Group 3 Database & Models | T-G3-DB-01 through T-G3-DB-04 | DONE |
| 14 - Group 3 Quantum ML Engine | T-G3-QML-01 through T-G3-QML-04 | DONE |
| 15 - Group 3 FastAPI Services | T-G3-API-01 through T-G3-API-04 | DONE |
| 16 - Group 3 Deployment & E2E | T-G3-TEST-01, T-G3-OPS-01 | DONE |
| 17 - Integration Layer | T-INT-01 through T-INT-05 | DONE |
| **Total** | **51 tasks** | **51 DONE / 0 REMAINING (100% COMPLETE)** |






---

## E2 - Quantum Visual Demonstrator (DEC-035)

Live ledger for the E2 track (Candidate B: fixed 8-qubit angle-encoded quantum feature map -> 16 local-Z observables -> classical head, as an EXPERIMENTAL SECONDARY signal). Classical primary path is untouched. TRAIN + VALIDATION only; test partition frozen. Status updated as work proceeds.

| ID | Task | Status |
|---|---|---|
| T-E2-DOC-01 | Commit DEC-035 to DECISIONS.md; open E2 task ledger | DONE |
| T-E2-CIRC-01 | `quantum_ml/visual_circuit.py`: fixed 8-qubit Ry angle-encoding circuit, NN CZ ring, <=2 re-uploading blocks, 0 trainable gates; frozen 16-observable ObservableSet (8 <Z_i> + 8 NN <Z_iZ_j>) | DONE |
| T-E2-CIRC-02 | Exact Aer statevector executor -> `|amp|^2 @ diag^T` 16-feature vector + connected correlations; validate vs Aer `save_expectation_value` <=1e-10 (measured max-dev 8.9e-16 primary / 1.1e-15 secondary) | DONE |
| T-E2-PREP-01 | `backend/ml/quantum_visual.py`: TRAIN-only PCA-576->8 + deterministic angle scaler ->[0,pi/2] (DEC-036), persisted with provenance/seed/artifact hash | DONE |
| T-E2-ANGLE-01 | `backend/evaluation/e2_angle_experiment.py`: TRAIN-only nested grouped-CV angle-range sweep; diagnosed cos(2 theta) fold at [0,pi], selected [0,pi/2] (DEC-036); validation PR-AUC 0.505 -> 0.879 | DONE |
| T-E2-CTRL-01 | Matched controls: RFF-16 (same dim, same PCA-8 input, same head) + raw-PCA-8 ablation | DONE |
| T-E2-DRV-01 | `backend/evaluation/e2_quantum_visual.py` driver: TRAIN fit / VALIDATION eval, PR-AUC + ROC-AUC/Brier/ECE, patient-blocked null >=200, variance/rank/degeneracy, KEEP/KILL, `test_partition_used: false` (both conditions KEEP) | DONE |
| T-E2-TEST-01 | pytest: circuit/encoding/readout/pipeline/leakage/controls/evaluation/backend + source-inspection firewall test forbidding test reads | DONE -- `tests/test_e2_quantum_visual.py`, 68 tests green in 2.9s. Firewall (no `.test`, iterates only ("train","validation"), oracle-only conditions, `test_partition_used: False`) on BOTH driver and angle experiment; angle experiment additionally asserted to read no `*_validation` field. Circuit topology/gate-count/0-trainable/Pauli-label; `validate_against_aer` <=1e-10; 2-block connected-corr >1e-2 vs 1-block <1e-9 (genuine-quantum-content property); preprocessor [0,pi/2]+clamp+round-trip, PCA==sklearn; RFF bounded/deterministic; `LogisticHead.from_readout_dict` reproduces sklearn readout to 1e-12 incl. dropped degenerate column; controls invariant {quantum,rff,pca} at 16/16/8; emitted-payload (skip-if-absent) test_partition_used False / overlap 0 / aer passed / portable-head tiny / entanglement>0 / label-null quantum-only / no advantage claim. Full suite still collects (507) clean. |
| T-E2-PERF-01 | Runtime: cold + warm + sustained per-image quantum-stage timing; K1 <=500 ms/image (measured cold 2.37ms / sustained 1.47ms -- PASS) | DONE |
| T-E2-INT-01 | `QuantumVisualSummary` additive schema + inference wiring; validate dim/finite/ranges; classical primary survives quantum failure; primary_*/final_probability/threshold/risk_level/FHIR unchanged | TODO |
| T-E2-PC-01 | Point-cloud: inspect for a genuine model; integrate via typed interface if real, else document absence (do not fabricate) -- INVESTIGATED: no genuine point-cloud model, dataset, artifact, or library exists anywhere in the repo (only this ledger row references the term); absence documented, nothing fabricated | DONE |
| T-E2-UI-01 | Flutter "Quantum Visual Analysis" experimental card (EXPERIMENTAL badge, qubits/depth/#features/exec time/backend, feature-vector viz, connected-correlation telemetry, matched-control metric); primary card unchanged | TODO |
| T-E2-RPT-01 | `docs/PHASE_E2_FINAL_REPORT.md` sections A-X + STATUS block (WORKING/FAILED per component, PASS/FAIL runtime & null, KEEP/KILL overall) | TODO |

## E3 - Quantum-Advantage Research Push (DEC-037, DEC-038, DEC-039, DEC-040, mission §30)

Live ledger for the E3 track: does a genuine quantum/hybrid configuration add information beyond the **best classical baseline C7** (validation PR-AUC 0.913038)? Additive, versioned, TRAIN+VALIDATION only, test partition frozen. Each rung answers the same question via `_increment_null` with C7 as reference. Descends the ladder until §29 stop condition (A defensible advantage [preferred] or B multiple approaches with no improvement).

| ID | Task | Status |
|---|---|---|
| T-E3-DRV-01 | `backend/evaluation/e3_hybrid_fusion.py` (`v1-e3-hybrid-fusion-1`): reuses E1.1 `build_inputs`/`candidate_matrices`/`_standardise`/`_logistic_arm`/`_cv_select`/`_metrics`/`bootstrap_pr_auc`/`_increment_null` + `grouped_folds`; new fold-honest arm scorer refits PCA/angle/RFF inside each fold-train only; arms C7 / quantum-16 / RFF-16 / PCA-8; C7->candidate increment tests; primary condition only; `test_partition_used: false` | DONE |
| T-E3-ANCHOR-01 | Correctness gates: rebuilt C7 reproduces **0.913038 to 1e-6** (assembly byte-identical to E1); quantum map vs Aer `save_expectation_value` **8.9e-16 <= 1e-10** before any quantum feature used | DONE |
| T-E3-RUN-01 | Full run (seed 42, 2000 bootstrap, 200 perms) -> `reports_e3_hybrid_fusion.json`. **Result (null, §29B):** standalone val PR-AUC C7 0.913038 > quantum 0.868335 > PCA-8 0.838846 > RFF-16 0.795360 (same order on train-CV OOF); C7->quantum increment +0.0171, column-perm null p=0.0945, does NOT survive p95; RFF/PCA increments equal-or-larger and equal-or-more-significant -> gain is generic, not quantum-specific. Quantum is best of the matched low-dim compressors but does not beat/complement C7 | DONE |
| T-E3-DOC-01 | `docs/PHASE_E3_QUANTUM_ADVANTAGE_RESEARCH.md` leaderboard + increment tests + honest conclusion + research ladder; DEC-037 in DECISIONS.md | DONE |
| T-E3-TEST-01 | pytest for Round 1: C7 anchor reproduction, Aer gate, test-partition firewall (driver reads only train/validation), fold-honest no-holdout-leak property (spy factory), arm/increment shape contract, matched-controls-present, verdict-consistency | DONE -- `tests/test_e3_hybrid_fusion.py`, 19 tests green |
| T-E3-QK-01 | **Round 2 (DEC-038):** quantum-kernel / QSVM -- fidelity kernel `K(x,x')=\|<psi(x)\|psi(x')>\|^2` over the exact 8-qubit states -> `SVC(kernel="precomputed")`, vs C7 and vs matched classical RBF-SVM, same folds, `_increment_null` gate, concentration witness (off-diag Gram sd). `backend/evaluation/e3_quantum_kernel.py` (`v1-e3-quantum-kernel-1`). **Result (null, §29B):** standalone val PR-AUC C7 0.913038 > rbf_svm 0.803026 > quantum_kernel 0.772148; kernel NOT concentrated (off-diag sd 0.185, range 4.8e-9->0.969) so failure is geometric not degenerate; C7->quantum increment **-0.000663 at p=0.592** (dead centre of null); quantum kernel **loses to its own RBF control** (0.772 < 0.803) -> no kernel-level quantum advantage. All advantage flags false | DONE |
| T-E3-QK-TEST-01 | pytest for Round 2: version/frozen-wiring, fidelity-kernel math (symmetric / unit-diagonal / bounded [0,1] / orthogonal->0 / identical->1), concentration-witness flags concentrated-vs-spread, fold-honest no-holdout-leak spy (per-fold `fit` sees only fold-train rows), aligned columns at Hilbert dim 256, RBF control at PCA-8 + gamma>0, emitted-payload (anchor 0.913038 / test unused / aer passed / 3 arms / witness present / increments / verdict consistency / leaderboard sorted) | DONE -- `tests/test_e3_quantum_kernel.py`, 21 tests green |
| T-E3-RUN-02 | Full Round 2 run (seed 42, 2000 bootstrap, 200 perms) -> `reports_e3_quantum_kernel.json`. C7 anchor reproduced 0.913038; Aer 8.9e-16; quantum_kernel val PR-AUC 0.772148 (OOF 0.692485, C 1.0, 170 SVs); rbf_svm 0.803026 (OOF 0.691096, gamma 0.4917); increment -0.000663 p=0.592 does NOT survive; rbf increment +0.013156 p=0.129 does NOT survive; `defensible_quantum_advantage_over_c7=false`, `quantum_specific_advantage=false`, `quantum_kernel_concentrated=false` | DONE |
| T-E3-VQC-01 | **Round 3 (DEC-039):** trainable shallow VQC (`sqrt(T/N)`-bounded, local-observable barren-plateau guard), evaluated identically vs C7, a capacity-matched classical MLP, and a no-entangler ablation. `backend/evaluation/e3_trainable_vqc.py` (`v1-e3-trainable-vqc-1`): fold-honest MobileNet->PCA-8->angle encoding + L in {1,2} trainable {RY;CZ-ring} layers -> 8 local `<Z_i>` -> logistic head; exact torch float64 autodiff (E2 RY+CZ are real, so no parameter-shift / shot noise); depth selected on TRAIN OOF. **Result (null, §29B):** val PR-AUC C7 0.913038 >> no-ent 0.778527 > vqc 0.770660 > mlp 0.768450; depth 1 selected (CZ-after-last-RY invisible to diagonal `<Z>`, so entangler==ablation exactly 0.718104); at depth 2 entangler 0.713016 < ablation 0.723102 -> `trainable_vqc_beats_no_entangler=false`; C7->vqc increment **-0.049934 at p=0.9552 (z -2.15)** does NOT survive; barren-plateau witness healthy (var 0.0137, not vanishing) so the null is representational, not a training failure. All advantage flags false | DONE |
| T-E3-VQC-TEST-01 | pytest for Round 3: real-arithmetic ansatz vs independent qiskit `Statevector.evolve` honesty gate (<=1e-10), depth-1 CZ-invisibility property (entangler==no-entangler at L=1), barren-plateau witness health + determinism, training dynamics (loss decreases), capacity-matched-MLP + no-entangler-ablation formulas, fold-honest no-holdout-leak spy, emitted-payload contract (anchor 0.913038 / test unused / aer+ansatz gates passed / 4 arms / witness present / increments / verdict consistency / leaderboard sorted) | DONE -- `tests/test_e3_trainable_vqc.py`, 32 tests green |
| T-E3-RUN-03 | Full Round 3 run (seed 42, 2000 bootstrap, 200 perms, 400 steps, lr 0.05, wd 0.001) -> `reports_e3_trainable_vqc.json`. C7 anchor reproduced 0.913038 (train-CV OOF 0.766915, fold-honest via `_score_arm` identity path -- byte-identical to Rounds 1/2); Aer 8.9e-16, ansatz-vs-qiskit 4.3e-16 (both <=1e-10, 24 checks each); trainable_vqc val 0.770660 (OOF 0.718104, depth 1, 17 params); trainable_vqc_no_entangler val 0.778527 (OOF 0.723102, depth 2, 25 params); classical_mlp val 0.768450 (OOF 0.675469, 21 params); vqc increment -0.049934 p=0.9552 does NOT survive; mlp increment -0.000212 p=0.5572 does NOT survive; `defensible_quantum_advantage_over_c7=false`, `quantum_specific_advantage=false`, `trainable_vqc_barren_plateau=false`, `trainable_vqc_beats_no_entangler=false`. Third independent §29B data point | DONE |
| T-E3-MAP-01 | **Round 4 (DEC-040):** richer feature map / structured observables / label-aware feature selection -- the one qualitatively distinct untested representation strategy after R1/R2/R3 all nulled. `backend/evaluation/e3_feature_map.py` (`v1-e3-feature-map-1`): fold-honest MobileNet->{PCA-8 \| MI-top-8} selection -> exact **Havlicek ZZ feature map** (reps in {1,2}, entanglement=full, hand-built from native h/p/cx and asserted bit-identical to qiskit `ZZFeatureMap`) -> **36 structured observables** (8 singles + 28 pairs) -> logistic head; config selected on TRAIN OOF; vs C7 and a matched **RFF-36** control; entanglement witness (connected correlations). **Result (null, §29B):** val PR-AUC C7 0.913038 >> rff36 0.726207 > rich_quantum 0.390688 (=base rate); the quantum map is **worse than its own matched classical control**; reps-1 ZZ read by Z-strings is **provably zero** (state is `U\|+>^n`, `U` Z-diagonal -> `<Z_S>`=0, verified 1.4e-16) so both reps-1 configs score the class base rate; reps-2 genuinely entangles (witness mean\|cc\| 0.052, max 0.689) yet **anti-generalizes** (val ROC 0.390 < 0.5); C7->quantum increment **-0.003502 at p=0.716** does NOT survive (rff36 +0.007149 p=0.264 also does NOT survive). All advantage flags false. **Fourth independent §29B data point -> stop-condition (B) exhaustive** | DONE |
| T-E3-MAP-TEST-01 | pytest for Round 4: native-gate-set (h/p/cx only), hand-built-vs-canonical-`ZZFeatureMap` fidelity gate (\|<lib\|hb>\|=1 to 1e-10), Aer readout gate (`\|psi\|^2@diag` vs `save_expectation_value` <=1e-10) + fail-closed corruption check, the reps-1-zero-readout property, entanglement-witness zero-on-product-state (features=pi -> `<Z>`=0) and nonzero-on-generic, MI selector label-awareness, fold-honest no-holdout-leak spy across **both** selectors (per-fold selector sees only fold-train rows), selected-config-is-best-on-TRAIN-OOF, matched-control-at-equal-width (DEC-033), verdict-consistency + no-advantage-without-survival fabrication guard, emitted-payload contract | DONE -- `tests/test_e3_feature_map.py`, 38 tests green |
| T-E3-RUN-04 | Full Round 4 run (seed 42, 2000 bootstrap, 200 perms) -> `reports_e3_feature_map.json`. C7 anchor reproduced 0.913038 (train-CV OOF 0.766915, fold-honest via `_score_arm` identity path -- byte-identical to Rounds 1/2/3); Aer readout gate 7.2e-16, canonical-ZZ gate 1.8e-15 (both <=1e-10); rich_quantum val 0.390688 (OOF 0.506997, PCA-8 + reps-2, C 0.001); rff36 val 0.726207 (OOF 0.693370, C 0.01); config grid pca/1 0.479070 pca/2 0.506997 mi/1 0.479070 mi/2 0.503239 (reps-1 = base rate 0.479070); witness mean\|cc\| 0.052049 -> `rich_quantum_entangling=true`; increment rich_quantum -0.003502 p=0.7164, rff36 +0.007149 p=0.2637, neither survives; `defensible_quantum_advantage_over_c7=false`, `quantum_specific_advantage=false`, `rich_quantum_beats_c7=false`. Fourth independent §29B data point | DONE |
| **E3 STATUS** | **§29B stop-condition (B) reached exhaustively.** Four rungs -- R1 fixed map read linearly (DEC-037), R2 induced fidelity kernel (DEC-038), R3 trainable VQC (DEC-039), R4 richer entangling map + structured observables + label-aware selection (DEC-040) -- span every qualitatively distinct quantum representation strategy the mission named; none beats or adds defensible information beyond C7 (0.913038), and R4 is worse than its matched classical control. No genuine quantum-enhanced configuration found on this task at this data scale. Any future rung requires a *materially new lever* (more data / different modality / justified hardware); no advantage is fabricated to keep the search alive (§29A) | DONE |

## E4 - Data Expansion for Quantum-Advantage Research (DEC-041, mission "Data Expansion")

Live ledger for the E4 track. The §29B stop was reached because SMART-OM may be a **data-scale / representation** bottleneck (215 TRAIN / 103 positives; megapixel images crushed to 8 dims). E4 acquires a **richer, legitimate** biomedical dataset as a better arena to test for a genuine quantum contribution -- selected on scientific suitability, **not** on any preliminary metric. Selected arena: **PTB-XL** (open CC BY 4.0 12-lead ECG, 21,799 records / 18,869 patients, naturally low-dim after feature extraction). Additive: SMART-OM, C7, E2, E3 records and the shipped pipeline are untouched. **Order is enforced: dataset audit -> classical failure map -> only then quantum.**

| ID | Task | Status |
|---|---|---|
| T-E4-DATA-01 | FIRST DELIVERABLE: pre-registered selection criteria (fixed before any modeling), top-5 candidate comparison table (PTB-XL / ISIC-2020 / NIH ChestX-ray14 / BreakHis / MedMNIST) with sources+licences+patient/sample counts, ranking, recommendation, and why-better-than-SMART-OM -> `docs/PHASE_E4_DATA_EXPANSION.md`. Provenance labelled honestly (PTB-XL/ISIC/BreakHis primary-verified this session; NIH/PCam/MedMNIST/tabular knowledge-pending). No metric shopping | DONE |
| T-E4-DL-01 | Reproducible PTB-XL v1.0.3 metadata acquisition into gitignored `backend/artifacts/dataset/ptbxl/` (ptbxl_database.csv + scp_statements.csv + SHA256SUMS.txt + VERSION.txt); resume-capable curl over slow link. `ptbxl_database.csv` SHA-256 **verified OK** vs PhysioNet SHA256SUMS.txt (`7600de9c…6859d216b`). **Signal files (records100/records500) not yet downloaded** -- metadata audit does not require them | DONE (metadata) |
| T-E4-IDX-01 | `backend/dataset/ptbxl.py` -- reproducible acquisition/audit module mirroring `smartom.py`/`split.py`: provenance constants (source URLs/version/licence), `EcgRecord`/`PtbxlIndex` (by_patient/by_superclass/by_partition), stringified-`scp_codes` parser, SCP->superclass mapping (skips non-diagnostic codes), `discover_ptbxl_root` (CARESCAN_PTBXL_ROOT env), `verify_checksums` (OK/mismatch/unknown), `assert_no_patient_leakage` (mirrors split.py: re-derives patient->partition from fold assignments, refuses spanning), `audit()` -> serialisable `PtbxlAuditReport`, CLI `--audit` | DONE |
| T-E4-AUDIT-01 | Formal metadata audit (`python -m backend.dataset.ptbxl`). **Verified:** version 1.0.3; checksum OK; **21,799 records / 18,869 patients** (match published); max 10 studies/patient; 0 missing age/sex; **411 unlabeled** (no diagnostic superclass); superclasses NORM 9,514 / MI 5,469 / STTC 5,235 / CD 4,898 / HYP 2,649; strat_fold partitions train 17,418(15,023 pt)/val 2,183(1,942 pt)/test 2,198(1,904 pt); **`patient_leakage_free: True`** (no patient spans folds -> frozen test safe). Report -> gitignored `audit_report.json` | DONE |
| T-E4-TEST-01 | pytest for the acquisition/audit module: stringified-dict parsing + garbage tolerance, non-diagnostic-code exclusion, index/superclass aggregation + float patient_id, fold->partition mapping, grouping helpers, **patient-level leakage guard (accepts disjoint, raises on span)**, checksum OK/mismatch/unknown, audit counts on synthetic metadata, absent-metadata error, and a **real-metadata regression lock** (skips if gitignored data absent) asserting 21,799/18,869/leakage-free/checksum-OK/NORM 9,514 | DONE -- `tests/test_ptbxl_dataset.py`, 12 tests green; full suite 650 green |
| T-E4-SIG-01 | Download PTB-XL signal files (records100 @100Hz) with checksum verification; record actual vs expected file counts. `backend/dataset/_dl_ptbxl_sample.sh` -- CRLF-safe, idempotent, 3 retries/file, empty-stub cleanup, explicit tally (an earlier attempt failed all 398 fetches on a trailing `\r` from the CRLF record list *and still exited 0*). **Result: 200/200 records, 400/400 files, 4,920,686 B, and 400/400 SHA-256 OK vs PhysioNet `SHA256SUMS.txt` (0 mismatch, 0 not-in-manifest).** Sample folds: train 127 rec/124 pt · val 42/34 · test 31/27, patient overlap train∩(val∪test) = 0 | DONE (bounded 200-record sample; full ~1.7 GB `records100` fetch is T-E4-BULK-01) |
| T-E4-WFDB-01 | `backend/dataset/wfdb_reader.py` -- dependency-free WFDB reader (~340 lines, numpy only), since `wfdb`/`neurokit2`/`pywt` are not installable here. Handles PTB-XL's single fully-specified variant (one `.dat`, format 16, 1 sample/frame, no skew, no byte offset); **deliberately strict** -- any unimplemented header feature raises `WfdbError` rather than being silently mis-read. Verifies the **WFDB per-signal checksum** from the header (integrity of the *decoded samples*, orthogonal to SHA-256 on the *file bytes*). Validated on real records: `(1000, 12)` @100 Hz, 12 standard leads, `checksums_ok: True` | DONE |
| T-E4-FEAT-01 | `backend/ml/features_ecg.py` (`FEATURE_SET_VERSION = "ecg-v1"`) -- **97 named features**: rhythm/HRV 9 · global morphology 4 (QRS/QT/QTc/JT) · per-lead 72 (12 leads x Q/R/S amp, ST60, ST slope, T amp) · frontal axis 2 · spectral 5 · acquisition quality 5. Conditioning 0.5-40 Hz -> Pan-Tompkins detection on the **across-lead RMS composite** -> **median** whole-beat template -> envelope delineation. **Fold honesty is structural:** every function is record-local, no statistic pooled, unmeasurable features emitted as `NaN`; all fitted steps deliberately excluded. **TRAIN-fold audit (n=127, 124 pt): 127/127 extracted, 0 failed, 0 flagged, 3 ms/record, 0 non-finite of 12,319 cells**; physiology vs published ranges -- HR med 68.89 (99.2% in [40,140]), QRS med 80 ms (88.2% in [70,120]), QT med 370 ms (95.3%), QTc med 397.42 ms (89.8%), QRS axis med 42.24° (95.3%), `ii_st60` / `v2_t_amp` / `rr_regularity` 100%. **Bug fixed:** per-side search bounds admitted 2x the configured QRS cap (p95 = 390 ms) -> explicit total-duration cap, post-fix max 200 ms | DONE (extractor; TRAIN-only fitting stage is T-E4-FIT-01) |
| T-E4-CAL-01 | QRS onset/offset threshold calibration under a rule **fixed before evaluation**: anchor the crossing level to the PR-segment noise floor (a bare fraction-of-peak truncates the low-amplitude slurs, each costing a full 10 ms bin at 100 Hz), then take *the largest fraction whose median QRS falls in the normal adult window 80-100 ms while clamping <5% at the 200 ms cap*. Selected **0.08** -> median QRS 80 ms, 1.6% clamped. Evaluated on **TRAIN folds only**, against **published physiology**, **no classification metric consulted** -- instrument calibration, not model selection | DONE (DEC-042) |
| T-E4-SIGTEST-01 | pytest for both new modules: `tests/test_wfdb_reader.py` (18) -- synthetic byte-for-byte records, physical-unit round-trip, baseline/zero-gain handling, checksum verify **and** mismatch, strict rejection of format≠16 / multi-sample-per-frame / skew / byte-offset / multi-file / truncation, comment tolerance, lead lookup; `tests/test_features_ecg.py` (35) -- synthetic beat train with known rate & lead geometry (beat count + HR at 50/60/75/100 bpm, analytic frontal-axis ratios, median-template outlier rejection, fiducial ordering, **parametrized regression lock on the 2x-duration-cap bug**, 97-dim layout stability, missing-lead NaN layout, NaN-not-filled, determinism, record-locality), plus **TRAIN-fold-only** real-data regression locks (`00001_lr`, confirmed `strat_fold` 3) | DONE -- full suite **703 green** (was 650 + 53) |
| T-E4-DISC-01 | Discipline correction: the sample was drawn as the *first* 200 stems and contains **42 validation + 31 frozen-test records**. All development/validation restricted to the **127 TRAIN-fold records**; the 73 held-out records downloaded but **deliberately not read**; real-data tests touch only a TRAIN-fold record. Future sample lists must be fold-filtered at construction | DONE (recorded in DEC-042 + docs §7.2) |
| T-E4-BULK-01 | Bulk **fold-filtered** `records100` download (folds 1-9 only -- fold 10 is deliberately never fetched) in resumable background batches. `backend/dataset/download_ptbxl_signals.sh`: 8 parallel curl jobs, 600-file batches, per-batch tally recomputed **from the filesystem** rather than from curl's exit code, `cygpath -m` for curl's output paths (native curl cannot write to `/c/Users/...`), one `find -printf` + `awk` pass for the missing-file scan (a per-file `stat` loop took ~20 min; this takes 0.066 s), and a **DNS circuit breaker** (abort after 2 consecutive all-dead batches). **Result: 19,601/19,601 records, 39,202/39,202 files, 0 missing, 482,260,870 B (459.9 MiB), exit 0.** The earlier "~1.7 GB" estimate was wrong -- it is the size of the *whole* PTB-XL release including `records500`; the folds-1-9 100 Hz corpus is 460 MiB | DONE |
| T-E4-INTEG-01 | Integrity pass over the bulk corpus -- "presence is not integrity". `python -m backend.dataset.ptbxl --verify-signals dev_records.txt --delete-corrupt`: SHA-256 every file against PhysioNet's published `SHA256SUMS.txt`, classify OK/mismatch/unknown, delete the corrupt so a re-fetch is idempotent, write a JSON report. **First pass found `fully_verified: False` -- 2,267 of 39,202 files had no published checksum.** Diagnosis: **our local `SHA256SUMS.txt` was itself a truncated download** (3,915,407 of 8,284,204 bytes, ending mid-hash with no newline, 0 `records500` entries, stopping at `records100/20000/20643_lr.dat`). Re-fetched the manifest (87,203 lines = 43,598 `records100` + 43,598 `records500` + metadata), proved the old copy was a strict byte-**prefix** with `cmp -n` (pure truncation, not a version difference, so the 36,935 checks already passed were against genuine hashes), kept the partial as `SHA256SUMS.txt.truncated-partial`, re-verified. **Final: 19,601 stems complete / 0 incomplete, 39,202 files hashed, 39,202 SHA-256 OK, 0 MISMATCH, 0 unknown, `fully_verified: True`.** Manifest SHA-256 `b7224b92…f51dc695d` | DONE |
| T-E4-PREREG-01 | **Pre-registration of the task and evaluation protocol, written before a single model of any kind was fitted on PTB-XL** (`docs/PHASE_E4_DATA_EXPANSION.md` §7.4). Fixes: primary task `NORM` vs abnormal (positive = any of MI/STTC/CD/HYP; negative = NORM and none of the four; **no diagnostic superclass = label *unknown*, never a negative**; NORM+abnormal ⇒ positive, the screening-conservative direction); secondary MI vs NORM, reported but **never used for selection**; **primary metric ROC-AUC** (PTB-XL is near-balanced -- C7's PR-AUC was primary because SMART-OM was not), with PR-AUC / balanced accuracy / sensitivity-at-fixed-specificity / per-superclass as secondary and **non-promotable**; folds 1-8 train / 9 validation / **10 frozen test evaluated exactly once**; bootstrap **over patients** (not records -- patients contribute multiple ECGs, so record resampling would understate uncertainty by pseudo-replication), 2,000 resamples, percentile 95% CI; and the **five fixed conditions** for a quantum contribution (beats the strongest classical baseline; paired bootstrap CI of the difference excludes 0; a matched classical control at the same input dimension does **not** also achieve it; permutation p < 0.01; survives the single frozen-test evaluation). "Failing any of these is a null, and a null reported honestly is an acceptable outcome of this phase" | DONE |
| T-E4-COHORT-01 | Cohort driver `backend/training/prepare_ecg_features.py` (`ECG_CACHE_VERSION = "ptbxl-ecg-1"`) -- runs the record-local extractor over the corpus and caches one `.npz` carrying the matrix **plus every identifier an honest evaluation needs** (`ecg_id`, `patient_id`, `strat_fold`, the §7.4 label, `label_known`, the 5-way superclass matrix, age/sex, beat counts, quality flags). Mirrors `prepare_pixels.py`: metadata travels **inside** the archive (no sidecar JSON can be separated from its provenance), no pickle, row-alignment validated on save, stale `feature_set_version` / changed feature layout **refused on load**. Three load-bearing properties: **nothing is fitted here**; **fold 10 is refused** without `allow_test_fold=True`; **an absent label stays `-1`, never a negative**. A record whose signal is unreadable is named in `failures` and still counted in the denominator (`rows + failures == n_candidates`), never zero-filled or silently dropped. **Full-corpus run: 19,601/19,601 extracted, 0 failed, 16,965 patients, dim 97, non-finite cells 0 / 1,901,297, 29 quality-flagged, 227.8 s (86 rec/s).** Labels: **abnormal 11,073 / normal 8,157 / unknown 371** (57.6% / 42.4% of the 19,230 labeled -- near-balanced, confirming §7.4's choice of ROC-AUC as primary). Independent cross-check: fold 9 = **2,183 rec / 1,942 pt**, exactly the DEC-041 metadata audit's figure, and TRAIN (15,023 pt) + VAL (1,942 pt) = **16,965 = the dev total**, so the partitions are patient-disjoint by arithmetic as well as by the explicit guard | DONE -- `tests/test_prepare_ecg_features.py`, 44 tests |
| T-E4-FIT-01 | TRAIN-only fitting stage as a **separate** module `backend/ml/ecg_transform.py` (`ECG_TRANSFORM_VERSION = "ecg-fit-1"`) -- deliberately **not** in `features_ecg.py`, which must stay unfitted to preserve structural fold honesty. Coverage filter -> TRAIN-median imputation -> TRAIN mean/std standardisation -> reduction (`pca` \| `select` \| `identity`) into the compact 8-32 dim near-term regime. **Refuses to fit on a non-TRAIN fold** unless `allow_non_train=True` ("Fitting a median, a scale or a PCA basis on validation data leaks its statistics into every subsequent evaluation"); the refusal is proved *effective*, not decorative, by a test that shifts every fold-9 row by +500 and shows the fitted medians/means/scales unchanged. Records `train_patient_ids` **inside the saved artifact**, so `assert_no_fit_eval_patient_overlap` turns "the folds are patient-disjoint" from a claim into a check that survives a reload. Supervised selection sees **only labeled TRAIN rows** (tested: unknown-label rows screaming a competing column lose). Plain arrays + JSON, no pickle. Not reusing `DimensionalityReducer`: that targets `2 ** n_qubits` for amplitude encoding and does no imputation; E4's regime is angle-encoded, so 8-32 need not be a power of two. **Full-corpus fit (folds 1-8, 17,418 rec / 15,023 pt, 17,084 labeled): 97 in -> 97 kept -> 16 out, 0 features dropped, 0.7745 variance retained**, both partitions finite, disjointness guard passed. The 2 features that were zero-variance on the 127-record sample **do vary** across the full corpus -- a sample-level finding that did not generalise. TRAIN-only variance spectrum: 8d **0.6003** · 16d 0.7745 · 32d **0.9275** (90% needs 28, 95% needs 37, 99% needs 58). Artifacts `pca08`/`pca16`/`pca32`/`select16` saved for the later train/validation-only dimension choice | DONE -- `tests/test_ecg_transform.py`, 40 tests |
| T-E4-BASE-01 | **Classical failure map on PTB-XL -- tabular rungs + the matched-dimension controls** (`backend/evaluation/e4_classical_baseline.py`, `E4_BASELINE_VERSION = "v1-e4-classical-1"`). Grid frozen in the module before the first fit: 3 models (simple **logistic** / feature-engineered **HistGBM** / kernel **RBF-SVM**) x 4 representations (`f97` identity, `pca32`, `pca16`, `pca08`) x 2 tasks, plus a **prevalence floor** arm. Every hyperparameter, threshold and calibration chosen by **leave-one-`strat_fold`-out CV inside folds 1-8** -- PTB-XL's own folds reused as the inner split, so the inner CV inherits the verified patient-disjointness instead of asserting a new one. `SVC(probability=True)` is deprecated in sklearn 1.9, so the kernel arm is **Platt-scaled from its TRAIN out-of-fold decision values** (monotone -> provably cannot move ROC-AUC; it exists so Brier/log-loss/ECE and the threshold are defined). **Primary task (fold 9, 2,146 rec / 1,917 pt, prevalence 0.5741): ceiling `gbm@f97` ROC-AUC 0.940234 [0.9297, 0.9494]**, then `rbf_svm@f97` 0.933624 · `rbf_svm@pca32` 0.931961 · `rbf_svm@pca16` 0.927408 · `gbm@pca32` 0.922905 · `gbm@pca16` 0.916917 · **`rbf_svm@pca08` 0.915006 [0.9024, 0.9265]** · `gbm@pca08` 0.907049 · `logistic@f97` 0.900270 · `logistic@pca32` 0.895591 · `logistic@pca16` 0.879296 · `logistic@pca08` 0.869805 · **`prevalence` exactly 0.500000** (the score/label alignment check). Secondary MI-vs-NORM reported only, `used_for_selection: false`, ceiling 0.973262. **Permutation null on the winner: 200 patient-blocked draws, model refitted every draw, null mean 0.5079 (sd 0.0236) max 0.5674, 0/200 reached the observed 0.940234, p = 0.004975** -- clears §7.4 condition 4. Honestly flagged and quantified: `within_patient_labels_consistent: False` because **1.90% of TRAIN patients (17.23% of multi-record patients, 4.03% of records) genuinely carry mixed labels**, so blocking shifts the permuted record prevalence 0.5760 -> 0.5525 (sd 0.0018); blocking still preserves *more* structure than record-level permutation, which is the conservative direction. 800.8 s grid + 920.7 s permutation run (867.3 s of it the 200 refits), **`test_partition_used: false`** asserted in both payloads | DONE (tabular; deep rung is T-E4-BASE-02) -- `tests/test_e4_classical_baseline.py`, 58 tests |
| T-E4-SIGCACHE-01 | Raw waveform cache `backend/training/prepare_ecg_signals.py` (`SIGNAL_CACHE_VERSION = "ptbxl-signal-1"`) -- the deep rung's input, decoded once instead of 19,601 WFDB records per epoch. One `(n, 12, 1000)` float32 `.npy` (memory-mappable; ~900 MiB resident otherwise) plus an `.npz` of `ecg_id`/`patient_id`/`strat_fold` + metadata + failures, ordered by `ecg_id` identically to `extract_cohort` so the two caches align without either trusting the other's ordering. Three enforced properties: **fold 10 refused** without an explicit override; **nothing fitted** -- physical millivolts exactly as the header declares them, no mean/scale/filter/resample pooled across records (per-record standardisation happens inside the model's input layer, where it is record-local and therefore fold-honest by construction); **the channel axis is validated, never coerced** -- a record whose lead set is not exactly the 12 standard leads, whose length or rate differs, or which decodes non-finite is named in `failures` with a reason and omitted, never padded or reordered, with `n_rows + n_failures == n_candidates` asserted before writing. Allocated as a memmap and filled in place (a list-of-arrays would peak at ~2x final size), then trimmed to written rows. **Full run: 19,601/19,601 cached, 0 failures, 0 records without a header checksum, `fitted_statistics: "none"`** | DONE |
| T-E4-SIGTEST-02 | pytest for the waveform cache -- the owed suite, organised around the failure modes that would be **silent**: lead permutation (4 parametrized non-standard sets: missing / extra / duplicate / all-duplicates), the frozen fold (refusal + §7.4 citation + default = `DEV_FOLDS` + override recorded in metadata), the array (shape/dtype, `ecg_id` ordering, **agreement with `extract_cohort` on the same corpus**, an explicit rotation check on a record written with shuffled leads, physical-mV decoding against `raw/gain`, **the no-fitted-statistics claim tested as a property** -- the same record cached inside two different corpora must be bit-identical -- and a flatline record cached rather than judged), failure policy (accounting identity, 3 parametrized defect reasons, trim-to-written-rows, empty selection, all-defect corpus, `limit` applied after sorting), `SignalCache` guards (row/id mismatch, unequal identifier lengths, duplicate `ecg_id`, `rows_for` order preservation, `rows_for` raising on a miss rather than dropping it), the loader (missing cache names the build command, version mismatch refused, memmap by default) and the CLI fold parser. **The suite immediately earned its keep: it caught a real defect.** `_lead_permutation` accepted a 13-channel record and silently dropped the extra column while its own docstring promised an extra lead is a failure -- fixed by requiring an exact 12-channel width. No measured number moves (`records100` is uniformly 12-lead and the real cache reports 0 failures) | DONE -- `tests/test_prepare_ecg_signals.py`, 40 tests |
| T-E4-BASE-02 | **Remaining classical rungs: modern deep 1D-CNN on the raw 12-lead signal, then best-reasonable fusion** with the tabular ceiling (`backend/evaluation/e4_deep_baseline.py`, `E4_DEEP_VERSION = "v1-e4-deep-1"`). Completes the four rungs the mission fixed. Conditioning is record-local only (0.5-40 Hz zero-phase band-pass, order 3; **no amplitude normalisation**); architecture chosen by inner ROC-AUC on **TRAIN fold 8** before fold 9 was scored once; the blender is a logistic fit on **TRAIN fold 8 only, with both members out-of-sample on it**. **Fold 9 (2,146 rec / 1,917 pt, prevalence 0.5741): `fusion@cnn+gbm` 0.946293 [0.9364, 0.9550]** (bal.acc 0.8700, sens@sp0.90 0.8409, ECE 0.0190) · **`cnn@resnet_small` 0.940476 [0.9297, 0.9501]** (126,649 params over 12,000 raw samples, ECE 0.0867) · **`gbm@f97` 0.940234** reproduced exactly from DEC-044. **The load-bearing result is a tie that is not a redundancy:** deep-vs-tabular paired delta **+0.000242, CI [-0.006101, +0.006374], spans zero** -- yet the fusion beats *both* with intervals that exclude zero (vs cnn +0.005817 [0.0020, 0.0098]; vs gbm +0.006058 [0.0037, 0.0086]), the blender keeps both members positive (deep 0.286, tabular 0.564), and fusion ECE is 0.019 against 0.040/0.087. Two unrelated function classes score the same in aggregate while disagreeing per record. Architecture selection: `resnet_small` inner 0.946739 (66 s/epoch) · `resnet_long_kernel` 0.946422 (83 s/epoch) · `resnet_wide` 0.945696 at **640 s/epoch** -- 73% of the 14,061 s run for the worst score, and a ~5x miss against the predicted 132 s/epoch, so the width term in the cost model was corrected. `patient_overlap 0`, `ecg_id_overlap 0`, **`test_partition_used: false`**, `test_folds_read: []` | DONE (DEC-046 -- **the quantum gate opens, at a higher bar**) |
| T-E4-PRIOR-01 | Focused prior-art / novelty audit for quantum-ML on ECG. Systematic arXiv API sweep paged to exhaustion: `abs:"PTB-XL"` (103 hits, **102 examined**, spanning **2020-04-28 to 2026-09-15**), `all:"quantum" AND all:"ECG"` (40 returned, ~8 genuinely ECG -- the rest collide with *Einsteinian cubic gravity* / *explicitly correlated Gaussians* / *extended Chaplygin gas*), `abs:"quantum machine learning" AND abs:"classical baselines"` (48 hits, 40 examined). **The honest result is that two of three candidate novelty claims are FALSE.** (1) **"First quantum model on PTB-XL" is false:** exactly one of 102 PTB-XL papers is quantum -- arXiv:2603.27269 (2026-03-28), a 6-qubit VQC **distillation student** of an ECGFounder teacher, and **the classical teacher won**. What is still open is the question that design structurally could not ask, since a distillation student is bounded above by its teacher. (2) **"Novel evaluation methodology" is false:** 2608.18155 already publishes a leakage-controlled, calibration-aware benchmark with a quantum-attribution audit *and* the "stands whether quantum wins, ties, or loses" commitment; 2607.15815 screens against five tuned classical twins; 2605.19233 establishes group-aware splits. E4's stance is downstream of that literature, not ahead of it. (3) **The one surviving claim is the narrow one DEC-046 had already pre-registered:** classical ECG work knows stacking wins (2609.12803 -- five architectures + three ensembling schemes, stacking tops macro AUROC/AUPRC/F1), yet every quantum paper in the sweep compares against a *single* model, parallel *twins*, or a teacher -- **no paper controls a quantum arm against a fused classical baseline of the same shape**. Corroboration for §7.8's pessimistic reading: 2608.14633 finds **data, not model capacity, is the limiting factor** on PTB-XL-pooled data. Hazard logged: 2507.11401 samples **400 entanglement topologies** and reports the 16% that beat baseline -- exactly what §7.4 pre-registration prevents. **Limits stated before findings:** arXiv-only (no PubMed/IEEE/Scopus -- `WebSearch` unavailable here, one `WebFetch` quota-failed), abstract-level only, so every claim is phrased "unattested in this sweep", never "nobody has done this" | DONE (DEC-047, docs §7.9) |
| T-E4-Q-01 | Only after the **complete** classical failure map (BASE-01 **and** BASE-02) and the prior-art audit (PRIOR-01) -- **all three now done**: transfer + re-evaluate the E-series quantum infrastructure (kernels/VQC/feature maps/fusion) on the compact ECG representation, under the same patient-level + matched-control + permutation + bootstrap standard that made the E3 nulls trustworthy. **The bars it must clear are now numbers, not placeholders: 0.946293 overall** (the fusion, DEC-046 -- *not* the superseded tabular 0.940234, which the platform already exceeds), **and at its own input dimension 0.915006 (d=8) / 0.927408 (d=16) / 0.931961 (d=32)**. **Plus the sixth condition DEC-046 pre-registered before any quantum fit:** a quantum arm entering a fusion must be controlled against a **classical** fusion of the same shape, blended the same way on the same inner fold. Beating a single classical model is no longer evidence of anything, because a second classical representation already does that -- and per DEC-047 that control is the phase's only surviving novelty claim, so it is the experiment, not a formality. **MEASURED (`backend/evaluation/e4_quantum.py`, `E4_QUANTUM_VERSION = "v1-e4-quantum-1"`, 7,127.9 s): the answer is a NULL.** **HEADLINE `fusion@cnn+gbm+zz` - `fusion@cnn+gbm+poly2` = +0.000641, CI [-0.000486, +0.001747], spans zero** -> by the rule §7.10 fixed *before* the first circuit was fitted, the quantum map contributed nothing a same-shape classical map did not, **regardless of how either arm scores against the bar**. That rule is load-bearing rather than ceremonial here: `fusion@cnn+gbm+zz` **0.946329** is the single highest number in the experiment, beating the recorded bar (+0.000036 [-0.000070, +0.000143]) and both classical-fusion controls -- "our quantum fusion set a new best" was a writable and indefensible sentence, and the pre-registration removed the choice before there was anything to choose. Arms (fold 9, 2,146 rec / 1,917 pt, patient-clustered CI): `q@zz` (reps 2, C 0.01) **0.760505** [0.7401, 0.7796] · `c@poly2` 0.884961 · `c@rff36` 0.897873 · `fusion@cnn+gbm` 0.946293 · **`fusion@cnn+gbm+zz` 0.946329** · `fusion@cnn+gbm+poly2` 0.945688 · `fusion@cnn+gbm+rff36` 0.945662; **4 grid cells per family, all three**. **The only interval that excludes zero points the wrong way:** `q@zz` vs `c@poly2` single arms **-0.124457 [-0.145524, -0.105109]** -- over the identical index set (8 singletons + 28 pairs, verified column-for-column against `all_pair_masks(8)`), identical head, identical budget, a plain degree-2 polynomial beats the Havlicek encoding decisively; the quantum map is the **worst** of the three maps of its own shape. **The finding with the longest reach: the fusion is saturated** -- adding *any* 36-column map to `cnn+gbm` moves it <±0.0007 with every interval spanning zero (zz +0.000036, poly2 -0.000605, rff36 -0.000631), confirming DEC-046's "residual error is a property of the task, not of model capacity" from a third direction, and retroactively vindicating the classical-fusion control: alone, zz's +0.000036 would have looked like a contribution when the slot is inert for everything. **Not an artifact:** both E3 gates passed at machine precision before any score was kept (readout-vs-Aer 1.80e-16 / 1.11e-15; circuit-vs-qiskit-`ZZFeatureMap` 6.66e-16 / 9.99e-16, tol 1e-10) and the entangling witness gives reps=2 mean \|C_ij\| **0.07320875** (max 0.71480617) -- the entanglement was present and measured and did not help, which is stronger than a product-state null. **Baseline is the one on record:** `recover_members` re-ran DEC-046's exact path (early stop epoch 25, best 17; refit 1-7 then 1-8) and reproduced **bit-identically** -- cnn 0.9404758944556537, gbm 0.9402343416976896, inner 0.946739. Caveat recorded, deliberately not repaired: `q@zz` selected C at the **grid edge** (0.01) -- widening only the losing arm's budget would break the same-shape contract and is metric shopping. **Defect found and fixed before any number was kept (ISS-011):** the degeneracy guard read `_standardiser`'s std, which floors a constant column at 1.0, so it **could never fire**; caught by an assertion printing `GUARD FAILED: reps=1 was accepted`, fixed to read the raw spread, verified firing at 7.03e-17 -- no recorded number affected, since `QUANTUM_REPS_GRID = (2,)` meant no scored cell reached it. `test_partition_used: false`, `patient_overlap 0`, `ecg_id_overlap 0`, `pca_sees_fold_9 false`; fold 10 unread | DONE (DEC-048 -- **a measured, pre-registered NULL**) -- `tests/test_e4_quantum.py`, 51 tests |
| T-PLAT-01 | **Platform spine**: `backend/ml/track.py` -- `DiseaseTrack` as a `runtime_checkable` Protocol (not a base class, so `InferenceService` is adapted from outside and never touched), `Modality`/`RiskBand` shared vocabularies, `InputSpec` publishing channel order, `TrackAssessment` with a per-track `detail` payload, thread-safe `TrackRegistry` refusing silent duplicate-id overwrites and isolating a broken track from the listing. **The contract carries the methodology**: `ValidationSummary.frozen_test_evaluated` has no default and `is_clinically_claimable` is false without it, so a track cannot register while implying validation it does not have. 20 tests | DONE |
| T-PLAT-02 | **Oral-lesion adapter**: `backend/ml/tracks/oral_lesion.py` wraps the existing `InferenceService`. No import runs toward it, `/api/screening/analyze` is byte-identical, and DEC-034 survives translation -- the classical band headlines while `probability_is_calibrated=False` marks its score as a ranking value, and the calibrated quantum probability is preserved separately as the number persisted to `final_probability`/FHIR. `to_assessment()` split out so the mapping is testable without loading artifacts | DONE |
| T-PLAT-03 | **ECG track**: `backend/ml/tracks/ecg.py`. Shipped **before** its model on purpose -- input contract, lead order, provenance and failure vocabulary fixed now; reports `ready=false` naming the missing bundle. Validates shape/finiteness/lead-major-vs-time-major before loading anything (so a bad payload reads 422, not 503), returns `RiskBand.INDETERMINATE` rather than banding an unreadable record LOW, refuses a bundle missing its threshold or built on another `feature_set_version`. `uses_quantum=False` / `quantum_role=None`: a statement about what is deployed | DONE (model pending -- BASE-02 has landed, but no arm is persisted for serving because every number so far was selected on a development partition) |
| T-PLAT-04 | **HTTP surface**: `backend/routes/track_routes.py` mounted at `/api/tracks` alongside the existing routes. `GET /api/tracks` lists **unready** tracks with a reason; separate JSON and multipart analyze endpoints; 404/422/503/413 mapping with tracebacks kept server-side. Wired in `backend/main.py` via idempotent `install_default_tracks()`. 36 tests, including one that fails if the platform routes ever stop being additive; existing `test_api`/`test_e2e`/`test_localize_route` (21) re-run green | DONE |
| T-PLAT-05 | `docs/PLATFORM_ARCHITECTURE.md` -- backend platform architecture (the root `ARCHITECTURE.md` covers only the Flutter client): why the layer is additive rather than a generalisation of `InferenceService`, the seven-stage shared pipeline with a per-track table, the contract, the four honesty constraints carried as data, the HTTP/error surface, **where the quantum component actually is** (oral: served and secondary; ECG: gated behind the measured **0.946293** fusion bar plus DEC-046's classical-fusion control), how to add a third condition, and an implementation-status table | DONE |
| T-PLAT-06 | **Flutter read surface for the track catalogue** -- `carescan/lib/features/tracks/models/screening_track.dart` (Dart mirror of `backend/ml/track.py`, verified field-for-field against the Pydantic models, which carry **no aliases**, so the wire is snake_case unlike the camelCase assessment schemas), `carescan/lib/data/datasources/track_remote_data_source.dart` (`dart:io` `HttpClient` only, house catch ladder verbatim), and additive `ApiConstants.tracksPath` / `trackDetailPath`. **The honesty property is structural on this side of the wire too**: `frozen_test_evaluated` has no default here either -- a descriptor arriving without it raises `ServerFailure` rather than being read as validated or hidden -- and the headline metric is reachable only through `headline`/`provenance`, which carry the partition and the frozen-test status with it, so the qualified caption is the path of least resistance instead of a step a screen must remember. Unknown *values* are tolerated (a fourth `modality` degrades to `unknown` rather than crashing an older client) while missing *required* fields are refused; absent `ready` reads as **not** ready. `TrackCatalogue` records every refusal in `rejected` with its reason and raises when a non-empty response yields zero usable tracks, because a descriptor vanishing silently is indistinguishable from a schema regression and would render as the false statement "no screening tracks configured". 404 maps to `ValidationFailure`, not `ServerFailure` -- "this build asked for a track the backend does not have" must not read as "the backend is down". **Read surface only, deliberately:** the `/api/tracks/{id}/analyze` endpoints are *not* wired, because the shipping capture flow runs through `/api/screening/analyze` and re-routing the one path the product depends on would be a destructive replacement of working code rather than the additive surface DEC-045 describes. The guard was proved **live, not decorative**: defanging it to `as bool? ?? false` behind `if (1 > 2)` turned **5** tests red across three groups. 44 tests | DONE (DEC-049) -- `carescan/test/data/datasources/track_remote_data_source_test.dart`, 44 tests |
| T-PLAT-07 | **Two stale client tests repaired** (ISS-013) -- `test/core/accessibility_test.dart` and `test/features/assessment/assessment_flow_test.dart` asserted a HomeScreen that T-SIH-03/04 had already replaced (`Icons.settings`, `'Take a Photo'`, `'Good morning, Alex'`, `'Start a New Assessment'`) and pumped `MyApp` without a prototype session, so `AppRouter`'s redirect held them at `/welcome`. Proved **pre-existing** before being touched, by reverting the `api_constants.dart` addition and observing identical failures. Repaired to the current intended UI rather than weakened: both enter via `appSession.continueAsGuest()` with `addTearDown(appSession.logout)`, and the navigation test now asserts the shell's own `currentIndex` instead of text, because `StatefulShellRoute.indexedStack` keeps a visited branch size-maintaining rather than offstage -- so text from a previously visited branch still matches and cannot establish *which* tab is selected. Mutating the expected branch index turns it red, so the assertion is live. Pre-existing `unnecessary_import` in `lib/core/image/image_quality_service.dart` also cleared | DONE -- `flutter test` **119 green**, `flutter analyze` **no issues found** |
 **All five rungs are measured. The quantum rung returned a NULL, and it is a pre-registered one.** FIRST DELIVERABLE complete (T-E4-DATA-01); metadata downloaded/checksum-verified/audited with folds patient-disjoint (T-E4-DL-01/IDX-01/AUDIT-01/TEST-01, DEC-041); bounded signal sample, dependency-free WFDB reader, and the fold-honest `ecg-v1` 97-feature extractor validated on TRAIN folds against published physiology (T-E4-SIG-01/WFDB-01/FEAT-01/CAL-01/SIGTEST-01, DEC-042); bulk folds-1-9 corpus acquired and fully integrity-verified, task pre-registered before any model, 19,601 x 97 matrix built, TRAIN-only `ecg-fit-1` transform fitted (T-E4-BULK-01/INTEG-01/PREREG-01/COHORT-01/FIT-01, DEC-043); tabular classical failure map + matched 8/16/32-dim controls + passing permutation null (T-E4-BASE-01, DEC-044); deep 1D-CNN rung and the fusion that completed the classical map (T-E4-SIGCACHE-01/SIGTEST-02/BASE-02, DEC-046); prior-art audit that falsified two of three novelty claims and left the classical-fusion control as the only survivor (T-E4-PRIOR-01, DEC-047); **and now the quantum rung itself (T-E4-Q-01, DEC-048).** **THE RESULT: `fusion@cnn+gbm+zz` - `fusion@cnn+gbm+poly2` = +0.000641, CI [-0.000486, +0.001747], spans zero -> the quantum feature map contributed nothing a classical map of the same shape did not.** The §7.10 rule that made this a null was fixed *before* the first circuit was fitted and is what makes the record defensible: **`fusion@cnn+gbm+zz` 0.946329 is the highest number in the whole phase**, above the DEC-046 bar (+0.000036) and above both classical-fusion controls -- "our quantum fusion set a new best on PTB-XL" was a true-in-isolation and indefensible sentence, and the pre-registration removed the choice before there was anything to choose. Three findings, all pointing the same way. **(1) The quantum map is the worst of the three maps of its own shape:** `q@zz` 0.760505 vs `c@poly2` 0.884961 = **-0.124457 [-0.145524, -0.105109]**, the only interval in the experiment that excludes zero, over an identical index set (8 singletons + 28 pairs, verified column-for-column against `all_pair_masks(8)`), identical head, identical 4-cell budget. **(2) The fusion slot is saturated:** adding *any* 36-column map to `cnn+gbm` moves it <±0.0007 with every interval spanning zero (zz +0.000036, poly2 -0.000605, rff36 -0.000631) -- an independent third confirmation of DEC-046's "the residual error is a property of the task, not of model capacity", and the retroactive vindication of the classical-fusion control, since alone zz's +0.000036 would have read as a quantum contribution when the slot is inert for everything. **(3) The null is not an artifact:** both E3 correctness gates passed at machine precision before a single score was kept (readout-vs-Aer 1.80e-16 / 1.11e-15; circuit-vs-qiskit-`ZZFeatureMap` 6.66e-16 / 9.99e-16, tol 1e-10) and the entangling witness gives reps=2 mean \|C_ij\| **0.07320875** (max 0.71480617) -- the entanglement was present, was measured, and did not help, which is a stronger statement than a product-state null. The baseline it was measured against is provably the one on record: `recover_members` re-ran DEC-046's exact path and reproduced **bit-identically** (cnn 0.9404758944556537, gbm 0.9402343416976896). Two things recorded rather than repaired, both deliberately: `q@zz` selected **C at the grid edge** (0.01) -- widening only the losing arm's budget would break the same-shape contract and is the metric shopping the mission forbids; and DEC-048 **rejects** rescuing the null by adding reps, changing the entanglement topology or varying the readout basis, citing arXiv:2507.11401 (400 topologies sampled, the winning 16% reported). **One defect found and fixed before any number was kept (ISS-011):** the degeneracy guard read `_standardiser`'s std, which floors a constant column at 1.0, so the guard **could never fire for any input**; a deliberate `reps=1` probe printed `GUARD FAILED: reps=1 was accepted`, the fix reads the raw encoded spread, and it now fires at 7.03e-17 -- no recorded number was affected, because `QUANTUM_REPS_GRID = (2,)` meant no scored cell ever reached `reps=1`. A coverage gap was closed alongside it (ISS-012): `e4_deep_baseline` produced half the bar with zero tests, and its reproduction assertion proves **determinism, not correctness** -- a systematic defect would reproduce exactly and pass. Mandate honoured end to end: every threshold/hyperparameter/architecture/reps chosen inside TRAIN or on inner fold 8; **fold-10 signals never downloaded, `test_partition_used: false` in every payload**, `patient_overlap 0`, `ecg_id_overlap 0`, `pca_sees_fold_9 false`; existing pipeline untouched, every artefact additive; suite **1,038 green** (957 + 51 quantum + 30 deep, 0 failures, 790.70 s) | **COMPLETE -- the arena was built correctly and it returned a null** |
