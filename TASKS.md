# CareScan — Implementation Tasks

Atomic implementation backlog for the CareScan Patient Mobile Application.

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





