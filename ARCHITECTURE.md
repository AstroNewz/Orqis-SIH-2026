# CareScan — Architecture

This document defines the technical architecture for the CareScan Patient Mobile Application.

**Principle**: Use the simplest architecture that supports maintainability, testability, and the stated requirements. Do not over-engineer.

---

## 1. Application Structure

```
lib/
├── main.dart                     # Entry point
├── app.dart                      # MaterialApp / root widget
├── core/
│   ├── theme/                    # Design tokens, ThemeData, colors, typography
│   ├── constants/                # App-wide constants
│   ├── errors/                   # Error types, failure classes
│   ├── utils/                    # Shared utility functions
│   └── extensions/               # Dart extensions
├── navigation/
│   └── app_router.dart           # Centralized route definitions
├── features/
│   ├── home/                     # Home Dashboard
│   ├── camera/                   # Camera Scan
│   ├── preview/                  # Image Preview
│   ├── analyzing/                # Analyzing state
│   ├── result/                   # Assessment Result
│   ├── history/                  # Assessment History
│   └── settings/                 # Settings
├── shared/
│   ├── widgets/                  # Reusable UI components
│   ├── models/                   # Shared data models
│   └── services/                 # Cross-feature services
└── data/
    ├── repositories/             # Repository implementations
    ├── datasources/              # Remote and local data sources
    └── models/                   # Data transfer objects / serialization
```

## 2. Feature Organization

Each feature folder follows a consistent structure:

```
features/<feature>/
├── screens/                      # Screen-level widgets
├── widgets/                      # Feature-specific widgets
├── state/                        # Feature state management
└── models/                       # Feature-specific models (if needed)
```

Features are self-contained. Cross-feature dependencies flow through shared services and models, not direct imports between feature folders.

## 3. Shared Components

`shared/widgets/` contains reusable presentation widgets used across multiple features:

- Loading indicators
- Error state widget
- Empty state widget
- Common buttons, cards, inputs
- Image display components

These widgets accept data and callbacks via constructor parameters — they do not manage application state.

## 4. Theme / Design Tokens

Centralized in `core/theme/`:

| File | Purpose |
|---|---|
| `app_theme.dart` | ThemeData construction |
| `app_colors.dart` | Color palette from Stitch |
| `app_typography.dart` | TextStyles from Stitch |
| `app_spacing.dart` | Spacing constants |
| `app_shapes.dart` | Border radii, card shapes |

All visual values come from the design token files. Hardcoded styling in feature code is avoided.

> **Note**: Exact token values require Stitch inspection. Placeholders will be used until verified, clearly marked as `// TODO: Verify with Stitch`.

## 5. Navigation

**Approach**: Flutter's declarative routing (GoRouter or Navigator 2.0).

Confirmed — Bottom navigation bar with central floating action button (Scan).

### Known Routes

| Route | Screen | Notes |
|---|---|---|
| `/` | Home Dashboard | App entry point |
| `/camera` | Camera Scan | Push from Home |
| `/preview` | Image Preview | Push from Camera (passes image data) |
| `/analyzing` | Analyzing | Push from Preview (passes image data) |
| `/result` | Assessment Result | Replace Analyzing on completion |
| `/history` | Assessment History | Accessible from Home |
| `/settings` | Settings | Accessible from Home |

### Assessment Flow

```
Home → Camera → Preview → Analyzing → Result
                  ↑ (retake)
```

The assessment flow is sequential. The user cannot skip steps. Back navigation from the Result screen returns to Home (not back through the analysis flow).

## 6. State Management

**Decision**: `REQUIRES CLARIFICATION` — will select the simplest approach that meets the requirements.

**Candidates** (in order of preference for simplicity):

1. **ChangeNotifier + Provider** — sufficient for moderate complexity
2. **Riverpod** — if stronger dependency injection and testability are needed
3. **Bloc/Cubit** — if explicit event-driven state transitions are preferred

### State Categories

| Category | Examples | Scope |
|---|---|---|
| Local UI state | Dialog visibility, selected tab, text input | Widget-local |
| Feature state | Camera state, analysis progress, current result | Feature-scoped |
| Application state | User session, navigation state | App-scoped |
| Persisted/remote data | Assessment history, user profile | Repository-managed |

### State Contract

Every async operation must represent these states explicitly:

```dart
// Conceptual — not prescriptive implementation
enum AsyncState { initial, loading, success, empty, error }
```

## 7. Data Layer

```
UI / Screen
    ↓
State / Controller
    ↓
Repository (interface)
    ↓
Data Source (remote API, local storage, mock)
```

### Repositories

Repositories expose domain-oriented methods and hide data-source implementation:

| Repository | Responsibility |
|---|---|
| `AssessmentRepository` | Submit image, get result, get history |
| `UserRepository` | User profile, session (if applicable) |
| `SettingsRepository` | App settings, preferences |

Repositories are defined as abstract interfaces. Concrete implementations (API-backed, mock, local) can be swapped.

### Data Sources

| Source | Purpose |
|---|---|
| Remote (API) | Assessment submission, history retrieval |
| Local (SharedPreferences / SQLite) | Settings, cached data |
| Mock | Development/testing without backend |

## 8. API Boundaries

The application communicates with the Group 3 FastAPI backend API:

| Endpoint | Method | Purpose | Payload / Response |
|---|---|---|---|
| `/health` | `GET` | Health check & system status | `{"status": "healthy", "quantum_backend": "aer_simulator"}` |
| `/api/screening/upload` | `POST` | Image file upload (multipart) | Returns `{"image_path": "...", "patient_id": "..."}` |
| `/api/screening/analyze` | `POST` | Full QML pipeline execution | Input: `ScreeningCreate`, Returns: `AssessmentResultResponse` |
| `/api/screening/{id}` | `GET` | Retrieve screening session | Returns: `AssessmentResponse` |
| `/api/results/{id}` | `GET` | Retrieve assessment result | Returns: `AssessmentResultResponse` |
| `/api/patients/{id}/history` | `GET` | Patient screening history | Returns: `List[HistoryEntryResponse]` |
| `/api/screening/{id}/fhir` | `GET` | Interoperable clinical report | Returns: HL7 FHIR R4 `Observation` & `RiskAssessment` |

**Boundary rule**: API interaction is encapsulated in data source classes. No HTTP calls in UI or state layers.


## 9. Image Handling

| Concern | Approach |
|---|---|
| Capture | Platform camera plugin (e.g., `camera` or `image_picker`) |
| Permissions | Request before access; handle denial gracefully |
| Preview | Display captured image at appropriate resolution |
| Compression | Resize/compress before upload to manage bandwidth and memory |
| Memory | Dispose image data and controllers when no longer needed |
| Orientation | Handle EXIF orientation correctly |
| Validation | Verify file exists, format is supported, size is reasonable |

Image handling utilities reside in `shared/services/` or a dedicated `core/image/` module.

## 10. ML / Inference Boundary

The application does **not** perform ML inference directly (unless explicitly required later).

**Architecture**:

```
UI → Service Interface → Backend API (or Mock)
```

The service interface abstracts the inference provider. During development, a mock implementation returns test data. In production, the real API is called.

**Critical**: Mock results must be clearly identifiable as test data. Never present mock clinical results as real.

## 11. Persistence Boundary

| Data | Storage | Justification |
|---|---|---|
| User preferences / settings | SharedPreferences | Simple key-value |
| Auth tokens (if applicable) | flutter_secure_storage | Sensitive credential |
| Assessment history cache | Local DB or in-memory | `REQUIRES CLARIFICATION` |
| Images (temporary) | Temporary app directory | Cleared after upload |

## 12. Error Handling

### Strategy

- Exceptions are caught at the repository/service boundary
- Repositories return typed results (success/failure) rather than throwing
- UI displays user-readable error messages
- Technical details logged (without sensitive data)
- Retry available where appropriate

### Error Categories

| Category | Handling |
|---|---|
| Network error | Error state + retry |
| API error | Error state + user message |
| Camera permission denied | Guidance to grant permission |
| Camera init failure | Error state + retry |
| Image processing failure | Error state + retry/retake |
| Assessment timeout | Error state + retry |
| Storage failure | Error state + fallback |
| Unexpected error | Generic error state + log |

## 13. Logging / Observability

- Use a lightweight logging utility (e.g., `logger` package or custom)
- Log errors, failed operations, and important state transitions
- **Never log**: passwords, tokens, sensitive patient data, image content
- Structured logging preferred for production diagnostics
- Crash reporting integration: `REQUIRES CLARIFICATION`

## 14. Security Boundaries

| Boundary | Control |
|---|---|
| Network | HTTPS only |
| Credentials | Secure storage, never in source |
| API keys | Environment config, never hardcoded |
| Logs | No sensitive data |
| Local data | Minimal sensitive persistence |
| Input | Validated at boundaries |
| Images | Temporary storage, cleaned up |

## 15. Testing Architecture

```
test/
├── unit/                         # Pure logic, models, utilities
├── widget/                       # Widget rendering and interaction
├── integration/                  # Full-flow tests
└── mocks/                        # Shared mock implementations
```

| Layer | Test Type | Tool |
|---|---|---|
| Models, utilities | Unit tests | flutter_test |
| Widgets, components | Widget tests | flutter_test |
| State management | Unit tests | flutter_test |
| Repositories | Unit tests (with mock data sources) | flutter_test |
| Screens | Widget tests | flutter_test |
| User flows | Integration tests | integration_test |
| Visual regression | Golden tests | flutter_test (matchesGoldenFile) |

Mock implementations of repositories and services are provided for testing. Tests never depend on real backend services.
