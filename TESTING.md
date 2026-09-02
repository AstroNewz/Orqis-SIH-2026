# CareScan — Testing Strategy

This document defines the testing strategy for the CareScan Patient Mobile Application.

---

## 1. Testing Pyramid

```
         ┌─────────────────┐
         │  Integration     │  Few — critical user flows
         │  Tests           │
         ├─────────────────┤
         │  Widget Tests    │  Medium — screens and components
         │                  │
         ├─────────────────┤
         │  Unit Tests      │  Many — models, logic, services
         │                  │
         └─────────────────┘
```

Most tests should be fast unit tests. Widget tests cover UI behavior. Integration tests cover critical end-to-end flows.

---

## 2. Static Analysis

### Tool: `flutter analyze` / `dart analyze`

| Expectation | Threshold |
|---|---|
| Warnings | 0 in production code |
| Info-level hints | Addressed or explicitly acknowledged |
| Lints | Follow `flutter_lints` or `very_good_analysis` |

### When to Run

- Before every commit
- Before marking any task complete
- In CI pipeline

### Configuration

`analysis_options.yaml` at project root with strict lint rules enabled.

---

## 3. Unit Tests

### Scope

| Target | Examples |
|---|---|
| Data models | Serialization, deserialization, equality, copyWith |
| Validation logic | Input validation, boundary checks |
| Error types | Failure class construction and matching |
| Utilities | Formatting, date handling, string manipulation |
| State logic | State transitions (independent of UI) |
| Repository logic | Correct data source delegation (with mocked sources) |

### Conventions

- Test files mirror source structure: `lib/core/errors/failure.dart` → `test/unit/core/errors/failure_test.dart`
- Use descriptive test names: `'should return Failure when API returns 500'`
- Avoid testing framework internals — test behavior, not implementation
- No real network calls — use mocked data sources

### Running

```bash
flutter test test/unit/
```

---

## 4. Widget Tests

### Scope

| Target | Examples |
|---|---|
| Shared widgets | `LoadingIndicator`, `ErrorStateWidget`, `EmptyStateWidget`, `AppButton` |
| Screen widgets | Each of the 7 patient screens |
| Component states | Loading, error, empty, success rendering |
| User interactions | Tap, input, scroll behavior |
| Navigation triggers | Verify correct navigation actions on interaction |

### Conventions

- Use `WidgetTester` for pumping and interacting with widgets
- Provide mock dependencies (repositories, state) via dependency injection
- Test all meaningful states: loading, success, error, empty
- Verify accessibility: semantic labels present, correct widget semantics

### Running

```bash
flutter test test/widget/
```

---

## 5. Integration Tests

### Scope

| Flow | Description |
|---|---|
| Assessment flow | Camera → Preview → Analyzing → Result |
| History browsing | Navigate to History, view entries, handle empty state |
| Settings | Navigate to Settings, change preferences |
| Navigation | Bottom nav switching, back navigation |

### Conventions

- Use `integration_test` package
- Mock backend services (no real API calls)
- Test on emulator/simulator or real device
- Focus on critical user journeys, not exhaustive permutations

### Running

```bash
flutter test integration_test/
```

---

## 6. Navigation Tests

### Scope

- All 7 screens are reachable via their defined routes
- Assessment flow enforces sequential navigation
- Back button returns to correct parent
- Bottom navigation switches between primary screens
- Deep links handled (if applicable — `REQUIRES CLARIFICATION`)

### Approach

- Widget tests with mocked router
- Verify correct routes are pushed/popped on user actions

---

## 7. Camera / Image Flow Tests

### Feasibility

Camera hardware interaction is difficult to test in automated environments.

### Approach

| Layer | Test Type | Feasibility |
|---|---|---|
| Permission handling logic | Unit test | ✅ Feasible |
| Camera state management | Unit test | ✅ Feasible |
| Image preview rendering | Widget test (with mock image) | ✅ Feasible |
| Camera initialization | Manual test on device | ⚠️ Manual only |
| Image capture | Manual test on device | ⚠️ Manual only |
| Image validation logic | Unit test | ✅ Feasible |
| Gallery selection | Manual test on device | ⚠️ Manual only |

Abstract camera interaction behind a service interface so logic can be tested without hardware.

---

## 8. Golden / Visual Regression Tests

### Purpose

Detect unintended visual changes to stable UI components and screens.

### Scope

| Target | Priority |
|---|---|
| Shared widgets (AppButton, AppCard, etc.) | HIGH |
| Screen layouts (stable screens) | MEDIUM |
| Loading, error, empty states | MEDIUM |

### Approach

- Use `matchesGoldenFile` from `flutter_test`
- Store golden files in `test/goldens/`
- Update goldens explicitly when design changes are intentional
- Run on a consistent environment (same OS, Flutter version) to avoid false diffs

### Running

```bash
# Generate/update goldens
flutter test --update-goldens test/widget/

# Verify against goldens
flutter test test/widget/
```

---

## 9. Accessibility Testing

### Automated Checks

| Check | Tool |
|---|---|
| Semantic labels present | Widget tests — verify `Semantics` widgets |
| Tap target size | Widget tests — verify minimum 48×48 dp |
| Contrast ratios | Manual inspection + design token verification |

### Manual Checks

- Enable TalkBack (Android) / VoiceOver (iOS) and navigate all screens
- Verify logical focus order
- Verify dynamic text scaling doesn't break layout
- Verify information isn't conveyed by color alone

---

## 10. Error State Testing

### Scope

Every screen that performs async operations must be tested with:

| Scenario | Expected |
|---|---|
| Network failure | Error state with retry |
| API error response | Error state with user message |
| Timeout | Error state with retry |
| Malformed response | Error state (not crash) |
| Camera permission denied | Permission guidance |
| Camera init failure | Error with retry |
| Empty data | Empty state with guidance |

### Approach

- Widget tests with mock repositories configured to return failures
- Verify error widget renders with correct message and retry action
- Verify retry triggers re-fetch

---

## 11. Performance Testing

### Checks

| Metric | Method |
|---|---|
| Startup time | Profile with Flutter DevTools |
| Frame rate | Profile scrolling, transitions with DevTools |
| Memory (images) | Monitor memory during image capture/preview flow |
| List rendering | Profile Assessment History with many entries |
| Build size | `flutter build apk --analyze-size` |

### Thresholds

- No visible jank during scrolling (target 60fps)
- No memory leak from undisposed image controllers
- Build size reasonable for a mobile application

---

## 12. Build Validation

### Android

```bash
flutter build apk --release
# Verify APK launches on device/emulator
```

### iOS

```bash
flutter build ios --release --no-codesign
# Verify build completes without errors
```

### Both Platforms

- `flutter analyze` passes
- `flutter test` passes
- No build warnings in release mode

---

## 13. Definition of Done for Testing

A feature's testing is complete when:

- [ ] `flutter analyze` passes with 0 warnings
- [ ] Unit tests written for all business logic (models, validation, state transitions)
- [ ] Widget tests written for the screen and its states (loading, success, error, empty)
- [ ] Navigation behavior tested
- [ ] Accessibility semantics verified in tests
- [ ] Error scenarios tested with mock failures
- [ ] Tests pass: `flutter test`
- [ ] Manual verification on device/emulator for camera/hardware-dependent features
- [ ] Golden tests updated if visual changes are intentional

---

## 14. Test Organization

```
test/
├── unit/
│   ├── core/
│   │   ├── errors/
│   │   └── utils/
│   ├── data/
│   │   ├── models/
│   │   └── repositories/
│   └── features/
│       └── <feature>/state/
├── widget/
│   ├── shared/
│   │   └── widgets/
│   └── features/
│       └── <feature>/screens/
├── goldens/
│   └── <component_name>.png
├── mocks/
│   ├── mock_repositories.dart
│   └── mock_services.dart
└── integration_test/
    ├── assessment_flow_test.dart
    ├── navigation_test.dart
    └── history_test.dart
```
