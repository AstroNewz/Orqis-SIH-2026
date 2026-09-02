# CareScan — Requirements

This document defines the functional and non-functional requirements for the CareScan Patient Mobile Application.

**Source of truth**: Current Stitch design + Patient App PRD. Requirements not confirmed by these sources are marked `REQUIRES CLARIFICATION`.

---

## 1. Functional Requirements

### 1.1 Home Dashboard

| ID | Requirement | Status |
|---|---|---|
| FR-HOME-01 | Display a dashboard as the primary entry point after launch | Confirmed |
| FR-HOME-02 | Provide quick access to the Camera Scan screen | Confirmed |
| FR-HOME-03 | Provide access to Assessment History | Confirmed |
| FR-HOME-04 | Provide access to Settings | Confirmed |
| FR-HOME-05 | Display a summary of recent assessments or status | `REQUIRES CLARIFICATION` — exact dashboard content pending Stitch inspection |
| FR-HOME-06 | Display user greeting or profile information | `REQUIRES CLARIFICATION` |

### 1.2 Camera Scan

| ID | Requirement | Status |
|---|---|---|
| FR-CAM-01 | Open device camera for image capture | Confirmed |
| FR-CAM-02 | Request camera permission before access | Confirmed |
| FR-CAM-03 | Handle camera permission denial gracefully (error state with guidance) | Confirmed |
| FR-CAM-04 | Support selecting an image from the device gallery | `REQUIRES CLARIFICATION` — Stitch shows camera; gallery selection TBD |
| FR-CAM-05 | Handle camera initialization failure | Confirmed |
| FR-CAM-06 | Provide a capture button with clear affordance | Confirmed |
| FR-CAM-07 | Support flash toggle if available | `REQUIRES CLARIFICATION` |
| FR-CAM-08 | Support camera switching (front/back) if applicable | `REQUIRES CLARIFICATION` |

### 1.3 Image Preview

| ID | Requirement | Status |
|---|---|---|
| FR-PREV-01 | Display the captured/selected image for user review | Confirmed |
| FR-PREV-02 | Allow the user to confirm the image and proceed to analysis | Confirmed |
| FR-PREV-03 | Allow the user to retake/reselect the image | Confirmed |
| FR-PREV-04 | Display the image at sufficient resolution for review | Confirmed |
| FR-PREV-05 | Handle image loading failure | Confirmed |

### 1.4 Analyzing

| ID | Requirement | Status |
|---|---|---|
| FR-ANAL-01 | Display a loading/progress state while assessment is processed | Confirmed |
| FR-ANAL-02 | Provide visual feedback that processing is active (animation or progress) | Confirmed |
| FR-ANAL-03 | Handle assessment processing timeout | Confirmed |
| FR-ANAL-04 | Handle assessment processing failure with error state and retry | Confirmed |
| FR-ANAL-05 | Allow user to cancel/navigate back during analysis | `REQUIRES CLARIFICATION` |
| FR-ANAL-06 | Transition to Assessment Result on success | Confirmed |

### 1.5 Assessment Result

| ID | Requirement | Status |
|---|---|---|
| FR-RES-01 | Display the assessment outcome clearly | Confirmed |
| FR-RES-02 | Display the assessed image alongside the result | `REQUIRES CLARIFICATION` |
| FR-RES-03 | Provide navigation back to Home Dashboard | Confirmed |
| FR-RES-04 | Provide option to start a new scan | `REQUIRES CLARIFICATION` |
| FR-RES-05 | Display confidence level or severity if provided by the assessment | `REQUIRES CLARIFICATION` — depends on ML/backend response |
| FR-RES-06 | Display recommended actions if provided | `REQUIRES CLARIFICATION` — clinical content must not be invented |
| FR-RES-07 | Handle missing or incomplete result data gracefully | Confirmed |

### 1.6 Assessment History

| ID | Requirement | Status |
|---|---|---|
| FR-HIST-01 | Display a chronological list of past assessments | Confirmed |
| FR-HIST-02 | Each history entry shows date, summary, and/or thumbnail | `REQUIRES CLARIFICATION` — exact fields pending Stitch inspection |
| FR-HIST-03 | Allow tapping an entry to view full assessment detail | `REQUIRES CLARIFICATION` |
| FR-HIST-04 | Handle empty history state (no assessments yet) | Confirmed |
| FR-HIST-05 | Handle history loading failure | Confirmed |
| FR-HIST-06 | Support pagination or lazy loading for large history lists | `REQUIRES CLARIFICATION` |

### 1.7 Settings

| ID | Requirement | Status |
|---|---|---|
| FR-SET-01 | Display user-configurable application settings | Confirmed |
| FR-SET-02 | Provide a way to sign out or manage account | `REQUIRES CLARIFICATION` |
| FR-SET-03 | Display application version information | `REQUIRES CLARIFICATION` |
| FR-SET-04 | Provide access to privacy policy / terms | `REQUIRES CLARIFICATION` |
| FR-SET-05 | Support notification preferences if applicable | `REQUIRES CLARIFICATION` |
| FR-SET-06 | Support theme selection (light/dark) if applicable | `REQUIRES CLARIFICATION` |

### 1.8 Navigation

| ID | Requirement | Status |
|---|---|---|
| FR-NAV-01 | Bottom navigation bar with central floating action button | Confirmed |
| FR-NAV-02 | Assessment flow is sequential: Camera → Preview → Analyzing → Result | Confirmed |
| FR-NAV-03 | Back navigation returns to the appropriate parent screen | Confirmed |
| FR-NAV-04 | Deep links handled if required | `REQUIRES CLARIFICATION` |

---

## 2. Non-Functional Requirements

### 2.1 Loading States

| ID | Requirement | Status |
|---|---|---|
| NFR-LOAD-01 | Every network or async operation has an explicit loading state | Confirmed |
| NFR-LOAD-02 | Loading indicators are appropriate to the operation (spinner, shimmer, progress bar) | Confirmed |
| NFR-LOAD-03 | No blank/white screens during loading | Confirmed |

### 2.2 Error States

| ID | Requirement | Status |
|---|---|---|
| NFR-ERR-01 | All operations that can fail display a user-readable error state | Confirmed |
| NFR-ERR-02 | Error states include a retry action where appropriate | Confirmed |
| NFR-ERR-03 | Raw exception messages are never displayed to users | Confirmed |
| NFR-ERR-04 | Camera permission denial shows an actionable error message | Confirmed |

### 2.3 Empty States

| ID | Requirement | Status |
|---|---|---|
| NFR-EMPTY-01 | Assessment History displays an appropriate empty state when no assessments exist | Confirmed |
| NFR-EMPTY-02 | Empty states provide guidance or a call-to-action | Confirmed |

### 2.4 Accessibility

| ID | Requirement | Status |
|---|---|---|
| NFR-A11Y-01 | Sufficient text contrast (WCAG AA minimum) | Confirmed |
| NFR-A11Y-02 | Minimum touch target size (48×48 dp) | Confirmed |
| NFR-A11Y-03 | All interactive elements have semantic labels for screen readers | Confirmed |
| NFR-A11Y-04 | Important information not conveyed by color alone | Confirmed |
| NFR-A11Y-05 | Dynamic text scaling considered | Confirmed |
| NFR-A11Y-06 | Logical focus/navigation order | Confirmed |

### 2.5 Offline / Poor-Network Resilience

| ID | Requirement | Status |
|---|---|---|
| NFR-NET-01 | Network failures produce a clear error state, not a crash | Confirmed |
| NFR-NET-02 | Retry is available after network failure | Confirmed |
| NFR-NET-03 | Previously loaded assessment history remains visible during temporary connectivity loss | `REQUIRES CLARIFICATION` |
| NFR-NET-04 | Image capture works without network (upload deferred) | `REQUIRES CLARIFICATION` |

### 2.6 Security / Privacy

| ID | Requirement | Status |
|---|---|---|
| NFR-SEC-01 | No hardcoded secrets (API keys, tokens, passwords) | Confirmed |
| NFR-SEC-02 | HTTPS for all network communication | Confirmed |
| NFR-SEC-03 | Sensitive patient data not logged | Confirmed |
| NFR-SEC-04 | Secure storage for credentials or tokens | Confirmed |
| NFR-SEC-05 | External data validated at application boundaries | Confirmed |
| NFR-SEC-06 | No unnecessary copies of sensitive data | Confirmed |

### 2.7 Performance

| ID | Requirement | Status |
|---|---|---|
| NFR-PERF-01 | Fast application startup | Confirmed |
| NFR-PERF-02 | Smooth scrolling (no jank) | Confirmed |
| NFR-PERF-03 | Responsive touch interactions | Confirmed |
| NFR-PERF-04 | Efficient image handling (resize, cache, dispose) | Confirmed |
| NFR-PERF-05 | Efficient list rendering for Assessment History | Confirmed |

### 2.8 Localization

| ID | Requirement | Status |
|---|---|---|
| NFR-L10N-01 | Architecture supports localization (intl/arb) | `REQUIRES CLARIFICATION` — confirm target languages |
| NFR-L10N-02 | User-facing strings externalized | `REQUIRES CLARIFICATION` — confirm if required for initial release |

### 2.9 Platform

| ID | Requirement | Status |
|---|---|---|
| NFR-PLAT-01 | Runs on Android (minimum API level TBD) | Confirmed |
| NFR-PLAT-02 | Runs on iOS (minimum version TBD) | Confirmed |
| NFR-PLAT-03 | Platform-appropriate permission flows | Confirmed |

---

## 3. Requirements Requiring Clarification

The following items need input from the product owner or Stitch design inspection before implementation:

1. **FR-HOME-05/06** — Exact Home Dashboard content and user greeting
2. **FR-CAM-04/07/08** — Gallery selection, flash toggle, camera switching
3. **FR-ANAL-05** — Cancel during analysis
4. **FR-RES-02/04/05/06** — Result screen detail fields and actions
5. **FR-HIST-02/03/06** — History entry fields, detail view, pagination
6. **FR-SET-02–06** — Settings screen exact options
7. **FR-NAV-01** — Navigation pattern (bottom nav, tabs, etc.)
8. **NFR-NET-03/04** — Specific offline capabilities
9. **NFR-L10N-01/02** — Localization targets and timeline
10. **NFR-PLAT-01/02** — Minimum OS version targets

> These items should be resolved by inspecting the current Stitch design and confirming with the product owner. Do not guess or invent functionality.
