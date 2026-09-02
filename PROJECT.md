# CareScan Patient Mobile Application

## Project Overview

CareScan is a patient-facing mobile application that enables users to capture images of skin conditions, receive AI-assisted assessments, and track their assessment history over time.

This project is established as a **new, independent project** in the workspace `/Users/pratyakshranjan/workinggggg`. It is not a continuation, fork, or migration of any prior project (including QuOra). All implementation decisions begin fresh from the current requirements and approved design.

## Product Goal

Deliver a polished, accessible, and secure mobile application that allows patients to:

1. Capture or select images for skin condition assessment
2. View AI-generated assessment results
3. Track their assessment history
4. Manage application settings

## Target Users

- **Primary**: Patients seeking convenient skin condition assessment from their mobile device
- **Secondary**: Healthcare system users directed to perform preliminary self-assessment

## Scope

### In Scope

- Seven patient-facing screens (see below)
- Camera capture and gallery image selection
- Image preview and confirmation workflow
- Assessment processing with loading/progress feedback
- Assessment result display
- Assessment history browsing
- Application settings management
- Offline/poor-network resilience where confirmed by requirements
- Accessibility (WCAG-informed)
- Android and iOS platform support

### Out of Scope

- Doctor/provider-facing features
- Backend/API server implementation
- ML model training or deployment
- Clinical diagnosis logic (UI only — abstracted behind service interfaces)
- Admin panel or web portal
- Push notifications (unless specified in requirements)
- Payment or billing
- Social features
- Flutter Web target (unless explicitly added later)

## Technology Stack

| Layer | Technology |
|---|---|
| Framework | Flutter |
| Language | Dart |
| Platforms | Android, iOS |
| Design Source | Google Stitch — "CareScan Mobile Patient Portal" |
| State Management | TBD — simplest appropriate solution |
| Testing | flutter_test, integration_test |
| Static Analysis | flutter analyze, dart analyze |

## Main Features

1. **Home Dashboard** — Landing screen with quick access to scanning and history
2. **Camera Scan** — Capture a new image using the device camera
3. **Image Preview** — Review and confirm a captured/selected image before submission
4. **Analyzing** — Processing/loading state while assessment is performed
5. **Assessment Result** — Display the outcome of an image assessment
6. **Assessment History** — Browse past assessments
7. **Settings** — User preferences and application configuration

## Patient Application Screens

| # | Screen | Purpose |
|---|---|---|
| 1 | Home Dashboard | Primary navigation and status overview |
| 2 | Camera Scan | Image capture interface |
| 3 | Image Preview | Review captured image before submitting |
| 4 | Analyzing... | Assessment processing indicator |
| 5 | Assessment Result | Display assessment outcome |
| 6 | Assessment History | List of past assessments |
| 7 | Settings | User preferences and configuration |

## Design Source of Truth

The **current approved Google Stitch project** ("CareScan Mobile Patient Portal") is the visual/UI source of truth.

Additional Stitch assets used as reference/documentation (not patient screens):

- Doctor Consultation Mobile App Design.jpeg
- Health Scan Flow
- Patient App PRD

**Rules:**

- Always prefer the current Stitch design over any older screenshot or prior implementation.
- If the current Stitch design cannot be directly inspected, state this clearly rather than guessing.
- Do not invent visual details not supported by the approved design.

## Source of Truth Hierarchy

| Priority | Source | Domain |
|---|---|---|
| 1 | Explicit user instruction | All |
| 2 | Current approved Stitch design | Visual / UI |
| 3 | Patient App PRD / confirmed requirements | Product / Functional |
| 4 | PROJECT.md | Project constraints and goals |
| 5 | REQUIREMENTS.md | Functional and non-functional requirements |
| 6 | ARCHITECTURE.md | Technical architecture |
| 7 | DESIGN.md | UI/UX implementation contract |
| 8 | DECISIONS.md | Durable architectural/product decisions |
| 9 | TASKS.md | Current implementation state |
| 10 | ISSUES.md | Known defects and blockers |
| 11 | Existing CareScan code | Implementation state (not design authority) |
| 12 | Human-created AAS skills | Engineering guidance |
| 13 | Official external docs | Only when genuinely necessary |

If an older implementation conflicts with the current Stitch design, do **not** automatically preserve the old implementation.

## Engineering Principles

1. **Simplicity** — Use the simplest architecture that satisfies the requirements
2. **Atomic execution** — Implement one task at a time; validate before moving on
3. **Faithful implementation** — Match the current Stitch design; do not redesign
4. **No invented requirements** — If it's not specified, mark it `REQUIRES CLARIFICATION`
5. **Clean boundaries** — Separate UI, state, services, and data access
6. **Accessibility first** — Accessibility is a requirement, not an afterthought
7. **Defensive UI** — Handle long text, missing images, errors, empty states, variable screen sizes
8. **Resource discipline** — Dispose controllers, manage image memory, avoid dependency bloat
9. **Security by default** — Never hardcode secrets; treat patient data as sensitive
10. **Measurable quality** — Do not claim performance or correctness without validation

## Quality Expectations

- Zero `flutter analyze` warnings in production code
- All screens handle loading, error, and empty states
- All interactive elements have accessibility labels
- Touch targets meet minimum size guidelines
- No layout overflow on standard device sizes
- Smooth scrolling and transitions
- All critical paths covered by tests

## Security Expectations

- No hardcoded API keys, tokens, passwords, or secrets
- Sensitive data uses platform-appropriate secure storage
- No sensitive patient data in logs or debug output
- Secure network communication (HTTPS)
- External data validated at application boundaries
- Secrets never committed to version control

## Definition of Done

A feature or task is **done** when:

- [ ] Implementation matches confirmed requirements
- [ ] UI matches current Stitch design (or deviation is documented)
- [ ] `flutter analyze` passes with no warnings
- [ ] Relevant tests pass
- [ ] Loading, error, and empty states are handled
- [ ] Accessibility labels are present
- [ ] No layout overflow on standard devices
- [ ] Code is reviewed for correctness, security, and maintainability
- [ ] TASKS.md is updated
- [ ] Any new decisions are recorded in DECISIONS.md
- [ ] Any discovered issues are recorded in ISSUES.md

## Current Project Status

| Item | Status |
|---|---|
| Project documentation | 🟡 Being established |
| Flutter project initialized | ❌ Not yet |
| Design system implemented | ❌ Not yet |
| Screens implemented | ❌ Not yet |
| Tests written | ❌ Not yet |
| Build verified | ❌ Not yet |
