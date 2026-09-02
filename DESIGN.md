# CareScan — Design System & UI/UX Contract

This document defines the UI/UX implementation contract for the CareScan Patient Mobile Application.

**Visual source of truth**: The current approved Google Stitch project — "CareScan Mobile Patient Portal".

---

## 1. Stitch as Source of Truth

The Stitch design is the authoritative reference for:

- Screen layouts
- Component styling
- Color palette
- Typography
- Spacing and sizing
- Iconography
- Navigation patterns
- State presentations (loading, error, empty)

### Rules

1. Implement the **current** approved Stitch design, not an older version.
2. If the current Stitch design cannot be directly inspected in a given session, state this clearly and mark values as `VERIFY WITH STITCH`.
3. Do not invent visual details not supported by Stitch or confirmed requirements.
4. Do not redesign Stitch screens based on personal preference.
5. Record any justified design deviations in DECISIONS.md.

---

## 2. Design Tokens

Design tokens are centralized in `lib/core/theme/` and provide the single source of implementation values for the design system.

### 2.1 Colors

> `VERIFY WITH STITCH` — Exact color values require Stitch inspection.

| Token | Purpose | Value |
|---|---|---|
| `primary` | Primary brand color | `VERIFY WITH STITCH` |
| `primaryVariant` | Darker/lighter primary | `VERIFY WITH STITCH` |
| `secondary` | Secondary accent | `VERIFY WITH STITCH` |
| `background` | Screen backgrounds | `VERIFY WITH STITCH` |
| `surface` | Card/container surfaces | `VERIFY WITH STITCH` |
| `error` | Error indicators | `VERIFY WITH STITCH` |
| `onPrimary` | Text/icons on primary | `VERIFY WITH STITCH` |
| `onBackground` | Text/icons on background | `VERIFY WITH STITCH` |
| `onSurface` | Text/icons on surface | `VERIFY WITH STITCH` |
| `onError` | Text/icons on error | `VERIFY WITH STITCH` |
| `divider` | Divider lines | `VERIFY WITH STITCH` |
| `disabled` | Disabled elements | `VERIFY WITH STITCH` |

**Rules**:
- All color usage references these tokens.
- No inline hex values in widget code.
- Ensure all text/background combinations meet WCAG AA contrast (4.5:1 for normal text, 3:1 for large text).

### 2.2 Typography

> `VERIFY WITH STITCH` — Font family and scale require Stitch inspection.

| Token | Usage | Properties |
|---|---|---|
| `headlineLarge` | Screen titles | `VERIFY WITH STITCH` |
| `headlineMedium` | Section headers | `VERIFY WITH STITCH` |
| `titleLarge` | Card titles | `VERIFY WITH STITCH` |
| `titleMedium` | Subtitles | `VERIFY WITH STITCH` |
| `bodyLarge` | Primary body text | `VERIFY WITH STITCH` |
| `bodyMedium` | Secondary body text | `VERIFY WITH STITCH` |
| `bodySmall` | Captions | `VERIFY WITH STITCH` |
| `labelLarge` | Button labels | `VERIFY WITH STITCH` |
| `labelMedium` | Tab labels, chips | `VERIFY WITH STITCH` |

**Rules**:
- All text uses `Theme.of(context).textTheme` references.
- No hardcoded font sizes or font families in widgets.
- Support dynamic text scaling (accessibility).

### 2.3 Spacing

> `VERIFY WITH STITCH` — Exact spacing scale requires Stitch inspection.

| Token | Value |
|---|---|
| `xs` | 4.0 |
| `sm` | 8.0 |
| `md` | 16.0 |
| `lg` | 24.0 |
| `xl` | 32.0 |
| `xxl` | 48.0 |

**Rules**:
- Padding, margins, and gaps use spacing tokens from `AppSpacing`.
- No arbitrary magic numbers for layout spacing.

### 2.4 Shapes

> `VERIFY WITH STITCH` — Exact radii require Stitch inspection.

| Token | Purpose | Value |
|---|---|---|
| `radiusSm` | Small elements (chips, badges) | `VERIFY WITH STITCH` |
| `radiusMd` | Cards, inputs | `VERIFY WITH STITCH` |
| `radiusLg` | Bottom sheets, dialogs | `VERIFY WITH STITCH` |
| `radiusFull` | Circular elements | `Radius.circular(999)` |

---

## 3. Components

### 3.1 Component Library

Reusable widgets in `lib/shared/widgets/`:

| Component | Purpose |
|---|---|
| `AppButton` | Primary, secondary, text button variants |
| `AppCard` | Standard card container |
| `LoadingIndicator` | Consistent loading state presentation |
| `ErrorStateWidget` | Consistent error display with retry |
| `EmptyStateWidget` | Consistent empty state with guidance |
| `AppScaffold` | Consistent screen scaffold with app bar patterns |
| `AssessmentListItem` | History list entry |
| `ImageDisplay` | Image display with loading/error fallback |

### 3.2 Component States

Every interactive component must handle:

| State | Visual |
|---|---|
| Default | Normal appearance |
| Pressed | Visual feedback (ripple, opacity change) |
| Disabled | Reduced opacity, non-interactive |
| Focused | Focus ring/highlight (accessibility) |
| Loading | Loading indicator replaces content where appropriate |

### 3.3 Touch Targets

- Minimum touch target: **48×48 dp** (Material Design / accessibility guidelines)
- Buttons and interactive elements must meet this minimum regardless of visual size

---

## 4. Navigation Patterns

> `VERIFY WITH STITCH` — Exact navigation pattern requires Stitch inspection.

**Expected patterns** (subject to Stitch confirmation):

| Pattern | Usage |
|---|---|
| Bottom navigation bar | Primary navigation between Home, History, Settings |
| Push navigation | Camera → Preview → Analyzing → Result flow |
| Back button | Standard platform back navigation |
| App bar | Screen title and actions |

### Navigation Rules

- Assessment flow is sequential and cannot be skipped
- Completing an assessment returns to Home (not back through the flow)
- Bottom navigation items remain accessible from all primary screens
- Transition animations follow platform conventions

---

## 5. Responsive Mobile Layouts

The application targets mobile phones (Android and iOS). Tablet optimization is not in initial scope.

### Layout Rules

- Use `SafeArea` to respect system UI insets
- Test on multiple screen sizes (small, medium, large phones)
- Avoid fixed-height containers that overflow on small screens
- Use flexible/scrollable layouts for content-heavy screens
- Handle landscape orientation: `REQUIRES CLARIFICATION` — lock to portrait or support both
- Handle notch/dynamic island areas

---

## 6. Accessibility

| Requirement | Implementation |
|---|---|
| Screen reader support | Semantic labels on all interactive elements |
| Contrast | All text meets WCAG AA contrast ratios |
| Touch targets | Minimum 48×48 dp |
| Color independence | Information not conveyed by color alone |
| Dynamic text | Text scales with system font size settings |
| Focus order | Logical tab/navigation order |
| Motion | Respect `reduceMotion` preference where applicable |

**If Stitch design and accessibility requirements conflict**: implement the most accessible reasonable interpretation and document the deviation in DECISIONS.md.

---

## 7. Loading States

Each screen and async operation has an explicit loading representation:

| Screen | Loading State |
|---|---|
| Home Dashboard | Skeleton/shimmer for content areas |
| Camera Scan | Camera initialization loading |
| Image Preview | Image loading indicator |
| Analyzing | Dedicated animated loading screen |
| Assessment Result | Loading indicator if result is being fetched |
| Assessment History | List shimmer or spinner |
| Settings | Loading indicator for async settings |

### Rules

- No blank/white screens during loading
- Loading states maintain layout stability (avoid layout shifts)
- Use consistent loading component from the shared library

---

## 8. Empty States

| Screen | Empty State |
|---|---|
| Assessment History | "No assessments yet" + CTA to start first scan |
| Home Dashboard | Fresh user state if applicable |

### Rules

- Empty states include an illustration or icon, message, and action
- Tone is encouraging, not clinical
- Exact empty state design: `VERIFY WITH STITCH`

---

## 9. Error States

Every screen that performs async operations must display an error state on failure:

| Element | Content |
|---|---|
| Icon | Warning or error icon |
| Title | Brief, user-readable error description |
| Message | Helpful context (not raw exception) |
| Action | Retry button where appropriate |

### Rules

- Use the shared `ErrorStateWidget`
- Never show raw exceptions or stack traces
- Error messages are human-readable and non-technical
- Retry available for transient failures (network, timeout)

---

## 10. Visual Fidelity Checks

Before marking a UI task complete, verify:

- [ ] Layout matches current Stitch screen
- [ ] Colors match design tokens (verified against Stitch)
- [ ] Typography matches Stitch text styles
- [ ] Spacing is consistent with Stitch
- [ ] Component dimensions and proportions are correct
- [ ] Icons match Stitch (size, style, color)
- [ ] States (loading, error, empty) are implemented
- [ ] Alignment is pixel-accurate where practical
- [ ] No layout overflow on standard device sizes
- [ ] Accessibility labels are present
- [ ] Touch targets meet minimum size
- [ ] Animations are smooth and platform-appropriate

> If Stitch cannot be directly inspected during verification, document this and mark the visual check as `PENDING STITCH VERIFICATION`.

---

## 11. Design Deviations

Design deviations from Stitch are permitted only when:

1. Explicitly requested by the user
2. Required for accessibility compliance
3. Required for platform behavior (e.g., iOS vs Android conventions)
4. Required to resolve an implementation conflict in the Stitch design

All significant deviations must be recorded in DECISIONS.md with rationale.
