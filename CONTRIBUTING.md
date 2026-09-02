# CareScan — Contributing Guide

Development rules for the CareScan Patient Mobile Application.

---

## 1. Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, validated code |
| `feature/<task-id>-<short-name>` | Feature development |
| `fix/<issue-id>-<short-name>` | Bug fixes |

- Work on feature or fix branches, not directly on `main`.
- Branch names reference task or issue IDs (e.g., `feature/T-SCREEN-01-home-dashboard`).
- Keep branches short-lived — merge when the task is complete and validated.

---

## 2. Code Quality

### Formatting

- Use `dart format` consistently
- Line length: 80 characters (Dart default)
- No trailing whitespace
- Final newline at end of file

### Naming

| Element | Convention | Example |
|---|---|---|
| Files | `snake_case` | `assessment_result_screen.dart` |
| Classes | `PascalCase` | `AssessmentResultScreen` |
| Variables/functions | `camelCase` | `fetchAssessmentHistory` |
| Constants | `camelCase` or `SCREAMING_SNAKE` for top-level | `maxRetryCount` |
| Private members | Prefix with `_` | `_isLoading` |
| Test files | `<source_file>_test.dart` | `assessment_result_screen_test.dart` |

### Linting

- Follow `analysis_options.yaml` rules strictly
- `flutter analyze` must pass with 0 warnings before committing
- Do not add `// ignore` comments without a documented reason

### Type Safety

- Use strong Dart typing — avoid `dynamic` unless absolutely necessary
- Use null safety correctly — no unnecessary `!` operators
- Use typed models for data, not `Map<String, dynamic>`

---

## 3. File Organization

- Follow the directory structure defined in ARCHITECTURE.md
- One primary class/widget per file
- Keep files focused — split when a file exceeds ~300 lines and responsibilities are separable
- Do not split files artificially for the sake of small files

---

## 4. Testing

### Requirements

- Unit tests for business logic, models, validation, utilities
- Widget tests for screens and shared components (all states)
- Integration tests for critical user flows
- `flutter test` must pass before committing

### Test Quality

- Test behavior, not implementation details
- Use descriptive test names
- Each test should test one thing
- Use mocks for external dependencies — no real network calls in tests

---

## 5. Documentation

### Code Comments

- Add comments to explain **why**, not **what**
- Do not comment obvious code
- Use `///` doc comments for public APIs
- Mark unverified design values: `// TODO: Verify with Stitch`
- Mark clarification items: `// REQUIRES CLARIFICATION: <description>`

### Project Documentation

- Update TASKS.md when starting/completing tasks
- Update DECISIONS.md for architectural or product decisions
- Update ISSUES.md for discovered defects or blockers
- Do not create documentation for its own sake

---

## 6. Commit Standards

### Commit Message Format

```
<type>(<scope>): <short description>

[Optional body with more detail]

[Optional footer: references task/issue IDs]
```

### Types

| Type | Usage |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructuring (no behavior change) |
| `style` | Formatting, no logic change |
| `test` | Adding or updating tests |
| `docs` | Documentation only |
| `chore` | Build, config, dependencies |

### Examples

```
feat(home): implement Home Dashboard screen

Build the Home Dashboard layout matching Stitch design.
Add loading and error state handling.

Task: T-SCREEN-01
```

```
fix(camera): handle permission denial on Android

Show guidance message when camera permission is denied
instead of blank screen.

Issue: ISS-006
```

### Rules

- Each commit represents one logical change
- Do not mix unrelated changes in one commit
- Reference task or issue IDs in commit messages
- Commit messages should be understandable without reading the code

---

## 7. Review Expectations

Before a change is merged:

- [ ] `flutter analyze` passes (0 warnings)
- [ ] `flutter test` passes
- [ ] Code follows naming and formatting conventions
- [ ] No hardcoded secrets or sensitive data
- [ ] No unnecessary dependencies added
- [ ] UI matches Stitch design (or deviation documented)
- [ ] Loading, error, and empty states handled (UI changes)
- [ ] Accessibility labels present (UI changes)
- [ ] TASKS.md / DECISIONS.md / ISSUES.md updated as needed
- [ ] Commit messages follow the format

---

## 8. Avoiding Unrelated Changes

- Each PR/commit addresses one task or issue
- Do not refactor unrelated code alongside feature work
- If you discover unrelated technical debt, record it in ISSUES.md
- Do not "clean up" files outside the scope of your task
- Do not change formatting in files you didn't modify substantively

---

## 9. Dependencies

- Do not add packages without justification
- Check Flutter/Dart built-in capabilities first
- Check existing project dependencies
- Prefer mature, maintained, well-documented packages
- Document significant dependency additions in DECISIONS.md

---

## 10. Security

- Never commit API keys, tokens, passwords, or secrets
- Use `.gitignore` to exclude sensitive files
- Use platform-appropriate secure storage for credentials
- Do not log sensitive patient information
- Validate all external data at boundaries
