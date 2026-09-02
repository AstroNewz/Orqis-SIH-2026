# CareScan — AI Agent Instructions

This document defines the operating rules and workflow for AI coding agents working on the CareScan Patient Mobile Application.

---

## 1. Agent Workflow

Every significant unit of work follows this sequence:

```
READ CONTEXT
  → INSPECT CURRENT STATE
    → PLAN
      → IMPLEMENT ONE ATOMIC TASK
        → TEST
          → REVIEW
            → FIX
              → UPDATE DOCUMENTATION
                → COMPLETE TASK
```

Do not skip steps. Do not batch multiple unrelated tasks.

---

## 2. Mandatory Reading

Before starting work, agents must read the appropriate project documents:

| Activity | Required Reading |
|---|---|
| Any significant implementation | PROJECT.md |
| Feature implementation | REQUIREMENTS.md |
| Architectural changes | ARCHITECTURE.md |
| UI / visual work | DESIGN.md |
| Selecting next task | TASKS.md |
| Understanding prior decisions | DECISIONS.md |
| Checking known issues | ISSUES.md |

---

## 3. Task Selection

1. Check TASKS.md for available tasks
2. Select the highest-priority `TODO` task whose dependencies are `DONE`
3. Mark it `IN PROGRESS` in TASKS.md
4. Work on that task only
5. Complete it fully before selecting the next task

Do not:
- Skip dependency chains
- Work on multiple tasks simultaneously
- Create work outside of TASKS.md without justification

---

## 4. Agent Rules

### 4.1 Documentation

- Record durable decisions in DECISIONS.md
- Record genuine discovered issues in ISSUES.md
- Update TASKS.md status as work progresses
- Do not generate documentation for its own sake

### 4.2 Skills

- Use existing human-created skills (`.agent/skills/`) as engineering guidance when relevant
- **Never** modify, create, recreate, or copy SKILL.md files
- **Never** generate AI replacements for human-created skills
- Skills inform engineering approach — they are not product requirements

### 4.3 Requirements

- **Never** invent product requirements
- **Never** invent clinical, ML, or medical functionality
- If a requirement is unclear, mark it `REQUIRES CLARIFICATION`
- Do not silently assume behavior that isn't specified

### 4.4 Design Fidelity

- Implement the **current** Stitch design faithfully
- **Never** claim Stitch was inspected when it was not accessible
- If Stitch cannot be inspected, mark values `VERIFY WITH STITCH`
- Design deviations require explicit justification in DECISIONS.md

### 4.5 Workspace Isolation

- **Never** inspect, read, reference, or copy from any directory outside `/Users/pratyakshranjan/workinggggg`
- **Never** inspect the QuOra directory or any prior project
- Treat all prior project work as nonexistent

### 4.6 Code Quality

- Validate code with `flutter analyze` before marking tasks complete
- Run relevant tests before marking tasks complete
- Dispose controllers and resources properly
- No hardcoded secrets in source code
- No sensitive data in logs
- Use typed models, not dynamic
- Follow consistent naming and formatting

### 4.7 Dependencies

- Do not add packages without justification
- Check if Flutter/Dart already provides the capability
- Prefer established, maintained packages
- Record significant dependency decisions in DECISIONS.md

### 4.8 Scope Control

- Each task modifies only what is necessary
- Do not perform unrelated refactoring
- If unrelated technical debt is discovered, record it in ISSUES.md
- Do not automatically fix unrelated issues

---

## 5. Task Completion Criteria

A task is **COMPLETE** only when:

| Check | Required |
|---|---|
| Code implemented | ✅ |
| `flutter analyze` passes | ✅ |
| Relevant tests pass | ✅ |
| States handled (loading, error, empty) | ✅ for UI tasks |
| Accessibility labels present | ✅ for UI tasks |
| No layout overflow | ✅ for UI tasks |
| TASKS.md updated | ✅ |
| New decisions recorded | ✅ if applicable |
| New issues recorded | ✅ if applicable |

**Never** mark a task complete merely because code was written.

For UI tasks specifically:

```
IMPLEMENT → RUN → CHECK LAYOUT → COMPARE TO STITCH → FIX → TEST → REVIEW → COMPLETE
```

---

## 6. Error Handling

- Catch exceptions at service/repository boundaries
- UI shows user-readable errors, never raw exceptions
- Provide retry where appropriate
- Log technical details without sensitive data

---

## 7. Clinical / ML Rules

- Do not present mock results as real clinical results
- Keep ML/backend abstraction replaceable:
  ```
  UI → Service Interface → Mock/Real Implementation
  ```
- Mock data must be clearly identifiable as test data
- Never generate medical diagnoses or clinical recommendations

---

## 8. Security Rules

- No hardcoded API keys, tokens, passwords, or secrets
- No sensitive patient data in logs or debug output
- Use secure storage for credentials
- HTTPS for all network communication
- Validate external data at boundaries
- Never commit secrets to version control

---

## 9. File Safety

Before modifying any existing file:

1. Inspect it
2. Understand its purpose
3. Confirm it belongs to the current CareScan workspace
4. Confirm modification is necessary

Never blindly overwrite files. Never run destructive commands without clear justification.

---

## 10. Autonomy Guidelines

### Proceed independently for:
- Inspecting workspace files
- Creating project source files
- Running `flutter` commands
- Running tests and analyzers
- Fixing implementation errors
- Updating documentation
- Implementing clearly specified functionality

### Ask the user when:
- A genuine product decision cannot be resolved from project context
- An architectural decision has significant trade-offs
- A security-sensitive change is needed
- A destructive operation is required
- Requirements are genuinely ambiguous

---

## 11. Source Priority

When resolving conflicts, use this priority order:

1. Explicit user instruction
2. Current approved Stitch design
3. Confirmed Patient App requirements / PRD
4. PROJECT.md
5. REQUIREMENTS.md
6. ARCHITECTURE.md
7. DESIGN.md
8. DECISIONS.md
9. TASKS.md
10. ISSUES.md
11. Existing CareScan implementation
12. Human-created AAS skills (engineering guidance)
13. Official external documentation (only when genuinely necessary)

---

## 12. Internet Research

Do not browse the internet by default.

External research is allowed only when:
- The user explicitly requests it, OR
- A required technical fact genuinely cannot be resolved from local sources

When external research is necessary, prefer official documentation and explain why it was needed.
