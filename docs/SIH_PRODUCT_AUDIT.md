# SIH product audit — 20 September 2026

Scope: existing `carescan/` Flutter client and actual FastAPI contracts. Existing uncommitted work is preserved. The Windows workspace is the user's explicitly named project; the obsolete macOS path in AGENTS.md does not identify this checkout.

## Findings before implementation

- Working foundations: go_router stateful shell, camera plugin, typed repositories/results, multipart upload, server history, model provenance, independent localization and substantial tests. Baseline `flutter analyze`: no issues.
- UI: purple placeholder tokens; hardcoded white navigation; fabricated Alex identity/ABHA connection; inactive upload/help/profile controls; document/QR camera labels; missing camera lifecycle handling; preview asserts unmeasured quality; null result defaults to LOW RISK; history displays the secondary band and cannot open details.
- No localization catalogs, language persistence, functioning theme preference, patient onboarding, articles or bundled instructional illustrations.
- Preserve `POST /api/screening/upload`, `/api/screening/analyze`, `/api/localize`, `GET /api/patients/{id}/history`, `/api/screening/{id}/image` and existing wire fields.
- Backend quality gate measures exposure, resolution, normalized Laplacian focus, clipping and ROI adequacy before inference. The ROI regressor is explicitly not an oral-presence classifier. No validated wrong-object detector is present.
- `/api/auth/login` and `/api/auth/me` authenticate clinic staff only. No patient signup/login exists. Clinic access must not masquerade as patient authentication.
- Android main manifest lacks INTERNET permission; release signing uses debug keys. iOS has camera/local-network descriptions and local-network ATS exception. Device capture and iOS builds require hardware/macOS verification.
- Research artifacts are experimental. Preserve primary risk, never display uncalibrated scores as probability, and separate quantum output. No clinical-validation or quantum-advantage claims.

## Plan and design authority

Follow the T-SIH sequence in TASKS.md. The user's teal/white/cool-grey brief authorizes redesign. Stitch was not accessible: legacy design comparison is **VERIFY WITH STITCH**. Illustrations will be original Flutter vector drawings, without external imagery.
