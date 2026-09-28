# CareScan (Flutter client)

Camera-based oral-cancer screening client for the Orqis backend. Flow:

> Camera → capture → preview → analyzing → backend inference → MobileNet ROI
> overlay → validated classical verdict (quantum readout secondary) → result →
> history

Networking uses `dart:io HttpClient` only. The backend base URL is **never
hardcoded** — the client resolves it, in order, from `ApiConstants.setBaseUrl()`,
then `--dart-define=BACKEND_BASE_URL`, then the platform default
(`10.0.2.2:8000` on Android emulator, `localhost:8000` otherwise).

## Demo: physical iPhone over LAN

> **Running the live iPhone demo on a Mac?** Follow the full step-by-step handoff
> runbook: [`docs/MAC_DEMO_RUNBOOK.md`](../docs/MAC_DEMO_RUNBOOK.md) (prerequisites,
> signing, troubleshooting). The summary below is the short version.

The phone and the machine running the backend must be on the **same Wi‑Fi**.

**1. Start the backend bound to the LAN** (not just localhost, or the phone
can't reach it):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**2. Find the backend machine's LAN IP:**

```bash
ipconfig getifaddr en0
```

(Use `ipconfig` on Windows, `hostname -I` on Linux; pick the Wi‑Fi address.)

**3. Confirm the phone can reach it** — open `http://<lan-ip>:8000/health` in
the phone's browser; you should get a JSON health response.

**4. Run the app pointed at that IP** (substitute the address from step 2 — do
not commit a real IP):

```bash
flutter run --dart-define=BACKEND_BASE_URL=http://<lan-ip>:8000
```

The demo drives patient `demo-patient-001`; results persist to history on the
backend.

## Test & analyze

```bash
flutter test
flutter analyze
```
