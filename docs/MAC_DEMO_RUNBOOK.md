# CareScan — Mac Demo Runbook (Physical iPhone over LAN)

**Audience:** the person with a Mac + iPhone who will run the live SIH demo.
**Goal:** get from a fresh Mac to a working end-to-end demo on a physical iPhone
without guessing.

The implementation is **frozen**. This document is the only thing you need to
follow. Do not change model artifacts, thresholds, or evaluation code to make
the demo work — everything required is already in place and verified.

---

## 0. What is already verified (do not re-litigate)

Validated on the development machine (Windows), all GREEN:

- Backend boots on `0.0.0.0:8000`; `/health` reports `model_ready: true`
  (`v1-handcrafted`, quantum `ideal_simulation`, `aer_simulator`, 8 qubits).
- Real inference (`isMock: false`) through `/api/screening/analyze`.
- `/api/screening/upload`, `/api/localize`, `/api/patients/{id}/history` all 200.
- Backend test suite: **439 passed, 0 failed**.
- Flutter: `flutter analyze` clean, `flutter test` **64 passed**.
- iOS `Info.plist` already grants **camera**, **local-network**, and
  **cleartext-HTTP-to-LAN** (`NSAllowsLocalNetworking`) — so the phone can talk
  to `http://<mac-lan-ip>:8000` without an ATS block.

**The only unvalidated step is running on a physical iPhone**, which needs
macOS + Xcode. That is what this runbook covers.

### The one thing that will bite you if you skip it
On a physical iPhone the app defaults its backend URL to `http://localhost:8000`,
which on the phone means *the phone itself*. You **must** pass the backend’s LAN
address at launch:

```
flutter run --dart-define=BACKEND_BASE_URL=http://<MAC-LAN-IP>:8000
```

---

## Final demo protocol (SIH) — run this exact sequence on the Mac

Execute in order. Sections §1–§10 below expand each step with detail and fixes;
this is the checklist to run on demo day.

1. `flutter doctor` — resolve any iOS ✗ lines. (§1)
2. `flutter devices` — the **physical iPhone must be listed by name**. (§1, §9)
3. `xcodebuild -version` — Xcode 15+. On the phone confirm: **unlocked**,
   **Developer Mode ON**, **computer trusted**, **same Wi‑Fi** as the backend, and
   an **Xcode signing Team** selected for the Runner target. (§1, §5, §9)
4. Backend, from repo root: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
   (wait for "Application startup complete"). (§2)
5. `curl http://localhost:8000/health` → `"model_ready": true`. (§2)
6. `ipconfig getifaddr en0` (try `en1` if empty) → this is `<MAC-LAN-IP>`. (§3)
7. **NETWORK GATE:** on the iPhone’s Safari open `http://<MAC-LAN-IP>:8000/health`
   — it must show the JSON. **Do not launch Flutter until this works.** (§4, §9)
8. From `carescan/`:
   `flutter run --dart-define=BACKEND_BASE_URL=http://<MAC-LAN-IP>:8000`
   — the override is **mandatory** on a physical iPhone (its default `localhost`
   points at the phone itself). (§5)
9. **Pre-warm** the localizer (one throwaway scan, or one `curl .../api/localize`),
   then perform **THREE complete real demo scans** through the full flow:
   Home → Camera → Capture → Preview → **Use This Image** → Analyze →
   MobileNet localization overlay → classical-primary risk result →
   experimental quantum secondary readout → History. (§6, §7)
10. **On-device pass/fail checklist** — confirm every item across the three scans:
    - [ ] camera permission granted; live preview shows
    - [ ] local-network permission granted (iOS prompt → Allow)
    - [ ] image capture works
    - [ ] upload succeeds (analyzing state appears)
    - [ ] analyzing state resolves (no stuck spinner)
    - [ ] localization overlay **visibly renders** a box on the capture
    - [ ] classical-primary **risk band** renders as the verdict
    - [ ] experimental **quantum secondary card** renders
    - [ ] quantum score does **not** change the classical verdict band
    - [ ] history entry **persists** (View History shows the scan)
    - [ ] no crashes; no stuck loading state
11. **Latency:** time scan #1 (expect ~4 s, one-time localizer cold load) vs scans
    #2–#3 (expect fast). Pre-warm before judging so #1 isn’t cold. (§7)
12. **Failure test (once):** stop the backend (or use an unreachable IP), run a
    scan → the app shows the network-failure state (“Unable to connect”), **no
    crash**; restart the backend → the next scan works. (§8)
13. **After device validation**, from repo root and `carescan/`:
    `flutter analyze` · `flutter test` · `pytest -q`.

### The two lanes the demo must communicate

Show this in the UI and say it out loud:

```
Image  →  MobileNet localization  →  Classical risk inference  →  RISK VERDICT   (PRIMARY)
Localized ROI  →  Experimental quantum analysis  →  Secondary research signal    (EXPERIMENTAL)
```

- The **classical** model is the **primary** verdict. The **quantum** result is an
  **experimental secondary readout** only.
- Never describe the quantum **PR‑AUC as "accuracy"**. Never claim quantum
  **advantage** or **superiority**. The quantum score must never silently alter the
  classical verdict.

---

## 1. Verify Mac prerequisites

Run each check; all must pass before continuing.

| # | Requirement | Check | Expected |
|---|-------------|-------|----------|
| 1 | Flutter | `flutter --version` | 3.47+ stable |
| 2 | Flutter toolchain | `flutter doctor` | Xcode + iOS lines are ✓ (checkmarks) |
| 3 | Xcode | `xcodebuild -version` | Xcode 15+ |
| 4 | Xcode license | `sudo xcodebuild -license accept` | no error |
| 5 | CocoaPods | `pod --version` | 1.11+ (see below if missing) |
| 6 | iPhone connected | `flutter devices` | your iPhone is listed by name |

**If CocoaPods is missing** (the `camera` plugin needs it):
```
sudo gem install cocoapods
```
`flutter run` runs `pod install` for you; you do **not** need to run it manually.

**On the iPhone itself:**
- **Unlock** the phone and keep it unlocked during the first build.
- **Trust this computer:** first USB connection shows *“Trust This Computer?”* →
  tap **Trust** and enter the passcode.
- **Developer Mode (iOS 16+):** Settings → Privacy & Security → **Developer Mode**
  → toggle **On** → the phone restarts → confirm after restart.
- **Same Wi‑Fi:** the iPhone and the backend host must be on the **same network**,
  and it must allow client-to-client traffic (many guest/public networks do not —
  use a phone hotspot or a home router if in doubt).

If `flutter devices` does not show the iPhone, see Troubleshooting §9.

---

## 2. Start the backend on the demo machine

From the repository root, in a Python 3.12 environment with dependencies
installed:

```
pip install -r requirements.txt        # first time only
git lfs install && git lfs pull        # fetch model artifacts if the repo uses Git LFS
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` is **required** so the phone can reach it. `--host 127.0.0.1`
  (the uvicorn default) is **not reachable** from the phone.
- Wait for `Application startup complete`.
- The backend host can be the Mac itself or any other machine on the same LAN;
  whichever it is, use *that machine’s* LAN IP everywhere `<MAC-LAN-IP>` appears.

**Confirm locally on the backend host:**
```
curl http://localhost:8000/health
```
Expect JSON containing `"model_ready": true`. If `model_ready` is `false`, the
model artifacts are missing — run `git lfs pull` (above) and restart.

---

## 3. Find the LAN IP of the backend host

On the Mac (Wi‑Fi):
```
ipconfig getifaddr en0        # try en1 if en0 is empty (Wi-Fi vs Ethernet)
```
Or: System Settings → Wi‑Fi → **Details…** → IP Address.

You want a private LAN address like `192.168.x.x` or `10.x.x.x`. Write it down;
this is `<MAC-LAN-IP>` for the rest of the runbook.

---

## 4. From the iPhone’s Safari, verify the backend is reachable

Before touching Flutter, open **Safari on the iPhone** and go to:
```
http://<MAC-LAN-IP>:8000/health
```
You should see the health JSON (`"model_ready": true`).

**If this page does not load, STOP.** The app cannot work until this does. Go to
Troubleshooting §9 (“phone cannot reach backend”, “wrong LAN IP”, “port 8000
blocked”). Do not proceed to §5 until Safari shows the JSON.

---

## 5. Launch the Flutter app on the iPhone

```
cd carescan
flutter devices                                   # confirm the iPhone is listed
flutter run --dart-define=BACKEND_BASE_URL=http://<MAC-LAN-IP>:8000
```

- Substitute the real IP from §3. **Do not commit a real IP anywhere.**
- The **first** build is slow (CocoaPods install + code signing + install to
  device). This is normal.
- If Xcode signing fails, see Troubleshooting §9 (“Xcode signing”).
- After install, if the app is killed by iOS on first launch, on the phone go to
  Settings → General → **VPN & Device Management** → trust your developer profile,
  then relaunch with the same `flutter run` command.

---

## 6. Execute the exact demo flow

Tap through, confirming each stage:

1. **Home** — tap **“Take a Photo”**.
2. **Camera** — the first time, iOS prompts for camera access → **Allow**. Frame
   the target and capture.
3. **Capture / Preview** — review the shot; tap **“Use This Image”** (or
   **“Retake”**).
4. **Analyze** — the analyzing screen appears while the backend runs
   upload → inference. (First run is slow; see §7.)
5. **Localization overlay** — on the result’s captured image, a **box** is drawn
   over the region of interest (the MobileNet localizer). If the localizer is
   unsure, no box is drawn and the image still shows — this is by design and never
   blocks the result.
6. **Classical risk result** — the headline **risk band** (e.g. `LOW RISK` /
   `MODERATE RISK`) is the **classical logistic-regression** verdict. This is the
   patient-facing result.
7. **Experimental quantum readout** — the “How this was scored” card also shows a
   **Quantum model (calibrated)** row. This is a *secondary, experimental* readout
   (see §10). It appears **without blocking** the classical result.
8. **History** — tap **“View History”**; the just-completed screening is listed
   (persisted server-side under patient `demo-patient-001`).

The first time the iPhone connects to the LAN backend, iOS shows a **“allow local
network access”** prompt → **Allow** (the app declares this permission).

---

## 7. Verify first-run vs warm-run latency

Measured server-side on localhost (device adds LAN image transfer + render on
top):

- **First scan after the backend starts: ~4 seconds.** This is a one-time cost:
  the ROI localizer lazily loads its Torch model on the first `/api/localize`
  call (~3.8 s cold). Every scan after that is fast.
- **Warm scans: ~0.2 s** of backend compute (upload ~20 ms + analyze ~120 ms +
  localize ~25 ms), plus the image upload over Wi‑Fi.

**Pre-warm before you present** so the audience never sees the 4‑second cold
start. On the backend host, after startup, fire one throwaway localize:
```
curl -F "file=@carescan/ios/Runner/Assets.xcassets/AppIcon.appiconset/*.png" http://localhost:8000/api/localize
```
or simply run one full scan on the phone and discard it. After that, the demo
scan is warm.

To observe on-device: time the analyzing screen with a stopwatch — expect ~4 s on
the first ever scan, well under a second thereafter.

---

## 8. Verify backend-unavailable behavior

Demonstrate graceful failure (optional but recommended):

- Stop the backend (Ctrl‑C) **or** launch the app with a wrong IP, then run a scan.
- The app surfaces a **network error** (“Unable to connect …”) and **does not
  crash**. Connect timeout is 10 s; receive timeout is 30 s, so a dead/blocked
  backend fails within ~10 s rather than hanging forever.
- Restart the backend and the next scan works — no app restart needed.

---

## 9. Troubleshooting

**Phone cannot reach backend (Safari §4 fails)**
- Confirm both devices are on the **same Wi‑Fi** and it isn’t “client isolation”
  Wi‑Fi (common on guest/enterprise networks). Try a personal hotspot.
- Confirm the backend was started with `--host 0.0.0.0` (not `127.0.0.1`).
- From the Mac: `curl http://<MAC-LAN-IP>:8000/health` — if that fails from the
  Mac too, it’s the backend/IP, not the phone.
- macOS firewall: System Settings → Network → Firewall → allow incoming
  connections for `python`/`uvicorn`, or turn the firewall off for the demo.

**Camera permission**
- If you tapped “Don’t Allow”, the camera stays black. Fix on the phone: Settings
  → **CareScan** → enable **Camera**, then relaunch.

**Developer Mode**
- Build installs but the app won’t launch / “Developer Mode required”: enable
  Settings → Privacy & Security → **Developer Mode**, restart the phone, retry.

**Xcode signing**
- `flutter run` fails with a signing error: open `carescan/ios/Runner.xcworkspace`
  in Xcode → target **Runner** → **Signing & Capabilities** → check **Automatically
  manage signing** → select your **Team** (a free personal Apple ID works). If the
  bundle id is taken, change **Bundle Identifier** to something unique
  (e.g. `com.<yourname>.carescan`). Re-run `flutter run`.
- On the phone the first launch may need: Settings → General → **VPN & Device
  Management** → trust the developer app.

**Wrong LAN IP**
- Symptom: Safari `/health` works but the app always times out, or vice-versa.
  Re-check §3 (use the **Wi‑Fi** interface address, not a VPN/Ethernet one).
  Re-launch `flutter run` with the corrected `--dart-define` — the IP is read at
  launch, so you must restart the app after changing it.

**Port 8000 blocked / in use**
- “Address already in use”: something else holds 8000. Either stop it, or run the
  backend on another port (`--port 8001`) **and** launch the app with
  `--dart-define=BACKEND_BASE_URL=http://<MAC-LAN-IP>:8001`.
- A corporate firewall may block 8000 across the LAN — use a hotspot or an allowed
  port as above.

**Flutter device not detected (`flutter devices` omits the iPhone)**
- Unlock the phone; re-plug the USB cable; tap **Trust** on the phone.
- `flutter doctor` and resolve any iOS ✗ lines.
- Open Xcode → Window → **Devices and Simulators** → confirm the phone appears and
  is “connected” (Xcode may need to “prepare” the device the first time).
- Enable **Developer Mode** (above) — a phone without it won’t appear as a run
  target.

---

## 10. Scientific honesty — what the numbers mean (say this in the demo)

**The classical model drives the verdict. The quantum score does not.**

- The patient-facing **risk band is the classical logistic-regression baseline’s**
  verdict — on this dataset the classical baseline is the strongest *validated*
  model, so it headlines the result (DEC-034).
- The **quantum VQC score is experimental** and shown only as a secondary,
  transparency readout. It is **not** the clinical verdict and does **not** gate or
  alter the classical result.
- The calibrated percentage carried internally (and in the FHIR export) is the
  quantum-calibrated probability because it is the only calibrated number
  available; even so, the **band the audience sees is classical**, and the
  classical score is a *ranking* value, never shown as a percentage.
- **No clinical validity is claimed.** The thresholds are experimental engineering
  choices, not validated clinical cut-offs. This is a screening-assist research
  prototype, not a diagnostic device.

Honest one-liner for the judges: *“The classical model makes the call; the quantum
component is an experimental readout we show for transparency — on this dataset it
hasn’t beaten the classical baseline, and we don’t pretend it has.”*
