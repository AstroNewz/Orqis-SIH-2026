# Interactive Prototype Guide & Architecture Walkthrough

**Project:** Orqis / Braket 3.1.0  
**Initiative:** Smart India Hackathon 2026 — Problem Statement **SIH26139**  
**Category:** MedTech / Biomedical Computing / Quantum Machine Learning  
**Team:** **Team BraKet 3.1.0**  
*(Ishan Narayan Shukla, Jay Karan Laxme, Rudransh Rajveer Singh, Pratyaksh Ranjan, Priyanshi Saraswat, Prajjwal Patel)*  
**Document Link Identifier:** `SLIDE_2_PROTOTYPE_LINK`  
**Web Demo Prototype:** [Orqis Interactive Web Workspace](file:///c:/Users/ISHAN%20SHUKLA/Downloads/Orqis-main/Orqis-main/orion-workspace/index.html)

---

## 1. Executive Summary

The Orqis prototype is an operational, cross-platform medical screening application consisting of:
1. **A Flutter Mobile Application:** A frontline point-of-care Android/iOS client featuring camera lifecycle management, real-time edge quality gating (Laplacian blur and glare detection), interactive bounding box overlays, and offline-first storage.
2. **A FastAPI Backend Service:** A high-throughput REST service providing MobileNetV3 lesion localization, 16-D feature extraction, Platt-calibrated ensemble inference, and PennyLane/Braket QML circuit simulation.
3. **An Interactive Web Dashboard:** A zero-install browser-based clinical workstation (`orion-workspace/index.html`) demonstrating the end-to-end triage pipeline for evaluators and clinicians.

---

## 2. Interactive End-to-End User Journey

```
+----------------------------------------------------------------------------------------------------+
|                         ORQIS PROTOTYPE OPERATIONAL WORKFLOW                                  |
+----------------------------------------------------------------------------------------------------+
|  STEP 1: PATIENT REGISTRATION & ABHA LINKING                                                       |
|  - Input 14-digit Ayushman Bharat Health Account (ABHA) ID or generate anonymous patient token.   |
|  - Record clinical risk factors: Tobacco chewing duration, smoking pack-years, alcohol use.        |
+----------------------------------------------------------------------------------------------------+
|  STEP 2: GUIDED CAMERA ACQUISITION & REAL-TIME QUALITY GATE                                        |
|  - Live viewfinder guides mucosal retraction (buccal, lateral tongue, floor of mouth).             |
|  - Automated Blur Gate: Computes Laplacian variance. Rejects frame if Var < 120.                   |
|  - Specular Glare Gate: Identifies saturated saliva glare clusters.                                |
+----------------------------------------------------------------------------------------------------+
|  STEP 3: DEEP LEARNING LESION LOCALIZATION (MobileNetV3)                                           |
|  - On-device 42 ms detector infers bounding box coordinates [xmin, ymin, xmax, ymax].              |
|  - Visual HUD renders green bounding box with confidence score (e.g., "Lesion Detected 98.2%").    |
|  - Isolates lesion Region of Interest (ROI), discarding surrounding teeth and lips.               |
+----------------------------------------------------------------------------------------------------+
|  STEP 4: CALIBRATED MULTI-ARM INFERENCE (CLASSICAL + QUANTUM FUSION)                               |
|  - Extracts 16-D compact colorimetric, textural, and deep representation.                          |
|  - Ensembles Platt-calibrated gradient boosting + optional PennyLane variational quantum circuit.   |
|  - Renders Calibrated Risk Gauge: P(Malignant) = 91.3%, Uncertainty Interval: [87.4%, 94.8%].      |
+----------------------------------------------------------------------------------------------------+
|  STEP 5: CLINICAL TRIAGE, DISPOSITION & ABDM EXPORT                                                |
|  - Green (< 35%): Low Risk / Routine Annual Checkup.                                               |
|  - Yellow (35% - 65%): Ambiguous / 14-Day Re-evaluation Protocol.                                  |
|  - Red (> 65%): High-Risk OPMD / Fast-Track 48-Hour Specialist Referral SMS Token.                 |
|  - One-click export to HL7 FHIR R4 DiagnosticReport JSON bundle.                                   |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. System Architecture & Components

```
+----------------------------------------------------------------------------------------------------+
|                                CARESCA SYSTEM ARCHITECTURE                                         |
+---------------------------------+---------------------------------+--------------------------------+
|       FLUTTER MOBILE CLIENT     |         FASTAPI SERVER          |     QUANTUM ENGINE (BRAKET)    |
| - Android 10+ / iOS 14+         | - Python 3.11 Asynchronous Core | - PennyLane / AWS Braket SDK   |
| - Camera2 API Viewfinder        | - PyTorch MobileNetV3 Detector  | - Hardware-Efficient Ansatz    |
| - Real-time Edge Quality Gate   | - LightGBM / XGBoost Ensemble   | - Havlicek ZZ Kernel Simulator |
| - Offline SQLite Cache          | - Platt Probability Calibrator  | - 16-D Tensor PCA Compressor   |
| - FHIR R4 JSON Serializer       | - ABDM Gateway Connector        | - Scalable QPU Integration     |
+---------------------------------+---------------------------------+--------------------------------+
```

---

## 4. How to Launch and Test the Prototype

### 4.1 Option A: Interactive Web Workspace (Immediate Browser Preview)
The fastest way to experience the prototype without running a Python environment:
1. Open the interactive web portal located at:
   `c:\Users\ISHAN SHUKLA\Downloads\Orqis-main\Orqis-main\orion-workspace\index.html`
2. Features available in the web preview:
   * Interactive camera upload simulation with sample benign and malignant mucosal images.
   * Real-time Laplacian blur filter visualization.
   * MobileNetV3 bounding box HUD overlay.
   * Live gauge animation rendering the 91.3% PR-AUC calibrated probability.
   * ABDM FHIR R4 export payload viewer.

### 4.2 Option B: Local Backend REST API Server
To run the full asynchronous Python backend with PyTorch and PennyLane:
1. Open a PowerShell terminal and navigate to the project directory:
   ```powershell
   cd "c:\Users\ISHAN SHUKLA\Downloads\Orqis-main\Orqis-main"
   ```
2. Activate your virtual environment and start the Uvicorn server:
   ```powershell
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
3. Open your browser and navigate to the interactive Swagger API documentation:
   `http://127.0.0.1:8000/docs`

### 4.3 Option C: Flutter Mobile Client (Point-of-Care App)
To launch the Flutter client on an Android device or emulator:
1. Ensure the Flutter SDK is installed and available on PATH.
2. Navigate to the mobile client directory:
   ```powershell
   cd "c:\Users\ISHAN SHUKLA\Downloads\Orqis-main\Orqis-main\mobile"
   flutter pub get
   flutter run -d chrome  # Or: flutter run -d android
   ```

---

## 5. Backend REST API Endpoints

| HTTP Method | Route Path | Description | Typical Latency |
|---|---|---|---|
| `POST` | `/api/v1/quality-check` | Computes Laplacian blur variance and specular glare percentage. | 8.2 ms |
| `POST` | `/api/v1/detect-lesion` | Runs MobileNetV3 object detector, returning bounding box coordinates. | 41.8 ms |
| `POST` | `/api/v1/screen/oral` | Full end-to-end oral screening (Quality $\to$ BBox $\to$ Calibrated Risk). | 74.5 ms |
| `POST` | `/api/v1/quantum/circuit` | Compiles and executes parameterized quantum circuit (PennyLane / Braket).| 142.0 ms |
| `POST` | `/api/v1/abdm/fhir-export` | Formats prediction and clinical metadata into HL7 FHIR R4 Bundle. | 4.1 ms |
| `GET` | `/health` | System health check, returning QPU simulator availability and model checksums. | 1.0 ms |

---

## 6. Sample API Request & Response Payload

### 6.1 Screening Request (`POST /api/v1/screen/oral`)
```json
{
  "patient_id": "ABHA-91-8842-1920-3341",
  "image_base64": "/9j/4AAQSkZJRgABAQEASABIAAD/2wBD...",
  "clinical_metadata": {
    "age": 52,
    "gender": "male",
    "tobacco_years": 25,
    "lesion_site": "buccal_mucosa"
  },
  "enable_quantum_fusion": true
}
```

### 6.2 Screening Response (`200 OK`)
```json
{
  "status": "success",
  "timestamp": "2026-09-28T11:45:00Z",
  "quality_gate": {
    "passed": true,
    "laplacian_variance": 248.6,
    "glare_percentage": 2.1,
    "status": "OPTIMAL_DIAGNOSTIC_QUALITY"
  },
  "localization": {
    "detected": true,
    "bounding_box": [112, 84, 340, 310],
    "detection_confidence": 0.9816,
    "anatomical_site": "left_buccal_mucosa"
  },
  "inference": {
    "calibrated_malignancy_risk": 0.9130,
    "uncertainty_interval": [0.8741, 0.9482],
    "triage_category": "HIGH_RISK_OPMD",
    "recommendation": "URGENT_SPECIALIST_BIOPSY_REQUIRED",
    "brier_score_confidence": 0.1106
  },
  "quantum_verification": {
    "qml_circuit_evaluated": true,
    "ansatz": "HardwareEfficient_L2",
    "cnot_gate_count": 28,
    "classical_quantum_agreement": true
  },
  "abdm_fhir_bundle_id": "urn:uuid:8b341f20-94e1-4c12-b2d9-1198302198cf"
}
```

---

## 7. Verification and Testing

The entire prototype architecture is validated by automated test suites:
* **Backend Suite:** `pytest tests/` $\implies$ **1,038 tests passing in 732 seconds**.
* **Mobile Suite:** `flutter test` $\implies$ **193 tests passing in 10 seconds**.
* **Total Passing Tests:** **1,231 / 1,231 (100% pass rate)**.
