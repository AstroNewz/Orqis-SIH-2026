# Video Demonstration Script: 3-Minute Technical Pitch & Operational Demo

**Project:** CareScan / Braket 3.1.0  
**Initiative:** Smart India Hackathon 2026 — Problem Statement **SIH26139**  
**Theme:** MedTech / Healthcare & Biomedical Computing  
**Category:** Software / Quantum Machine Learning  
**Team:** **Team BraKet 3.1.0**  
*(Ishan Narayan Shukla, Jay Karan Laxme, Rudransh Rajveer Singh, Pratyaksh Ranjan, Priyanshi Saraswat, Prajjwal Patel)*  
**Document Link Identifier:** `SLIDE_2_VIDEO_LINK`  
**Target Duration:** Exactly 3 Minutes (180 Seconds)  
**Mapping:** Aligned with all 6 slides of the official presentation deck `RESEARCH AND REFERENCE (3).pdf`

---

## 1. Production Overview & Speaker Roster

```
+----------------------------------------------------------------------------------------------------+
|                               VIDEO DEMONSTRATION ROSTER & TIMINGS                                 |
+---------+--------------------------+-----------------------+---------------------------------------+
| Time    | Phase / Presentation Slide | Presenting Lead       | Visual & Audio Focus                  |
+---------+--------------------------+-----------------------+---------------------------------------+
| 0:00-0:35 | Hook & Clinical Need (S1-S2) | Ishan Narayan Shukla  | Oral cancer epidemic, late detection  |
| 0:35-1:10 | Mobile App & Edge Gate (S2-S3)| Pratyaksh Ranjan      | Live phone UI, blur gate, MobileNetV3 |
| 1:10-1:45 | Quantum Rigor & Ansätze (S3-S4)| Jay Karan Laxme     | CNOT scaling, 7 QML families, physics |
| 1:45-2:20 | Audited Machine Results (S5-S6)| Priyanshi & Prajjwal | 91.3% PR-AUC, PTB-XL, 1,231 tests    |
| 2:20-3:00 | ABDM, Scalability & Close (S5-S6)| Rudransh & Ishan    | ABHA FHIR export, CDSCO SaMD, vision  |
+---------+--------------------------+-----------------------+---------------------------------------+
```

---

## 2. Minute-by-Minute Production Script

### Scene 1: The Urgent Clinical Crisis & The CareScan Solution (0:00 - 0:35)
* **Slides Referenced:** Slide 1 (Title) & Slide 2 (Problem & Solution Flow)
* **On-Screen Visual:** 
  * High-impact statistics on oral cancer in India (77,000 deaths annually; 70% presenting at Stage III/IV).
  * Split screen: Rural primary health center without diagnostic tools vs. CareScan mobile interface in an ASHA worker's hand.
  * Title overlay: **Problem Statement SIH26139 — CareScan / Braket 3.1.0**.
* **Speaker (Ishan Narayan Shukla - Team Lead):**
  > *"Every year in India, over seventy-seven thousand lives are lost to oral cancer. The tragedy is that oral cancer is curable when detected early—yet nearly seventy percent of rural patients present at Stage Three or Four, when five-year survival drops below thirty percent. 
  > 
  > Rural clinics have no oncologists, and optical tools like VELscope suffer from fifty-percent false positives. To solve this, Team BraKet 3.1.0 built **CareScan**: an accessible, hybrid quantum-classical screening platform delivering ninety-one point three percent Precision-Recall AUC on standard smartphones."*

---

### Scene 2: Live Mobile Demonstration & Real-Time Edge Quality Gate (0:35 - 1:10)
* **Slides Referenced:** Slide 2 (Solution Card) & Slide 3 (Technical Approach: Stages 1–3)
* **On-Screen Visual:** 
  * Direct screen recording of the Flutter mobile app running on an Android smartphone.
  * The camera captures a deliberately shaken, blurred image: The app immediately flashes red with an auditory alert: *"Image Blurred: Retake Photo"*.
  * The user steadies the device: The green HUD instantly locks onto an oral leukoplakia lesion on the lateral border of the tongue with a green bounding box and confidence score (`0.982`).
* **Speaker (Pratyaksh Ranjan - Software Architect & Mobile Lead):**
  > *"Point-of-care screening fails if the input image is blurry or ruined by saliva glare. CareScan solves this at the edge. 
  > 
  > Watch our real-time viewfinder: If an ASHA worker's hand shakes, our on-device Laplacian filter instantly rejects the corrupted frame in under eight milliseconds. 
  > 
  > Once steady, our optimized MobileNetV3 detector isolates the lesion in just forty-two milliseconds—achieving a ninety-eight point two percent localization rate across three hundred and eighty-one clinical test images, stripping away teeth and facial shortcuts."*

---

### Scene 3: Methodological Rigor & Quantum Circuit Compilation (1:10 - 1:45)
* **Slides Referenced:** Slide 3 (Feature Dimensionality) & Slide 4 (Technical Feasibility & QML Ansätze)
* **On-Screen Visual:** 
  * Animation comparing the exponential $O(2^n)$ CNOT explosion (>250,000 gates) collapsing $T_2^*$ coherence vs. CareScan's 16-D tensor-network dimensional compression.
  * PennyLane circuit diagram transpiling a 4-qubit Hardware-Efficient Ansatz with 28 CNOTs executing in 142 ms.
* **Speaker (Jay Karan Laxme - Quantum Lead):**
  > *"Most biomedical quantum papers suffer from a fatal flaw: attempting to load high-dimensional images directly into qubits, triggering an exponential CNOT gate explosion that obliterates physical coherence on NISQ hardware. 
  > 
  > CareScan resolves this through tensor-network dimensionality reduction, compressing features into a sixteen-dimensional latent manifold. We evaluated seven quantum algorithm families—from Havlicek ZZ-maps to Projected Quantum Kernels and Matrix Product States. Our compiled circuits execute under forty-eight CNOT gates, operating strictly within physical hardware coherence budgets."*

---

### Scene 4: Machine-Audited Empirical Results & Transparent Science (1:45 - 2:20)
* **Slides Referenced:** Slide 5 (Impacts) & Slide 6 (Research, Validation & Results)
* **On-Screen Visual:** 
  * High-resolution Precision-Recall curve ($0.913038$) and ROC curve ($0.933948$).
  * Test partition confusion matrix ($N=102$): 38 TP, 4 FN, 50 TN, 10 FP.
  * Forest plot of PTB-XL ECG benchmark ($N=19,601$): Showing classical 1D-ResNet vs. QML fusion.
  * Terminal window displaying `pytest` and `flutter test` logs: `1,231 / 1,231 PASSED`.
* **Speakers (Priyanshi Saraswat & Prajjwal Patel - Clinical AI & Signal Leads):**
  > **Priyanshi:** *"Our primary oral screening engine achieves a verified Precision-Recall AUC of ninety-one point three percent, with ninety point five percent clinical sensitivity and eighty point seven percent specificity on zero-overlap, patient-disjoint test splits. Our Platt-calibrated Brier score is zero point one one zero six, meaning risk percentages reflect true histological reality."*
  > 
  > **Prajjwal:** *"We practice honest science: On two-dimensional oral projections, classical RBF SVMs outperform NISQ ZZ-kernels by twelve point four percent. Furthermore, on nineteen thousand six hundred ECG records, our hybrid fusion achieved zero point nine four zero eight ROC-AUC. Every claim is validated by one thousand two hundred and thirty-one automated tests with zero failures."*

---

### Scene 5: National ABDM Integration, Scalability, & Closing Vision (2:20 - 3:00)
* **Slides Referenced:** Slide 4 (Operational Feasibility) & Slide 5 (Healthcare System Benefits)
* **On-Screen Visual:** 
  * App UI: Generating a 14-digit ABHA ID and displaying a one-click generated HL7 FHIR R4 JSON bundle.
  * Fast-track tele-triage referral SMS sent to a district hospital oncologist.
  * National map showing scalable screening across primary health centers in India.
  * Closing slide with team roster and GitHub repository QR code.
* **Speakers (Rudransh Rajveer Singh & Ishan Narayan Shukla):**
  > **Rudransh:** *"CareScan is natively integrated with India's Ayushman Bharat Digital Mission. In one tap, the app generates an HL7 FHIR Release Four DiagnosticReport bundle with standard SNOMED CT and ICD-11 coding, automatically routing high-risk patients to district hospitals within forty-eight hours."*
  > 
  > **Ishan:** *"By uniting smartphone edge computer vision, rigorous classical learning, and scalable quantum computing, CareScan transforms oral cancer screening from a late-stage death sentence into an accessible, point-of-care cure. We are Team BraKet 3.1.0. Thank you."*

---

## 3. Video Editing & Asset Guidelines

| Segment | Video Asset Required | Audio Track |
|---|---|---|
| **0:00 - 0:35** | B-roll of clinical oral exams + CareScan logo animation. | Confident, measured narrative voiceover; subtle ambient medical synth. |
| **0:35 - 1:10** | 1080p screen recording of Flutter app on physical Android device. | Clear click/haptic sound effects; crisp instructional tone. |
| **1:10 - 1:45** | 3D motion graphics of Bloch spheres and PennyLane circuit transpilation. | Dynamic technological pace; scientific graphics. |
| **1:45 - 2:20** | Direct screen capture of terminal test runs and vector matplotlib plots. | High-energy, data-driven presentation; metric callout badges. |
| **2:20 - 3:00** | Interactive ABDM JSON modal + animated map of India healthcare flow. | Inspiring, forward-looking orchestral crescendo to finish. |
