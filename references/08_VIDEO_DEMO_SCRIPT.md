# Video Demonstration Script: 3-Minute Technical Pitch & Operational Demo

**Project:** Orqis / Braket 3.1.0  
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
| 0:35-1:10 | Mobile App & Edge Gate (S2-S3)| Pratyaksh & Rudransh  | Live phone UI, blur gate, MobileNetV3 |
| 1:10-1:45 | Quantum Rigor & QPU Run (S3-S4)| Jay Karan & Ishan    | Heron ibm_fez QPU, 10-min free trial  |
| 1:45-2:20 | Audited Machine Results (S5-S6)| Priyanshi & Prajjwal | >93% ROC-AUC, 55%->93% journey, tests |
| 2:20-3:00 | ABDM, Scalability & Close (S5-S6)| Rudransh & Ishan    | ABHA FHIR export, CDSCO SaMD, vision  |
+---------+--------------------------+-----------------------+---------------------------------------+
```

---

## 2. Minute-by-Minute Production Script

### Scene 1: The Urgent Clinical Crisis & The Orqis Solution (0:00 - 0:35)
* **Slides Referenced:** Slide 1 (Title) & Slide 2 (Problem & Solution Flow)
* **On-Screen Visual:** 
  * High-impact statistics on oral cancer in India (77,000 deaths annually; 70% presenting at Stage III/IV).
  * Split screen: Rural primary health center without diagnostic tools vs. Orqis mobile interface in an ASHA worker's hand.
  * Title overlay: **Problem Statement SIH26139 — Orqis / Braket 3.1.0**.
* **Speaker (Ishan Narayan Shukla - Team Leader & Quantum Lead):**
  > *"Every year in India, over seventy-seven thousand lives are lost to oral cancer. The tragedy is that oral cancer is curable when detected early—yet nearly seventy percent of rural patients present at Stage Three or Four, when five-year survival drops below thirty percent. 
  > 
  > Rural clinics have no oncologists, and optical tools like VELscope suffer from fifty-percent false positives. To solve this, Team BraKet 3.1.0 built **Orqis**: an accessible, hybrid quantum-classical screening platform delivering over ninety-three percent ROC-AUC and ninety-four point seven percent PR-AUC on standard smartphones and physical IBM Quantum hardware."*

---

### Scene 2: Live Mobile Demonstration & Real-Time Edge Quality Gate (0:35 - 1:10)
* **Slides Referenced:** Slide 2 (Solution Card) & Slide 3 (Technical Approach: Stages 1–3)
* **On-Screen Visual:** 
  * Direct screen recording of the Flutter mobile app running on an Android smartphone.
  * The camera captures a deliberately shaken, blurred image: The app immediately flashes red with an auditory alert: *"Image Blurred: Retake Photo"*.
  * The user steadies the device: The green HUD instantly locks onto an oral leukoplakia lesion on the lateral border of the tongue with a green bounding box and confidence score (`0.982`).
* **Speaker (Pratyaksh Ranjan & Rudransh Rajveer Singh - Frontend & Mobile Leads):**
  > *"Point-of-care screening fails if the input image is blurry or ruined by saliva glare. Orqis solves this directly at the edge. 
  > 
  > Watch our real-time viewfinder: If an ASHA worker's hand shakes, our on-device Laplacian filter instantly rejects the corrupted frame in under eight milliseconds. 
  > 
  > Once steady, our optimized MobileNetV3 detector isolates the lesion in just forty-two milliseconds—achieving a ninety-eight point two percent localization rate across three hundred and eighty-one clinical test images, stripping away teeth and facial shortcuts."*

---

### Scene 3: Methodological Rigor & IBM Quantum Hardware Testing (1:10 - 1:45)
* **Slides Referenced:** Slide 3 (Feature Dimensionality) & Slide 4 (Technical Feasibility & QML Ansätze)
* **On-Screen Visual:** 
  * Animation comparing exponential $O(2^n)$ CNOT explosion (>250,000 gates) vs. Orqis's 16-D tensor-network dimensional compression.
  * Real IBM Quantum Platform dashboard showing job execution on the 156-qubit Heron processor `ibm_fez`.
  * Telemetry card: Depth 32, 42 CX gates, runtime 382.4s of 10-minute free trial, Pearson correlation $r = 0.9642$.
* **Speakers (Jay Karan Laxme - AI Lead & Ishan Narayan Shukla - Quantum Lead):**
  > **Jay Karan:** *"Most biomedical quantum papers suffer from a fatal flaw: attempting to load raw pixels directly into qubits, triggering an exponential CNOT gate explosion that obliterates physical coherence. Orqis resolves this through tensor-network dimensionality reduction, compressing features into a sixteen-dimensional latent manifold."*
  > 
  > **Ishan:** *"We didn't just simulate circuits; we validated our pipeline on physical superconducting quantum hardware—specifically IBM's 156-qubit Heron processor, ibm_fez, within IBM's 10-minute monthly free trial. Executing fifty circuits in 382.4 seconds with zero cloud cost, SABRE routing and XY4 dynamical decoupling with TREX error mitigation achieved an empirical Pearson correlation of zero point nine six four against ideal statevector simulation."*

---

### Scene 4: Machine-Audited Empirical Results & Transparent Science (1:45 - 2:20)
* **Slides Referenced:** Slide 5 (Impacts) & Slide 6 (Research, Validation & Results)
* **On-Screen Visual:** 
  * 5-Stage evolutionary journey graphic: 55.2% $\to$ 74.2% $\to$ 84.6% $\to$ 87.7% $\to$ 93.4%.
  * High-resolution Precision-Recall curve ($0.947275$) and ROC curve ($0.933948$).
  * Test partition confusion matrix ($N=102$): 38 TP, 4 FN, 50 TN, 10 FP.
  * Terminal window displaying `pytest` and `flutter test` logs: `1,231 / 1,231 PASSED`.
* **Speakers (Priyanshi Saraswat & Prajjwal Patel - Clinical & Signal Leads):**
  > **Priyanshi:** *"Our platform charts an audited five-stage evolutionary journey: advancing from a fifty-five point two percent raw-pixel baseline up to a verified ninety-three point four percent ROC-AUC and ninety-four point seven percent PR-AUC on patient-disjoint test splits. Our Platt-calibrated Brier score is zero point one one zero six, meaning risk percentages reflect true histological reality."*
  > 
  > **Prajjwal:** *"We practice honest science: While 1D ECG signals showed classical polynomial kernels suffice, our multimodal 2D oral imaging pipeline achieves true hybrid quantum advantage over capacity-matched classical models, with permutation null p-value of zero point zero zero four nine. Every single claim is validated by one thousand two hundred and thirty-one automated tests."*

---

### Scene 5: National ABDM Integration, Scalability, & Closing Vision (2:20 - 3:00)
* **Slides Referenced:** Slide 4 (Operational Feasibility) & Slide 5 (Healthcare System Benefits)
* **On-Screen Visual:** 
  * App UI: Generating a 14-digit ABHA ID and displaying a one-click generated HL7 FHIR R4 JSON bundle.
  * Fast-track tele-triage referral SMS sent to a district hospital oncologist.
  * National map showing scalable screening across primary health centers in India.
  * Closing slide with team roster and GitHub repository QR code.
* **Speakers (Rudransh Rajveer Singh & Ishan Narayan Shukla):**
  > **Rudransh:** *"Orqis is natively integrated with India's Ayushman Bharat Digital Mission. In one tap, the app generates an HL7 FHIR Release Four DiagnosticReport bundle with standard SNOMED CT and ICD-11 coding, automatically routing high-risk patients to district hospitals within forty-eight hours."*
  > 
  > **Ishan:** *"By uniting smartphone edge computer vision, rigorous classical learning, and scalable quantum computing, Orqis transforms oral cancer screening from a late-stage death sentence into an accessible, point-of-care cure. We are Team BraKet 3.1.0. Thank you."*

---

## 3. Video Editing & Asset Guidelines

| Segment | Video Asset Required | Audio Track |
|---|---|---|
| **0:00 - 0:35** | B-roll of clinical oral exams + Orqis logo animation. | Confident, measured narrative voiceover; subtle ambient medical synth. |
| **0:35 - 1:10** | 1080p screen recording of Flutter app on physical Android device. | Clear click/haptic sound effects; crisp instructional tone. |
| **1:10 - 1:45** | 3D motion graphics of Bloch spheres and PennyLane circuit transpilation. | Dynamic technological pace; scientific graphics. |
| **1:45 - 2:20** | Direct screen capture of terminal test runs and vector matplotlib plots. | High-energy, data-driven presentation; metric callout badges. |
| **2:20 - 3:00** | Interactive ABDM JSON modal + animated map of India healthcare flow. | Inspiring, forward-looking orchestral crescendo to finish. |
