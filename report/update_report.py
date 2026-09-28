import re

with open("report/main.tex", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update preamble: hyperref and artifact macro
old_preamble_target = r"\usepackage[hidelinks,bookmarks=true,bookmarksnumbered=true]{hyperref}"
new_preamble = r"\usepackage[hidelinks,bookmarks=true,bookmarksnumbered=true,plainpages=false,pdfpagelabels=true,hypertexnames=false]{hyperref}"
content = content.replace(old_preamble_target, new_preamble)

old_artifact = r"\newcommand{\artifact}[1]{\texttt{\footnotesize #1}}"
new_artifact = r"\newcommand{\artifact}[1]{\nolinkurl{#1}}"
content = content.replace(old_artifact, new_artifact)

# 2. Update Title Page to include Team BraKet 3.1.0 Roster
new_titlepage = r"""\begin{titlepage}
\centering
\vspace*{0.4cm}

{\color{csteal}\rule{\textwidth}{2.2pt}}
\vspace{0.35cm}

{\Huge \bfseries \color{csteal} CareScan / Braket 3.1.0}\\[0.30cm]
{\Large \bfseries A Hybrid Quantum--Classical Biomedical Screening Platform and a Rigorously Measured Quantum Null}\\[0.20cm]
{\large \itshape Engineering Architecture, Quality Gates, Patient-Disjoint Validation, and Multi-Family QML Evaluation}

\vspace{0.25cm}
{\color{csteal}\rule{\textwidth}{1.0pt}}

\vspace{0.5cm}
\textbf{Smart India Hackathon 2026 --- Problem Statement SIH26139}\\
\textbf{Category:} MedTech / Bio-Computing / Quantum Machine Learning\\
\textbf{Focus:} Early Oral Cancer Screening from Smartphone Photographs \& Biomedical Signal Extension

\vspace{0.5cm}
% =====================================================================
% TEAM ROSTER BOX: TEAM BRAKET 3.1.0
% =====================================================================
\fcolorbox{csteal}{cslight}{
\begin{minipage}{0.95\textwidth}
\centering
\vspace{0.15cm}
{\large\bfseries\color{csteal} Team BraKet 3.1.0 --- Project Engineering Roster}\\[0.2cm]
\scriptsize
\begin{tabularx}{\linewidth}{>{\bfseries\color{csteal}}p{3.8cm} >{\raggedright\arraybackslash}X}
Ishan Aran Shukla & \textbf{Team Lead \& Lead Technical Auditor} \newline Machine Learning Architecture, QML Algorithms, Pre-Registration Gating \\
\addlinespace[0.15cm]
Pratyaksh Ranjan & \textbf{Software Architect \& Mobile Systems Lead} \newline Flutter Client Engineering, Offline-First SQLite, Camera Lifecycle \\
\addlinespace[0.15cm]
Clinical AI Specialist & \textbf{Computer Vision \& Image Quality Assurance Lead} \newline MobileNetV3 Lesion Localization, 7-Parameter Deterministic QC Gate \\
\addlinespace[0.15cm]
Quantum Algorithm Engineer & \textbf{Quantum Compilation \& Hardware Benchmarking} \newline Havlicek ZZ Maps, VQC Ansätze, State-Prep CNOT Complexity Analysis \\
\addlinespace[0.15cm]
Biomedical Signal Lead & \textbf{Biomedical Signal Processing Lead} \newline 12-Lead ECG Processing, 1D ResNet Architecture, PTB-XL Benchmarking \\
\addlinespace[0.15cm]
Full-Stack Systems Lead & \textbf{FastAPI Backend, FHIR R4 \& Cloud Security} \newline 20 REST Endpoints, HL7 FHIR Interoperability, Test Suite Automation \\
\end{tabularx}
\vspace{0.15cm}
\end{minipage}
}

\vspace{0.5cm}
\scriptsize
\begin{tabular}{ll}
\textbf{Repository State:} & Tag \texttt{v3.1.0-audit} (September 2026) \\
\textbf{Audit Execution Date:} & 27--28 September 2026 \\
\textbf{Environment of Record:} & Python 3.12.10, NumPy 1.26.4, Flutter 3.47.2 (stable) \\
\textbf{Hardware Target:} & Standard Commodity CPU Inference (Demonstrated on Localhost) \\
\textbf{Test Verification:} & 1,231 Automated Tests Passing (1,038 Backend + 193 Mobile) \\
\textbf{Primary Empirical Finding:} & Classical C7 PR-AUC = 0.913038; QML Headline Fusion Delta = +0.000641 (Null)
\end{tabular}

\vspace{0.4cm}
\begin{minipage}{0.92\textwidth}
\tiny
\textbf{Core Scientific Stance of this Report:}\\
This document is a technical and empirical audit of the CareScan codebase. It is \textbf{not} a speculative proposal or marketing brochure. Every metric reported here is transcribed from machine-generated evaluation artifacts. We report an end-to-end smartphone triage application alongside a disciplined quantum evaluation across seven distinct circuit families. In our matched, pre-registered benchmarks, the evaluated quantum feature maps \textbf{did not outperform} matched classical controls. Rather than obscuring this negative result, this report documents the rigorous statistical controls and circuit proofs that make the null finding a genuine scientific contribution.
\end{minipage}

\vfill
{\footnotesize Smart India Hackathon 2026 $\cdot$ Team BraKet 3.1.0 Technical Submission}
\end{titlepage}"""

pattern_title = re.compile(r"\\begin\{titlepage\}.*?\\end\{titlepage\}", re.DOTALL)
content = pattern_title.sub(lambda m: new_titlepage, content, count=1)

# 3. Replace Chapter 2 Figure with fig1_pipeline_architecture.png
new_fig_ch2 = r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig1_pipeline_architecture.png}
\caption{CareScan / Braket 3.1.0 End-to-End System Pipeline Architecture. The modular pipeline decouples client-side guidance, deterministic server quality gating, MobileNetV3 lesion localization, multimodal classical feature extraction, quantum feature mapping under the same-shape contract, calibrated classification, and HL7 FHIR export.}
\label{fig:pipeline_arch}
\end{figure}"""
old_fig_ch2 = re.compile(r"\\begin\{figure\}\[H\].*?\\caption\{CareScan Seven-Stage Screening and Triage Architecture.*?\\end\{figure\}", re.DOTALL)
content = old_fig_ch2.sub(lambda m: new_fig_ch2, content, count=1)

# 4. Replace Chapter 3 Figure with fig2_data_provenance_funnel.png
new_fig_ch3 = r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig2_data_provenance_funnel.png}
\caption{Dataset Provenance, Curation Funnel, and Leak-Free Patient-Disjoint Partitioning ($k=0$ Overlap). (A) Deduplication of 7,731 raw records into 2,469 canonical unique files, filtered into 2,436 working images across 328 patients via the 33-exclusion ledger. (B) Patient-disjoint partition distribution across Train, Validation, and untouched Test sets.}
\label{fig:cohort_funnel}
\end{figure}"""
old_fig_ch3 = re.compile(r"\\begin\{figure\}\[H\].*?\\caption\{Oral Cancer Cohort Derivation Funnel.*?\\end\{figure\}", re.DOTALL)
content = old_fig_ch3.sub(lambda m: new_fig_ch3, content, count=1)

# 5. Insert fig3_lesion_localization_sample.png in Chapter 5/6
new_fig_ch5_loc = r"""\caption{Frozen Localizer Performance on Held-Out Test Partition.}
\label{tab:localizer_test}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig3_lesion_localization_sample.png}
\caption{Representative Clinical Mucosal Photograph Processed through the CareScan Triage Pipeline. Overlay demonstrates the real-time QA telemetry HUD (focus, luminance, clipping, and red-tissue checks) alongside the MobileNetV3-Small predicted lesion bounding box (pinned checkpoint SHA-256: \nolinkurl{27d6036e...}, evaluated IoU 0.5279, acceptance rate 0.9816) and resulting high-risk referral banner.}
\label{fig:lesion_localization_sample}
\end{figure}"""
target_loc = r"\caption{Frozen Localizer Performance on Held-Out Test Partition.}" + "\n" + r"\label{tab:localizer_test}" + "\n" + r"\end{table}"
content = content.replace(target_loc, new_fig_ch5_loc)

# 6. Fix Table 7 (QC thresholds) column widths
old_tab_qc = r"\begin{tabularx}{\textwidth}{llrrX}"
new_tab_qc = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash\ttfamily\tiny}p{3.8cm} >{\raggedright\arraybackslash}p{1.4cm} >{\raggedright\arraybackslash}p{2.2cm} >{\raggedright\arraybackslash}X}"
content = content.replace(old_tab_qc, new_tab_qc)

# 7. Fix Table 8 (Classical oral val) column widths and add Figures 4 & 5
old_tab_c7 = r"\begin{tabularx}{\textwidth}{lrrrrrr}"
new_tab_c7 = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X r >{\raggedright\arraybackslash}p{2.4cm} rrrr}"
content = content.replace(old_tab_c7, new_tab_c7)

new_ch7_fig4 = r"""\caption{Validation Performance across Classical Feature Blocks under Primary Condition ($A_\text{lesion\_polygon}$, 52 Images, 21 Positive).}
\label{tab:classical_oral_val}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig4_pr_roc_curves.png}
\caption{Precision-Recall and ROC Curves across Classical Feature Blocks under Primary Condition $A_\text{lesion\_polygon}$ ($N=52$, 21 Positives, Prevalence 0.4038). Candidate C7 achieves the highest overall discrimination (PR-AUC 0.913038, ROC-AUC 0.933948) compared to single-modality baselines C1--C6.}
\label{fig:pr_roc_curves}
\end{figure}"""
target_c7_tab = r"\caption{Validation Performance across Classical Feature Blocks under Primary Condition ($A_\text{lesion\_polygon}$, 52 Images, 21 Positive).}" + "\n" + r"\label{tab:classical_oral_val}" + "\n" + r"\end{table}"
content = content.replace(target_c7_tab, new_ch7_fig4)

target_ch7_cm = r"\item \textbf{Confusion Matrix:} True Positive = 19, False Positive = 6, True Negative = 25, False Negative = 2."
new_ch7_cm = r"""\item \textbf{Confusion Matrix:} True Positive = 19, False Positive = 6, True Negative = 25, False Negative = 2.
\end{itemize}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig5_confusion_matrix.png}
\caption{Validation Confusion Matrix and Operating Point Diagnostic Performance for Primary Candidate C7 ($N=52$). The calibrated model identifies 19 of 21 malignant lesions (sensitivity 90.48\%) while correctly clearing 25 of 31 benign lesions (specificity 80.65\%), statistically verified via patient-blocked permutation ($p = 0.004975$).}
\label{fig:confusion_matrix}
\end{figure}"""
content = content.replace(target_ch7_cm + "\n" + r"\end{itemize}", new_ch7_cm)

# 8. Fix Table 9 (Oral test results)
old_tab_test = r"\begin{tabular}{lrrrrr}"
new_tab_test = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X rrrrr}"
content = content.replace(old_tab_test, new_tab_test)
content = content.replace(r"\end{tabular}" + "\n" + r"\caption{Held-Out Oral Test Partition Performance", r"\end{tabularx}" + "\n" + r"\caption{Held-Out Oral Test Partition Performance")

# 9. Fix Table 10 (CX scaling) and insert fig6_quantum_stateprep_scaling.png
old_tab_cx = r"\begin{tabular}{lrr}"
new_tab_cx = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X rr}"
content = content.replace(old_tab_cx, new_tab_cx)
content = content.replace(r"\end{tabular}" + "\n" + r"\caption{CNOT State Preparation Scaling vs Parameterized Ansatz Depth.}",
                          r"\end{tabularx}" + "\n" + r"\caption{CNOT State Preparation Scaling vs Parameterized Ansatz Depth.}")

target_ch8_cx_fig = r"\caption{CNOT State Preparation Scaling vs Parameterized Ansatz Depth.}" + "\n" + r"\label{tab:cx_scaling}" + "\n" + r"\end{table}"
new_ch8_cx_fig = target_ch8_cx_fig + r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig6_quantum_stateprep_scaling.png}
\caption{Exponential State Preparation Gate Scaling vs Parameterized Ansatz Gate Depth. For 16-qubit amplitude encoding, exact state synthesis requires 62,940 CNOT gates (upper bound 131,038 CNOTs), exceeding current physical NISQ coherence budgets by two orders of magnitude and creating a $3,934\times$ gate imbalance relative to the shallow ansatz.}
\label{fig:quantum_scaling}
\end{figure}"""
content = content.replace(target_ch8_cx_fig, new_ch8_cx_fig)

# 10. Fix Table 11 (E3 QML results)
old_tab_e3 = r"\begin{tabularx}{\textwidth}{lrrX}"
new_tab_e3 = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{4.2cm} rr >{\raggedright\arraybackslash}X}"
content = content.replace(old_tab_e3, new_tab_e3)

# 11. Replace old pgfplots forest plot with fig7_paired_bootstrap_forest_plot.png
new_forest_fig = r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig7_paired_bootstrap_forest_plot.png}
\caption{Forest Plot of Pre-Registered Empirical Hypotheses with 95\% Patient-Clustered Bootstrap Confidence Intervals. Headline QML fusion yields a verified null ($\Delta = +0.000641$, CI spans zero); single-arm Havlicek ZZ map suffers a statistically significant deficit ($\Delta = -0.124457$, CI excludes zero); and Classical C7 exhibits significant feature signal ($p = 0.004975$).}
\label{fig:forest_plot}
\end{figure}"""
old_forest_fig = re.compile(r"\\begin\{figure\}\[H\].*?\\caption\{Forest Plot of Four Decisive Paired-Bootstrap Comparisons.*?\\end\{figure\}", re.DOTALL)
content = old_forest_fig.sub(lambda m: new_forest_fig, content, count=1)

# 12. Fix Table 12 (ECG leaderboard) and add fig8_ecg_multimodal_leaderboard.png
old_tab_ecg = r"\begin{tabular}{rllr}"
new_tab_ecg = r"\begin{tabularx}{\textwidth}{r >{\raggedright\arraybackslash\ttfamily\small}p{3.2cm} >{\raggedright\arraybackslash}X r}"
content = content.replace(old_tab_ecg, new_tab_ecg)
content = content.replace(r"\end{tabular}" + "\n" + r"\caption{PTB-XL Classical Baseline Hierarchy",
                          r"\end{tabularx}" + "\n" + r"\caption{PTB-XL Classical Baseline Hierarchy")

target_ch11_ecg_fig = r"\caption{PTB-XL Classical Baseline Hierarchy under Strict Patient-Disjoint Partitions (Fold 9, N=2,146).}" + "\n" + r"\label{tab:ecg_leaderboard}" + "\n" + r"\end{table}"
new_ch11_ecg_fig = target_ch11_ecg_fig + r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig8_ecg_multimodal_leaderboard.png}
\caption{13-Arm Multimodal Biomedical Signal Benchmark on the PTB-XL ECG Corpus ($N=19,601$ Verified Records across 16,965 Patients). Supervised 1D-ResNet achieves dominant discriminatory performance (ROC-AUC 0.940234); classical and hybrid fusion converge at $\approx 0.884$, while fixed quantum feature maps experience substantial degradation.}
\label{fig:ecg_leaderboard_fig}
\end{figure}"""
content = content.replace(target_ch11_ecg_fig, new_ch11_ecg_fig)

# 13. Insert fig9_mobile_application_mockup.png in Mobile Application Architecture chapter
target_ch12_mobile_fig = r"\chapter{Mobile Application Architecture and Implementation}"
new_ch12_mobile_fig = r"""\chapter{Mobile Application Architecture and Implementation}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig9_mobile_application_mockup.png}
\caption{CareScan Flutter Mobile Application Architecture and Screening Workflow. The five-step user journey comprises offline patient intake, real-time smart camera guidance, deterministic quality gate validation, MobileNetV3 lesion localization, and calibrated tri-band risk triage with HL7 FHIR R4 export.}
\label{fig:mobile_mockup}
\end{figure}"""
content = content.replace(target_ch12_mobile_fig, new_ch12_mobile_fig)

# 14. Fix long strings in Security chapter
content = content.replace(r'\texttt{"insecure-default-key-for-dev-only..."}', r'\nolinkurl{"insecure-default-key-for-dev-only..."}')

# 15. Fix Table 15 (Security controls) column widths
old_tab_sec = r"\begin{tabularx}{\textwidth}{llX}"
new_tab_sec = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{3.2cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{4.8cm}}"
content = content.replace(old_tab_sec, new_tab_sec)

# 16. Fix Table 16 (Traceability Matrix in Appendix B) column widths to prevent page overflow
old_tab_app_b = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{4.0cm} >{\raggedright\arraybackslash}p{5.2cm} >{\raggedright\arraybackslash}X}"
new_tab_app_b = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{4.2cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{4.8cm}}"
content = content.replace(old_tab_app_b, new_tab_app_b)

with open("report/main.tex", "w", encoding="utf-8") as f:
    f.write(content)

print("report/main.tex successfully updated!")
