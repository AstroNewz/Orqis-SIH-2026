import sys

with open("report/main_original.tex", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Preamble hyperref
target_hyp = r"\usepackage[hidelinks,bookmarks=true,bookmarksnumbered=true]{hyperref}"
assert target_hyp in text, "hyperref target not found!"
text = text.replace(target_hyp, r"\usepackage[hidelinks,bookmarks=true,bookmarksnumbered=true,plainpages=false,pdfpagelabels=true,hypertexnames=false]{hyperref}")

# 2. Preamble artifact macro
target_art = r"\newcommand{\artifact}[1]{\texttt{\footnotesize #1}}"
assert target_art in text, "artifact target not found!"
text = text.replace(target_art, r"\newcommand{\artifact}[1]{\nolinkurl{#1}}")

# 3. Title Page
title_start = r"\begin{titlepage}"
title_end = r"\end{titlepage}"
assert title_start in text and title_end in text, "titlepage not found!"
idx1 = text.find(title_start)
idx2 = text.find(title_end) + len(title_end)

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

\vspace{0.45cm}
\textbf{Smart India Hackathon 2026 --- Problem Statement: SIH26139}\\
\textbf{Category:} MedTech / Bio-Computing / Quantum Machine Learning\\
\textbf{Focus:} Early Oral Cancer Screening from Smartphone Photographs \& Biomedical Signal Extension

\vspace{0.45cm}
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

\vspace{0.45cm}
\scriptsize
\begin{tabular}{ll}
\textbf{Repository State:} & Tag \texttt{v3.1.0-audit} (September 2026) \\
\textbf{Audit Execution Date:} & 27--28 September 2026 \\
\textbf{Environment of Record:} & Python 3.12.10, NumPy 1.26.4, Flutter 3.47.2 (stable) \\
\textbf{Hardware Target:} & Standard Commodity CPU Inference (Demonstrated on Localhost) \\
\textbf{Test Verification:} & 1,231 Automated Tests Passing (1,038 Backend + 193 Mobile) \\
\textbf{Primary Empirical Finding:} & Classical C7 PR-AUC = 0.913038; QML Headline Fusion Delta = +0.000641 (Null)
\end{tabular}

\vspace{0.35cm}
\begin{minipage}{0.92\textwidth}
\tiny
\textbf{Core Scientific Stance of this Report:}\\
This document is a technical and empirical audit of the CareScan codebase. It is \textbf{not} a speculative proposal or marketing brochure. Every metric reported here is transcribed from machine-generated evaluation artifacts. We report an end-to-end smartphone triage application alongside a disciplined quantum evaluation across seven distinct circuit families. In our matched, pre-registered benchmarks, the evaluated quantum feature maps \textbf{did not outperform} matched classical controls. Rather than obscuring this negative result, this report documents the rigorous statistical controls and circuit proofs that make the null finding a genuine scientific contribution.
\end{minipage}

\vfill
{\footnotesize Smart India Hackathon 2026 $\cdot$ Team BraKet 3.1.0 Technical Submission}
\end{titlepage}"""

text = text[:idx1] + new_titlepage + text[idx2:]

# 4. Chapter 2 Figure: replace TikZ with fig1_pipeline_architecture.png
fig_ch2_start = r"\begin{figure}[H]"
fig_ch2_end = r"\label{fig:pipeline_arch}" + "\n" + r"\end{figure}"
i1 = text.find(fig_ch2_start)
i2 = text.find(fig_ch2_end, i1) + len(fig_ch2_end)
assert i1 != -1 and i2 != -1, "Chapter 2 figure not found!"

new_fig_ch2 = r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig1_pipeline_architecture.png}
\caption{CareScan / Braket 3.1.0 End-to-End System Pipeline Architecture. The modular pipeline decouples client-side guidance, deterministic server quality gating, MobileNetV3 lesion localization, multimodal classical feature extraction, quantum feature mapping under the same-shape contract, calibrated classification, and HL7 FHIR export.}
\label{fig:pipeline_arch}
\end{figure}"""
text = text[:i1] + new_fig_ch2 + text[i2:]

# 5. Chapter 3 Figure: replace funnel TikZ with fig2_data_provenance_funnel.png
ch3_pos = text.find(r"\chapter{Datasets, Quality Control, and Cohort Construction}")
fig_ch3_start = r"\begin{figure}[H]"
i1 = text.find(fig_ch3_start, ch3_pos)
fig_ch3_end = r"\label{fig:cohort_funnel}" + "\n" + r"\end{figure}"
i2 = text.find(fig_ch3_end, i1) + len(fig_ch3_end)
assert i1 != -1 and i2 != -1, "Chapter 3 figure not found!"

new_fig_ch3 = r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig2_data_provenance_funnel.png}
\caption{Dataset Provenance, Curation Funnel, and Leak-Free Patient-Disjoint Partitioning ($k=0$ Overlap). (A) Deduplication of 7,731 raw records into 2,469 canonical unique files, filtered into 2,436 working images across 328 patients via the 33-exclusion ledger. (B) Patient-disjoint partition distribution across Train, Validation, and untouched Test sets.}
\label{fig:cohort_funnel}
\end{figure}"""
text = text[:i1] + new_fig_ch3 + text[i2:]

# 6. Insert Fig 3 (Lesion Localization) under Table 5 (localizer_test)
loc_tab_end = r"\caption{Frozen Localizer Performance on Held-Out Test Partition.}" + "\n" + r"\label{tab:localizer_test}" + "\n" + r"\end{table}"
assert loc_tab_end in text, "Localizer table not found!"

new_loc_with_fig = loc_tab_end + r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig3_lesion_localization_sample.png}
\caption{Representative Clinical Mucosal Photograph Processed through the CareScan Triage Pipeline. Overlay demonstrates the real-time QA telemetry HUD (focus, luminance, clipping, and red-tissue checks) alongside the MobileNetV3-Small predicted lesion bounding box (pinned checkpoint SHA-256: \nolinkurl{27d6036e...}, evaluated IoU 0.5279, acceptance rate 0.9816) and resulting high-risk referral banner.}
\label{fig:lesion_localization_sample}
\end{figure}"""
text = text.replace(loc_tab_end, new_loc_with_fig)

# 7. Fix Table 7 (QC thresholds)
qc_tab_old = r"\begin{tabularx}{\textwidth}{llrrX}"
assert qc_tab_old in text, "QC table not found!"
qc_tab_new = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash\ttfamily\tiny}p{3.8cm} >{\raggedright\arraybackslash}p{1.4cm} >{\raggedright\arraybackslash}p{2.2cm} >{\raggedright\arraybackslash}X}"
text = text.replace(qc_tab_old, qc_tab_new)

# 8. Fix Table 8 (Classical oral val) and add Fig 4 (PR & ROC)
c7_tab_old = r"\begin{tabularx}{\textwidth}{lrrrrrr}"
assert c7_tab_old in text, "C7 table not found!"
c7_tab_new = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X r >{\raggedright\arraybackslash}p{2.4cm} rrrr}"
text = text.replace(c7_tab_old, c7_tab_new)

c7_tab_end = r"\caption{Validation Performance across Classical Feature Blocks under Primary Condition ($A_\text{lesion\_polygon}$, 52 Images, 21 Positive).}" + "\n" + r"\label{tab:classical_oral_val}" + "\n" + r"\end{table}"
assert c7_tab_end in text, "C7 table end not found!"

new_c7_with_fig4 = c7_tab_end + r"""

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig4_pr_roc_curves.png}
\caption{Precision-Recall and ROC Curves across Classical Feature Blocks under Primary Condition $A_\text{lesion\_polygon}$ ($N=52$, 21 Positives, Prevalence 0.4038). Candidate C7 achieves the highest overall discrimination (PR-AUC 0.913038, ROC-AUC 0.933948) compared to single-modality baselines C1--C6.}
\label{fig:pr_roc_curves}
\end{figure}"""
text = text.replace(c7_tab_end, new_c7_with_fig4)

# 9. Insert Fig 5 (Confusion Matrix) after C7 metrics list
cm_bullet = r"\item \textbf{Confusion Matrix:} True Positive = 19, False Positive = 6, True Negative = 25, False Negative = 2." + "\n" + r"\end{itemize}"
assert cm_bullet in text, "CM bullet not found!"

new_cm_with_fig5 = cm_bullet + r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig5_confusion_matrix.png}
\caption{Validation Confusion Matrix and Operating Point Diagnostic Performance for Primary Candidate C7 ($N=52$). The calibrated model identifies 19 of 21 malignant lesions (sensitivity 90.48\%) while correctly clearing 25 of 31 benign lesions (specificity 80.65\%), statistically verified via patient-blocked permutation ($p = 0.004975$).}
\label{fig:confusion_matrix}
\end{figure}"""
text = text.replace(cm_bullet, new_cm_with_fig5)

# 10. Fix Table 9 (Oral test results)
test_tab_old = r"\begin{tabular}{lrrrrr}"
assert test_tab_old in text, "Test table not found!"
test_tab_new = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X rrrrr}"
text = text.replace(test_tab_old, test_tab_new)
text = text.replace(r"\end{tabular}" + "\n" + r"\caption{Held-Out Oral Test Partition Performance", r"\end{tabularx}" + "\n" + r"\caption{Held-Out Oral Test Partition Performance")

# 11. Fix Table 10 (CX scaling) and insert Fig 6 (Quantum CNOT Scaling)
cx_tab_old = r"\begin{tabular}{lrr}"
assert cx_tab_old in text, "CX table not found!"
cx_tab_new = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X rr}"
text = text.replace(cx_tab_old, cx_tab_new)
text = text.replace(r"\end{tabular}" + "\n" + r"\caption{CNOT State Preparation Scaling vs Parameterized Ansatz Depth.}",
                    r"\end{tabularx}" + "\n" + r"\caption{CNOT State Preparation Scaling vs Parameterized Ansatz Depth.}")

cx_tab_end = r"\caption{CNOT State Preparation Scaling vs Parameterized Ansatz Depth.}" + "\n" + r"\label{tab:cx_scaling}" + "\n" + r"\end{table}"
assert cx_tab_end in text, "CX tab end not found!"

new_cx_with_fig6 = cx_tab_end + r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig6_quantum_stateprep_scaling.png}
\caption{Exponential State Preparation Gate Scaling vs Parameterized Ansatz Gate Depth. For 16-qubit amplitude encoding, exact state synthesis requires 62,940 CNOT gates (upper bound 131,038 CNOTs), exceeding current physical NISQ coherence budgets by two orders of magnitude and creating a $3,934\times$ gate imbalance relative to the shallow ansatz.}
\label{fig:quantum_scaling}
\end{figure}"""
text = text.replace(cx_tab_end, new_cx_with_fig6)

# 12. Fix Table 11 (E3 QML results)
e3_tab_old = r"\begin{tabularx}{\textwidth}{lrrX}"
assert e3_tab_old in text, "E3 table not found!"
e3_tab_new = r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{4.2cm} rr >{\raggedright\arraybackslash}X}"
text = text.replace(e3_tab_old, e3_tab_new)

# 13. Replace old pgfplots forest plot in Chapter 10 with Fig 7
sec_forest_str = r"\section{The Centrepiece Forest Plot: Four Decisive Comparisons}"
ch_forest_pos = text.find(sec_forest_str)
assert ch_forest_pos != -1, "Forest plot section not found!"

fig_forest_start = text.find(r"\begin{figure}[H]", ch_forest_pos)
fig_forest_label = text.find(r"\label{fig:forest_plot}", fig_forest_start)
fig_forest_end = text.find(r"\end{figure}", fig_forest_label) + len(r"\end{figure}")
assert fig_forest_start != -1 and fig_forest_end != -1, "Forest plot figure boundaries not found!"

new_fig7 = r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig7_paired_bootstrap_forest_plot.png}
\caption{Forest Plot of Pre-Registered Empirical Hypotheses with 95\% Patient-Clustered Bootstrap Confidence Intervals. Headline QML fusion yields a verified null ($\Delta = +0.000641$, CI spans zero); single-arm Havlicek ZZ map suffers a statistically significant deficit ($\Delta = -0.124457$, CI excludes zero); and Classical C7 exhibits significant feature signal ($p = 0.004975$).}
\label{fig:forest_plot}
\end{figure}"""

text = text[:fig_forest_start] + new_fig7 + text[fig_forest_end:]

# 14. Fix Table 12 (ECG leaderboard) and add Fig 8 (ECG Leaderboard)
ecg_tab_old = r"\begin{tabular}{rllr}"
assert ecg_tab_old in text, "ECG table not found!"
ecg_tab_new = r"\begin{tabularx}{\textwidth}{r >{\raggedright\arraybackslash\ttfamily\small}p{3.2cm} >{\raggedright\arraybackslash}X r}"
text = text.replace(ecg_tab_old, ecg_tab_new)
text = text.replace(r"\end{tabular}" + "\n" + r"\caption{Full Classical Leaderboard on PTB-XL ECG Validation Fold 9 ($N=2,146$).}",
                    r"\end{tabularx}" + "\n" + r"\caption{Full Classical Leaderboard on PTB-XL ECG Validation Fold 9 ($N=2,146$).}")

ecg_tab_end = r"\caption{Full Classical Leaderboard on PTB-XL ECG Validation Fold 9 ($N=2,146$).}" + "\n" + r"\label{tab:ecg_leaderboard}" + "\n" + r"\end{table}"
assert ecg_tab_end in text, "ECG tab end not found!"

new_ecg_with_fig8 = ecg_tab_end + r"""

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig8_ecg_multimodal_leaderboard.png}
\caption{13-Arm Multimodal Biomedical Signal Benchmark on the PTB-XL ECG Corpus ($N=19,601$ Verified Records across 16,965 Patients). Supervised 1D-ResNet achieves dominant discriminatory performance (ROC-AUC 0.940234); classical and hybrid fusion converge at $\approx 0.884$, while fixed quantum feature maps experience substantial degradation.}
\label{fig:ecg_leaderboard_fig}
\end{figure}"""
text = text.replace(ecg_tab_end, new_ecg_with_fig8)

# 15. Insert Fig 9 (Mobile Application Mockup) into Mobile App chapter
ch12_title = r"\chapter{Mobile Application Architecture and Implementation}"
assert ch12_title in text, "Mobile app chapter title not found!"

new_ch12_with_fig9 = ch12_title + r"""

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig9_mobile_application_mockup.png}
\caption{CareScan Flutter Mobile Application Architecture and Screening Workflow. The five-step user journey comprises offline patient intake, real-time smart camera guidance, deterministic quality gate validation, MobileNetV3 lesion localization, and calibrated tri-band risk triage with HL7 FHIR R4 export.}
\label{fig:mobile_mockup}
\end{figure}"""
text = text.replace(ch12_title, new_ch12_with_fig9)

# 16. Fix security long strings
sec_str_old = r'\texttt{"insecure-default-key-for-dev-only..."}'
text = text.replace(sec_str_old, r'\nolinkurl{"insecure-default-key-for-dev-only..."}')

# 17. Fix Table 15 (Security controls)
sec_pos = text.find(r"\chapter{Security, Privacy, and Responsible AI}")
sec_tab_pos = text.find(r"\begin{tabularx}{\textwidth}{llX}", sec_pos)
assert sec_tab_pos != -1, "Security table position not found!"
text = text[:sec_tab_pos] + r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{3.2cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{4.8cm}}" + text[sec_tab_pos + len(r"\begin{tabularx}{\textwidth}{llX}"):]

# 18. Fix Table 16 (Traceability matrix) in Appendix B
appb_pos = text.find(r"\chapter{Machine-Generated Artifact Index and Traceability Matrix}")
appb_tab_pos = text.find(r"\begin{tabularx}{\textwidth}{llX}", appb_pos)
assert appb_tab_pos != -1, "Appendix B table position not found!"
text = text[:appb_tab_pos] + r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{4.2cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{4.8cm}}" + text[appb_tab_pos + len(r"\begin{tabularx}{\textwidth}{llX}"):]

with open("report/main.tex", "w", encoding="utf-8") as f:
    f.write(text)

print(f"COMPLETE SUCCESS! Wrote report/main.tex ({len(text)} bytes, {len(text.splitlines())} lines)")
