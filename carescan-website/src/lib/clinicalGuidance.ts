/**
 * Clinical Decision Support & Oral Pathology Guidance.
 *
 * Provides evidence-based clinical interpretations, care pathways,
 * differential diagnoses, and examination protocols for healthcare providers,
 * dentists, oral & maxillofacial surgeons, and screening clinics.
 *
 * Aligned with WHO Collaborating Centre for Oral Cancer & Precancer guidance
 * and American Dental Association (ADA) clinical guidelines for oral mucosal lesions.
 */

export interface ClinicalProtocolStep {
  id: string;
  category: 'examination' | 'intervention' | 'timeline' | 'counseling';
  title: string;
  action: string;
  mandatory: boolean;
}

export interface ClinicalRiskGuidance {
  band: 'LOW RISK' | 'MODERATE RISK' | 'HIGH RISK';
  triageCategory: string;
  triageLevel: 1 | 2 | 3;
  clinicalImpression: string;
  diagnosticSummary: string;
  recallWindow: string;
  urgencyLabel: string;
  urgencyTone: 'low' | 'moderate' | 'high';
  morphologicalIndicators: string[];
  differentialDiagnoses: {
    condition: string;
    description: string;
    riskNote?: string;
  }[];
  clinicalCareProtocols: ClinicalProtocolStep[];
  patientCounselingNotes: string[];
  highRiskAnatomicalSites: string[];
}

export const CLINICAL_GUIDANCE: Record<'low' | 'moderate' | 'high', ClinicalRiskGuidance> = {
  low: {
    band: 'LOW RISK',
    triageCategory: 'Category 1 · Routine Surveillance (Benign / Non-Dysplastic Mucosa)',
    triageLevel: 1,
    clinicalImpression:
      'Computer-assisted screening indicates normal epithelial architecture or typical low-suspicion benign alterations without features suggestive of epithelial dysplasia or malignant transformation.',
    diagnosticSummary:
      'Mucosal tissue exhibits uniform chromatic and structural characteristics well within normal physiological variance. Quantitative risk score is below the screening threshold. No overt morphological indicators of oral potentially malignant disorders (OPMD) or microvascular anomalies detected.',
    recallWindow: '6 – 12 months (Standard Preventive Dental Recall)',
    urgencyLabel: 'Routine / Non-Urgent',
    urgencyTone: 'low',
    morphologicalIndicators: [
      'Uniform epithelial texture and mucosal hydration',
      'Intact mucosal barrier without suspicious erythroplakia or non-scrapable white plaques',
      'Normal vascular pattern without irregular capillary tortuosity or contact bleeding',
      'Absence of submucosal induration or rolled mucosal borders',
    ],
    differentialDiagnoses: [
      {
        condition: 'Normal Anatomical Variation',
        description: 'Fordyce granules, leukoedema, prominent mucosal veins, or linea alba.',
      },
      {
        condition: 'Frictional Keratosis',
        description: 'Reversible reactive hyperkeratosis secondary to chronic mechanical friction.',
      },
      {
        condition: 'Minor Aphthous Ulceration',
        description: 'Self-limiting shallow ulcerative lesion with characteristic erythematous halo.',
      },
      {
        condition: 'Traumatic Fibroma',
        description: 'Benign reactive fibrous hyperplasia commonly associated with habitual biting.',
      },
    ],
    clinicalCareProtocols: [
      {
        id: 'doc-baseline',
        category: 'examination',
        title: 'Document Baseline Mucosal Appearance',
        action: 'Record visual mucosal status and capture intra-oral photographic record in patient chart for longitudinal comparison.',
        mandatory: true,
      },
      {
        id: 'palpation-check',
        category: 'examination',
        title: 'Routine Oral Soft Tissue Screen',
        action: 'Perform routine visual and bimanual palpation of buccal mucosa, palate, gingiva, and tongue.',
        mandatory: false,
      },
      {
        id: 'recall-schedule',
        category: 'timeline',
        title: 'Preventive Recall Interval',
        action: 'Maintain standard 6 to 12-month routine preventive dental examination recall schedule.',
        mandatory: true,
      },
      {
        id: 'patient-rule-14d',
        category: 'counseling',
        title: 'Two-Week Persistence Rule Education',
        action: 'Instruct patient to self-monitor and report promptly if any new red/white patch, non-healing sore, or swelling persists beyond 14 days.',
        mandatory: true,
      },
    ],
    patientCounselingNotes: [
      'Reassure patient that automated screening shows low suspicion for neoplastic changes.',
      'Emphasize the importance of regular dental checkups every 6–12 months.',
      'Advise patient to avoid tobacco (smoking and chewing) and reduce alcohol consumption to maintain long-term mucosal health.',
    ],
    highRiskAnatomicalSites: [
      'Lateral border of the tongue',
      'Floor of the mouth',
      'Retromolar trigone',
      'Soft palate & anterior tonsillar pillar',
    ],
  },

  moderate: {
    band: 'MODERATE RISK',
    triageCategory: 'Category 2 · Monitored Re-evaluation & Active Surveillance (Atypical / Potential OPMD)',
    triageLevel: 2,
    clinicalImpression:
      'Screening identifies atypical mucosal features exceeding standard decision threshold. Findings are consistent with an Oral Potentially Malignant Disorder (OPMD) or persistent chronic inflammatory/keratotic lesion requiring direct clinical correlation and monitored re-evaluation.',
    diagnosticSummary:
      'Tissue exhibits localized chromatic heterogeneity, surface texturing abnormalities, or mild hyperkeratosis. While not conclusively dysplastic, the lesion demonstrates suspicious optical features that warrant rigorous physical examination, elimination of potential local irritants, and scheduled short-interval follow-up.',
    recallWindow: '14 – 21 calendar days (Strict Protocol)',
    urgencyLabel: 'Monitored Follow-Up Required',
    urgencyTone: 'moderate',
    morphologicalIndicators: [
      'Localized epithelial hyperkeratosis, opacity, or chromatic mottling',
      'Irregular or ill-defined mucosal demarcation from surrounding healthy mucosa',
      'Subtle textural alterations (granular, corrugated, or plaque-like surface)',
      'Possible underlying chronic inflammation without classic acute ulcer halo',
    ],
    differentialDiagnoses: [
      {
        condition: 'Oral Leukoplakia (Homogeneous)',
        description: 'Predominantly white plaque of questionable risk; must exclude other definable entities.',
        riskNote: 'Estimated malignant transformation rate 1% – 5% annually depending on site and dysplasia.',
      },
      {
        condition: 'Oral Lichen Planus / Lichenoid Reaction',
        description: 'Chronic mucocutaneous immune-mediated condition (reticular, plaque, or erosive presentation).',
        riskNote: 'Requires periodic monitoring for erosive changes and secondary dysplasia.',
      },
      {
        condition: 'Chronic Hyperplastic Candidiasis',
        description: 'Candidal leukoplakia characterized by firm white plaques that cannot be wiped off.',
        riskNote: 'Respond to 14-day antifungal therapy; persistent lesions require biopsy.',
      },
      {
        condition: 'Persistent Traumatic Keratosis',
        description: 'Reaction to fractured cusps, rough dental restorations, or ill-fitting prostheses.',
        riskNote: 'Must resolve within 2 weeks of removing mechanical irritant.',
      },
      {
        condition: 'Oral Submucous Fibrosis (OSMF)',
        description: 'Fibrotic blanching, burning sensation, and progressive trismus (areca nut/gutkha association).',
        riskNote: 'High potential for malignant transformation (7% – 13%).',
      },
    ],
    clinicalCareProtocols: [
      {
        id: 'bimanual-palpate',
        category: 'examination',
        title: 'Bimanual Lesion Palpation',
        action: 'Palpate lesion for deep submucosal induration, depth of infiltration, tissue mobility, and tenderness.',
        mandatory: true,
      },
      {
        id: 'cervical-lymph-eval',
        category: 'examination',
        title: 'Cervical Lymph Node Examination',
        action: 'Palpate submandibular, submental, and jugulodigastric chains (Levels I–III) bilaterally for firmness or lymphadenopathy.',
        mandatory: true,
      },
      {
        id: 'eliminate-etiology',
        category: 'intervention',
        title: 'Eliminate Local Traumatic/Infectious Factors',
        action: 'Smooth sharp tooth margins, adjust prosthesis borders, and prescribe topical antifungal trial if candidiasis is suspected.',
        mandatory: true,
      },
      {
        id: 'recall-14d',
        category: 'timeline',
        title: 'Mandatory 14–21 Day Clinical Re-Evaluation',
        action: 'Schedule formal in-clinic recall in 2–3 weeks. If lesion fails to regress completely after eliminating irritants, biopsy or specialist referral is indicated.',
        mandatory: true,
      },
      {
        id: 'adjunctive-delineation',
        category: 'intervention',
        title: 'Adjunctive Diagnostic Aids (Optional)',
        action: 'Consider toluidine blue vital rinse or tissue autofluorescence (e.g. VELscope) to assist in delineating subtle borders.',
        mandatory: false,
      },
    ],
    patientCounselingNotes: [
      'Explain that the lesion requires careful monitored follow-up and does not represent a confirmed cancer diagnosis.',
      'Strongly emphasize immediate cessation of tobacco (beedis, cigarettes, gutkha, khaini) and areca nut / paan products.',
      'Schedule a confirmed follow-up appointment within 14 to 21 days; stress that attendance is critical.',
      'Instruct patient to contact clinic immediately if the lesion becomes painful, ulcerated, or bleeds spontaneously.',
    ],
    highRiskAnatomicalSites: [
      'Lateral border of the tongue (highest incidence of OSCC)',
      'Floor of the mouth (high-risk lymphatic drainage)',
      'Ventral tongue & lingual frenulum',
      'Retromolar trigone & anterior faucial pillar',
    ],
  },

  high: {
    band: 'HIGH RISK',
    triageCategory: 'Category 3 · Urgent Fast-Track Specialist Referral & Biopsy (High Suspicion of Dysplasia/OSCC)',
    triageLevel: 3,
    clinicalImpression:
      'Screening algorithm flags prominent high-risk mucosal abnormalities with marked structural heterogeneity, irregular margins, or suspicious chromatic density. Findings carry high clinical suspicion for advanced epithelial dysplasia, carcinoma-in-situ, or Oral Squamous Cell Carcinoma (OSCC). Urgent specialist referral and histopathological verification are mandatory.',
    diagnosticSummary:
      'Image feature vectors and multi-modal classifiers identified severe optical anomalies, asymmetric border delineation, and tissue characteristics associated with epithelial dysplastic proliferation or micro-invasive malignancy. Prompt diagnostic tissue biopsy is required for definitive histopathological staging.',
    recallWindow: 'Urgent Fast-Track (< 14 calendar days)',
    urgencyLabel: 'Immediate Action & Biopsy Indicated',
    urgencyTone: 'high',
    morphologicalIndicators: [
      'Erythroplakic (velvety red) or mixed speckled erythroleukoplakic plaque',
      'Persistent non-healing ulceration (> 14 days) with raised, rolled, or everted margins',
      'Distinct palpable submucosal induration, tissue fixation, or loss of mucosal compliance',
      'Exophytic, papillary, or verrucous mucosal projection',
      'Tissue friability with spontaneous bleeding on gentle contact',
    ],
    differentialDiagnoses: [
      {
        condition: 'Erythroplakia / Speckled Erythroleukoplakia',
        description: 'Fiery red patch that cannot be characterized clinically or pathologically as any other disease.',
        riskNote: 'Highest malignant transformation rate (> 85% harbor severe dysplasia, CIS, or invasive carcinoma).',
      },
      {
        condition: 'Oral Squamous Cell Carcinoma (OSCC)',
        description: 'Malignant epithelial neoplasm; early presentation often masquerades as painless red/white ulcer.',
        riskNote: 'Requires urgent clinical TNM staging and multidisciplinary oncology team management.',
      },
      {
        condition: 'Proliferative Verrucous Leukoplakia (PVL)',
        description: 'Progressive, multi-focal, non-smoking associated leukoplakia with refractory behavior.',
        riskNote: 'Extremely high long-term transformation risk (60% – 70%).',
      },
      {
        condition: 'Severe Epithelial Dysplasia / Carcinoma In Situ',
        description: 'Full-thickness architectural atypia with intact basement membrane.',
        riskNote: 'Immediate complete surgical excision or photodynamic therapy recommended.',
      },
      {
        condition: 'Deep Chronic Ulcerative Granuloma',
        description: 'Deep fungal or mycobacterial ulcer (e.g. histoplasmosis, tuberculosis); mimics carcinoma.',
        riskNote: 'Requires biopsy with special stains and microbial culture.',
      },
    ],
    clinicalCareProtocols: [
      {
        id: 'fast-track-referral',
        category: 'timeline',
        title: 'Fast-Track Specialist Referral (< 14 Days)',
        action: 'Issue immediate urgent 2-week fast-track referral to Oral & Maxillofacial Surgery (OMFS), Head & Neck Surgical Oncology, or Oral Medicine.',
        mandatory: true,
      },
      {
        id: 'incisional-biopsy',
        category: 'intervention',
        title: 'Diagnostic Tissue Biopsy Requisition',
        action: 'Perform or order incisional/punch biopsy. Target the most suspicious, indurated, or erythroplakic margin (avoid central necrotic tissue).',
        mandatory: true,
      },
      {
        id: 'cervical-staging',
        category: 'examination',
        title: 'Full Head & Neck Lymphatic Staging',
        action: 'Systematically palpate bilateral cervical lymph nodes (Levels I through V). Document number, size, mobility, and firmness of any palpable nodes.',
        mandatory: true,
      },
      {
        id: 'imaging-workup',
        category: 'intervention',
        title: 'Diagnostic Imaging Coordination',
        action: 'Coordinate contrast-enhanced CT neck or MRI if deep soft tissue invasion, bone involvement, or nodal disease is clinically suspected.',
        mandatory: false,
      },
      {
        id: 'immediate-counseling',
        category: 'counseling',
        title: 'Urgent Patient Counseling & Family Support',
        action: 'Clearly communicate the necessity of immediate specialist evaluation without inducing panic. Provide printed clinical summary and direct clinic contact.',
        mandatory: true,
      },
    ],
    patientCounselingNotes: [
      'Communicate clearly that an urgent specialist assessment and tissue sample (biopsy) are needed to determine an accurate diagnosis.',
      'Explain that early detection and prompt biopsy significantly improve therapeutic success and treatment options.',
      'Emphasize absolute cessation of tobacco, betel quid/gutkha, and alcohol immediately.',
      'Ensure the patient understands that the referral is non-negotiable and confirm appointment logistics with hospital/clinic.',
    ],
    highRiskAnatomicalSites: [
      'Lateral border of the tongue (most frequent intraoral cancer site)',
      'Floor of mouth and ventral tongue (high risk of occult nodal metastasis)',
      'Retromolar trigone and tonsillar pillars',
      'Soft palate & uvular complex',
    ],
  },
};

/** Get clinical guidance for a specific band or fallback to moderate/low. */
export function getClinicalGuidance(band: string | null | undefined): ClinicalRiskGuidance {
  if (!band) return CLINICAL_GUIDANCE.moderate;
  const upper = band.toUpperCase();
  if (upper.includes('HIGH')) return CLINICAL_GUIDANCE.high;
  if (upper.includes('MODERATE') || upper.includes('MEDIUM')) return CLINICAL_GUIDANCE.moderate;
  return CLINICAL_GUIDANCE.low;
}
