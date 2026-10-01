"""Helix Medical / HelixRAG — synthetic mini-corpus + golden eval set.

Tiny stand-in for the real 2.3M-document corpus. Deliberately seeded with a
STALE and a CURRENT warfarin+amiodarone guideline so the contraindication
thread from the lectures runs through every lab. Swap this for your own corpus
(and a 300-pair clinician golden set) for real work.
"""

# (doc_id, source_type, text)
CORPUS = [
    ("g-101", "guideline",
     "Warfarin with amiodarone: amiodarone potentiates warfarin via CYP2C9 inhibition. "
     "Reduce the warfarin dose by 30-50% on initiation and monitor INR closely for 4 weeks. "
     "This combination requires active management. (Current, 2023)"),
    ("g-044", "guideline",
     "Warfarin and amiodarone may be co-administered with routine monitoring. "
     "No special dose adjustment is generally required. (Superseded, 2014)"),
    ("d-201", "drug_label",
     "AMIODARONE HYDROCHLORIDE | NDC 0555-0902 | Drug Interactions: increases the "
     "anticoagulant effect of warfarin; reduce warfarin dose. Metformin: no clinically "
     "significant interaction reported. Dosing: load 800-1600 mg/day in divided doses."),
    ("d-202", "drug_label",
     "WARFARIN SODIUM | NDC 0056-0176 | Many drugs potentiate warfarin including amiodarone, "
     "fluconazole, and metronidazole. Monitor INR when starting or stopping interacting drugs."),
    ("r-330", "research",
     "RE-LY randomized trial (NEJM 2009): dabigatran 150 mg twice daily reduced stroke and "
     "systemic embolism versus warfarin (1.11% vs 1.69% per year, RR 0.66). See Table 2 for "
     "major bleeding outcomes (2.71% vs 3.36% per year)."),
    ("r-331", "research",
     "ARISTOTLE trial: apixaban reduced stroke versus warfarin (1.27% vs 1.60% per year) with "
     "lower major bleeding (2.13% vs 3.09% per year). Table 3 summarizes outcomes by subgroup."),
    ("s-012", "sop",
     "Helix Medical anticoagulation pathway: a pharmacist MUST verify interacting medications "
     "before warfarin initiation. Document the baseline INR and interaction review. "
     "Pathway reviewed quarterly by the Anticoagulation Stewardship Committee."),
    ("p-808", "patient_note",
     "[de-identified] 71-year-old on amiodarone for atrial fibrillation, now starting warfarin. "
     "INR target 2-3. Flagged by pharmacy for interaction review per anticoagulation pathway."),
    ("p-809", "patient_note",
     "[de-identified] 64-year-old, metformin for type 2 diabetes, started on amiodarone. "
     "No anticoagulant. No interaction flag raised."),
]

# Clinician-written question / reference-answer / source-of-truth triples.
# Real HelixRAG golden set = 300 pairs. This is enough to exercise every metric.
GOLDEN = [
    {"q": "Is it safe to co-prescribe warfarin and amiodarone?",
     "answer": "No - amiodarone potentiates warfarin; reduce the warfarin dose 30-50% and monitor INR closely.",
     "gold_doc": "g-101"},
    {"q": "Is there an interaction between amiodarone and metformin?",
     "answer": "No clinically significant interaction is documented between amiodarone and metformin.",
     "gold_doc": "d-201"},
    {"q": "What did the RE-LY trial find for dabigatran 150mg versus warfarin on stroke?",
     "answer": "Dabigatran 150mg twice daily reduced stroke and systemic embolism (1.11% vs 1.69% per year).",
     "gold_doc": "r-330"},
    {"q": "What is Helix's required step before starting a patient on warfarin?",
     "answer": "A pharmacist must verify interacting medications and document the interaction review before warfarin initiation.",
     "gold_doc": "s-012"},
    {"q": "How did apixaban compare to warfarin for major bleeding in ARISTOTLE?",
     "answer": "Apixaban had lower major bleeding than warfarin (2.13% vs 3.09% per year).",
     "gold_doc": "r-331"},
    {"q": "Which common drugs potentiate warfarin according to its label?",
     "answer": "Amiodarone, fluconazole, and metronidazole potentiate warfarin; monitor INR.",
     "gold_doc": "d-202"},
]

DOC_IDS = [d[0] for d in CORPUS]
TEXTS = [d[2] for d in CORPUS]
SOURCE_TYPE = {d[0]: d[1] for d in CORPUS}

# Course target arc (for reference in reports)
TARGETS = {"faithfulness": 0.95, "context_recall": 0.90, "cost_per_query": 0.04, "p95_latency_s": 3.0}
BASELINE = {"faithfulness": 0.71, "context_recall": 0.62, "cost_per_query": 0.11, "p95_latency_s": 6.4}
