# EdgeTriage — M.Tech Project Pack

**EdgeTriage: An Offline-First, Uncertainty-Aware Edge-AI System for Predicting Hospital Admission at Emergency Department Triage**

Student: Gagan Chandra · M.Tech CSE (AI & Data Science) · Roll No. 2026KPAD1017
Duration: 12 weeks · Type: Solo research + working prototype

---

## One-paragraph summary

EdgeTriage predicts, at the moment a nurse first sees an emergency patient, the probability that the patient will need hospital admission. It is designed for hospitals where staff, EHR systems and internet are limited. The model runs fully on a local device (laptop, low-end PC or phone browser) and sends zero bytes of patient data over the network. It shows *why* it gave its estimate, and it says "uncertain — clinician review" when it should not be trusted. It is built and evaluated on the public Yale ED dataset (560,486 visits, 972 variables, 3 hospitals).

## How to read this pack

| # | File | What it contains | Read when |
|---|------|------------------|-----------|
| 0 | `README.md` | This index and summary | First |
| 1 | `01_proposal.md` | Title, abstract, problem, research gap, objectives, research questions, contributions, scope | Before meeting supervisor |
| 2 | `02_dataset_and_features.md` | Dataset facts, feature tiers, leakage rules, data splits | Week 1–4 |
| 3 | `03_methodology.md` | Full research method: models, calibration, uncertainty, explainability, compression, robustness | Week 4–9 |
| 4 | `04_system_design.md` | Offline-first architecture, tech stack, app screens, live demo script | Week 9–11 |
| 5 | `05_evaluation_plan.md` | All metrics, result tables to fill in, success criteria, statistical tests | Throughout |
| 6 | `06_timeline_and_risks.md` | 12-week plan, milestones, risk register, weekly update template | Planning |
| 7 | `07_limitations_and_ethics.md` | Limitations, ethics, data licence, responsible-AI statement | Writing |
| 8 | `08_getting_started.md` | Environment setup, repo structure, Week-1 code, checklists | Day 1 |
| 9 | `09_dissertation_outline.md` | Chapter structure, figures list, viva questions | Week 10–12 |
| 10 | `10_references.md` | Core references and resources | Writing |

## What improved from the first version

| Area | First version | This version |
|------|---------------|--------------|
| Novelty | Train 4 models and build an app (largely repeats Hong et al., 2018) | Five clear contributions the original paper did not study: nurse-only input tier, cross-hospital validation, conformal uncertainty with abstention, missing-vital robustness, measured edge deployment |
| India relevance | Mentioned only as a limitation | **Tier-1 model uses no ESI and no EHR history**, because most Indian EDs have neither. This is the main model, not an afterthought |
| Generalisation | Single random split | **Leave-one-hospital-out** validation across the 3 Yale sites (academic trauma centre, community hospital, free-standing ED) |
| Uncertainty | "Display uncertainty" (undefined) | Calibration (isotonic/Platt) + **split conformal prediction** with an explicit "Uncertain → clinician review" output, tested **under hospital shift** with a local-adaptation curve |
| Clinical usefulness | Accuracy/AUC only | Adds **decision-curve analysis** (net benefit), calibration curves, sensitivity at fixed specificity |
| Edge AI | "Offline" claimed | **Measured**: artifact size, p50/p95 latency, peak RAM on a resource-limited device, and network bytes per prediction (target: 0) |
| Chief complaint | Treated as free text | Correctly handled as the dataset's **200 coded categories** (searchable dropdown in app) |
| Sensitive attributes | Not discussed | Race, religion, insurance etc. **excluded from model inputs**, used only for a fairness audit |
| Models | LR, RF, XGBoost, CatBoost | + **LightGBM** (small, fast) and optional **EBM** glass-box model for in-browser explanations |
| Data split | "Patient-level if possible" | Explicit plan for the fact that the public file may lack patient IDs (202,953 patients over 560,486 visits) |
| Plan | Week list | Week list + deliverables + milestones + risk register + buffer + dissertation outline |

## Revision 2 — after novelty check (Nov 2026)

| Finding | Change made |
|---------|-------------|
| **Abdulai et al. (2025)** already applied conformal "Don't know" prediction to the **same Yale dataset** (random split only) | Conformal on a random split is now framed as a *replication*. RQ3 moved to the open question: **does uncertainty survive a new hospital, and how many local visits repair it?** |
| **Dave et al. (2026)** already did calibrated, explainable admission prediction at triage | Calibration/SHAP kept as necessary components, not claimed as novelty |
| "Under 60 seconds to enter" was claimed but never tested | Added a small **usability pilot** (time-to-enter + SUS score, 5–8 participants) |
| Leave-one-hospital-out gave only a before/after number | Added the **local-adaptation curve** (n = 0…5000 local visits, no retraining) — the project's signature figure |
| Admission rates likely differ a lot between the 3 hospitals | Prevalence shift now analysed explicitly; new risks R12–R14 |

## The one-line pitch for your supervisor

> "A model and its uncertainty estimate built in one hospital are moved, offline, to a different hospital. What breaks, and how little local data fixes it? I test this with only what a nurse can type in 60 seconds, on a cheap offline device — and I demo it live with Wi-Fi switched off."
