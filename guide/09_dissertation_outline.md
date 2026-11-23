# 09 · Dissertation Outline and Viva Preparation

## 1. Chapter structure (target ~60–80 pages)

| Ch. | Title | Contents | Pages |
|-----|-------|----------|-------|
| — | Front matter | Title, certificate, declaration, acknowledgements, abstract, lists of figures/tables/abbreviations | — |
| 1 | Introduction | ED crowding and admission planning; low-resource and low-connectivity context; problem statement; aim, objectives, RQs; contributions; thesis organisation | 6–8 |
| 2 | Literature Review | ED triage systems (ESI, colour triage); admission prediction studies; tabular ML (boosting); calibration in clinical ML; conformal prediction; explainability (SHAP, glass-box); edge/TinyML in healthcare; research gap table | 12–15 |
| 3 | Data and Methodology | Dataset; data dictionary; leakage audit; tiers; splits; preprocessing; models and tuning; calibration; thresholds; conformal; SHAP; compression; robustness; fairness; decision curves; statistics | 15–18 |
| 4 | Results | E1–E12 with Tables A–G and figures, organised by RQ | 15–18 |
| 5 | System Design and Implementation | Architecture; deployment levels; model bundle; app screens; offline verification; edge benchmark | 8–10 |
| 6 | Discussion | Answers to RQ1–RQ4; comparison with Hong et al.; what transfers and what does not; relevance to Indian EDs; limitations | 6–8 |
| 7 | Conclusion and Future Work | Summary; contributions; future work (Indian data, free-text/Hindi complaints, prospective study, federated recalibration across hospitals) | 2–3 |
| — | References, Appendices | Data dictionary, hyper-parameters, extra figures, TRIPOD+AI checklist, model card, user manual | — |

## 2. Writing rules

- Every claim about performance has a number and a 95% CI.
- Always say "admission prediction", never "triage-level prediction" or "diagnosis".
- Use "associated with", never "causes", for SHAP findings.
- Every figure has a caption that states what the reader should notice.
- Reported results come only from the single frozen test-set run (plus leave-one-hospital-out).

## 3. Future work ideas (good for the conclusion and for a paper later)

1. Validation on Indian ED data (e.g., partnership with a district hospital or AIIMS-type institution).
2. Free-text and Hindi/regional-language chief complaints via a small on-device language model.
3. Federated recalibration: hospitals share calibration statistics, not data.
4. Prospective silent-mode study: model runs alongside nurses without showing results.
5. Joint prediction of admission and ICU need or length of stay.

## 4. Likely viva questions (prepare answers)

**Problem and data**
0. Abdulai et al. (2025) already used conformal prediction on this exact dataset. What is new in your work? *(Answer: they tested on a random split of the same hospitals; I test what happens at an unseen hospital, how many local visits repair it, with nurse-only inputs, on an offline edge device.)*
1. Why predict admission instead of ESI level?
2. Your data is from the US. Why is this relevant to India?
3. How did you make sure there is no data leakage?
4. The same patient can appear many times. How did you handle that?
5. Why did you exclude race, religion and insurance?

**Methods**
6. Why gradient boosting over deep learning for this data?
7. Why tune on log-loss and not accuracy?
8. What is the difference between Platt scaling and isotonic regression? When would each fail?
9. Explain conformal prediction in two minutes. What guarantee does it give, and what does it assume?
10. Why did conformal coverage drop on the unseen hospital?
11. What does a SHAP value mean? Why is it not causal?
12. What is decision-curve analysis and why is it better than AUC for clinical usefulness?
13. How did you choose thresholds without touching the test set?

**Engineering**
14. How did you prove the app makes zero network calls?
15. Why ONNX? What did you lose by compressing?
16. How would you update the model in a hospital with no internet?

**Results**
17. How much did ESI add? How much did history add? Is the T1 model good enough to be useful?
18. Which subgroup performed worst, and why might that be?
19. If a hospital wanted to use this tomorrow, what would you tell them?

## 5. Presentation (15 minutes)

| Slide | Content |
|-------|---------|
| 1 | Title |
| 2 | Problem: early admission knowledge, low-resource constraints |
| 3 | Gap vs prior work (table) |
| 4 | RQs |
| 5 | Dataset and tiers |
| 6 | Method pipeline |
| 7 | RQ1 results: tier comparison |
| 8 | RQ2 results: leave-one-hospital-out |
| 9 | RQ3 results: conformal under hospital shift + local-adaptation curve |
| 10 | RQ4 results: Pareto plot + benchmark |
| 11 | Live demo (Wi-Fi off) |
| 12 | Limitations and ethics |
| 13 | Conclusion and future work |
