# 10 · References and Resources

## 1. Dataset and baseline

- Hong WS, Haimovich AD, Taylor RA. **Predicting hospital admission at emergency department triage using machine learning.** *PLoS ONE* 13(7): e0201016, 2018. https://doi.org/10.1371/journal.pone.0201016
- Yale ED dataset (Zenodo record 1308993): https://zenodo.org/records/1308993
- Original code (R): https://github.com/yaleemmlc/admissionprediction

## 1b. Closest recent work (must cite and position against)

- Abdulai ASB, Storm J, Ehrlich M. "I don't know": An uncertainty-aware machine learning model for predicting patient disposition at emergency department triage. *International Journal of Medical Informatics* 201:105957, 2025. — Same Yale data; conformalized XGBoost; random split only.
- Dave K, Robert A, Russell WA, Buckeridge D. Explainable machine learning for predicting hospital admission at emergency department triage: retrospective cohort study. *JMIR Medical Informatics* 14:e98273, 2026. https://doi.org/10.2196/98273 — Calibration + explainability at 2 McGill hospitals.
- Meimeti F, et al. Validating emergency department admission predictions based on local data through MIMIC-IV. arXiv:2503.22706, 2025. — Local model validated on a large external dataset.

## 2. Models

- Chen T, Guestrin C. XGBoost: A scalable tree boosting system. *KDD*, 2016.
- Ke G, et al. LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*, 2017.
- Prokhorenkova L, et al. CatBoost: Unbiased boosting with categorical features. *NeurIPS*, 2018.
- Breiman L. Random forests. *Machine Learning* 45:5–32, 2001.
- Nori H, et al. InterpretML: A unified framework for machine learning interpretability. arXiv:1909.09223, 2019. (EBM)

## 3. Calibration and clinical usefulness

- Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. *ICML*, 2005.
- Van Calster B, et al. Calibration: the Achilles heel of predictive analytics. *BMC Medicine* 17:230, 2019.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making* 26(6), 2006.
- DeLong ER, et al. Comparing the areas under two or more correlated ROC curves. *Biometrics* 44(3), 1988.

## 4. Uncertainty

- Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511 (also *Foundations and Trends in ML*, 2023).
- Vovk V, Gammerman A, Shafer G. *Algorithmic Learning in a Random World.* Springer, 2005.
- MAPIE library documentation: https://mapie.readthedocs.io

## 5. Explainability

- Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS*, 2017.
- Lundberg SM, et al. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence* 2, 2020.

## 6. Reporting standards

- Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 385:e078378, 2024.

## 7. Edge deployment tools

- ONNX Runtime: https://onnxruntime.ai
- onnxmltools / skl2onnx: https://github.com/onnx/onnxmltools · https://github.com/onnx/sklearn-onnx
- m2cgen (model → code): https://github.com/BayesWitnesses/m2cgen
- Streamlit: https://streamlit.io
- Workbox (service workers for PWAs): https://developer.chrome.com/docs/workbox

## 8. Optional external dataset

- MIMIC-IV-ED (PhysioNet, credentialed access): https://physionet.org/content/mimic-iv-ed/

## 9. Literature search starters (Week 1)

Search Google Scholar / PubMed for 2022–2026 papers using:
- "emergency department" "admission prediction" machine learning triage
- conformal prediction clinical risk / emergency
- calibration external validation emergency department model
- edge AI OR on-device healthcare prediction low-resource
- triage India emergency department machine learning

Aim for 25–35 references in the final dissertation, with at least 8–10 from the last three years.
