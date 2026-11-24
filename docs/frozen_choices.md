# Frozen choices and pre-registered success criteria (Phase 5)

Frozen on 2026-11-24, **before** any metric was computed on the S1 test set.

## Research models (Table A–D, E10–E12)

| Tier | Model | Calibrator | t_low (NPV≥0.95) | t_high (PPV≥0.80) | val AUC |
|---|---|---|---|---|---|
| T1 | lgbm | isotonic | 0.1139 | 0.6201 | 0.8573 |
| T2 | lgbm | platt | 0.1737 | 0.6418 | 0.8761 |
| T3 | cat | platt | 0.1758 | 0.4260 | 0.9279 |

Conformal: Mondrian split conformal, α = 0.10 (also 0.05, 0.20 reported), fit on S3 calibration rows.

## Shipped edge models (app)

| Slot | Variant | Size KB | val AUC | rule met |
|---|---|---|---|---|
| t1 | V4 LR top-222 | 9.3 | 0.8461 | False |
| t3lite | V3 ONNX d4 t200 top-50 | 214.9 | 0.9184 | True |
| t2 | T1 edge recipe + ESI (223 features) | nan | 0.8677 | n/a |

## Pre-registered success criteria (guide 05 §4, made concrete)

| Criterion | Pass condition |
|---|---|
| Reproduction | T2 and T3 best-model test AUC within 0.02 of Hong et al. (0.87 triage, 0.92 full) |
| Tier-1 usefulness | T1 test AUC with 95% CI; T1 net benefit > max(treat-all, treat-none) at every threshold 0.10–0.70 |
| Calibration | Post-calibration ECE ≤ 0.05 on S1 test, all three selected models |
| Conformal | Marginal coverage within ±0.02 of 1−α on S1 test for α = 0.05, 0.10, 0.20 (T1, T2) |
| Compression | Edge model test ΔAUC ≤ 0.01 vs V0 at ≥ 10× smaller (T1 already expected to miss on val: Δ0.011) |
| Offline | Demo works with networking disabled; 0 bytes measured (Docker --network none) |
| Local adaptation | Smallest n restoring coverage within ±0.02 identified per held-out site (from S2; already computed, not test-dependent) |
| Usability | Median T1 entry time and SUS for ≥ 5 participants (requires humans — not run by the agent) |

## Artifact hashes (MD5) at freeze

```
{
 "bundles": {
  "t1_v1": {
   "calibrator.json": "8436dd072d82a5b0a28a07dbcd7c9063",
   "conformal.json": "9c7906d74940a95cae7da4fb9dff2305",
   "model.onnx": "809eed0715efaeb669478e0cdc809b37",
   "thresholds.json": "46ebd5da76822e7303baa693eb3d6434"
  },
  "t2_v1": {
   "calibrator.json": "b11528fea4b461668a4cbda7cfe946b9",
   "conformal.json": "fa978fa44aaba24b3f7d9c7fcba69f97",
   "model.onnx": "1ccb91a72835652cf0ec579402cb6a5e",
   "thresholds.json": "739c941033117bf04728e9d830344cb6"
  },
  "t3lite_v1": {
   "calibrator.json": "231f1f52b0c739aa4b566afd9533ac4b",
   "conformal.json": "b46a986e2f8eed3261e456c04b67d2ab",
   "model.onnx": "9cb91ae50c3fb80704c66f17ad6a8ad2",
   "thresholds.json": "81b4371615c7397ea7e32751d4c851fb"
  }
 },
 "raw_models": {
  "t1_cat.joblib": "1d7b1b1122f0c78209f8815c40b8467a",
  "t1_lgbm.joblib": "b7422852f6ad9e6a208fcb8fd4bc6c1b",
  "t1_lr.joblib": "47798ce836db2e5189c23f52de604747",
  "t1_rf.joblib": "71474548d7dadb5db4b4962d12974be3",
  "t1_xgb.joblib": "3b780b97d626743cf1be483d97695faa",
  "t2_cat.joblib": "a5e9a44f4eed14a61532aab3882610de",
  "t2_lgbm.joblib": "b1bf9b32c6375e13e512827749f509dc",
  "t2_lr.joblib": "3ee1cde42238f9238e4cf25b7110cb84",
  "t2_rf.joblib": "e262486d2046be896da4adca9f4b8bdc",
  "t2_xgb.joblib": "d62598c6dd0acc26d3c4b4e729dcfb50",
  "t3_cat.joblib": "b267623f61f544cbf9922967035c1fa4",
  "t3_lgbm.joblib": "e9321b85e8f8bb6c33cd921a36655976",
  "t3_lr.joblib": "55762755c4595c149f49eb1ece3e5594",
  "t3_rf.joblib": "d841d57bf4c8d053b91fd955ca329fd5",
  "t3_xgb.joblib": "4f90dac8da7d8ff407b4b2b8ac571154"
 }
}
```
