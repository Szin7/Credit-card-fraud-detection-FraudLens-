# Model card — synthetic demonstration

## Purpose

This model demonstrates batch classification, model comparison, and a transaction-review workflow. It is not approved for blocking payments or claiming reductions in real fraud.

## Training data

The generator creates 24,000 legitimate rows and 120 fraudulent rows, plus 25 deliberately injected duplicate rows. The duplicates are rejected during validation, leaving 24,120 valid transactions. Fraud examples are generated with explicit shifts in several artificial features, so the task is easier than real fraud detection.

The stratified demonstration split is approximately 64/16/20 for training, validation, and test. There are 19 fraud cases in validation and 24 in the test set.

## Recorded holdout result

These figures are reproduced from `docs/results/synthetic-benchmark.json` (run at 2026-10-13T06:48:27Z). The test results below refer to the selected model at its frozen validation-selected threshold.

| Measure | Result |
|---|---:|
| Champion | Logistic Regression + SMOTE |
| Validation average precision | 0.9596 |
| Test average precision | 0.9447 |
| Test ROC-AUC | 0.9993 |
| Selected threshold | 0.95 |
| Test precision | 0.9130 |
| Test recall | 0.8750 |
| Test F1 | 0.8936 |
| True positives / false negatives | 21 / 3 |
| False positives / true negatives | 2 / 4,798 |

The approximate 95% Wilson interval for recall is 0.690 to 0.957. Only 24 fraud examples appear in the holdout, so this estimate is uncertain even before considering the limitations of synthetic data.

Average precision is the step-weighted precision-recall summary returned by scikit-learn; the repository calls it `pr_auc` for compatibility with the UI. It is not the same as trapezoidal integration of the drawn curve.

## How the model was selected

Preprocessing and sampling were fitted on training data only. Validation average precision selected the candidate model. The threshold was chosen by validation F1 on a grid from 0.05 to 0.95, with ties broken by precision and then the higher threshold. The test holdout was evaluated only after these choices.

Scores are not calibrated fraud probabilities. Low/review/high risk categories are demonstration triage labels, not a payment authorization policy. Anonymous components `V1`–`V28` cannot be interpreted as known customer attributes.

## What remains

Run a separate chronological benchmark on the real reference dataset, compare false positives and false negatives, and record full reproducibility metadata. The results above must never be presented as real-data performance. See `docs/real_data_experiments.md` for the experiment plan.
