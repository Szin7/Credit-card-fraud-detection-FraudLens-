# Model card — synthetic demonstration

## Intended use

Demonstrate batch ML engineering and review workflows. Not approved for blocking
payments, making financial decisions, or claiming fraud reduction.

## Data

The generator creates 24,000 legitimate rows, 120 fraud rows, and 25 injected
duplicates. Fraud features receive explicit distribution shifts. This makes the
classification problem easier and more artificial than real fraud.

After validation: 24,120 rows; 25 duplicates rejected. The stratified split uses
64/16/20 train/validation/test proportions. Validation has 3,860
rows and 19 frauds; test has 4,824 rows and 24 frauds.

## Recorded benchmark

Run on 2026-10-13T06:48:27.473625+00:00. See [machine-readable results](results/synthetic-benchmark.json)
for source checksum, model comparison, and exact values.

| Measure | Result |
|---|---:|
| Champion | logistic_regression_smote |
| Validation average precision | 0.9605 |
| Test average precision | 0.9427 |
| Test ROC-AUC | 0.9993 |
| Selected threshold | 0.95 |
| Test precision | 0.9130 |
| Test recall | 0.8750 |
| Test F1 | 0.8936 |
| True positives / false negatives | 21 / 3 |
| False positives / true negatives | 2 / 4798 |

The approximate 95% Wilson interval for recall is 0.690–0.957.
That interval reflects only the small holdout count; it does not account for synthetic
data bias, model-selection uncertainty, or deployment shift.

The code calls average precision `pr_auc` for continuity with the dashboard.
Average precision is a step-weighted precision-recall summary, not trapezoidal
integration of the plotted curve.

## Selection and interpretation

Candidates fit preprocessing on training only. The SMOTE candidate resamples only
training. Validation average precision selects the champion. Validation F1 selects
a threshold from 0.05 through 0.95; ties favor precision, then the higher threshold.
The selected operating point is saved with the release before holdout reporting.
The illustrative cost sweep is shown for inspection, not used as actual business costs.

A high score does not mean a calibrated probability. Risk bands support demonstration
triage; they do not prescribe a payment policy. Global feature importance does not
explain an individual transaction. Optional SHAP support may be unavailable.

## Limits and next experiment

Only 24 test fraud examples are available. The independent Gaussian synthetic
features do not represent real card behavior, attackers, merchant activity, delayed
feedback, or concept drift. Model rankings and the high threshold need not transfer.

Obtain the reference dataset lawfully, record its source and checksum, and run a
chronological benchmark with no temporal overlap. Preserve both successful and weak
results. Use confidence intervals and review the false positives/negatives. Repeated
experiments on the same holdout turn it into development data; reserve a new final
holdout when making further modeling decisions. There is no real-data benchmark in
this release.
