# Real-data model evaluation

## Why this experiment matters

Fraud is rare in the credit-card transaction dataset. A model that labels every transaction legitimate looks accurate but fails to catch any fraud. This experiment compares five classifiers using average precision, then checks the selected model on transactions that occur later in the dataset.

## Dataset and preparation

The reference data are from the [ULB/Kaggle credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The original CSV contains 284,807 rows, including 492 labeled fraud cases. Validation rejected 1,081 exact duplicates, leaving 283,726 rows with 473 fraud cases.

There are 30 input features: elapsed `Time`, `Amount`, and anonymized `V1`–`V28`. The label is `Class`. We cannot give the anonymized features plausible-sounding names because their definitions are not published.

The split is chronological by `Time` and keeps equal timestamps together:

| Partition | Rows | Fraud cases |
|---|---:|---:|
| Train | 173,176 | 343 |
| Validation | 50,662 | 54 |
| Test | 59,888 | 76 |

Transformations and SMOTE are fitted only to training data. Validation average precision selects a classifier; validation F1 selects the decision threshold. The test partition is evaluated afterward.

## Model selection

| Candidate | Validation average precision |
|---|---:|
| Dummy baseline | 0.0011 |
| Logistic Regression, class-weighted | 0.7444 |
| Logistic Regression + SMOTE | 0.7557 |
| Random Forest | 0.7455 |
| XGBoost | **0.7677** |

XGBoost was the selected classifier. The decision threshold was 0.95. This threshold is intentionally separate from the dashboard's LOW / REVIEW / HIGH categories.

## Holdout results

| Metric | XGBoost |
|---|---:|
| Precision | 0.8852 |
| Recall | 0.7105 |
| F1 | 0.7883 |
| Average precision (labeled PR-AUC in code) | 0.7954 |
| ROC-AUC | 0.9737 |
| True positives | 54 |
| False positives | 7 |
| True negatives | 59,805 |
| False negatives | 22 |

At the selected threshold the model detected 54 of the 76 fraud cases and missed 22. Among 61 flagged test transactions, 54 were genuinely fraudulent. Avoid interpreting high ROC-AUC as sufficient evidence of usefulness in a heavily imbalanced problem.

Average precision is scikit-learn's step-weighted precision–recall summary, not the trapezoidal area under the plotted curve.

## Local demonstration

The saved model release was downloaded from Kaggle and checked before loading into the local Python 3.12 environment. A separate SQLite database held 3,076 transactions selected from the chronological holdout: all 76 fraud cases plus 3,000 sampled legitimate transactions. The pipeline inserted 3,076 predictions, flagged 54, marked 57 HIGH risk, and inserted no duplicates on an identical retry.

The deliberately enriched 3,076-row sample is for demonstrating the batch dashboard, **not** for another independent test or an estimate of the natural fraud or alert rate. The real evaluation is the complete 59,888-row test partition above.

## Limits and deployment status

This is a research and portfolio demonstration, not a live payment-authorization service. The dataset spans a short period and contains no cardholder, merchant, or device identifiers for group-based evaluation. We have not measured long-term drift, deployment performance, or probability calibration. At 0.95, false negatives may matter substantially; a real operational threshold would require an explicit review policy and error-cost study.

The saved Kaggle-trained XGBoost release was later imported into the Docker/PostgreSQL stack in a [GitHub Actions acceptance run](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41322535119). Airflow used that release to score 6,080 synthetic integration-test transactions without retraining. A replay inserted no duplicate decisions, and Streamlit read the registered model metrics from PostgreSQL. This demonstrates that the real-trained model runs in the containerized pipeline; the fraud-detection performance figures above still come from the separate real-data test. The GitHub runner was temporary, not a persistent public deployment.

The aggregate values above are recorded in [`results/real-data-benchmark.json`](results/real-data-benchmark.json). The raw dataset and serialized model are not committed to Git. The checked model archive is available separately as a [public GitHub Release](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/releases/tag/kaggle-model).
