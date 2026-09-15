# Model Card — FraudLens Fraud Classifier

## Model purpose

Scores a credit-card transaction with a probability of being fraudulent,
for a human reviewer to triage via risk bands (LOW/REVIEW/HIGH) and for a
binary flag at a tuned decision threshold. It is a **triage aid**, not an
autonomous block/approve system — see Intended Use below.

## Dataset

Trained and evaluated on `data/raw/creditcard_sample.csv`, a **synthetic**
dataset bundled with this repository (see `data/raw/README.md` for exactly
how it was generated and how to substitute the real Kaggle dataset).
Every number in this document was produced by actually running
`src/models/train.py` and `scripts/optimize_threshold.py` against that
synthetic sample — **none of it describes real-world fraud-detection
performance**, and it will change if you regenerate the sample with a
different seed or supply the real dataset.

At the time these numbers were generated: 24,145 transactions, 120 labeled
fraud (0.497%), 25 exact duplicate rows removed by validation, leaving
24,120 valid rows fed to training.

## Features

`src/features/build_features.py` builds three groups of inputs, fit once
during training and reused unchanged at inference (a single
`ColumnTransformer`, to prevent training/serving skew):

- **Amount** — log1p-transformed (right-skewed) then standard-scaled.
- **Time** — folded to hour-of-day and encoded as sin/cos, so hour 23 and
  hour 0 sit next to each other in feature space instead of at opposite
  ends of a raw 0–23 scale.
- **V1–V28** — passed through unchanged. These are PCA-anonymized in the
  real dataset (and modeled as independent Gaussian components with an
  arbitrary fraud-row shift in the synthetic sample) — this model, and this
  document, never assign them real-world semantic meaning.

## Train / validation / test methodology

**Stratified random split** (64% train / 16% validation / 20% test,
`random_state=42`), not chronological. `Time` spans only ~48 hours with no
calendar or seasonal structure, and with well under 1% positives, a
chronological cut risks starving a split of fraud examples entirely
depending on which hours land where — stratification keeps the fraud ratio
stable in every split, which matters more here than simulating a
train-on-past/score-on-future production setup. A dataset with real
multi-week or multi-month history would change this tradeoff; see
Limitations.

Resulting split sizes: 15,436 train rows (77 fraud), 3,860 validation rows
(19 fraud), 4,824 test rows (24 fraud).

## Imbalance handling

Two strategies, measured against each other rather than assumed:

- **Class weighting** (`class_weight="balanced"` for Logistic Regression /
  Random Forest, `scale_pos_weight` for XGBoost) — the default across every
  model, adds no dependency, and can't leak between folds.
- **SMOTE oversampling**, compared head-to-head against class weighting on
  the same base model (Logistic Regression), via `imbalanced-learn`'s
  `Pipeline` (not scikit-learn's — it doesn't support a resampling step
  that changes row counts). SMOTE runs **only** inside that training
  pipeline, fit exclusively on the training fold; the persisted inference
  pipeline never includes a resampling step, since resampling at predict
  time is meaningless.

Measured result: SMOTE modestly outperformed class weighting alone for
Logistic Regression on this dataset (validation PR-AUC 0.9605 vs. 0.9565).

## Models compared

All five trained inside a `Pipeline(features, model)`, evaluated on the
validation split at threshold 0.5:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Dummy (always "legitimate") | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0049 |
| Logistic Regression (class-weighted) | 0.7826 | 0.9474 | 0.8571 | 0.9991 | 0.9565 |
| **Logistic Regression + SMOTE (champion)** | 0.8182 | 0.9474 | 0.8780 | 0.9992 | **0.9605** |
| Random Forest (class-weighted) | 1.0000 | 0.5263 | 0.6897 | 0.9993 | 0.9289 |
| XGBoost (`scale_pos_weight`) | 1.0000 | 0.7895 | 0.8824 | 0.9992 | 0.9301 |

The dummy baseline's PR-AUC (0.0049) lands almost exactly at the
validation split's fraud rate (19/3860 = 0.0049), which is the expected
floor for a classifier with no signal — a useful sanity check that the
other numbers are measuring something real.

**Champion selection metric: validation PR-AUC**, not accuracy or even
F1 at a single threshold — see "Why accuracy is misleading" in
`notebooks/01_exploration.ipynb` for a concrete demonstration on this
dataset (a trivial "always legitimate" classifier scores 99.50% accuracy
while catching zero fraud).

Logistic Regression outperforming the tree ensembles here is a property of
*this synthetic dataset's construction* — fraud rows are generated with a
straightforward mean/variance shift on a handful of components, which is
close to linearly separable by design. That is not a general claim that
linear models beat gradient boosting for fraud detection; on the real,
non-synthetic dataset this ranking could easily invert.

## Final (test-set) metrics — champion model

Computed once, on the test split, at the configured decision threshold
(0.60) — the test set is never used for model selection or threshold
tuning, only this single final report:

| Metric | Value |
|---|---|
| Precision | 0.8077 |
| Recall | 0.8750 |
| PR-AUC | 0.9427 |
| ROC-AUC | 0.9993 |
| Confusion matrix | TN=4794, FP=6, FN=3, TP=21 |

## Threshold

`scripts/optimize_threshold.py` sweeps thresholds 0.05–0.95 on the
**validation** split (never test) and reports precision/recall/F1/expected
cost at each one. For the current champion, F1 and the illustrative
expected-cost metric both optimize at **threshold = 0.60**
(F1 = 0.923, precision = 0.90, recall = 0.947). That value is what's
configured in `config/settings.yaml > model > decision_threshold` — it was
copied there by a human after reviewing this output, not written
automatically by the script (see the script's docstring for why).

This is a separate concept from the dashboard's risk bands
(LOW/REVIEW/HIGH, `config/settings.yaml > risk_bands`), which classify the
same probability into UI triage categories independent of the operating
threshold.

## Business-cost framing

`config/settings.yaml > cost_model` defines `false_negative_cost` and
`false_positive_cost` (currently 500 and 5) used to compute
`expected_cost = FN × FN_cost + FP × FP_cost` across the threshold sweep.
**These are illustrative, configurable placeholders for demonstrating why
threshold selection is a business decision, not a real bank's actual cost
figures** — nothing in this repository claims otherwise, and changing
either value in `config/settings.yaml` changes which threshold minimizes
cost.

## Known limitations

- All measured numbers describe the bundled **synthetic** sample, not the
  real dataset — see Dataset above.
- The synthetic sample's fraud-row feature shift makes classes closer to
  linearly separable than real fraud likely is; do not read the model
  ranking above as a general result.
- Validation and test splits contain only 19 and 24 fraud examples
  respectively — individual metric values (especially recall, which moves
  in increments of ~1/19 or ~1/24) are noisy at this sample size. The real
  284,807-row dataset (492 fraud) would give much more stable estimates.
- `transaction_id` is a UUID minted at validation time, not a real
  upstream identifier — re-running the pipeline against the same static
  sample file inserts a fresh batch of "new" transactions each time rather
  than being idempotent, which is a demo-data artifact, not how a
  production feed with real transaction IDs would behave.
- No model-drift detection or scheduled retraining is implemented; see
  README > Future Improvements.

## Ethical considerations

- Predictions are probabilistic and imperfect at any threshold — false
  positives (legitimate transactions flagged) impose a real cost on
  customers, which is exactly why threshold selection is exposed as a
  configurable, documented tradeoff rather than hardcoded at 0.5.
  `V1`-`V28` carry no disclosed demographic or identity information, but
  any model of this kind should still be monitored for disparate impact
  before any real deployment — not done here.
- No PII is used or stored anywhere in this system: `data/raw/README.md`
  and `docs/data_dictionary.md` cover exactly what each field is.

## Intended use

- A portfolio/reference implementation of a batch fraud-scoring pipeline
  and the ML engineering practices around it (leakage-safe splitting,
  imbalance handling measured rather than assumed, threshold analysis,
  train/serve-consistent feature preprocessing).
- A starting point for evaluating the same pipeline against the real
  Kaggle dataset or a similarly-shaped internal dataset.

## Non-intended use

- Not a production fraud-decisioning system as-is: no drift monitoring, no
  real-time/streaming scoring path, no human-review workflow beyond the
  dashboard's risk bands, and metrics above are synthetic-data results.
- Not validated for fairness or disparate impact across any protected
  attribute — none exist in this dataset, but a real deployment would need
  that analysis against its own data before go-live.
