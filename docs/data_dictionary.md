# Data Dictionary

## Source schema (raw CSV — real or synthetic)

Both `data/raw/creditcard.csv` (real, if supplied) and
`data/raw/creditcard_sample.csv` (bundled synthetic sample — see
`data/raw/README.md`) share this schema.

| Column | Type | Description |
|---|---|---|
| `Time` | float | Seconds elapsed since the first transaction in the batch. Not a calendar timestamp — see "On `Time`" below. |
| `V1`–`V28` | float | PCA-anonymized transaction features. **No disclosed real-world meaning** — the real dataset's publisher applied PCA specifically to anonymize the original fields, and did not release what those fields were. This codebase never assigns them one either (no "merchant", "device", "country", etc.). In the synthetic sample, `V1`–`V28` are independently generated Gaussian values with an arbitrary shift on `V1,V2,V3,V4,V5,V9,V14` for fraud rows, purely so a classifier has separable signal to find. |
| `Amount` | float | Transaction amount. Always ≥ 0 after validation. |
| `Class` | int | Ground truth label: `0` = legitimate, `1` = fraud. The real Kaggle dataset is publicly documented as 492 fraud out of 284,807 transactions (~0.17%) — this repository has not verified that by running code against it, since the file isn't included (see `data/raw/README.md`). The bundled synthetic sample targets ~0.5% instead (see below). |

**On `Time`:** the dataset spans roughly 48 hours with no calendar or
seasonal structure attached — it's elapsed seconds, not a date. This is
why the train/val/test split uses stratified random sampling rather than a
chronological cutoff; see `docs/model_card.md`.

**On the synthetic sample's fraud rate:** the real dataset's fraud rate
(~0.17%) is thin enough that a small demo sample would carry only a
handful of positive examples per split. The bundled sample targets ~0.5%
instead, documented in `data/raw/README.md`, specifically so validation/
test splits retain enough positives to produce a stable PR curve. This is
a demo-scale adjustment, not a claim about real-world fraud prevalence.

## `data/processed/transactions_valid.csv` (validation output)

Same columns as the source, plus:

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string (UUID4 hex) | Minted at validation time — the raw schema has no natural identifier. Stable from this point through features, prediction, and Postgres. A production system would receive this from the upstream payment processor instead. |

## `data/processed/predictions.csv` / `fraud_predictions` table

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | References the transaction. |
| `amount`, `time_elapsed` | float | Carried through for display without a join. |
| `fraud_probability` | float, [0,1] | `pipeline.predict_proba(X)[:, 1]` from the persisted champion model. |
| `risk_level` | enum: LOW / REVIEW / HIGH | UI triage category, from `config/settings.yaml > risk_bands`. Independent of `is_flagged`. |
| `is_flagged` | bool | `fraud_probability >= decision_threshold`. The actual operating decision, from `config/settings.yaml > model > decision_threshold`. |
| `model_version` | string | `{model_type}@{trained_at}` of the model that produced this row. |
| `threshold` | float | The decision threshold in effect when this prediction was made (recorded per-row so historical predictions remain interpretable after the configured threshold changes). |
| `prediction_timestamp` | timestamptz | When this row was scored. |

## Database tables

See `sql/schema.sql` for full DDL (columns, constraints, indexes) and
`sql/useful_queries.sql` for example queries against each table.

| Table | Purpose |
|---|---|
| `pipeline_runs` | One row per execution of any pipeline stage (ingestion/validation/training/inference). Every other table's provenance traces back here. |
| `validation_results` | The detailed rule-by-rule breakdown behind a validation-type `pipeline_runs` row. |
| `transactions` | Transaction-level summary for the app layer — `Time`/`Amount`/`Class`, not `V1`-`V28` (see `docs/architecture.md` for why). |
| `model_runs` | One row per trained model: metrics, decision threshold, and a `curve_data` JSONB blob (PR curve, ROC curve, threshold sweep, confusion matrix, feature importance). At most one row has `is_champion = true`. |
| `fraud_predictions` | One row per (transaction, model_version) scored. |

## What's deliberately *not* in the data dictionary

`V1`–`V28` do not get individual per-column descriptions beyond "PCA
component, no disclosed meaning" — inventing labels like "V4: transaction
velocity" would be fabricating information the dataset's publisher never
released, for either the real dataset or this project's synthetic stand-in.
