# Data dictionary

| Field | Meaning / rule |
|---|---|
| `Time` | Finite, nonnegative seconds relative to the dataset origin; not a calendar timestamp |
| `Amount` | Finite, nonnegative transaction amount; currency is not encoded in this schema |
| `V1`–`V28` | Finite numeric features; anonymous components in the reference dataset and fabricated values in the demo |
| `Class` | Ground truth 0/1; required for training, optional/nullable for inference |
| `transaction_id` | Optional upstream string ID; otherwise a deterministic SHA-256 digest of features |
| `rejection_reason` | Semicolon-separated row validation failures |
| `fraud_probability` | Stored `predict_proba` score; calibration is not established |
| `risk_level` | LOW/REVIEW/HIGH based on risk bands saved in the model release |
| `is_flagged` | Whether score reaches the release's decision threshold |
| `model_version` | Model type and training timestamp |
| `threshold` | Actual threshold used for this decision |
| `prediction_timestamp` | UTC time of scoring |

`transactions.content_hash` detects reuse of an upstream ID with changed features.
`actual_class` may be updated from unknown to known; contradicting a known label
raises an error. An existing transaction keeps its initial ingestion/source-run link.

`pipeline_runs` records stage status and counts; `validation_results` stores the
summary for a validation stage. `model_runs` stores model metrics and chart data.
`fraud_predictions` keeps decisions unique by transaction, model version, threshold.
See `sql/schema.sql` for types and indexes.

The 24-hour cyclical feature is a periodic encoding of elapsed time relative to the
source origin. Without the origin's wall-clock timestamp it is not a known local
hour of day.
