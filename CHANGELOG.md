# Changes

## Reliability and portfolio review

- Allow unlabeled inference and reject non-finite feature values.
- Preserve upstream IDs or derive stable fallback IDs from features.
- Save each batch to its own snapshot/output directory.
- Commit transactions and predictions together, with replay-safe uniqueness.
- Store nullable labels and reject conflicting reuse of an upstream ID.
- Publish versioned model releases with checksums, frozen thresholds, and risk bands.
- Select thresholds on validation; support chronological holdouts.
- Serialize curve metadata as strict JSON, including the ROC infinity sentinel.
- Isolate application dependencies from Airflow and slim the dashboard image.
- Add regression tests, a PostgreSQL test target, CI, and an executable local demo.
- Update setup, operations, model evidence, résumé, and GitHub documentation.

Original development history and co-author attribution are retained.
