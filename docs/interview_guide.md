# Interview walkthrough

Practice answering these questions while opening the relevant code. Show a command
or a test, then explain the tradeoff in your own words.

1. **Why isn't accuracy the main metric?** Most rows are legitimate. A model that
   always predicts legitimate can have high accuracy and zero fraud recall. The
   project compares validation average precision and also reports precision, recall,
   F1, ROC-AUC, and confusion counts.
2. **Where can leakage occur?** Scaling or SMOTE before the split exposes validation
   information. Each candidate fits preprocessing and any resampling on training
   only. Model and threshold selection use validation. Test metrics follow selection.
3. **Why use two split strategies?** Stratification stabilizes the small synthetic
   positive class. Chronological splitting tests later timestamps and keeps ties in
   one partition. It fails if either class is absent. No customer/card identifier is
   available to assess entity leakage.
4. **Can live transactions be unlabeled?** Yes. Inference requires only features.
   `Class` is nullable and a label can arrive later. Training requires complete labels.
5. **What happens when the same file is submitted twice?** Stable IDs and a unique
   decision key prevent extra transactions/predictions for the same release and
   threshold. Audit run records are still created. Demonstrate `scripts.demo`.
6. **Why not hash every real transaction?** Two legitimate transactions can have
   identical anonymized features. The fallback can collapse them. Stable upstream
   IDs preserve distinct events and should be used for actual payment data.
7. **What if predictions fail to save?** Transactions and predictions share a database
   transaction, so both roll back. A post-commit failure is recovered by a safe retry.
8. **How do you avoid a mixed model and threshold?** Each release stores both; scoring
   resolves the active pointer once. A checksum detects accidental artifact corruption.
   It does not make an untrusted pickle safe.
9. **Why separate Airflow's Python environment?** Its dependency requirements conflict
   with the application's SQLAlchemy 2 stack. External-Python tasks call the same CLI
   workflow code from an isolated venv.
10. **Is this production-ready?** No. It still needs real-data validation and full
    Docker/PostgreSQL acceptance. Operational gaps include retention, access control,
    drift monitoring, calibration, and distributed publication recovery.
11. **What does 87.5% recall mean in the recorded demo?** It means 21 of 24 holdout
    fraud examples were detected. A small denominator makes the estimate uncertain;
    it is synthetic-data behavior, not proof of real-world fraud detection.
12. **Is the model score a calibrated probability?** No calibration has been measured.
    Weighting/oversampling changes the training distribution. A score of 0.9 does
    not justify claiming a 90% real-world fraud chance.

## Demonstration sequence

Run tests; show the strict-JSON regression; train the synthetic demo; score an
unlabeled file twice; inspect unchanged prediction counts; open the four dashboard
pages; explain the held-out evaluation separately from operational dashboard volume.

## Exercises before using the project in an interview

Add an invalid row and explain the rejection. Change a supplied transaction ID's
features and observe the conflict error. Explain the threshold tie-break. Locate a
failed pipeline run. Trace a prediction to its model version. Repeat a chronological
benchmark on a lawful real dataset and write down what changed.
