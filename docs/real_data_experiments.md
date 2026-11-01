# Testing FraudLens on real transaction data

The synthetic sample is useful for building and testing the pipeline, but it cannot tell us whether a fraud detector would work on actual transactions. This experiment keeps the application code the same while changing the data used to train and evaluate the models.

## Research question

How do different classifiers and imbalance-handling methods perform when fraudulent transactions are rare, and what tradeoff do we make between catching fraud and incorrectly flagging legitimate transactions?

## Dataset

Use the public research dataset linked at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud, following its license and access conditions. It contains 284,807 transactions with 492 fraud labels. `Time` is elapsed seconds rather than a real-world timestamp; `V1` to `V28` are anonymous components. Do not give these components invented meanings.

Keep the dataset outside Git. Do not rename the included synthetic sample to `creditcard.csv` or report its metrics as results from the real dataset.

## Experiment design

1. Inspect shape, schema, missing values, duplicate rows, class balance and amount distribution. Save only aggregate findings and figures.
2. Prefer a chronological split that keeps identical timestamps in the same partition. The existing pipeline uses approximately 64% training, 16% validation and 20% test by distinct timestamps; row proportions may vary. If any partition lacks one class, the run stops instead of giving misleading metrics.
3. Fit imputers, transformations, scalers and sampling methods on training data only. SMOTE is inside the training pipeline and never sees validation or test records.
4. Compare the existing five candidates using **validation average precision** (shown as PR-AUC in the code). Use validation data to select the decision threshold. Do not inspect test scores while choosing a model or threshold.
5. Evaluate the frozen champion **once** on the holdout set. Record precision, recall, F1, average precision, ROC-AUC, and the confusion matrix. Discuss false negatives and false positives in ordinary language.
6. Check probability calibration separately before treating a model score as a probability of fraud. Use an additional calibration partition or cross-validation; the untouched final test remains untouched.
7. Preserve the source file SHA-256 checksum, split method, model version, metric values, and software versions. Write down weak as well as strong results.

### Important limitation

A chronological split on elapsed `Time` is better aligned with future prediction than random stratification, but this dataset does not include cardholder, merchant, or device IDs. It cannot support full entity-disjoint evaluation or demonstrate performance under long-term fraud-pattern drift. The small number of positive examples also makes recall estimates uncertain.

## Run on Kaggle

The project requires Python **3.12** with the versions in `requirements.txt`. A Kaggle session can use a separate Python 3.12 environment even if the notebook kernel uses another version. An internet-enabled session may be needed once to install dependencies. Kaggle storage and compute limits may change.

1. Create a notebook and attach the `creditcardfraud` dataset from Kaggle.
2. Put a complete copy of the FraudLens repository in `/kaggle/working/FraudLens` (for example by uploading a repository archive; cloning also works when notebook internet access is enabled).
3. Make sure `scripts/benchmark_real_data.py` is present in the repository. Open the companion notebook `notebooks/02_real_data_benchmark.ipynb` and follow its cells.
4. Review the outputs. Do not publish the new release by default; publish only when the metadata and results are acceptable.
5. Download the generated `fraudlens-real-model-release.zip` file. Match its Python/scikit-learn dependencies before importing or loading it. Only load joblib artifacts you created or trust.

You can also run this locally once a database-free Python environment is ready:

```bash
python -m scripts.benchmark_real_data --input /path/to/creditcard.csv
```

After reviewing the model:

```bash
python -m scripts.benchmark_real_data --input /path/to/creditcard.csv --publish
```

`--publish` updates the current active model pointer **only in that environment**. It does not upload the model to your laptop or update the production database. A database-backed run should register the release and its metadata as part of the normal training pipeline.

After downloading a Kaggle release archive, import it into the local checkout without activating it:

```bash
python -m scripts.import_model_release --archive /path/to/fraudlens-real-model-release.zip
```

This verifies the archive structure, model checksum and feature schema. It does **not** deserialize the model, register its metrics in the database, or make it active by default. Once you trust the artifact and have matched environment versions, `--publish` updates the local model pointer. Database registration and dashboard acceptance remain separate checks.

## Recorded experiment results (1 November 2026)

The experiment completed on Kaggle CPU using Python 3.12.15. From 284,807 source transactions, validation removed 1,081 duplicate rows. The chronological test partition contained 59,888 transactions, including 76 fraud cases. Results below are from the **full held-out test partition**.

| Measure | Verified value |
|---|---:|
| Selected model | XGBoost |
| Validation average precision | 0.7677 |
| Selected threshold | 0.95 |
| Test average precision | 0.7954 |
| Test precision | 0.8852 |
| Test recall | 0.7105 |
| Test F1 | 0.7883 |
| True positives / false negatives | 54 / 22 |
| False positives / true negatives | 7 / 59,805 |

The trained model was exported, checksum-verified, imported into WSL and used successfully in a local batch/SQLite/Streamlit check. A deliberately enriched dashboard sample contained 3,076 holdout-feature rows and led to 54 flags; it is **not an independent performance evaluation**.

See [`real_model_card.md`](real_model_card.md) for interpretation and [`results/real-data-benchmark.json`](results/real-data-benchmark.json) for recorded values. The Docker/Airflow end-to-end test remains open.

## Discussion prompts

- How different is the class ratio in this dataset from the synthetic demonstration?
- Which method catches the most fraud at a comparable false-alarm rate?
- What happens when the threshold changes? Which errors become more frequent?
- Does the result suggest overfitting, leakage, or data drift? What evidence would you need to distinguish them?
- Why is a model's `predict_proba` output not necessarily a calibrated probability?

## Before claiming completion

The runner must finish, a model release must load in a compatible environment, and all reported metrics must agree with the saved metadata. A full database/Streamlit acceptance run is a separate check.
