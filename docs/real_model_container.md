# Running the Kaggle-trained model with Docker

This guide uses the **existing** XGBoost release trained on the ULB/Kaggle credit-card data. It does not retrain the model, replace the original GitHub model card, or store the serialized model in Git history.

The release was evaluated separately on 59,888 chronologically held-out real transactions: precision 0.8852, recall 0.7105 and average precision 0.7954. The integration checks below test whether that exact saved release can score transactions inside Airflow and write results to PostgreSQL. They are **not another independent measurement of accuracy**.

## What changes

The optional `model-import` container reads an explicitly supplied ZIP file from `/incoming`, verifies the archive SHA-256 and the model's internal checksum, installs it into the shared Docker model volume, registers its metadata in PostgreSQL, and updates the active model pointer. It does not deserialize the model. During inference, the existing application loads the trusted joblib artifact under its pinned Python environment.

Normal `docker compose up` is unchanged. `model-import` is defined in the additional `docker-compose.real.yml` file and is only started explicitly. Importing the same valid release twice is safe. **Leave `fraud_training_pipeline` paused:** running training again would intentionally select and publish a different model.

### Check the ZIP on your laptop, without Docker

In WSL, from the FraudLens repository:

```bash
cd ~/Credit-card-fraud-detection-FraudLens-
source .venv/bin/activate
mkdir -p data/incoming
cp /path/to/fraudlens-model-release.zip data/incoming/fraudlens-model-release.zip
sha256sum data/incoming/fraudlens-model-release.zip
```

Record the 64-character checksum. Use it for the next command; **do not replace it with the hash of `model.joblib`**.

```bash
python -m scripts.deploy_real_model \
  --archive data/incoming/fraudlens-model-release.zip \
  --expected-archive-sha256 YOUR_ZIP_SHA256 \
  --check-only
```

This check reads the release archive and its metadata. It does not open PostgreSQL, install the model or change your existing local active model.

## Option A: GitHub-hosted Docker verification

This method avoids Docker image builds on your laptop. **GitHub Releases in a public repository are publicly downloadable.** Only do this if you are comfortable publishing your trained-model archive. Check the model/dataset's licensing terms first. Do not publish the raw Kaggle CSV or the 3,076-row real demonstration CSV as a release asset.

1. Open the published [`kaggle-model` release](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/releases/tag/kaggle-model). Its trusted archive is attached as `fraudlens-model-release.zip`. The archive is public; do not upload raw transaction data.
2. Open **Actions → Real model pipeline check → Run workflow** on `main`. Enter `kaggle-model` and the exact **ZIP SHA-256** recorded when the trusted archive was built.
3. Open the run and inspect the model import, inference, replay and dashboard checks. The test runs on a temporary GitHub-hosted machine.

The workflow builds containers on GitHub's runner, downloads the ZIP from the repository release, verifies its SHA-256, registers the Kaggle-trained release, and scores **6,080 synthetic test transactions** with it. It then checks that replay does not duplicate predictions and that Streamlit reads the registered real-model metrics from PostgreSQL. The synthetic **scoring inputs are for system acceptance**; do not report the resulting alerts as fraud-detection accuracy on real transactions.

The GitHub runner and its database/model volumes are temporary. The workflow does not deploy a publicly reachable, persistent FraudLens server.

## Option B: Local Docker with your 3,076 real-feature holdout examples

Use this path only when Docker Desktop's Linux engine and WSL integration are healthy and the machine has enough free disk space for the Airflow images and database volumes. The GitHub-hosted option avoids a large local build.

Copy the same trusted ZIP into `data/incoming` as above, and copy the already-downloaded unlabeled sample:

```bash
cp /path/to/fraudlens-holdout-demo.csv \
  data/incoming/fraudlens-holdout-demo.csv
```

In your untracked `.env`, set `INFERENCE_INPUT=/incoming/fraudlens-holdout-demo.csv`, `POSTGRES_PORT=5434`, and `DASHBOARD_PORT=8504`. Keep `TRAINING_INPUT` empty. Ensure host ports 5434 and 8504 are free. Use the **same project name** in each command to keep its database and model volume separate from other labs.

```bash
# Builds require several gigabytes of free disk space.
docker compose -p fraudlens-real build dashboard airflow-init
docker compose -p fraudlens-real up -d --wait postgres
docker compose -p fraudlens-real up --exit-code-from airflow-init airflow-init

export MODEL_ARCHIVE_SHA256=YOUR_ZIP_SHA256
docker compose -p fraudlens-real -f docker-compose.yml -f docker-compose.real.yml \
  run --rm model-import

# Start only what is needed for inference and the dashboard.
docker compose -p fraudlens-real up -d airflow-scheduler dashboard
```

After Airflow has registered the `fraud_inference_pipeline` DAG, trigger it:

```bash
docker compose -p fraudlens-real exec airflow-scheduler airflow dags list
docker compose -p fraudlens-real exec airflow-scheduler airflow dags unpause fraud_inference_pipeline
docker compose -p fraudlens-real exec airflow-scheduler airflow dags trigger fraud_inference_pipeline
```

Use `docker compose -p fraudlens-real logs airflow-scheduler` and check Pipeline Health. The dashboard should be at <http://localhost:8504>. Model Performance should show the Kaggle XGBoost holdout results, and Overview should show the scored real-feature sample (3,076 rows). The earlier WSL run flagged 54, but independently confirm the count here rather than assuming it.

To check stored predictions:

```bash
docker compose -p fraudlens-real exec postgres sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -c "SELECT count(*) FROM fraud_predictions"'
```

Trigger inference again and confirm the count remains unchanged. Shut down safely with `docker compose -p fraudlens-real down` (**without** `-v`). The model and database remain in this project's named volumes.

## Important distinctions

- **Trained on Kaggle:** a real-data XGBoost model with held-out evaluation.
- **GitHub cloud integration:** that saved model scores synthetic input inside Docker; checks deployment correctness, not accuracy.
- **Local real-feature integration (optional):** the same saved model scores the 3,076 real-feature holdout examples and displays their predictions in Streamlit.
- **Security:** joblib is a Python serialization format that can execute code when loaded. Only import and run model archives you produced or independently trust. The SHA-256 proves that the file matches the one you checked; it does not by itself prove the model's source is safe.
