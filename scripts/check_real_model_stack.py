"""Acceptance test: score a batch using the imported real-trained model."""

from __future__ import annotations

import argparse
import time
import urllib.request

from scripts.check_full_stack import count, db, run, wait_for_dags, wait_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-rows", type=int, default=6080)
    args = parser.parse_args()

    if count("SELECT count(*) FROM model_runs WHERE is_champion AND model_type='xgboost' AND curve_data->>'dataset_kind'='real'") != 1:
        raise AssertionError("The registered champion is not a real-data XGBoost model")
    version = db("SELECT model_version FROM model_runs WHERE is_champion").strip()
    if count("SELECT count(*) FROM pipeline_runs WHERE run_type='training'"):
        raise AssertionError("Unexpected training execution; this test should reuse the Kaggle release")
    if count("SELECT count(*) FROM fraud_predictions"):
        raise AssertionError("The database must start without previous predictions")
    print(f"Imported model registered: {version}", flush=True)

    wait_for_dags()
    # Only inference is enabled. The training DAG stays paused.
    print(run("docker", "compose", "exec", "-T", "airflow-scheduler", "airflow", "dags", "unpause", "fraud_inference_pipeline"))
    before = count("SELECT count(*) FROM pipeline_runs WHERE run_type='inference' AND status='succeeded'")
    print(run("docker", "compose", "exec", "-T", "airflow-scheduler", "airflow", "dags", "trigger", "fraud_inference_pipeline"))
    wait_stage("inference", minimum=before + 1)

    stored = count("SELECT count(*) FROM fraud_predictions")
    if stored != args.expected_rows:
        raise AssertionError(f"Expected {args.expected_rows} predictions, found {stored}")
    matched = count(
        "SELECT count(*) FROM fraud_predictions "
        "WHERE model_version=(SELECT model_version FROM model_runs WHERE is_champion)"
    )
    if matched != stored:
        raise AssertionError("Predictions were not made with the imported release")
    if count("SELECT count(*) FROM transactions") != stored:
        raise AssertionError("Mismatch between stored transactions and predictions")
    print(f"Scored {stored:,} transactions with the real-trained model", flush=True)

    before = count("SELECT count(*) FROM pipeline_runs WHERE run_type='inference' AND status='succeeded'")
    print(run("docker", "compose", "exec", "-T", "airflow-scheduler", "airflow", "dags", "trigger", "fraud_inference_pipeline"))
    wait_stage("inference", minimum=before + 1)
    if count("SELECT count(*) FROM fraud_predictions") != stored:
        raise AssertionError("Replay generated duplicate predictions")
    print("Repeated inference added no new decisions", flush=True)

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=5) as resp:
                if resp.status == 200:
                    code = (
                        "from src.database import repository as r; "
                        "s=r.get_overview_stats(); m=r.get_champion_model(); "
                        f"assert s['total_transactions']=={stored},s; "
                        "assert m and m['model_type']=='xgboost',m; "
                        "assert m['curve_data']['dataset_kind']=='real',m; "
                        "print('Streamlit database and imported metrics OK')"
                    )
                    print(run("docker", "compose", "exec", "-T", "dashboard", "python", "-c", code))
                    print("Real-model Docker/Airflow/PostgreSQL acceptance passed", flush=True)
                    return
        except OSError:
            pass
        time.sleep(5)
    raise TimeoutError("Streamlit health endpoint did not respond")


if __name__ == "__main__":
    main()
