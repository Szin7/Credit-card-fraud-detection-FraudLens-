"""Exercise the Airflow/SQL/Streamlit stack without installing Docker locally.

Designed for the optional GitHub Actions Compose smoke test. Requires a
running Docker Compose stack with PostgreSQL and both FraudLens DAGs.
"""
from __future__ import annotations

import subprocess
import time
import urllib.request
import shlex


def run(*args: str) -> str:
    process = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(args)}\n{process.stdout[-3000:]}")
    return process.stdout.strip()


def db(sql: str) -> str:
    # DB name and user are read inside the container; credentials are not printed.
    script = 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -c ' + shlex.quote(sql)
    return run("docker", "compose", "exec", "-T", "postgres", "sh", "-c", script).strip()


def count(sql: str) -> int:
    return int(db(sql).splitlines()[-1])


def wait_stage(kind: str, minimum: int, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = db(
            "SELECT COALESCE((SELECT status FROM pipeline_runs "
            f"WHERE run_type='{kind}' ORDER BY started_at DESC LIMIT 1), 'pending')"
        )
        if result == "failed":
            reason = db(
                "SELECT COALESCE(error_message,'unknown') FROM pipeline_runs "
                f"WHERE run_type='{kind}' ORDER BY started_at DESC LIMIT 1"
            )
            raise RuntimeError(f"{kind} failed: {reason}")
        done = count(f"SELECT count(*) FROM pipeline_runs WHERE run_type='{kind}' AND status='succeeded'")
        if done >= minimum:
            print(f"{kind} succeeded ({done} recorded run(s))", flush=True)
            return
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for {kind}; inspect Airflow task logs")



def wait_for_dags(timeout: int = 360) -> None:
    """Wait until Airflow registers both DAGs in its metadata database."""
    sql = (
        "SELECT count(*) FROM dag WHERE is_active = true "
        "AND dag_id IN ("
        "'fraud_training_pipeline', 'fraud_inference_pipeline')"
    )
    shell = (
        'psql -U "$POSTGRES_USER" -d "$AIRFLOW_DB" -At -c '
        + shlex.quote(sql)
    )

    deadline = time.monotonic() + timeout
    last_error = None

    while time.monotonic() < deadline:
        try:
            output = run(
                "docker", "compose", "exec", "-T",
                "postgres", "sh", "-c", shell
            )
            if int(output.splitlines()[-1]) == 2:
                print("Both Airflow DAGs registered.", flush=True)
                return
        except (RuntimeError, ValueError) as exc:
            last_error = str(exc)

        time.sleep(5)

    raise TimeoutError(
        f"Airflow DAG registration timed out. Last error: {last_error}"
    )


def main() -> None:
    assert count("SELECT count(*) FROM information_schema.tables WHERE table_name='pipeline_runs'") >= 1
    print("PostgreSQL application schema found", flush=True)
    wait_for_dags()

    # Keep scheduled inference paused until the training release exists.
    print(run("docker", "compose", "exec", "-T", "airflow-scheduler", "airflow", "dags", "unpause", "fraud_training_pipeline"))
    print(run("docker", "compose", "exec", "-T", "airflow-scheduler", "airflow", "dags", "trigger", "fraud_training_pipeline"))
    wait_stage("training", minimum=1)
    champion = count("SELECT count(*) FROM model_runs WHERE is_champion=true")
    if champion != 1:
        raise AssertionError(f"Expected one champion model, got {champion}")

    print(run("docker", "compose", "exec", "-T", "airflow-scheduler", "airflow", "dags", "unpause", "fraud_inference_pipeline"))
    before_runs = count("SELECT count(*) FROM pipeline_runs WHERE run_type='inference' AND status='succeeded'")
    print(run("docker", "compose", "exec", "-T", "airflow-scheduler", "airflow", "dags", "trigger", "fraud_inference_pipeline"))
    wait_stage("inference", minimum=before_runs + 1)
    stored = count("SELECT count(*) FROM fraud_predictions")
    transactions = count("SELECT count(*) FROM transactions")
    if not (stored == transactions == 6080):
        raise AssertionError(f"Unexpected database counts: {stored} predictions, {transactions} transactions")
    print(f"PostgreSQL contains {stored:,} scored synthetic transactions", flush=True)

    # Replay the identical data; uniqueness rules should prevent new decisions.
    before_replay = count("SELECT count(*) FROM pipeline_runs WHERE run_type='inference' AND status='succeeded'")
    print(run("docker", "compose", "exec", "-T", "airflow-scheduler", "airflow", "dags", "trigger", "fraud_inference_pipeline"))
    wait_stage("inference", minimum=before_replay + 1)
    after = count("SELECT count(*) FROM fraud_predictions")
    if after != stored:
        raise AssertionError(f"Replay duplicated predictions: before={stored} after={after}")

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=5) as response:
                if response.status == 200:
                    print("Containerized Streamlit health endpoint responded OK", flush=True)
                    check = (
                        "from src.database import repository as r; "
                        "s=r.get_overview_stats(); m=r.get_champion_model(); "
                        "assert s['total_transactions']==6080, s; "
                        "assert m and m['model_type'], m; "
                        "print('Dashboard database read and champion metadata: OK')"
                    )
                    print(run("docker", "compose", "exec", "-T", "dashboard", "python", "-c", check))
                    print("Docker/Airflow/PostgreSQL smoke test passed", flush=True)
                    return
        except OSError:
            pass
        time.sleep(5)
    raise TimeoutError("Streamlit health endpoint did not respond")


if __name__ == "__main__":
    main()
