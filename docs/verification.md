# Verification record

This record separates execution evidence from configured but unexecuted checks.

| Check | Result |
|---|---|
| Unit/regression suite | 54 passed |
| PostgreSQL integration tests | 2 skipped locally; CI service configured |
| Line coverage of `src` and `scripts` | 85% in the recorded pytest run |
| Ruff correctness checks (`E9`, `F`) | Passed |
| Full five-candidate synthetic training | Completed |
| Unlabeled batch inference | 24,120 valid rows scored |
| Same-input replay | Zero extra transactions and predictions; source checksum unchanged |
| Dashboard pages | All four passed Streamlit AppTest without application exceptions |
| Docker image builds and Airflow DAG execution | Not executed in this environment |
| GitHub Actions | Workflow supplied; no remote run yet |
| Real-data model evaluation | Not performed |
| AWS deployment | Not performed |

Runtime: Linux, Python 3.12; installed application versions match requirements.txt.
The optional notebook/SHAP extras were not needed for the tested application paths.
The dashboard check exercises page execution, not a browser-based visual acceptance
review. Coverage reflects the test suite only; it does not include the separate full
training or dashboard checks.

PostgreSQL could not be started here because the runtime does not support the OS
user setup required by the server. Docker is not installed. CI supplies PostgreSQL
16 and builds the images; its success must be checked after the repository is pushed.
The release should not be described as fully container-verified until then.

The supplied CSV and notebook used Windows line endings. These have been normalized
to LF without changing their values or notebook content. The notebook is exploratory;
current model results are in `docs/results/synthetic-benchmark.json`.

## Material changes

Unlabeled scoring, finite-value validation, stable identity, atomic/replay-safe
persistence, delayed labels, separate batch paths, immutable model releases,
validation-only threshold selection, chronological splitting, strict JSON, isolated
Airflow application dependencies, CI, regression tests, and a SQLite demo command.

## Remaining acceptance work

Run the GitHub workflow, execute the Docker runbook, and add a real-data experiment.
These remaining gates are why this release is not labeled “production-ready” or
assigned a guaranteed résumé score.
