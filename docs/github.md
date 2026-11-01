# Repository guide

FraudLens is maintained in this repository: [Szin7 / Credit-card-fraud-detection-FraudLens-](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-).

This guide is for navigating, demonstrating and updating the existing project. The [README](../README.md) is the main entry point for new readers.

## Start here

| Goal | File or location |
| --- | --- |
| Understand the project and run a demo | [README](../README.md) |
| Review dataset preparation and real-model results | [Real-data experiments](real_data_experiments.md) and [model card](real_model_card.md) |
| Understand the application design | [Architecture](architecture.md) |
| Run the application or investigate failures | [Runbook](runbook.md) |
| Check the Kaggle-trained model inside Docker | [Real-model container guide](real_model_container.md) |
| Verify that the tested functionality works | [Verification records](verification.md) |
| Prepare an interview demonstration | [Interview guide](interview_guide.md) |

## Repository structure

The repository separates application code (`src/`), the Streamlit user interface (`dashboard/`), Airflow DAGs (`airflow/dags/`), runnable scripts (`scripts/`), experiments (`notebooks/`), tests (`tests/`) and documentation (`docs/`). Docker and CI configuration live at the repository root and in `.github/workflows/`.

The raw ULB/Kaggle dataset, local SQLite databases and serialized model files are intentionally not committed. The trusted XGBoost archive is distributed separately through the [`kaggle-model` GitHub Release](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/releases/tag/kaggle-model).

## Check a change before merging it

Use a separate branch for edits and keep unrelated changes out of the same commit. In WSL, from the repository root:

```bash
git switch main
git pull --ff-only origin main
git switch -c docs-or-fix-description

# Make the changes, then inspect what actually changed.
git diff --check
git diff --stat
ruff check .
pytest -q
```

Commit only the intended files, push the branch and open a pull request into `main`. Review the diff and wait for CI checks before merging. Check that there are no local datasets, secrets, `.env` files, database files or model binaries in the staged changes.

For workflow evidence, use the [current CI run](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41326322968), [full-stack acceptance](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41317199864) and [real-model integration test](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41322535119). The latter two were run against earlier project commits; rerun manual workflows on the current branch when fresh acceptance evidence is required.

## Presentation notes

Open the four Streamlit pages and explain the data flow, model-selection decision and the difference between held-out test metrics and synthetic integration-test inputs. Screenshots or a short demo recording should come from an actual successful run. Don't describe a successful GitHub Actions test as a permanently deployed website.
