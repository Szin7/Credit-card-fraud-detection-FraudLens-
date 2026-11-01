# Verification and evidence

This page records what FraudLens has demonstrated, with links to actual GitHub Actions runs. A passed workflow verifies the code and configuration it ran; it is not evidence of a permanent public deployment.

## Verified checks

| Check | Result | Evidence |
| --- | --- | --- |
| Python tests, linting, container builds and DAG imports | Passed on the current `main` commit `ed60d4a` | [CI run](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41326322968) |
| Training, inference, PostgreSQL persistence and Streamlit connectivity | Passed with synthetic test data | [Full-stack acceptance](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41317199864) |
| Saved Kaggle-trained XGBoost release imported into Docker | Passed; model registered and used without retraining | [Real-model pipeline check](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41322535119) |
| Repeat inference adds no identical decisions | Passed in local and containerized checks | [Real-model pipeline check](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41322535119) |
| Local real-model dashboard | Four pages opened and reviewed; SQLite demonstration completed | [Real-data model card](real_model_card.md) |
| Local Python checks | Last reported WSL check: 65 passed, 2 skipped; Ruff clean | [Completion checklist](completion_checklist.md) |

The full-stack and real-model acceptance runs were completed on earlier commits. The current single-commit `main` snapshot contains the same implementation, but the **real-model acceptance workflow has not yet been rerun on `ed60d4a`**. That distinction matters when referring to evidence by commit SHA.

## Research evaluation versus system testing

The selected XGBoost model was evaluated on **59,888 real-data holdout rows**, including **76 known fraud cases**. It recorded precision 0.8852, recall 0.7105, F1 0.7883, average precision 0.7954 and ROC-AUC 0.9737.

The local 3,076-row dashboard demonstration intentionally included every fraud example in a selected holdout subset. It is not an independent estimate of deployment alert volume or model accuracy. The container acceptance used **6,080 synthetic scoring inputs** with the real-trained model; that verifies deployment compatibility and data flow, not real-world detection performance.

For the source figures and method, see [recorded benchmark results](results/real-data-benchmark.json) and the [real-data model card](real_model_card.md).

## Repeat the checks

In a configured Python 3.12 environment:

```bash
ruff check .
pytest -q
```

The CI workflow runs tests against PostgreSQL and builds the images. To repeat full integration acceptance, open GitHub Actions and manually run **Full stack check** or **Real model pipeline check** on the current `main` branch. The latter requires the trusted `kaggle-model` release and its archive SHA-256; see the [container guide](real_model_container.md).

The runners and databases are temporary. Local Docker/Desktop recovery, dashboard authentication, automated drift monitoring and production security review remain outside these acceptance checks.
