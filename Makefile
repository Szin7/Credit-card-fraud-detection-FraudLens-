# Convenience wrapper around the same commands documented in README.md.
# Assumes a Unix-style venv at .venv/bin/python (created by `make setup`).
# On native Windows without WSL/Git Bash's `make`, run the command shown
# in each target directly with `.venv\Scripts\python.exe` instead.

PYTHON := .venv/bin/python

.PHONY: setup sample-data ingest validate train threshold predict pipeline \
        test dashboard notebook up down logs db-init clean

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

sample-data:
	$(PYTHON) -m scripts.generate_sample_data

ingest:
	$(PYTHON) -m src.ingestion.ingest

validate:
	$(PYTHON) -m src.validation.validator

train:
	$(PYTHON) -m src.models.train

threshold:
	$(PYTHON) -m scripts.optimize_threshold

predict:
	$(PYTHON) -m src.models.predict

# Full local run: train a model, then score the same validated batch with it.
pipeline:
	$(PYTHON) -m scripts.run_pipeline training
	$(PYTHON) -m scripts.run_pipeline inference

test:
	$(PYTHON) -m pytest tests/ -v

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

notebook:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		notebooks/01_exploration.ipynb --ExecutePreprocessor.kernel_name=python3

db-init:
	$(PYTHON) -c "from src.database.connection import init_db; init_db()"

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
