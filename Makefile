PYTHON ?= .venv/bin/python
.PHONY: setup demo test lint pipeline dashboard up down db-init
setup:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt
demo:
	$(PYTHON) -m scripts.demo --dashboard
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check .
pipeline:
	$(PYTHON) -m scripts.run_pipeline training --dataset-kind synthetic
	$(PYTHON) -m scripts.run_pipeline inference
dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py
db-init:
	$(PYTHON) -c "from src.database.connection import init_db; init_db()"
up:
	docker compose up --build -d
down:
	docker compose down
