FROM python:3.12-slim-bookworm AS application
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv --copies /opt/fraudlens-venv \
    && /opt/fraudlens-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

FROM apache/airflow:2.10.5-python3.12
USER root
COPY --from=application /opt/fraudlens-venv /opt/fraudlens-venv
COPY --chown=airflow:root src /opt/fraudlens/src
COPY --chown=airflow:root scripts /opt/fraudlens/scripts
COPY --chown=airflow:root config /opt/fraudlens/config
COPY --chown=airflow:root data/raw/creditcard_sample.csv /opt/fraudlens/data/raw/creditcard_sample.csv
RUN mkdir -p /opt/fraudlens/data/processed /opt/fraudlens/models \
    && chown -R airflow:root /opt/fraudlens
ENV PYTHONPATH=/opt/fraudlens FRAUDLENS_ROOT=/opt/fraudlens
USER airflow
