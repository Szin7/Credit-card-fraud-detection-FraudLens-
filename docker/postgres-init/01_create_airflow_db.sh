#!/bin/bash
# Airflow needs its own metadata database. Rather than running a second
# Postgres container for it, this creates a second database inside the
# same instance — a portfolio-scale simplification; a production setup
# would typically isolate these. See docs/architecture.md.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE "${AIRFLOW_DB}";
EOSQL
