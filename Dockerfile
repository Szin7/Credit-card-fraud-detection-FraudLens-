# Dashboard image. Only needs src/database + src/utils (config, logging) at
# runtime — the dashboard reads exclusively from Postgres, never touches
# models/ or data/ directly, so nothing ML-related needs to be baked in
# beyond what's already in requirements.txt for the import chain to resolve.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
