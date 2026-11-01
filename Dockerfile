FROM python:3.12-slim-bookworm
WORKDIR /app
COPY requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements-dashboard.txt
RUN useradd --create-home --uid 10001 app
COPY --chown=app:app src ./src
COPY --chown=app:app dashboard ./dashboard
COPY --chown=app:app config ./config
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
