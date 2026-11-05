# Usa una imagen base de Python oficial y ligera
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8003

# /health answers 503 when the recommendation engine failed to load its data.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/health', timeout=5)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]