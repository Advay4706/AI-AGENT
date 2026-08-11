# FastAPI screening service.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install runtime deps first for better layer caching.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# App code + frozen corpus (the API loads data/corpus.json at request time).
COPY src ./src
COPY data/corpus.json ./data/corpus.json

EXPOSE 8000

# ANTHROPIC_API_KEY must be provided at run time, e.g.:
#   docker build -t aml-screen .
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... aml-screen
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
