FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PILL_APP_CONFIG=configs/default.yaml

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
COPY configs ./configs
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
