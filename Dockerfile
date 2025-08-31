FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps (se precisar compilar pacotes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# requirements
COPY app/requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# código + assets
COPY app ./app

# garante diretório de uploads
RUN mkdir -p /app/app/assets/uploads

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]