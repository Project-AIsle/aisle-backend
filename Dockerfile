FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps (se precisar compilar pacotes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

# requirements
COPY app/requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# código + assets
COPY . /app

# garante diretório de uploads
RUN mkdir -p /app/app/assets/uploads/items
ENV UPLOAD_PATH /app/app/assets/uploads

EXPOSE 8443
CMD ["hypercorn", "-c", "hypercorn.toml", "app.main:app"]