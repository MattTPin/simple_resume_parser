# ---- Use slim Python 3.11 ----
FROM python:3.11-slim

# ---- Metadata / Environment ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC \
    TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache \
    API_AUTO_START=true

WORKDIR /app

# ---- System dependencies ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        git \
        wget \
        bash \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/cache

# ---- Copy requirements and install Python dependencies ----
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ---- Copy app code ----
COPY . .

# ---- Fix entrypoint.sh permissions and line endings ----
RUN sed -i 's/\r$//' /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# ---- Entrypoint ----
ENTRYPOINT ["bash", "/app/entrypoint.sh"]
