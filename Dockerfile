FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/mplcache

RUN apt-get update \
    && apt-get install -y --no-install-recommends libfreetype6 libpng16-16 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/mplcache

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "ydeposits.py" ]
