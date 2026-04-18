FROM python:3.10-slim

ENV PYTHONDONTWRITECODE=1 \
    PYTHONUNBUFFERED=1


WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 5000


CMD ["python","app.py"]