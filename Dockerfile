
FROM python:3.12-slim

WORKDIR /app

# Install system libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY yolo11n.pt ./yolo11n.pt

ENV HOST=0.0.0.0
ENV PORT=5000
ENV FLASK_DEBUG=0
ENV APP_ENV=production
ENV CLASSPULSE_DB_PATH=/data/classpulse.db
ENV CLASSPULSE_PHONE_MODEL=/app/yolo11n.pt

VOLUME ["/data"]
EXPOSE 5000
