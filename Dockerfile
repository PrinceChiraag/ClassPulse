
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python && \
    pip install --no-cache-dir --no-deps opencv-python-headless

COPY . .

ENV HOST=0.0.0.0
ENV PORT=5000
ENV FLASK_DEBUG=0
ENV APP_ENV=production
ENV CLASSPULSE_DB_PATH=/data/classpulse.db
ENV CLASSPULSE_PHONE_MODEL=/app/yolo11n.pt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "wsgi:app"]
