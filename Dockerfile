
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Validate the OpenCV API Ultralytics imports and confirm bundled model weights.
# Full model deserialization is left to runtime startup/health checks.
RUN python -c "from pathlib import Path; import cv2; from ultralytics import YOLO; weights = Path('/app/yolo11n.pt'); assert hasattr(cv2, 'imshow'), 'OpenCV build is incompatible with Ultralytics'; assert weights.is_file() and weights.stat().st_size > 0, 'Bundled phone detector weights are missing or empty'; print('OpenCV, Ultralytics, and bundled detector weights are ready')"

ENV HOST=0.0.0.0
ENV PORT=5000
ENV FLASK_DEBUG=0
ENV APP_ENV=production
ENV CLASSPULSE_DB_PATH=/data/classpulse.db
ENV CLASSPULSE_PHONE_MODEL=/app/yolo11n.pt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "wsgi:app"]
