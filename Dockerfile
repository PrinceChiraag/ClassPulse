
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

# Validate that the detector package and bundled weights are present without
# deserializing the PyTorch model during the image build. Full model loading is
# performed by the app at runtime and exposed through /health.
RUN python -c "from pathlib import Path; from ultralytics import YOLO; weights = Path('/app/yolo11n.pt'); assert weights.is_file() and weights.stat().st_size > 0, 'Bundled phone detector weights are missing or empty'; print('Ultralytics installed; bundled phone detector weights present')"

ENV HOST=0.0.0.0
ENV PORT=5000
ENV FLASK_DEBUG=0
ENV APP_ENV=production
ENV CLASSPULSE_DB_PATH=/data/classpulse.db
ENV CLASSPULSE_PHONE_MODEL=/app/yolo11n.pt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "wsgi:app"]
