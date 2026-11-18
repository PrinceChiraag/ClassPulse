
FROM python:3.12-slim

WORKDIR /app

# System dependencies for OpenCV, MediaPipe and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
# Install CPU-only PyTorch first so Ultralytics does not pull CUDA runtime wheels.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source and required model files
COPY . .

# Verify both models and run CPU inference for phone and general objects.
RUN python -c "import numpy as np; from ultralytics import YOLO; \
general = YOLO('/app/yolo11n.pt'); \
phone = YOLO('/app/models/phone_best.pt'); \
assert any('person' in str(n).lower() for n in general.names.values()), 'Person class missing'; \
phone_ids = [i for i, n in phone.names.items() if str(n).strip().lower() in {'phone', 'smartphone', 'cell phone', 'mobile phone', 'handset', 'cellular phone'}]; \
assert phone_ids, f'Phone class missing: {phone.names}'; \
frame = np.zeros((480, 640, 3), dtype=np.uint8); \
phone_result = phone.predict(source=frame, classes=phone_ids, conf=0.25, imgsz=640, device='cpu', verbose=False)[0]; \
object_result = general.predict(source=frame, conf=0.25, imgsz=640, device='cpu', verbose=False)[0]; \
assert phone_result.boxes is not None and object_result.boxes is not None, 'CPU inference returned no boxes object'; \
print(f'Models loaded; phone classes={phone_ids}; phone and general CPU inference completed')"

# Production configuration
ENV HOST=0.0.0.0
ENV PORT=10000
ENV FLASK_DEBUG=0
ENV APP_ENV=production
ENV CLASSPULSE_PHONE_MODEL=/app/models/phone_best.pt

EXPOSE 10000

# Start Flask through Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "4", "--timeout", "120", "wsgi:app"]
