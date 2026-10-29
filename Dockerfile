FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for OpenCV headless, MediaPipe, and PostgreSQL compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . .

# Verification check to confirm YOLO model loads properly
RUN python -c "from ultralytics import YOLO; model = YOLO('/app/yolo11n.pt'); assert any('phone' in str(name).lower() for name in model.names.values()), 'Bundled model has no phone class'"

# Expose Render default port
ENV HOST=0.0.0.0
ENV PORT=10000

EXPOSE 10000

# Keep worker alive with Gunicorn on 0.0.0.0:10000
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "4", "--timeout", "120", "wsgi:app"]
