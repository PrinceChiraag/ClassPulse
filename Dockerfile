
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
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and required model files
COPY . .

# Verify both YOLO models load correctly
RUN python -c "from ultralytics import YOLO; \
general = YOLO('/app/yolo11n.pt'); \
phone = YOLO('/app/models/phone_best.pt'); \
assert any('person' in str(n).lower() for n in general.names.values()), 'Person class missing'; \
assert any('phone' == str(n).lower() for n in phone.names.values()), 'Phone class missing'; \
print('Both models loaded successfully')"

# Production configuration
ENV HOST=0.0.0.0
ENV PORT=10000
ENV FLASK_DEBUG=0
ENV APP_ENV=production
ENV CLASSPULSE_PHONE_MODEL=/app/models/phone_best.pt

EXPOSE 10000

# Start Flask through Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "4", "--timeout", "120", "wsgi:app"]
