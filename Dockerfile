FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# YOLO model verification
RUN python -c "from ultralytics import YOLO; model = YOLO('/app/yolo11n.pt'); assert any('phone' in str(name).lower() for name in model.names.values()), 'Bundled model has no phone class'"

ENV HOST=0.0.0.0
ENV PORT=10000

EXPOSE 10000

CMD ["python", "app.py"]