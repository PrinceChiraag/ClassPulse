FROM python:3.12-slim

WORKDIR /app

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

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:5000", "wsgi:app"]
