web: CLASSPULSE_PHONE_MODEL=./yolo11n.pt gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 --access-logfile - --error-logfile - wsgi:app
