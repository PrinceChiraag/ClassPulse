
web: CLASSPULSE_PHONE_MODEL=./models/phone_best.pt gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 --access-logfile - --error-logfile - wsgi:app
