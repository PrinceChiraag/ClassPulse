# ClassPulse

ClassPulse is a final-year CSE project prototype for on-demand classroom engagement and fatigue cues. A teacher starts a browser camera or submits a captured image; Flask runs face-landmark and optional phone-object analysis and returns room-level metrics. The project demonstrates a complete web application, computer-vision pipeline, authenticated API, SQLite persistence, and container deployment.

It is a decision-support prototype, not a validated measure of learning, attention, fatigue, or phone use. Results can be wrong and should never be used to grade, discipline, diagnose, or make decisions about an individual student. Obtain institutional approval and informed consent before any real classroom use.

## Capabilities

- Teacher registration and sign-in with password hashes and signed sessions.
- Responsive teacher portal with camera controls, scan settings, live summaries, and light/dark themes.
- MediaPipe face landmarks and posture cues, with an OpenCV fallback; optional Ultralytics phone-object detection.
- Versioned bearer-token API for a future native mobile client (`docs/mobile-api.md`, `docs/openapi.yaml`).
- SQLite teacher accounts and opt-in named scan records. Camera frames are processed in memory by default; a teacher can explicitly opt into retaining a named scan from Overview.
- Security headers, bounded request size and decoded image dimensions, authenticated application APIs, and production checks for secret configuration.

## Run locally

Requirements: Python 3.10–3.12, a webcam for live capture, and enough memory for OpenCV, MediaPipe, and the YOLO model. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000/login`. The local demo account is `teacher@classpulse.com` / `Teacher@123`; replace it before any shared or public deployment. HTTPS is unnecessary for the localhost camera workflow. Browsers require HTTPS for camera access on non-localhost hosts.

## Deploy with Docker Compose

The image bundles the provided `yolo11n.pt` weights so inference does not need a first-request model download. It uses a single Gunicorn worker because the SQLite database and temporal face tracks are local to one process. Persistent storage is mounted at `/data`. Put the service behind a TLS-terminating reverse proxy on the deployment host; do not expose its port directly to the public internet.

1. Generate a secret: `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
2. Create a `.env` file containing `SECRET_KEY=<generated value>`; keep it out of source control.
3. Build and start: `docker compose up --build -d`.
4. Configure the reverse proxy to forward HTTPS and host headers; the app trusts one proxy hop. Set the platform health check to `/health`.

`compose.yaml` binds the app to `127.0.0.1:5000`, sets production mode, requires the secret, and persists the database in a named volume. Back up that volume and restrict access. SQLite plus in-memory scan tracking are single-instance choices; migrate the database and session/track state before using multiple replicas. Set CPU and memory limits appropriate for inference.

For a managed container host, use the same image, set `APP_ENV=production`, `TRUST_PROXY=1`, `SECRET_KEY`, and `CLASSPULSE_DB_PATH=/data/classpulse.db`, and mount persistent storage at `/data`. The platform must provide HTTPS and a single trusted proxy hop. Production removes the seeded demo account. A valid secret of at least 32 characters is required at startup.

## Configuration

Copy `.env.example` for local configuration. Common deployment settings:

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | Stable, private signing key; required in production. |
| `APP_ENV` | Set to `production` to enable secure cookies, HSTS, and demo-account removal. |
| `TRUST_PROXY` | Set to `1` only behind one trusted TLS-terminating proxy. |
| `CLASSPULSE_DB_PATH` | SQLite database location; mount persistent storage there. |
| `CLASSPULSE_PHONE_MODEL` | Optional path to a compatible Ultralytics model. |
| `FLASK_DEBUG` | Keep `0` outside local development. |

## Privacy and limitations

- Camera permission is requested by the browser after the teacher starts the camera. Stop the camera when the scan is done.
- Submitted frames are processed in memory. A named scan, including its source image and derived metrics, is stored only when the teacher checks the explicit save option and provides a name, class, and roll number. Protect and periodically review the SQLite file; it contains personal data when this feature is used.
- Temporal eye and mouth signals are heuristic; the fallback uses coarse image brightness, edges, and face size and is not a reliable attention detector. Phone detection finds object-like phones and cannot prove ownership or use.
- This repository does not include a representative, consented classroom dataset or validated accuracy claims. The pretrained object model and heuristic metrics require evaluation for the intended population and environment.
- Student scan records are not automatically expired. Apply an institutional retention policy and delete the database when appropriate.

## Final-year project material

See [`docs/final-year-project.md`](docs/final-year-project.md) for the abstract, objectives, system design, evaluation plan, limitations, and suggested report/demo structure. See [`docs/mobile-api.md`](docs/mobile-api.md) for the API contract and [`docs/openapi.yaml`](docs/openapi.yaml) for its OpenAPI definition.
