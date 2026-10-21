import base64
import logging
import os
import re
import secrets
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, g, jsonify, redirect, render_template, request, session, url_for
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - fallback when dependency is unavailable
    mp = None

from classpulse.metrics import compute_attention_fatigue
from classpulse.phone_detection import detect_phones, phone_detector_runtime_status
from classpulse.store import TeacherStore, TelemetryStore, validate_email


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")
_scan_tracks = {}
_scan_tracks_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _face_bbox(face_landmarks, frame_width: int, frame_height: int) -> dict:
    if not face_landmarks:
        return {"x": 0, "y": 0, "width": 0, "height": 0}

    xs = [point.x for point in face_landmarks]
    ys = [point.y for point in face_landmarks]
    x0 = min(xs) * frame_width
    y0 = min(ys) * frame_height
    x1 = max(xs) * frame_width
    y1 = max(ys) * frame_height
    return {
        "x": int(round(x0)),
        "y": int(round(y0)),
        "width": max(1, int(round(x1 - x0))),
        "height": max(1, int(round(y1 - y0))),
    }


def build_face_summary(metrics: dict) -> dict:
    attention = float(metrics.get("attention", 0.0))
    fatigue = float(metrics.get("fatigue", 0.0))
    eye_engagement = float(metrics.get("eye_engagement", 0.0))
    posture_score = float(metrics.get("posture_score", 0.0))
    face_detected = bool(metrics.get("face_detected", metrics.get("face_count", 0) > 0))

    drowsiness_state = metrics.get("drowsiness_state", "Alert")
    fatigue_index = float(metrics.get("fatigue_index", 0.0))
    microsleep = bool(metrics.get("microsleep", False))
    yawning = bool(metrics.get("yawning", False))

    if not face_detected:
        status = "No clear face detected"
        risk_level = "Needs reset"
        recommendation = "Adjust lighting, keep the student facing the camera, and retry the scan with the face fully visible."
        confidence = 0.0
    elif drowsiness_state == "Yawning":
        status = "Yawning"
        risk_level = "High"
        recommendation = "Yawning is increasing. Pause for a short reset and re-engage the class with a quick verbal prompt."
        confidence = round(max(0.0, min(100.0, (attention * 0.45) + (fatigue_index * 0.35) + (posture_score * 100 * 0.2))), 1)
    elif drowsiness_state == "Drowsy" or microsleep:
        status = "Drowsy"
        risk_level = "High"
        recommendation = "The student shows drowsiness signs. Encourage movement, hydration, or a short activity break."
        confidence = round(max(0.0, min(100.0, (attention * 0.5) + (fatigue_index * 0.4) + (posture_score * 100 * 0.1))), 1)
    elif attention >= 80:
        status = "Focused"
        risk_level = "Low"
        recommendation = "Strong attention and stable posture. Continue the current teaching flow."
        confidence = round(max(0.0, min(100.0, (attention * 0.6) + (posture_score * 100 * 0.25) + (eye_engagement * 100 * 0.15))), 1)
    elif attention >= 60:
        status = "Watch"
        risk_level = "Moderate"
        recommendation = "Slight fatigue risk. Encourage a quick engagement reset or brief pause."
        confidence = round(max(0.0, min(100.0, (attention * 0.55) + (posture_score * 100 * 0.3) + (eye_engagement * 100 * 0.15))), 1)
    else:
        status = "High fatigue risk"
        risk_level = "High"
        recommendation = "Attention is dipping. Pause and re-engage the class with a clear instruction or activity change."
        confidence = round(max(0.0, min(100.0, (attention * 0.5) + (posture_score * 100 * 0.35) + (eye_engagement * 100 * 0.15))), 1)

    overall_score = round((attention * 0.55) + (posture_score * 100 * 0.25) + (eye_engagement * 100 * 0.2), 1)

    summary = {
        "attention": round(attention, 1),
        "fatigue": round(fatigue, 3),
        "eye_engagement": round(eye_engagement, 3),
        "posture_score": round(posture_score, 3),
        "confidence": confidence,
        "overall_score": round(max(0.0, min(100.0, overall_score)), 1),
        "status": status,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "face_detected": face_detected,
        "drowsiness_state": drowsiness_state,
        "fatigue_index": round(fatigue_index, 1),
        "microsleep": microsleep,
        "yawning": yawning,
        "face_count": metrics.get("face_count", 1 if face_detected else 0),
        "faces": metrics.get("faces", []),
        "analysis_mode": metrics.get("analysis_mode", "landmark"),
    }
    if "student_details" in metrics:
        summary["student_details"] = metrics["student_details"]
    return summary


def _fallback_face_metrics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = ()
    # Try the more tolerant frontal models when the default cascade misses a
    # face, so a MediaPipe failure does not automatically become an empty scan.
    for cascade_name, neighbors in (
        ("haarcascade_frontalface_default.xml", 4),
        ("haarcascade_frontalface_alt2.xml", 3),
        ("haarcascade_frontalface_alt.xml", 3),
    ):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
        if cascade.empty():
            continue
        faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=neighbors, minSize=(32, 32))
        if len(faces):
            break

    if len(faces) == 0:
        return {
            "attention": 0.0,
            "fatigue": 1.0,
            "eye_engagement": 0.0,
            "posture_score": 0.0,
            "face_detected": False,
            "face_count": 0,
            "faces": [],
        }

    entries = []
    frame_area = max(frame.shape[0] * frame.shape[1], 1)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    for index, (x, y, w, h) in enumerate(faces):
        face = gray[y : y + h, x : x + w]
        face_area = w * h
        brightness = float(face.mean()) / 255.0
        edge_focus = float(laplacian[y : y + h, x : x + w].var()) / 255.0
        posture_score = max(0.0, min(1.0, 0.55 + (brightness * 0.25) + (min(edge_focus, 1.0) * 0.2)))
        attention = max(0.0, min(100.0, (face_area / frame_area) * 1000 + brightness * 45 + edge_focus * 25))
        eye_engagement = max(0.0, min(1.0, brightness * 1.5 + edge_focus * 0.4))
        fatigue = max(0.0, min(1.0, (1.0 - (attention / 100.0)) * 0.8 + (1.0 - posture_score) * 0.35))
        entries.append({"attention": attention, "fatigue": fatigue, "fatigue_index": round(fatigue * 100, 1),
                        "drowsiness_state": "Basic estimate", "microsleep": False, "yawning": False,
                        "eye_engagement": eye_engagement,
                        "posture_score": posture_score, "label": f"Student {index + 1}",
                        "face_index": index, "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}})
    if len(entries) == 1:
        return {**entries[0], "face_detected": True, "face_count": 1, "faces": entries, "analysis_mode": "basic-fallback"}
    return {"attention": round(sum(e["attention"] for e in entries) / len(entries), 1),
            "fatigue": round(sum(e["fatigue"] for e in entries) / len(entries), 3),
            "eye_engagement": round(sum(e["eye_engagement"] for e in entries) / len(entries), 3),
            "posture_score": round(sum(e["posture_score"] for e in entries) / len(entries), 3),
            "face_detected": True, "face_count": len(entries), "faces": entries, "analysis_mode": "basic-fallback"}


def _aggregate_face_scan(face_entries):
    if not face_entries:
        return build_face_summary({"attention": 0.0, "fatigue": 1.0, "eye_engagement": 0.0, "posture_score": 0.0, "face_detected": False, "face_count": 0, "faces": []})

    average_attention = sum(item["attention"] for item in face_entries) / len(face_entries)
    average_eye = sum(item["eye_engagement"] for item in face_entries) / len(face_entries)
    average_posture = sum(item["posture_score"] for item in face_entries) / len(face_entries)
    average_fatigue = sum(item["fatigue"] for item in face_entries) / len(face_entries)
    average_fatigue_index = sum(item.get("fatigue_index", 0.0) for item in face_entries) / len(face_entries)
    aggregate = {
        "attention": round(average_attention, 1),
        "fatigue": round(average_fatigue, 3),
        "eye_engagement": round(average_eye, 3),
        "posture_score": round(average_posture, 3),
        "drowsiness_state": max(face_entries, key=lambda item: item.get("fatigue_index", 0.0)).get("drowsiness_state", "Alert"),
        "fatigue_index": round(average_fatigue_index, 1),
        "face_count": len(face_entries),
        "faces": face_entries,
        "student_details": [
            {
                "label": item.get("label", f"Student {index + 1}"),
                "attention": round(item.get("attention", 0.0), 1),
                "fatigue": round(item.get("fatigue", 0.0), 3),
                "status": item.get("drowsiness_state", "Alert"),
                "bbox": item.get("bbox", {}),
            }
            for index, item in enumerate(face_entries)
        ],
    }
    return build_face_summary(aggregate)


def _add_phone_scan(summary, frame, detection_settings=None):
    settings = detection_settings or {}
    if settings.get("phone_detection", True) is False:
        summary["phone_detections"] = []
        summary["phone_detection_status"] = "disabled"
        summary["phone_alert"] = False
        return summary
    detections, detector_status = detect_phones(frame, confidence=float(settings.get("phone_confidence", 0.2)))
    summary["phone_detections"] = detections
    summary["phone_detection_status"] = detector_status
    summary["phone_alert"] = bool(detections)
    return summary


def analyze_camera_frame(frame, detection_mode="multi-face", detection_settings=None, scan_session="default"):
    if mp is None or not hasattr(mp, "solutions"):
        fallback = _fallback_face_metrics(frame)
        fallback["analysis_mode"] = "basic-fallback"
        return _add_phone_scan(build_face_summary(fallback), frame, detection_settings)

    face_mesh = None
    pose = None
    try:
        settings = detection_settings or {}
        max_faces = 1 if str(detection_mode or "multi-face").lower() == "single-user" else 20
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Pose is a single-person model. Skip it entirely in classroom mode;
        # a whole-frame pose failure must not discard successful face landmarks.
        pose_result = None
        if max_faces == 1 and settings.get("posture_analysis", True):
            try:
                pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                pose_result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception:
                logger.exception("Pose analysis failed; continuing with face-landmark posture fallback")
                if pose is not None:
                    try:
                        pose.close()
                    except Exception:
                        logger.exception("Failed to close pose analyzer")
                    pose = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = face_mesh.process(rgb)

        face_entries = []
        if face_result.multi_face_landmarks:
            timestamp = time.monotonic()
            with _scan_tracks_lock:
                tracks = _scan_tracks.setdefault(str(scan_session), [])
                tracks[:] = [track for track in tracks if timestamp - track["seen"] < 8.0]
                if len(_scan_tracks) > 128:
                    oldest = sorted(_scan_tracks, key=lambda key: max((t["seen"] for t in _scan_tracks[key]), default=0))[:32]
                    for key in oldest:
                        _scan_tracks.pop(key, None)
                used_tracks = set()
                for index, face in enumerate(face_result.multi_face_landmarks[:max_faces]):
                    bbox = _face_bbox(face.landmark, frame.shape[1], frame.shape[0])
                    center = ((bbox["x"] + bbox["width"] / 2) / frame.shape[1], (bbox["y"] + bbox["height"] / 2) / frame.shape[0])
                    candidates = [(i, track) for i, track in enumerate(tracks) if i not in used_tracks and ((track["center"][0] - center[0]) ** 2 + (track["center"][1] - center[1]) ** 2) ** .5 < .16]
                    if candidates:
                        track_index, track = min(candidates, key=lambda item: (item[1]["center"][0] - center[0]) ** 2 + (item[1]["center"][1] - center[1]) ** 2)
                        used_tracks.add(track_index)
                        track["center"] = center
                        track["seen"] = timestamp
                        local_state = track["state"]
                    else:
                        local_state = {"ear": deque(maxlen=90), "mar": deque(maxlen=90), "pitch": deque(maxlen=90)}
                        tracks.append({"center": center, "seen": timestamp, "state": local_state})
                    # Pose processes the whole frame and can lock onto a different
                    # person in a multi-face classroom. Face landmarks are more
                    # reliable for the individual eye based evaluation; keep the
                    # neutral posture fallback when pose is ambiguous.
                    pose_landmarks = getattr(pose_result, "pose_landmarks", None) if pose_result is not None else None
                    metrics = compute_attention_fatigue(face.landmark, pose_landmarks, settings, rolling_state=local_state)
                    if not settings.get("fatigue_detection", True):
                        metrics.update({"fatigue": 0.0, "fatigue_index": 0.0, "microsleep": False, "yawning": False, "drowsiness_state": "Disabled"})
                    metrics["label"] = f"Student {index + 1}"
                    metrics["face_index"] = index
                    metrics["bbox"] = bbox
                    metrics["confidence"] = round(max(0.0, min(100.0,
                        (metrics.get("attention", 0.0) * 0.55)
                        + (metrics.get("posture_score", 0.0) * 25)
                        + (metrics.get("eye_engagement", 0.0) * 20))), 1)
                    metrics["overall_score"] = metrics["confidence"]
                    face_entries.append(metrics)

        if face_entries:
            return _add_phone_scan(_aggregate_face_scan(face_entries), frame, settings)
        return _add_phone_scan(build_face_summary(_fallback_face_metrics(frame)), frame, settings)
    except Exception:
        logger.exception("MediaPipe face/fatigue analysis failed; using basic fallback")
        fallback = _fallback_face_metrics(frame)
        fallback["analysis_mode"] = "basic-fallback"
        return _add_phone_scan(build_face_summary(fallback), frame, detection_settings)
    finally:
        for analyzer in (face_mesh, pose):
            if analyzer is not None:
                try:
                    analyzer.close()
                except Exception:
                    logger.exception("Failed to close MediaPipe analyzer")


def create_app():
    project_root = Path(__file__).resolve().parent.parent
    template_folder = project_root / "classpulse" / "templates"
    static_folder = project_root / "classpulse" / "static"

    app = Flask(
        __name__,
        template_folder=str(template_folder),
        static_folder=str(static_folder),
    )
    secret_key = os.getenv("SECRET_KEY")
    production = os.getenv("FLASK_ENV", "").lower() == "production" or os.getenv("APP_ENV", "").lower() == "production"
    if production and (not secret_key or len(secret_key) < 32):
        raise RuntimeError("Production requires SECRET_KEY to be set to at least 32 characters.")
    app.secret_key = secret_key or secrets.token_hex(32)
    mobile_token_signer = URLSafeTimedSerializer(app.secret_key, salt="classpulse-mobile-api-v1")
    mobile_token_ttl = 60 * 60 * 12
    is_https = production or os.getenv("ENABLE_HTTPS", "0") == "1"
    app.config.update(
        SESSION_COOKIE_SECURE=is_https,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_NAME="classpulse_session",
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        PREFERRED_URL_SCHEME="https" if is_https else "http",
    )
    if os.getenv("TRUST_PROXY", "0") == "1":
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    app.telemetry_store = TelemetryStore()
    db_path = Path(os.getenv("CLASSPULSE_DB_PATH", project_root / "classpulse" / "data" / "classpulse.db"))
    app.teacher_store = TeacherStore(db_path)
    if production:
        app.teacher_store.remove_demo_teacher()

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob: data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        if is_https:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    @app.before_request
    def enforce_https_redirect():
        if not is_https or request.is_secure:
            return None
        if request.host and request.host.startswith("127.0.0.1"):
            return redirect("https://127.0.0.1:5000" + request.path, code=301)
        return None

    @app.before_request
    def authenticate_mobile_api():
        if not request.path.startswith("/api/v1/") or request.path == "/api/v1/auth/login":
            return None
        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return jsonify({"success": False, "error": "unauthorized", "message": "A bearer access token is required."}), 401
        try:
            claims = mobile_token_signer.loads(token, max_age=mobile_token_ttl)
            teacher = app.teacher_store.get_teacher_by_id(int(claims["teacher_id"]))
        except (BadSignature, SignatureExpired, KeyError, TypeError, ValueError):
            teacher = None
            claims = {}
        if teacher is None:
            return jsonify({"success": False, "error": "unauthorized", "message": "The access token is invalid or expired."}), 401
        g.mobile_teacher = teacher
        g.mobile_scan_session = claims.get("scan_session", f"teacher-{teacher['id']}")
        return None

    def get_authenticated_teacher():
        mobile_teacher = getattr(g, "mobile_teacher", None)
        if mobile_teacher is not None:
            return mobile_teacher
        teacher_id = session.get("teacher_id")
        if teacher_id is None:
            return None
        return app.teacher_store.get_teacher_by_id(int(teacher_id))

    @app.before_request
    def require_teacher_for_api():
        if not request.path.startswith("/api/"):
            return None
        if request.path in {"/api/auth/login", "/api/auth/register", "/api/v1/auth/login"}:
            return None
        if get_authenticated_teacher() is None:
            return jsonify({"success": False, "error": "unauthorized", "message": "Sign in as a teacher to use this endpoint."}), 401
        return None

    @app.get("/")
    def index():
        teacher = get_authenticated_teacher()
        if teacher is None:
            return render_template("login.html")
        return render_template("overview.html", teacher=teacher)

    @app.get("/overview")
    def overview_page():
        teacher = get_authenticated_teacher()
        if teacher is None:
            return redirect(url_for("login_page"))
        return render_template("overview.html", teacher=teacher)

    @app.get("/manifest.webmanifest")
    def web_manifest():
        return jsonify({
            "name": "ClassPulse Teacher Portal",
            "short_name": "ClassPulse",
            "description": "Privacy-first classroom engagement and fatigue checks.",
            "start_url": "/overview",
            "scope": "/",
            "display": "standalone",
            "background_color": "#f4f8f7",
            "theme_color": "#087f78",
            "orientation": "any",
            "icons": [
                {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
                {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
                {"src": "/static/icons/icon-512-maskable.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
            ],
        })

    @app.get("/service-worker.js")
    def service_worker():
        response = app.send_static_file("service-worker.js")
        response.headers["Service-Worker-Allowed"] = "/"
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/classroom")
    def classroom_page():
        teacher = get_authenticated_teacher()
        if teacher is None:
            return redirect(url_for("login_page"))
        return render_template("classroom.html", teacher=teacher)

    @app.get("/login")
    def login_page():
        return render_template("login.html", show_demo_credentials=not production)

    @app.get("/camera")
    def camera_page():
        teacher = get_authenticated_teacher()
        if teacher is None:
            return redirect(url_for("login_page"))
        return render_template("camera.html", teacher=teacher)

    @app.get("/camera-settings")
    def camera_settings_page():
        teacher = get_authenticated_teacher()
        if teacher is None:
            return redirect(url_for("login_page"))
        return render_template("camera_settings.html", teacher=teacher)

    @app.get("/settings")
    def settings_page():
        teacher = get_authenticated_teacher()
        if teacher is None:
            return redirect(url_for("login_page"))
        return render_template("settings.html", teacher=teacher)

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "service": "ClassPulse",
            "version": "1.0.0",
            "phone_detector": phone_detector_runtime_status(),
        })

    @app.post("/api/auth/register")
    def register_teacher():
        raw = request.get_json(silent=True) or request.form or {}
        name = raw.get("name", "").strip()
        email = raw.get("email", "").strip().lower()
        password = raw.get("password", "")
        department = raw.get("department", "").strip()

        if not name or not email or not password:
            if request.is_json:
                return jsonify({"success": False, "message": "Name, email, and password are required."}), 400
            return render_template("login.html", error="Name, email, and password are required.")

        if not validate_email(email):
            if request.is_json:
                return jsonify({"success": False, "message": "A valid email address is required."}), 400
            return render_template("login.html", error="A valid email address is required.")

        if len(password) < 8:
            if request.is_json:
                return jsonify({"success": False, "message": "Password must be at least 8 characters long."}), 400
            return render_template("login.html", error="Password must be at least 8 characters long.")

        teacher = app.teacher_store.register_teacher(name, email, password, department)
        if teacher is None:
            if request.is_json:
                return jsonify({"success": False, "message": "Teacher already exists or the details are invalid."}), 409
            return render_template("login.html", error="Teacher already exists or the details are invalid.")

        session.clear()
        session["teacher_id"] = teacher["id"]
        if request.is_json:
            return jsonify({"success": True, "teacher": teacher})
        return redirect(url_for("index"))

    @app.post("/api/auth/login")
    def login_teacher():
        raw = request.get_json(silent=True) or request.form or {}
        email = raw.get("email", "").strip().lower()
        password = raw.get("password", "")

        if not email or not password:
            if request.is_json:
                return jsonify({"success": False, "message": "Email and password are required."}), 400
            return render_template("login.html", error="Email and password are required.")

        if not validate_email(email):
            if request.is_json:
                return jsonify({"success": False, "message": "A valid email address is required."}), 400
            return render_template("login.html", error="A valid email address is required.")

        teacher = app.teacher_store.authenticate_teacher(email, password)
        if teacher is None:
            if request.is_json:
                return jsonify({"success": False, "message": "Invalid email or password."}), 401
            return render_template("login.html", error="Invalid email or password.")

        session.clear()
        session.clear()
        session["teacher_id"] = teacher["id"]
        if request.is_json:
            return jsonify({"success": True, "teacher": teacher})
        return redirect(url_for("index"))

    @app.post("/api/v1/auth/login")
    def mobile_login_teacher():
        payload = request.get_json(silent=True) or {}
        email = str(payload.get("email", "")).strip().lower()
        password = payload.get("password", "")
        if not email or not password:
            return jsonify({"success": False, "error": "validation_error", "message": "Email and password are required."}), 400
        teacher = app.teacher_store.authenticate_teacher(email, password)
        if teacher is None:
            return jsonify({"success": False, "error": "invalid_credentials", "message": "Invalid email or password."}), 401
        token = mobile_token_signer.dumps({
            "teacher_id": int(teacher["id"]),
            "scan_session": secrets.token_urlsafe(16),
        })
        return jsonify({
            "success": True,
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": mobile_token_ttl,
            "teacher": teacher,
        })

    @app.post("/api/v1/auth/logout")
    def mobile_logout_teacher():
        return jsonify({"success": True, "message": "Discard the bearer token on the device."})

    @app.get("/api/auth/me")
    @app.get("/api/v1/auth/me")
    def get_authenticated_teacher_api():
        teacher = get_authenticated_teacher()
        if teacher is None:
            return jsonify({"success": False, "teacher": None}), 401
        return jsonify({"success": True, "teacher": teacher})

    @app.post("/api/auth/logout")
    def logout_teacher():
        session.clear()
        return jsonify({"success": True})

    @app.get("/api/dashboard-config")
    @app.get("/api/v1/dashboard-config")
    def dashboard_config():
        teacher = get_authenticated_teacher()
        teacher_payload = teacher or {"name": "Demo Teacher", "department": "Administration"}
        return jsonify(
            {
                "teacher": teacher_payload,
                "analysis_modes": [
                    "Lecture mode",
                    "Exam mode",
                    "Group review",
                    "Hands-on lab",
                ],
                "default_mode": "Lecture mode",
                "camera_devices": [
                    {"id": "default", "label": "Default camera"},
                    {"id": "front", "label": "Front camera"},
                    {"id": "rear", "label": "Rear camera"},
                ],
                "controls": {
                    "camera_feed": True,
                    "student_tracking": True,
                    "fatigue_detection": True,
                    "posture_analysis": True,
                    "privacy_mode": True,
                    "auto_refresh": True,
                    "face_blur": False,
                    "preview_blur": False,
                    "phone_detection": True,
                    "phone_confidence": 0.2,
                    "show_phone_boxes": True,
                    "sound_alerts": False,
                    "capture_interval": 5,
                    "focus_mode": "lecture",
                    "detection_mode": "multi-face",
                    "ear_blink_cutoff": 0.18,
                    "mar_yawn_sensitivity": 0.24,
                    "posture_tilt_degrees": 18.0,
                    "alert_threshold": 0.7,
                    "camera_source": "default",
                },
                "portal_details": {
                    "focus_stability": "Stable",
                    "face_coverage": "Waiting for live scan",
                    "last_capture": "Not captured yet",
                    "camera_status": "Ready for live scan",
                    "student_count": 0,
                    "recommended_action": "Start the classroom camera or capture a group photo for temporary, teacher-reviewed cues. Named records are saved only through the explicit opt-in workflow.",
                },
                "student_roster": [],
            }
        )

    @app.get("/api/telemetry")
    @app.get("/api/v1/telemetry")
    def get_telemetry():
        teacher_id = int(get_authenticated_teacher()["id"])
        summary = app.telemetry_store.get_summary(teacher_id)
        return jsonify(
            {
                "summary": summary,
                "history": app.telemetry_store.get_history(teacher_id),
            }
        )

    @app.get("/api/students")
    @app.get("/api/v1/students")
    def get_students():
        return jsonify({"students": app.telemetry_store.get_student_snapshot()})

    @app.get("/api/student-scans")
    def get_student_scans():
        teacher = get_authenticated_teacher()
        if teacher is None:
            return jsonify({"error": "Authentication required."}), 401
        class_name = request.args.get("class", "").strip()
        roll_number = request.args.get("roll", "").strip()
        scans = app.teacher_store.list_student_scans(int(teacher["id"]), class_name, roll_number)
        return jsonify({"scans": scans})

    @app.get("/api/student-scans/<int:scan_id>/image")
    def get_student_scan_image(scan_id: int):
        teacher = get_authenticated_teacher()
        if teacher is None:
            return jsonify({"error": "Authentication required."}), 401
        image = app.teacher_store.get_student_scan_image(int(teacher["id"]), scan_id)
        if image is None:
            return jsonify({"error": "Scan image not found."}), 404
        return app.response_class(image[0], mimetype=image[1], headers={"Cache-Control": "private, no-store"})

    @app.get("/api/alerts")
    @app.get("/api/v1/alerts")
    def get_alerts():
        return jsonify(app.telemetry_store.get_alerts())

    @app.post("/api/scan")
    @app.post("/api/v1/scan")
    def scan_face():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "A JSON object is required.", "status": "invalid-request"}), 400
        image_data = payload.get("image_data", "")
        if not isinstance(image_data, str):
            return jsonify({"error": "image_data must be a base64 string.", "status": "invalid-image"}), 400
        identity = payload.get("student") or {}
        if not isinstance(identity, dict):
            return jsonify({"error": "student must be an object.", "status": "invalid-student"}), 400
        identity_values = [str(identity.get(key, "")).strip() for key in ("name", "class_name", "roll_number")]
        if any(identity_values) and not all(identity_values):
            return jsonify({"error": "Student name, class, and roll number are all required to save a scan.", "status": "invalid-student"}), 400
        if not image_data:
            return jsonify({
                "attention": 0.0,
                "fatigue": 1.0,
                "status": "no-image",
                "risk_level": "High",
                "recommendation": "No face image was captured. Please allow the camera and retry the scan.",
                "eye_engagement": 0.0,
                "posture_score": 0.0,
                "confidence": 0.0,
                "overall_score": 0.0,
                "face_detected": False,
            }), 400

        try:
            if image_data.startswith("data:image"):
                image_header, image_data = image_data.split(",", 1)
                image_mime = image_header[5:].split(";", 1)[0]
            else:
                image_mime = "image/jpeg"
            if image_mime not in {"image/jpeg", "image/png", "image/webp"}:
                return jsonify({"error": "Only JPEG, PNG, or WebP images can be saved.", "status": "invalid-image"}), 400
            image_bytes = base64.b64decode(image_data, validate=True)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({
                    "attention": 0.0,
                    "fatigue": 1.0,
                    "status": "invalid-image",
                    "risk_level": "High",
                    "recommendation": "The image could not be read. Please retry with a clear face view.",
                    "eye_engagement": 0.0,
                    "posture_score": 0.0,
                    "confidence": 0.0,
                    "overall_score": 0.0,
                    "face_detected": False,
                }), 400
            if frame.shape[0] * frame.shape[1] > 12_000_000:
                return jsonify({"error": "Image dimensions exceed the 12 megapixel limit.", "status": "invalid-image"}), 413

            detection_settings = payload.get("detection_settings") or {}
            detection_mode = payload.get("detection_mode") or "multi-face"
            scan_session = getattr(g, "mobile_scan_session", None) or session.setdefault("scan_session", secrets.token_urlsafe(12))
            metrics = analyze_camera_frame(frame, detection_mode=detection_mode, detection_settings=detection_settings, scan_session=scan_session)
            app.telemetry_store.append(
                attention=metrics.get("attention", 0),
                fatigue=metrics.get("fatigue", 0),
                timestamp=payload.get("timestamp"),
                teacher_id=int(get_authenticated_teacher()["id"]),
                eye_engagement=metrics.get("eye_engagement", 0),
                posture_score=metrics.get("posture_score", 0),
            )
            teacher = get_authenticated_teacher()
            if teacher is not None and all(identity_values):
                try:
                    saved = app.teacher_store.save_student_scan(int(teacher["id"]), identity, image_bytes, metrics, image_mime)
                    metrics["saved_scan"] = saved
                except ValueError as exc:
                    return jsonify({"error": str(exc), "status": "invalid-student"}), 400

            return jsonify({"status": "ok", "frame_width": int(frame.shape[1]), "frame_height": int(frame.shape[0]), **metrics})
        except Exception as exc:  # pragma: no cover - guard for invalid browser payloads
            logger.exception("Scan request failed")
            message = "Analysis could not complete. Check the image and camera, then try again."
            if not production:
                message = f"Analysis failed: {type(exc).__name__}: {exc}"
            return jsonify({
                "status": "error",
                "error": message,
            }), 500

    @app.post("/api/telemetry")
    @app.post("/api/v1/telemetry")
    def receive_telemetry():
        payload = request.get_json(silent=True) or {}
        app.telemetry_store.append(
            attention=payload.get("attention", 0),
            fatigue=payload.get("fatigue", 0),
            timestamp=payload.get("timestamp"),
            teacher_id=int(get_authenticated_teacher()["id"]),
            eye_engagement=payload.get("eye_engagement", 0),
            posture_score=payload.get("posture_score", 0),
        )
        return jsonify({"received": True, "payload": payload})

    return app
