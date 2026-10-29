import base64
import sqlite3
import uuid
from pathlib import Path

import cv2
import numpy as np
from types import SimpleNamespace

from app import app
from classpulse import backend
from classpulse.metrics import LEFT_EYE, RIGHT_EYE
from classpulse.store import TeacherStore
from werkzeug.security import generate_password_hash


def authenticated_client():
    client = app.test_client()
    email = f"teacher-test-{uuid.uuid4()}@example.com"
    response = client.post(
        "/api/auth/register",
        json={
            "name": "Test Teacher",
            "username": f"teacher-test-{uuid.uuid4().hex[:10]}",
            "email": email,
            "password": "secret123",
            "department": "Testing",
        },
    )
    assert response.status_code == 200
    return client


def test_browser_launch_target_uses_teacher_login_page():
    from app import get_start_url

    assert get_start_url() == "http://localhost:5000/login"


def test_dashboard_config_exposes_controls():
    client = authenticated_client()

    response = client.get("/api/dashboard-config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["analysis_modes"]
    assert "fatigue_detection" in payload["controls"]
    assert "student_tracking" in payload["controls"]
    assert "camera_feed" in payload["controls"]


def test_dashboard_template_renders_student_controls():
    client = app.test_client()
    email = f"teacher-dashboard-{uuid.uuid4()}@example.com"

    client.post(
        "/api/auth/register",
        json={
            "name": "Teacher Dashboard",
            "username": f"teacher-dash-{uuid.uuid4().hex[:10]}",
            "email": email,
            "password": "secret123",
            "department": "Math",
        },
    )

    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True).lower()
    assert "student roster" in html
    assert "classroom insights" in html
    assert "camera-toggle" in html


def test_camera_page_explains_browser_camera_permissions():
    client = app.test_client()
    email = f"teacher-camera-{uuid.uuid4()}@example.com"

    client.post(
        "/api/auth/register",
        json={
            "name": "Teacher Camera",
            "username": f"teacher-cam-{uuid.uuid4().hex[:10]}",
            "email": email,
            "password": "secret123",
            "department": "English",
        },
    )

    response = client.get("/camera")

    assert response.status_code == 200
    html = response.get_data(as_text=True).lower()
    assert "capture quality checklist" in html
    assert "open overview" in html


def test_login_page_exposes_teacher_registration_form():
    client = app.test_client()

    response = client.get("/login")

    assert response.status_code == 200
    html = response.get_data(as_text=True).lower()
    assert "register as teacher" in html
    assert "create account" in html
    assert "teacher name" in html
    assert 'name="username"' in html
    assert "email or username" in html


def test_dashboard_config_includes_camera_and_threshold_controls():
    client = authenticated_client()

    response = client.get("/api/dashboard-config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["camera_devices"]
    assert "alert_threshold" in payload["controls"]
    assert "camera_source" in payload["controls"]
    assert "detection_mode" in payload["controls"]
    assert "ear_blink_cutoff" in payload["controls"]
    assert "mar_yawn_sensitivity" in payload["controls"]
    assert "posture_tilt_degrees" in payload["controls"]


def test_settings_page_exposes_detection_mode_and_threshold_controls():
    client = app.test_client()
    email = f"teacher-settings-{uuid.uuid4()}@example.com"

    client.post(
        "/api/auth/register",
        json={
            "name": "Teacher Settings",
            "username": f"teacher-set-{uuid.uuid4().hex[:10]}",
            "email": email,
            "password": "secret123",
            "department": "CS",
        },
    )

    response = client.get("/settings")

    assert response.status_code == 200
    html = response.get_data(as_text=True).lower()
    assert "single-user" in html or "multi-face" in html
    assert "ear blink cutoff" in html
    assert "mar yawn sensitivity" in html
    assert "posture tilt" in html


def test_dashboard_does_not_render_saved_records_and_uses_live_session_tools():
    client = app.test_client()
    email = f"teacher-live-tools-{uuid.uuid4()}@example.com"

    client.post(
        "/api/auth/register",
        json={
            "name": "Teacher Live",
            "username": f"teacher-live-{uuid.uuid4().hex[:10]}",
            "email": email,
            "password": "secret123",
            "department": "AI",
        },
    )

    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True).lower()
    assert "ava chen" not in html
    assert "noah patel" not in html
    assert "live classroom camera" in html


def test_dashboard_config_uses_live_group_scan_payload_and_no_seeded_roster():
    client = authenticated_client()
    response = client.get("/api/dashboard-config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["student_roster"] == []
    assert payload["controls"]["detection_mode"] == "multi-face"
    assert payload["portal_details"]["student_count"] in (0, "0")


def test_registration_rejects_invalid_email_and_short_password():
    client = app.test_client()

    invalid_email = client.post(
        "/api/auth/register",
        json={
            "name": "Teacher Invalid",
            "username": "invalid-user",
            "email": "not-an-email",
            "password": "short",
            "department": "Math",
        },
    )
    invalid_email_body = invalid_email.get_json()
    assert invalid_email.status_code == 400
    assert invalid_email_body["success"] is False
    assert "email" in invalid_email_body["message"].lower()

    short_password = client.post(
        "/api/auth/register",
        json={
            "name": "Teacher Weak",
            "username": "weak-user",
            "email": "weak@example.com",
            "password": "short",
            "department": "Math",
        },
    )
    short_password_body = short_password.get_json()
    assert short_password.status_code == 400
    assert short_password_body["success"] is False
    assert "at least 8" in short_password_body["message"].lower()


def test_security_headers_are_present_on_responses():
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert "frame-ancestors" in response.headers.get("Content-Security-Policy", "")
    assert "camera=(self)" in response.headers.get("Permissions-Policy", "")
    assert "media-src" in response.headers.get("Content-Security-Policy", "")
    assert response.get_json()["database"] in {"sqlite", "postgresql"}


def test_student_snapshot_endpoint_returns_live_session_data_only():
    client = authenticated_client()

    response = client.get("/api/students")

    assert response.status_code == 200
    data = response.get_json()
    assert data["students"] == []


def test_heatmap_and_alerts_endpoint_has_no_seeded_classroom_data():
    client = authenticated_client()

    response = client.get("/api/alerts")

    assert response.status_code == 200
    payload = response.get_json()
    assert "heatmap" in payload
    assert "alerts" in payload
    assert payload["alerts"] == []
    assert payload["heatmap"]["zones"] == []


def test_scan_endpoint_accepts_camera_frame_and_returns_metrics(monkeypatch):
    client = authenticated_client()
    monkeypatch.setattr(backend, "detect_phones", lambda frame, confidence=0.32: ([], "ready"))
    ok, encoded = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    assert ok
    image_data = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    response = client.post(
        "/api/scan",
        json={"image_data": image_data, "source": "camera"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert "attention" in payload
    assert "fatigue" in payload
    assert "status" in payload
    assert "risk_level" in payload
    assert "recommendation" in payload
    assert "confidence" in payload
    assert "overall_score" in payload
    assert payload["phone_detection_status"] == "ready"


def test_aggregate_face_scan_returns_nonzero_scores_for_detected_faces():
    result = backend._aggregate_face_scan([
        {
            "attention": 72.0,
            "fatigue": 0.2,
            "fatigue_index": 20.0,
            "eye_engagement": 0.35,
            "posture_score": 0.7,
            "drowsiness_state": "Alert",
            "microsleep": False,
            "yawning": False,
            "confidence": 0,
            "overall_score": 0,
            "label": "Student 1",
            "bbox": {"x": 10, "y": 10, "width": 100, "height": 100},
        }
    ])

    assert result["face_detected"] is True
    assert result["attention"] > 0
    assert result["eye_engagement"] > 0
    assert result["posture_score"] > 0
    assert result["overall_score"] > 0


def test_multi_face_scan_returns_scores_without_starting_single_person_pose(monkeypatch):
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
    face_mesh_instances = []

    class FakeFaceMesh:
        def __init__(self, **kwargs):
            self.closed = False
            face_mesh_instances.append(self)

        def process(self, _image):
            return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=landmarks)])

        def close(self):
            self.closed = True

    def unexpected_pose(**_kwargs):
        raise AssertionError("multi-face mode should not start single-person pose analysis")

    fake_mp = SimpleNamespace(solutions=SimpleNamespace(
        face_mesh=SimpleNamespace(FaceMesh=FakeFaceMesh),
        pose=SimpleNamespace(Pose=unexpected_pose),
    ))
    monkeypatch.setattr(backend, "mp", fake_mp)

    def phone_detector(_frame, confidence=0.2):
        assert face_mesh_instances[0].closed is True
        return [], "ready"

    monkeypatch.setattr(backend, "detect_phones", phone_detector)
    monkeypatch.setattr(backend, "compute_attention_fatigue", lambda *_args, **_kwargs: {
        "attention": 72.0, "fatigue": 0.2, "fatigue_index": 20.0,
        "eye_engagement": 0.35, "posture_score": 0.7,
        "drowsiness_state": "Alert", "microsleep": False, "yawning": False,
    })

    result = backend.analyze_camera_frame(np.zeros((240, 320, 3), dtype=np.uint8))

    assert result["face_detected"] is True
    assert result["attention"] > 0
    assert result["eye_engagement"] > 0
    assert result["posture_score"] > 0
    assert result["overall_score"] > 0


def test_face_detection_fallback_returns_face_and_explains_estimate(monkeypatch):
    class MissedMesh:
        def __init__(self, **_kwargs):
            pass

        def process(self, _image):
            return SimpleNamespace(multi_face_landmarks=None)

        def close(self):
            pass

    detection = SimpleNamespace(
        location_data=SimpleNamespace(relative_bounding_box=SimpleNamespace(xmin=.25, ymin=.2, width=.4, height=.5)),
        score=[.81],
    )

    class FaceDetector:
        def __init__(self, **_kwargs):
            pass

        def process(self, _image):
            return SimpleNamespace(detections=[detection])

        def close(self):
            pass

    fake_mp = SimpleNamespace(solutions=SimpleNamespace(
        face_mesh=SimpleNamespace(FaceMesh=MissedMesh),
        pose=SimpleNamespace(Pose=lambda **_kwargs: None),
        face_detection=SimpleNamespace(FaceDetection=FaceDetector),
    ))
    monkeypatch.setattr(backend, "mp", fake_mp)
    monkeypatch.setattr(backend, "detect_phones", lambda _frame, confidence=0.2: ([], "ready"))

    result = backend.analyze_camera_frame(np.zeros((240, 320, 3), dtype=np.uint8))

    assert result["face_detected"] is True
    assert result["analysis_mode"] == "face-detection-fallback"
    assert result["attention"] > 0
    assert result["fatigue_index"] > 0
    assert result["overall_score"] > 0
    assert result["recommendation"]


def test_phone_detector_uses_recall_friendly_model_settings(monkeypatch):
    from classpulse import phone_detection

    captured = {}

    class FakeModel:
        names = {67: "cell phone"}

        def predict(self, _frame, **kwargs):
            captured.update(kwargs)
            return [SimpleNamespace(boxes=None)]

    monkeypatch.setattr(phone_detection, "_get_model", lambda: FakeModel())
    detections, status = phone_detection.detect_phones(np.zeros((120, 120, 3), dtype=np.uint8))

    assert status == "ready"
    assert detections == []
    assert captured["conf"] == 0.12
    assert captured["imgsz"] == 960


def test_phone_detector_uses_a_single_inference_for_fast_large_frame_scans(monkeypatch):
    from classpulse import phone_detection

    calls = []

    class FakeModel:
        names = {67: "cell phone"}

        def predict(self, frame, **kwargs):
            calls.append(frame.shape[:2])
            return [SimpleNamespace(boxes=None)]

    monkeypatch.setattr(phone_detection, "_get_model", lambda: FakeModel())
    detections, status = phone_detection.detect_phones(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert status == "ready"
    assert detections == []
    assert len(calls) == 1
    assert calls[0] == (720, 1280)


def test_phone_detection_failure_does_not_fail_face_scan(monkeypatch):
    monkeypatch.setattr(backend, "mp", None)
    monkeypatch.setattr(backend, "detect_phones", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model failed")))
    ok, encoded = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    assert ok
    image_data = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    response = authenticated_client().post("/api/scan", json={"image_data": image_data})

    assert response.status_code == 200
    result = response.get_json()
    assert "attention" in result
    assert "overall_score" in result
    assert result["phone_detection_status"] == "error"


def test_analyze_camera_frame_falls_back_when_mediapipe_is_unavailable():
    original_mp = backend.mp
    backend.mp = None
    try:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = backend.analyze_camera_frame(frame)
        assert 0.0 <= result["attention"] <= 100.0
        assert 0.0 <= result["overall_score"] <= 100.0
        assert "status" in result
    finally:
        backend.mp = original_mp


def test_teacher_registration_and_login_store_details_in_db():
    client = app.test_client()
    email = f"teacher-unique-{uuid.uuid4()}@example.com"

    register = client.post(
        "/api/auth/register",
        json={
            "name": "Ms. Parker",
            "username": f"parker-{uuid.uuid4().hex[:10]}",
            "email": email,
            "password": "secret123",
            "department": "Science",
        },
    )

    assert register.status_code == 200
    assert register.get_json()["success"] is True

    login = client.post(
        "/api/auth/login",
        json={"identifier": email, "password": "secret123"},
    )

    assert login.status_code == 200
    payload = login.get_json()
    assert payload["success"] is True
    assert payload["teacher"]["email"] == email
    username = payload["teacher"]["username"]

    username_login = client.post(
        "/api/auth/login",
        json={"identifier": username.upper(), "password": "secret123"},
    )
    assert username_login.status_code == 200
    assert username_login.get_json()["teacher"]["username"] == username

    mobile_login = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": "secret123"},
    )
    assert mobile_login.status_code == 200
    assert mobile_login.get_json()["access_token"]

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.get_json()["teacher"]["name"] == "Ms. Parker"


def test_login_reports_database_unavailable_separately_from_bad_password(monkeypatch):
    client = app.test_client()

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("database connection refused")

    monkeypatch.setattr(app.teacher_store, "authenticate_teacher", unavailable)
    response = client.post("/api/auth/login", json={"identifier": "teacher@example.com", "password": "secret123"})

    assert response.status_code == 503
    assert response.get_json()["error"] == "account_store_unavailable"


def test_registration_reports_username_conflict_clearly():
    client = app.test_client()
    username = f"taken-{uuid.uuid4().hex[:10]}"
    first = client.post("/api/auth/register", json={
        "name": "First Teacher", "username": username,
        "email": f"first-{uuid.uuid4().hex[:8]}@example.com", "password": "secret123",
    })
    assert first.status_code == 200

    conflict = client.post("/api/auth/register", json={
        "name": "Second Teacher", "username": username.upper(),
        "email": f"second-{uuid.uuid4().hex[:8]}@example.com", "password": "secret123",
    })
    assert conflict.status_code == 409
    assert "username" in conflict.get_json()["message"].lower()


def test_existing_sqlite_teacher_accounts_receive_login_safe_usernames():
    db_path = Path(__file__).resolve().parent / f"legacy-{uuid.uuid4().hex}.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute("""CREATE TABLE teachers (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
        department TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    connection.execute(
        "INSERT INTO teachers (id, name, email, password_hash, department) VALUES (?, ?, ?, ?, ?)",
        (7, "Legacy Teacher", "legacy@example.com", generate_password_hash("secret123"), "Science"),
    )
    connection.commit()
    connection.close()

    try:
        store = TeacherStore(db_path)
        assert store.authenticate_teacher("legacy@example.com", "secret123")["username"] == "teacher7"
        assert store.authenticate_teacher("teacher7", "secret123")["email"] == "legacy@example.com"
    finally:
        db_path.unlink(missing_ok=True)


def test_all_sidebar_pages_are_available_and_protected():
    client = app.test_client()
    email = f"teacher-sidebar-{uuid.uuid4()}@example.com"

    client.post(
        "/api/auth/register",
        json={
            "name": "Sidebar Teacher",
            "username": f"teacher-side-{uuid.uuid4().hex[:10]}",
            "email": email,
            "password": "secret123",
            "department": "Computer Science",
        },
    )

    pages = [
        ("/overview", "overview"),
        ("/classroom", "classroom"),
        ("/camera", "camera"),
        ("/camera-settings", "camera settings"),
        ("/settings", "settings"),
    ]

    for route, expected_text in pages:
        response = client.get(route)
        assert response.status_code == 200
        html = response.get_data(as_text=True).lower()
        assert expected_text in html


def test_fatigue_metrics_include_temporal_drowsiness_fields():
    from classpulse.metrics import compute_attention_fatigue

    face_landmarks = [
        type("Point", (), {"x": 0.5, "y": 0.45})() for _ in range(468)
    ]
    for index in [LEFT_EYE["upper"], LEFT_EYE["lower"], RIGHT_EYE["upper"], RIGHT_EYE["lower"]]:
        face_landmarks[index].x = 0.5
        face_landmarks[index].y = 0.5

    for index in [LEFT_EYE["outer"], LEFT_EYE["inner"], RIGHT_EYE["outer"], RIGHT_EYE["inner"]]:
        face_landmarks[index].x = 0.7
        face_landmarks[index].y = 0.5

    metrics = compute_attention_fatigue(face_landmarks, None)

    assert "fatigue_index" in metrics
    assert "drowsiness_state" in metrics
    assert "microsleep" in metrics
    assert "yawning" in metrics
    assert 0 <= metrics["fatigue_index"] <= 100
