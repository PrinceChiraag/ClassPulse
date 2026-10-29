import os
from flask import Flask, jsonify, request
from database import init_db, SessionLocal, User, Detection

# Initialize Flask application
app = Flask(__name__)

# Run database schema initialization on startup
try:
    init_db()
except Exception as e:
    print(f"Database initialization warning: {e}")


# --- Health & Root Routes ---

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "ClassPulse API",
        "database": "Supabase PostgreSQL connected"
    }), 200


# --- Database Helper Functions ---

def register_user(username: str, email: str, password_hash: str):
    db = SessionLocal()
    try:
        existing = db.query(User).filter((User.username == username) | (User.email == email)).first()
        if existing:
            return {"status": "error", "message": "User already exists"}
        new_user = User(username=username, email=email, password=password_hash)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"status": "success", "user_id": new_user.id}
    except Exception as err:
        db.rollback()
        return {"status": "error", "message": str(err)}
    finally:
        db.close()


def authenticate_user(email: str, password_hash: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email, User.password == password_hash).first()
        if user:
            return {
                "status": "success",
                "user": {"id": user.id, "username": user.username, "email": user.email}
            }
        return {"status": "error", "message": "Invalid email or password"}
    finally:
        db.close()


def log_detection(object_name: str, confidence: float, camera_id: str = "classroom_cam"):
    db = SessionLocal()
    try:
        detection = Detection(
            camera_id=camera_id,
            object_name=object_name,
            confidence=float(confidence)
        )
        db.add(detection)
        db.commit()
    except Exception as err:
        db.rollback()
        print(f"Failed to log detection: {err}")
    finally:
        db.close()


# --- API Endpoints ---

@app.route("/api/register", methods=["POST"])
def handle_register():
    data = request.get_json() or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    result = register_user(username, email, password)
    status_code = 200 if result["status"] == "success" else 400
    return jsonify(result), status_code


@app.route("/api/login", methods=["POST"])
def handle_login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"status": "error", "message": "Missing email or password"}), 400

    result = authenticate_user(email, password)
    status_code = 200 if result["status"] == "success" else 401
    return jsonify(result), status_code


@app.route("/api/detections", methods=["GET"])
def list_detections():
    db = SessionLocal()
    try:
        records = db.query(Detection).order_by(Detection.timestamp.desc()).limit(50).all()
        return jsonify({
            "detections": [
                {
                    "id": r.id,
                    "camera_id": r.camera_id,
                    "object_name": r.object_name,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in records
            ]
        }), 200
    finally:
        db.close()


# Keep server persistently alive and bind to dynamic Render environment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting ClassPulse server on {host}:{port}...")
    app.run(host=host, port=port)