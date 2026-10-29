import os
from flask import Flask, jsonify, request, render_template
from database import init_db, SessionLocal, User, Detection

# Locate the actual templates and static directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "classpulse", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "classpulse", "static")

if not os.path.exists(TEMPLATE_DIR):
    TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

if not os.path.exists(STATIC_DIR):
    STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# Initialize Supabase schema
try:
    init_db()
except Exception as e:
    print(f"Database initialization warning: {e}")


# ==========================================
# 1. FRONTEND PAGE ROUTES
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return render_template("dashboard.html")

@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/classroom", methods=["GET"])
def classroom_page():
    return render_template("classroom.html")

@app.route("/camera", methods=["GET"])
def camera_page():
    return render_template("camera.html")

@app.route("/camera-settings", methods=["GET"])
def camera_settings_page():
    return render_template("camera_settings.html")

@app.route("/overview", methods=["GET"])
def overview_page():
    return render_template("overview.html")

@app.route("/settings", methods=["GET"])
def settings_page():
    return render_template("settings.html")


# ==========================================
# 2. BACKEND API ROUTES
# ==========================================

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "ClassPulse API",
        "database": "Supabase PostgreSQL connected",
        "template_dir": TEMPLATE_DIR
    }), 200

@app.route("/api/register", methods=["POST"])
def handle_register():
    data = request.get_json() or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    db = SessionLocal()
    try:
        existing = db.query(User).filter((User.username == username) | (User.email == email)).first()
        if existing:
            return jsonify({"status": "error", "message": "Username or email already in use"}), 400

        new_user = User(username=username, email=email, password=password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return jsonify({"status": "success", "user_id": new_user.id}), 200
    except Exception as err:
        db.rollback()
        return jsonify({"status": "error", "message": str(err)}), 500
    finally:
        db.close()

@app.route("/api/login", methods=["POST"])
def handle_login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"status": "error", "message": "Missing email or password"}), 400

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email, User.password == password).first()
        if user:
            return jsonify({
                "status": "success",
                "user": {"id": user.id, "username": user.username, "email": user.email}
            }), 200
        return jsonify({"status": "error", "message": "Invalid email or password"}), 401
    finally:
        db.close()

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


# ==========================================
# 3. SERVER ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting ClassPulse on {host}:{port}...")
    app.run(host=host, port=port)