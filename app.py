from database import init_db, SessionLocal, User, Detection

# Initialize tables when the server boots
try:
    init_db()
except Exception as e:
    print(f"Database initialization error: {e}")

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

def log_detection(object_name: str, confidence: float, camera_id: str = "classroom_cam"):
    db = SessionLocal()
    try:
        detection = Detection(camera_id=camera_id, object_name=object_name, confidence=float(confidence))
        db.add(detection)
        db.commit()
    except Exception as err:
        db.rollback()
        print(f"Failed to log detection: {err}")
    finally:
        db.close()