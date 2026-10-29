import re
import sqlite3
import threading
import time
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional

from werkzeug.security import check_password_hash, generate_password_hash


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")


def validate_email(email: str) -> bool:
    return bool(email) and bool(EMAIL_PATTERN.fullmatch(email.strip()))


def validate_username(username: str) -> bool:
    return bool(username) and bool(USERNAME_PATTERN.fullmatch(username.strip()))


class _PostgresConnection:
    """Small DB-API adapter so existing parameterized queries work on both backends."""

    def __init__(self, connection):
        self.connection = connection

    def execute(self, query, parameters=()):
        return self.connection.execute(query.replace("?", "%s"), parameters)

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.connection.__exit__(exc_type, exc_value, traceback)


class _SQLiteConnection:
    """Close SQLite handles after transactions instead of retaining file locks."""

    def __init__(self, connection):
        self.connection = connection

    def execute(self, query, parameters=()):
        return self.connection.execute(query, parameters)

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.connection.__exit__(exc_type, exc_value, traceback)
        finally:
            self.connection.close()


class TeacherStore:
    def __init__(self, db_path: str | Path = Path(__file__).resolve().parent / "data" / "classpulse.db"):
        self.sqlite_source_path = Path(db_path)
        database_url = os.getenv("DATABASE_URL", "").strip()
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        if database_url.startswith("postgresql+psycopg://"):
            database_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
        self.database_url = database_url
        self.use_postgres = self.database_url.startswith(("postgresql://", "postgresql+"))
        self.database_backend = "postgresql" if self.use_postgres else "sqlite"
        self.db_path = self.sqlite_source_path if not self.use_postgres else None
        if not self.use_postgres:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        if self.use_postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ModuleNotFoundError as exc:
                if exc.name == "psycopg":
                    raise RuntimeError(
                        "DATABASE_URL selects PostgreSQL, but the Psycopg 3 driver is missing. "
                        "Rebuild the deployment image after installing requirements.txt."
                    ) from exc
                raise

            return _PostgresConnection(psycopg.connect(self.database_url, row_factory=dict_row))
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return _SQLiteConnection(connection)

    def init_db(self) -> None:
        id_type = "BIGSERIAL PRIMARY KEY" if self.use_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
        ref_type = "BIGINT" if self.use_postgres else "INTEGER"
        blob_type = "BYTEA" if self.use_postgres else "BLOB"
        text_now = "(CURRENT_TIMESTAMP::text)" if self.use_postgres else "CURRENT_TIMESTAMP"
        with self._connect() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS teachers (
                    id {id_type},
                    name TEXT NOT NULL,
                    username TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    department TEXT,
                    created_at TEXT DEFAULT {text_now}
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS scan_history (
                    id {id_type}, teacher_id {ref_type}, attention REAL, fatigue REAL,
                    posture_score REAL, timestamp TEXT, created_at TEXT DEFAULT {text_now},
                    FOREIGN KEY(teacher_id) REFERENCES teachers(id)
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS students (
                    id {id_type},
                    teacher_id {ref_type} NOT NULL,
                    full_name TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    roll_number TEXT NOT NULL,
                    created_at TEXT DEFAULT {text_now},
                    UNIQUE(teacher_id, class_name, roll_number),
                    FOREIGN KEY(teacher_id) REFERENCES teachers(id)
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS student_scans (
                    id {id_type},
                    teacher_id {ref_type} NOT NULL,
                    student_id {ref_type} NOT NULL,
                    image_data {blob_type} NOT NULL,
                    image_mime TEXT NOT NULL DEFAULT 'image/jpeg',
                    attention REAL NOT NULL,
                    fatigue REAL NOT NULL,
                    posture_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    fatigue_level TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT {text_now},
                    FOREIGN KEY(teacher_id) REFERENCES teachers(id),
                    FOREIGN KEY(student_id) REFERENCES students(id)
                )
                """
            )
            # Migrate existing SQLite/PostgreSQL installations without
            # invalidating accounts created before username sign-in.
            if self.use_postgres:
                connection.execute("ALTER TABLE teachers ADD COLUMN IF NOT EXISTS username TEXT")
                connection.execute("UPDATE teachers SET username = 'teacher' || id::text WHERE username IS NULL OR username = ''")
                connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_teachers_username_ci ON teachers (LOWER(username))")
                connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_teachers_email_ci ON teachers (LOWER(email))")
            else:
                columns = {row["name"] for row in connection.execute("PRAGMA table_info(teachers)").fetchall()}
                if "username" not in columns:
                    connection.execute("ALTER TABLE teachers ADD COLUMN username TEXT")
                connection.execute("UPDATE teachers SET username = 'teacher' || id WHERE username IS NULL OR username = ''")
                connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_teachers_username_ci ON teachers (username COLLATE NOCASE)")
                connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_teachers_email_ci ON teachers (email COLLATE NOCASE)")
            connection.execute(
                "INSERT INTO teachers (id, name, username, email, password_hash, department) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                (
                    1,
                    "Demo Teacher",
                    "demoteacher",
                    "teacher@classpulse.com",
                    generate_password_hash("Teacher@123"),
                    "Administration",
                ),
            )
            if self.use_postgres:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS classpulse_legacy_imports (source_path TEXT PRIMARY KEY, imported_at TEXT DEFAULT (CURRENT_TIMESTAMP::text))"
                )
                connection.execute(
                    "SELECT setval(pg_get_serial_sequence('teachers', 'id'), GREATEST(COALESCE((SELECT MAX(id) FROM teachers), 1), 1), true)"
                )
        if self.use_postgres:
            self._import_legacy_sqlite_accounts()

    def _import_legacy_sqlite_accounts(self) -> None:
        """Copy existing login records once when a mounted SQLite DB is available."""
        source_path = self.sqlite_source_path
        if not source_path.is_file() or source_path.stat().st_size == 0:
            return
        source_key = str(source_path.resolve())
        with self._connect() as connection:
            imported = connection.execute(
                "SELECT 1 FROM classpulse_legacy_imports WHERE source_path = ?", (source_key,)
            ).fetchone()
        if imported:
            return

        source = sqlite3.connect(source_path)
        source.row_factory = sqlite3.Row
        try:
            columns = {row["name"] for row in source.execute("PRAGMA table_info(teachers)").fetchall()}
            if not {"id", "name", "email", "password_hash"}.issubset(columns):
                return
            username_column = "username" if "username" in columns else None
            with self._connect() as connection:
                for row in source.execute("SELECT * FROM teachers ORDER BY id").fetchall():
                    email = str(row["email"] or "").strip().lower()
                    if not validate_email(email) or not row["password_hash"]:
                        continue
                    existing = connection.execute(
                        "SELECT id FROM teachers WHERE lower(email) = ?", (email,)
                    ).fetchone()
                    if existing:
                        continue
                    username = str(row[username_column] or "").strip().lower() if username_column else ""
                    if not validate_username(username):
                        username = f"teacher{row['id']}"
                    for attempt in range(100):
                        collision = connection.execute(
                            "SELECT id FROM teachers WHERE lower(username) = ?", (username,)
                        ).fetchone()
                        if not collision:
                            break
                        username = f"teacher{row['id']}-{attempt + 1}"
                    connection.execute(
                        "INSERT INTO teachers (name, username, email, password_hash, department) VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                        (str(row["name"] or "Teacher")[:100], username, email, row["password_hash"], row["department"] if "department" in columns else ""),
                    )
                connection.execute(
                    "INSERT INTO classpulse_legacy_imports (source_path) VALUES (?) ON CONFLICT DO NOTHING",
                    (source_key,),
                )
        finally:
            source.close()

    def remove_demo_teacher(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM student_scans WHERE teacher_id = 1")
            connection.execute("DELETE FROM scan_history WHERE teacher_id = 1")
            connection.execute("DELETE FROM students WHERE teacher_id = 1")
            connection.execute(
                "DELETE FROM teachers WHERE id = 1 AND email = ?",
                ("teacher@classpulse.com",),
            )

    def _serialize_teacher(self, row: sqlite3.Row) -> Dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "username": row["username"],
            "email": row["email"],
            "department": row["department"],
            "created_at": row["created_at"],
        }

    def register_teacher(self, name: str, username: str, email: str, password: str, department: str = "") -> Optional[Dict]:
        clean_name = (name or "").strip()
        clean_username = (username or "").strip().lower()
        clean_email = (email or "").strip().lower()
        if not clean_name or not clean_username or not clean_email or not password:
            return None
        if not validate_email(clean_email) or not validate_username(clean_username):
            return None
        if len(password) < 8:
            return None

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM teachers WHERE lower(email) = ? OR lower(username) = ?",
                (clean_email, clean_username),
            ).fetchone()
            if existing is not None:
                return None

            cursor = connection.execute(
                "INSERT INTO teachers (name, username, email, password_hash, department) VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING RETURNING id",
                (clean_name, clean_username, clean_email, generate_password_hash(password), department.strip()),
            )
            inserted = cursor.fetchone()
            if inserted is None:
                return None
            teacher_id = inserted["id"]

        return self.get_teacher_by_id(teacher_id)

    def authenticate_teacher(self, identifier: str, password: str) -> Optional[Dict]:
        clean_identifier = (identifier or "").strip().lower()
        if not clean_identifier or not password:
            return None
        field = "email" if "@" in clean_identifier else "username"
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM teachers WHERE lower({field}) = ?",
                (clean_identifier,),
            ).fetchone()
            if row is None:
                return None
            if not check_password_hash(row["password_hash"], password):
                return None
            return self._serialize_teacher(row)

    def get_teacher_by_id(self, teacher_id: int) -> Optional[Dict]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM teachers WHERE id = ?",
                (teacher_id,),
            ).fetchone()
            if row is None:
                return None
            return self._serialize_teacher(row)

    def record_scan(self, teacher_id: int, attention: float, fatigue: float, posture_score: float = 0.0) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO scan_history (teacher_id, attention, fatigue, posture_score, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    teacher_id,
                    float(attention),
                    float(fatigue),
                    float(posture_score),
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ),
            )

    def save_student_scan(self, teacher_id: int, student: Dict, image_data: bytes, metrics: Dict, image_mime: str = "image/jpeg") -> Dict:
        name = str(student.get("name", "")).strip()
        class_name = str(student.get("class_name", "")).strip()
        roll_number = str(student.get("roll_number", "")).strip()
        if not name or not class_name or not roll_number:
            raise ValueError("Student name, class, and roll number are required.")
        if len(name) > 100 or len(class_name) > 50 or len(roll_number) > 30:
            raise ValueError("Student name, class, or roll number is too long.")
        fatigue = float(metrics.get("fatigue", 0))
        fatigue_level = "High" if fatigue >= 0.7 else "Moderate" if fatigue >= 0.4 else "Low"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO students (teacher_id, full_name, class_name, roll_number) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(teacher_id, class_name, roll_number) DO UPDATE SET full_name=excluded.full_name",
                (teacher_id, name, class_name, roll_number),
            )
            student_row = connection.execute(
                "SELECT id FROM students WHERE teacher_id=? AND class_name=? AND roll_number=?",
                (teacher_id, class_name, roll_number),
            ).fetchone()
            cursor = connection.execute(
                """INSERT INTO student_scans
                (teacher_id, student_id, image_data, image_mime, attention, fatigue, posture_score, confidence, fatigue_level, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id""",
                (teacher_id, student_row["id"], bytes(image_data), image_mime, float(metrics.get("attention", 0)), fatigue,
                 float(metrics.get("posture_score", 0)), float(metrics.get("confidence", 0)), fatigue_level, timestamp),
            )
            scan_id = cursor.fetchone()["id"]
            return {"id": scan_id, "student_id": student_row["id"], "name": name,
                    "class_name": class_name, "roll_number": roll_number, "fatigue_level": fatigue_level,
                    "fatigue": fatigue, "attention": float(metrics.get("attention", 0)), "timestamp": timestamp}

    def list_student_scans(self, teacher_id: int, class_name: str = "", roll_number: str = "") -> List[Dict]:
        query = """SELECT ss.id, s.full_name AS name, s.class_name, s.roll_number, ss.attention,
                   ss.fatigue, ss.posture_score, ss.confidence, ss.fatigue_level, ss.timestamp
                   FROM student_scans ss JOIN students s ON s.id=ss.student_id WHERE ss.teacher_id=?"""
        params: list = [teacher_id]
        if class_name:
            query += " AND s.class_name=?"; params.append(class_name)
        if roll_number:
            query += " AND s.roll_number=?"; params.append(roll_number)
        query += " ORDER BY ss.timestamp DESC LIMIT 250"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def get_student_scan_image(self, teacher_id: int, scan_id: int):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT image_data, image_mime FROM student_scans WHERE id=? AND teacher_id=?",
                (scan_id, teacher_id),
            ).fetchone()
            return (bytes(row["image_data"]), row["image_mime"]) if row else None


class TelemetryStore:
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=500))
        self.students: List[Dict] = []
        self._lock = threading.Lock()

    def append(self, attention: float, fatigue: float, timestamp: str | None = None, teacher_id: int | str = "anonymous", eye_engagement: float = 0.0, posture_score: float = 0.0) -> None:
        item = {
            "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attention": round(float(attention), 1),
            "fatigue": round(float(fatigue), 3),
            "eye_engagement": round(float(eye_engagement), 3),
            "posture_score": round(float(posture_score), 3),
        }
        with self._lock:
            self.history[str(teacher_id)].append(item)

    def get_history(self, teacher_id: int | str = "anonymous") -> List[Dict]:
        with self._lock:
            return list(self.history[str(teacher_id)])

    def get_student_snapshot(self) -> List[Dict]:
        return []

    def get_alerts(self) -> Dict:
        return {"heatmap": {"zones": []}, "alerts": []}

    def get_summary(self, teacher_id: int | str = "anonymous") -> Dict:
        history = self.get_history(teacher_id)
        total = len(history)
        average_attention = sum(item["attention"] for item in history) / max(total, 1)
        fatigue_alerts = sum(1 for item in history if item["fatigue"] > 0.75)
        return {
            "average_attention": round(average_attention, 1),
            "fatigue_alerts": fatigue_alerts,
            "active_students": total,
        }
