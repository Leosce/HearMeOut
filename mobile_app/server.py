from __future__ import annotations

import argparse
import cgi
import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
LEGACY_DB_PATH = APP_DIR / "data" / "hearmeout.db"
DATA_DIR = PROJECT_ROOT / "runs" / "hearmeout_app_data"
RUNS_DIR = PROJECT_ROOT / "runs" / "mobile_app_outputs"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "hearmeout_uploads"
DB_PATH = DATA_DIR / "hearmeout.db"

TOKEN_BYTES = 32
PBKDF2_ROUNDS = 200_000
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
TRANSLATOR_LOCK = threading.Lock()
TRANSLATOR = None
SERVER_CONFIG: dict[str, Any] = {"mock": False}
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "avhubert": {
        "key": "avhubert",
        "name": "Aurora Sentence Reader",
        "short_name": "Aurora",
        "description": "Best for general lip reading, full-sentence clips, live recording, and uploaded videos.",
        "supports_live": True,
        "supports_upload": True,
    },
    "lrw_ar": {
        "key": "lrw_ar",
        "name": "Esma3ny",
        "short_name": "Esma3ny",
        "description": "Best for TV-news styled Arabic clips with one clear spoken word.",
        "supports_live": False,
        "supports_upload": True,
    },
    "grid_lipnet": {
        "key": "grid_lipnet",
        "name": "Lyra GRID Reader",
        "short_name": "Lyra",
        "description": "Best for GRID-based lip reading with short, front-facing English phrases.",
        "supports_live": False,
        "supports_upload": True,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_filename(name: str) -> str:
    name = Path(name or "video.webm").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "video.webm"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists() and LEGACY_DB_PATH.exists():
        shutil.copy2(LEGACY_DB_PATH, DB_PATH)
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                input_name TEXT NOT NULL,
                predicted_text TEXT NOT NULL,
                arabic_text TEXT,
                confidence REAL,
                output_dir TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS settings (
                user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY(user_id, key),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "model_key" not in columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN model_key TEXT")


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS
    )
    return digest.hex(), salt.hex()


def verify_password(password: str, digest_hex: str, salt_hex: str) -> bool:
    candidate, _ = hash_password(password, salt_hex)
    return hmac.compare_digest(candidate, digest_hex)


def make_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def public_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "email": row["email"],
        "display_name": row["display_name"],
    }


def row_to_prediction(row: sqlite3.Row) -> dict[str, Any]:
    model_key = row["model_key"] or "avhubert"
    model_info = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["avhubert"])
    return {
        "id": row["id"],
        "source": row["source"],
        "input_name": row["input_name"],
        "predicted_text": row["predicted_text"],
        "arabic_text": row["arabic_text"],
        "confidence": row["confidence"],
        "model_key": model_key,
        "model_name": model_info["name"],
        "output_dir": row["output_dir"],
        "created_at": row["created_at"],
    }


def get_user_by_token(token: str | None) -> sqlite3.Row | None:
    if not token:
        return None
    with get_db() as conn:
        return conn.execute(
            """
            SELECT users.*
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()


def user_settings(user_id: int) -> dict[str, str]:
    defaults = {"detector": "hog", "scan_device": "auto"}
    with get_db() as conn:
        rows = conn.execute(
            "SELECT key, value FROM settings WHERE user_id = ?", (user_id,)
        ).fetchall()
    settings = defaults.copy()
    settings.update({row["key"]: row["value"] for row in rows})
    return settings


def save_user_setting(user_id: int, key: str, value: str) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO settings(user_id, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value
            """,
            (user_id, key, value),
        )


def update_job(job_id: str, **values: Any) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)


def public_model_list() -> list[dict[str, Any]]:
    return [dict(info) for info in MODEL_REGISTRY.values()]


def prediction_command(
    video_path: Path,
    out_dir: Path,
    detector: str,
    model_key: str,
    source: str,
    stable_mode: bool = False,
) -> list[str]:
    if model_key == "lrw_ar":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "decode_lrw_ar.py"),
            "--video",
            str(video_path),
            "--out",
            str(out_dir),
            "--device",
            "auto",
            "--phrase-mode",
            "--audio-threshold-db",
            "auto",
            "--min-silence-ms",
            "500",
            "--min-word-ms",
            "250",
            "--max-word-ms",
            "2200",
            "--pre-roll-ms",
            "120",
            "--post-roll-ms",
            "160",
        ]
    if model_key == "grid_lipnet":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "decode_grid_lipnet.py"),
            "--video",
            str(video_path),
            "--out",
            str(out_dir),
            "--device",
            "auto",
        ]
    if model_key == "avhubert":
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT_ROOT / "run_split_and_decode.ps1"),
            str(video_path),
            "-Out",
            str(out_dir),
            "-Detector",
            "hog" if stable_mode else detector,
            "-ScanDevice",
            "cpu" if stable_mode else "auto",
        ]
        if stable_mode:
            cmd += ["-Stable"]
        return cmd
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "decode_video.py"),
        "--video",
        str(video_path),
        "--out",
        str(out_dir),
        "--detector",
        detector,
    ]


def read_prediction_output(out_dir: Path, model_key: str) -> tuple[str, float | None, Any]:
    confidence = None
    top_predictions = None
    if model_key in {"lrw_ar", "grid_lipnet"}:
        prediction_json = out_dir / "prediction.json"
        if prediction_json.exists():
            prediction_data = json.loads(prediction_json.read_text(encoding="utf-8"))
            text = str(prediction_data.get("text") or "").strip()
            confidence = prediction_data.get("confidence")
            top_predictions = prediction_data.get("top")
            return text, confidence, top_predictions

    summary_file = out_dir / "summary.txt"
    if summary_file.exists():
        pieces: list[str] = []
        for line in summary_file.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            hyp = parts[-1].strip() if parts else ""
            if hyp and hyp not in {"(no output)", "None"}:
                pieces.append(hyp)
        if pieces:
            return " ".join(pieces).strip(), confidence, top_predictions

    hyp_file = out_dir / "hypotheses.txt"
    if hyp_file.exists():
        return hyp_file.read_text(encoding="utf-8").strip(), confidence, top_predictions

    raise RuntimeError("model finished but did not write hypotheses.txt, summary.txt, or prediction.json")


def friendly_prediction_error(output: str, model_key: str, source: str) -> str:
    lower = output.lower()
    if (
        "no face segments found" in lower
        or "no faces were detected" in lower
        or "face_hits=0" in lower
    ):
        if source == "live":
            return (
                "No face was detected in the recording. Please record again with your "
                "face centered, closer to the camera, and in stronger lighting."
            )
        return (
            "No face was detected in the video. Please upload a clip where the face "
            "and mouth are clearly visible."
        )
    if "model returned an empty prediction" in lower:
        return "The model ran, but it could not produce readable speech from this clip."
    return output[-4000:] or f"{MODEL_REGISTRY.get(model_key, MODEL_REGISTRY['avhubert'])['short_name']} failed to finish."


def run_prediction_job(
    job_id: str,
    user_id: int,
    source: str,
    video_path: Path,
    original_name: str,
    detector: str,
    model_key: str,
    stable_mode: bool = False,
) -> None:
    out_dir = RUNS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_info = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["avhubert"])
    update_job(
        job_id,
        status="processing",
        progress=10,
        message=f"Preparing video for {model_info['short_name']}...",
        started_at=utc_now(),
    )
    try:
        confidence = None
        top_predictions = None
        if SERVER_CONFIG.get("mock"):
            time.sleep(1.2)
            text = "عليكم" if model_key == "lrw_ar" else "Hello, how are you?"
            confidence = 92.0 if model_key == "lrw_ar" else None
            top_predictions = (
                [{"label": "عليكم", "confidence": 92.0, "class_index": 18}]
                if model_key == "lrw_ar"
                else None
            )
            command_output = "[mock] prediction completed"
        else:
            message = f"Running {model_info['short_name']}"
            if stable_mode and model_key == "avhubert":
                message += " in stable mode"
            update_job(job_id, progress=25, message=message + "...")
            cmd = prediction_command(video_path, out_dir, detector, model_key, source, stable_mode)
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            if stable_mode and model_key == "avhubert":
                env["HMO_STABLE_MODE"] = "1"
                env.setdefault("PYTHONHASHSEED", "0")
                env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60 * 30,
            )
            command_output = proc.stdout or ""
            if proc.returncode != 0:
                raise RuntimeError(friendly_prediction_error(command_output, model_key, source))
            text, confidence, top_predictions = read_prediction_output(out_dir, model_key)
            if not text:
                raise RuntimeError("model returned an empty prediction")

        update_job(job_id, progress=90, message="Saving result...")
        with get_db() as conn:
            cur = conn.execute(
                """
                INSERT INTO predictions(
                    user_id, source, input_name, predicted_text, arabic_text,
                    confidence, output_dir, created_at, model_key
                )
                VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    source,
                    original_name,
                    text,
                    confidence,
                    str(out_dir),
                    utc_now(),
                    model_key,
                ),
            )
            prediction_id = int(cur.lastrowid)
        update_job(
            job_id,
            status="succeeded",
            progress=100,
            message="Prediction ready",
            result={
                "id": prediction_id,
                "predicted_text": text,
                "confidence": confidence,
                "source": source,
                "input_name": original_name,
                "model_key": model_key,
                "model_name": model_info["name"],
                "stable_mode": bool(stable_mode and model_key == "avhubert"),
                "top_predictions": top_predictions,
                "output_dir": str(out_dir),
            },
            log_tail=command_output[-2000:],
            completed_at=utc_now(),
        )
    except Exception as exc:  # noqa: BLE001
        update_job(
            job_id,
            status="failed",
            progress=100,
            message=str(exc),
            error=str(exc),
            completed_at=utc_now(),
        )


def translate_to_arabic(text: str) -> str:
    global TRANSLATOR
    if not text.strip():
        return ""
    with TRANSLATOR_LOCK:
        if TRANSLATOR is None:
            sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
            from live_lipread import Translator  # type: ignore

            TRANSLATOR = Translator(enabled=True)
        if not getattr(TRANSLATOR, "enabled", False):
            raise RuntimeError("Arabic translation model is not available")
        return TRANSLATOR.translate(text)


class HearMeOutHandler(SimpleHTTPRequestHandler):
    server_version = "HearMeOutLocal/1.0"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[%s] %s\n" % (utc_now(), fmt % args))

    @property
    def route_path(self) -> str:
        return urllib.parse.urlparse(self.path).path

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def bearer_token(self) -> str | None:
        header = self.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            return header.split(" ", 1)[1].strip()
        return None

    def require_user(self) -> sqlite3.Row | None:
        user = get_user_by_token(self.bearer_token())
        if user is None:
            self.send_json(401, {"error": "Please log in again."})
        return user

    def do_GET(self) -> None:  # noqa: N802
        path = self.route_path
        if path == "/api/mode":
            self.send_json(200, {"simulation": bool(SERVER_CONFIG.get("mock"))})
            return
        if path == "/api/models":
            self.send_json(200, {"models": public_model_list()})
            return
        if path == "/api/me":
            user = self.require_user()
            if user is not None:
                self.send_json(200, {"user": public_user(user)})
            return
        if path == "/api/history":
            user = self.require_user()
            if user is None:
                return
            with get_db() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM predictions
                    WHERE user_id = ?
                    ORDER BY id DESC
                    LIMIT 100
                    """,
                    (user["id"],),
                ).fetchall()
            self.send_json(200, {"items": [row_to_prediction(row) for row in rows]})
            return
        if path == "/api/settings":
            user = self.require_user()
            if user is not None:
                self.send_json(200, {"settings": user_settings(user["id"])})
            return
        if path.startswith("/api/jobs/"):
            user = self.require_user()
            if user is None:
                return
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None or job.get("user_id") != user["id"]:
                self.send_json(404, {"error": "Job not found."})
            else:
                safe_job = {k: v for k, v in job.items() if k != "user_id"}
                self.send_json(200, {"job": safe_job})
            return
        if path.startswith("/api/public/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None or not job.get("public"):
                self.send_json(404, {"error": "Job not found."})
            else:
                safe_job = {k: v for k, v in job.items() if k != "user_id"}
                self.send_json(200, {"job": safe_job})
            return
        self.serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = self.route_path
            if path == "/api/signup":
                self.handle_signup()
                return
            if path == "/api/login":
                self.handle_login()
                return
            if path == "/api/settings":
                self.handle_settings()
                return
            if path == "/api/predict/upload":
                self.handle_predict_upload()
                return
            if path == "/api/predict/standalone":
                self.handle_predict_upload(public=True)
                return
            if path == "/api/translate/standalone":
                self.handle_public_translate()
                return
            if path == "/api/translate":
                self.handle_translate()
                return
            self.send_json(404, {"error": "Not found"})
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[post-error] {type(exc).__name__}: {exc}\n")
            try:
                self.send_json(500, {"error": str(exc)})
            except Exception:
                pass

    def do_DELETE(self) -> None:  # noqa: N802
        path = self.route_path
        if path.startswith("/api/history/"):
            user = self.require_user()
            if user is None:
                return
            item_id = path.rsplit("/", 1)[-1]
            with get_db() as conn:
                conn.execute(
                    "DELETE FROM predictions WHERE id = ? AND user_id = ?",
                    (item_id, user["id"]),
                )
            self.send_json(200, {"ok": True})
            return
        self.send_json(404, {"error": "Not found"})

    def handle_signup(self) -> None:
        data = self.read_json()
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))
        display_name = str(data.get("display_name", "")).strip()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            self.send_json(400, {"error": "Enter a valid email address."})
            return
        if len(password) < 6:
            self.send_json(400, {"error": "Password must be at least 6 characters."})
            return
        if not display_name:
            display_name = email.split("@", 1)[0].replace(".", " ").title()
        digest, salt = hash_password(password)
        try:
            with get_db() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO users(email, password_hash, salt, display_name, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (email, digest, salt, display_name, utc_now()),
                )
                user_id = int(cur.lastrowid)
                token = make_token()
                conn.execute(
                    "INSERT INTO sessions(token, user_id, created_at) VALUES (?, ?, ?)",
                    (token, user_id, utc_now()),
                )
                user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            self.send_json(201, {"token": token, "user": public_user(user)})
        except sqlite3.IntegrityError:
            self.send_json(409, {"error": "An account with that email already exists."})

    def handle_login(self) -> None:
        data = self.read_json()
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if user is None or not verify_password(
                password, user["password_hash"], user["salt"]
            ):
                self.send_json(401, {"error": "Email or password is incorrect."})
                return
            token = make_token()
            conn.execute(
                "INSERT INTO sessions(token, user_id, created_at) VALUES (?, ?, ?)",
                (token, user["id"], utc_now()),
            )
        self.send_json(200, {"token": token, "user": public_user(user)})

    def handle_settings(self) -> None:
        user = self.require_user()
        if user is None:
            return
        data = self.read_json()
        detector = str(data.get("detector", "")).strip()
        scan_device = str(data.get("scan_device", "")).strip()
        if detector in {"hog", "fa"}:
            save_user_setting(user["id"], "detector", detector)
        if scan_device in {"auto", "cpu", "cuda"}:
            save_user_setting(user["id"], "scan_device", scan_device)
        self.send_json(200, {"settings": user_settings(user["id"])})

    def handle_predict_upload(self, public: bool = False) -> None:
        user: sqlite3.Row | None
        if public:
            user = None
            user_id = 0
        else:
            user = self.require_user()
            if user is None:
                return
            user_id = int(user["id"])
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            self.send_json(400, {"error": "Upload a video file as multipart/form-data."})
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "video" not in form:
            self.send_json(400, {"error": "No video was uploaded."})
            return
        video_item = form["video"]
        original_name = clean_filename(getattr(video_item, "filename", "") or "recording.webm")
        source = str(form.getfirst("source", "upload") or "upload")
        if source not in {"upload", "live"}:
            source = "upload"
        settings = user_settings(user_id) if not public else {"detector": "hog", "scan_device": "auto"}
        detector = str(form.getfirst("detector", settings.get("detector", "hog")) or "hog")
        if detector not in {"hog", "fa"}:
            detector = "hog"
        requested_model = str(form.getfirst("model", "avhubert") or "avhubert")
        model_key = requested_model
        model_info = MODEL_REGISTRY.get(model_key)
        if model_info is None:
            self.send_json(400, {"error": f"Unknown model: {requested_model}"})
            return
        if source == "upload" and not model_info.get("supports_upload"):
            self.send_json(400, {"error": f"{model_info['name']} does not support uploaded videos."})
            return
        if source == "live" and not model_info.get("supports_live"):
            self.send_json(400, {"error": f"{model_info['name']} does not support live recording."})
            return
        stable_raw = str(form.getfirst("stable", "") or "").strip().lower()
        stable_mode = model_key == "avhubert" and stable_raw in {"1", "true", "yes", "on"}
        if stable_mode:
            detector = "hog"
        job_id = secrets.token_hex(12)
        saved_name = f"{job_id}_{original_name}"
        video_path = UPLOAD_DIR / saved_name
        with video_path.open("wb") as f:
            shutil.copyfileobj(video_item.file, f)
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "user_id": user_id,
                "public": public,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "source": source,
                "input_name": original_name,
                "model_key": model_key,
                "model_name": model_info["name"],
                "stable_mode": stable_mode,
                "created_at": utc_now(),
            }
        worker = threading.Thread(
            target=run_prediction_job,
            args=(job_id, user_id, source, video_path, original_name, detector, model_key, stable_mode),
            daemon=True,
        )
        worker.start()
        self.send_json(202, {"job_id": job_id})

    def handle_translate(self) -> None:
        user = self.require_user()
        if user is None:
            return
        data = self.read_json()
        text = str(data.get("text", "")).strip()
        prediction_id = data.get("prediction_id")
        try:
            translated = translate_to_arabic(text)
            if prediction_id:
                with get_db() as conn:
                    conn.execute(
                        """
                        UPDATE predictions
                        SET arabic_text = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (translated, prediction_id, user["id"]),
                    )
            self.send_json(200, {"arabic_text": translated})
        except Exception as exc:  # noqa: BLE001
            self.send_json(503, {"error": str(exc)})

    def handle_public_translate(self) -> None:
        data = self.read_json()
        text = str(data.get("text", "")).strip()
        if not text:
            self.send_json(400, {"error": "Text is required."})
            return
        try:
            translated = translate_to_arabic(text)
            if not translated:
                raise RuntimeError("Arabic translation model returned no text")
            self.send_json(200, {"arabic_text": translated})
        except Exception as exc:  # noqa: BLE001
            self.send_json(503, {"error": str(exc)})

    def serve_static(self, path: str) -> None:
        rel = "index.html" if path in {"", "/"} else urllib.parse.unquote(path).lstrip("/")
        if rel == "hear-me-out-v3.html":
            file_path = PROJECT_ROOT / "hear-me-out-v3.html"
            raw = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        file_path = (STATIC_DIR / rel).resolve()
        try:
            file_path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(403)
            return
        if not file_path.exists() or file_path.is_dir():
            file_path = STATIC_DIR / "index.html"
        raw = file_path.read_bytes()
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        if file_path.suffix == ".webmanifest":
            content_type = "application/manifest+json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def local_ip_hint() -> str:
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "YOUR-LAPTOP-IP"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mock", action="store_true",
                        help="Return a demo prediction without running AV-HuBERT.")
    args = parser.parse_args()

    SERVER_CONFIG["mock"] = args.mock
    init_db()
    server = ThreadingHTTPServer((args.host, args.port), HearMeOutHandler)
    ip = local_ip_hint()
    print(f"HearMeOut app running on http://localhost:{args.port}")
    print(f"Phone URL on the same Wi-Fi: http://{ip}:{args.port}")
    if args.mock:
        print("Mock mode is ON. Predictions return demo text.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HearMeOut app.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
