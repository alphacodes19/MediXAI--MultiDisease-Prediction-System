"""
utils/database.py
=================
Handles ALL database operations for MediXAI.

Database: SQLite (medixai_history.db — auto-created next to app.py)
No server needed. Works on any machine with Python.

Tables:
  users         → login accounts (username + SHA-256 hashed password)
  predictions   → every prediction, linked to a user
  chat_sessions → AI chatbot history, multiple sessions per user

To reset everything: just delete medixai_history.db and restart the app.
"""

import os
import json
import sqlite3
import hashlib
import datetime

# ── Find the project root (one level up from this utils/ folder) ──
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "medixai_history.db")


# ─────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # lets you access columns by name: row["disease"]
    return conn


# ─────────────────────────────────────────────────────────────────
# SCHEMA SETUP  (called once when app starts)
# ─────────────────────────────────────────────────────────────────
def init_db():
    """Create all tables if they don't already exist."""
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL DEFAULT 0,
            ts          TEXT    NOT NULL,
            disease     TEXT    NOT NULL,
            prediction  INTEGER NOT NULL,
            risk_pct    REAL    NOT NULL,
            risk_level  TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            features    TEXT    NOT NULL,
            shap_values TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS chat_sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            title      TEXT    NOT NULL DEFAULT 'New Chat',
            chat_type  TEXT    NOT NULL DEFAULT 'normal',
            messages   TEXT    NOT NULL DEFAULT '[]',
            created_at TEXT    NOT NULL,
            updated_at TEXT    NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────────────────────────
def _hash(password: str) -> str:
    """One-way SHA-256 hash. Passwords are NEVER stored as plain text."""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username: str, password: str) -> bool:
    """
    Register a new account.
    Returns True if created successfully.
    Returns False if the username is already taken (IntegrityError).
    """
    try:
        conn = _db()
        conn.execute(
            "INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
            (username.strip(), _hash(password),
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def login_user(username: str, password: str):
    """
    Check credentials.
    Returns the user's integer id if correct, None if wrong.
    """
    conn = _db()
    row  = conn.execute(
        "SELECT id FROM users WHERE username=? AND password_hash=?",
        (username.strip(), _hash(password))
    ).fetchone()
    conn.close()
    return row["id"] if row else None


# ─────────────────────────────────────────────────────────────────
# PREDICTION HISTORY
# ─────────────────────────────────────────────────────────────────
def save_prediction(result: dict, user_id: int = 0):
    """
    Save one prediction to the database.

    Expected keys in result:
      disease, prediction, risk_percent, risk_level,
      confidence, features (dict), shap_values (dict)
    """
    conn = _db()
    conn.execute(
        "INSERT INTO predictions "
        "(user_id,ts,disease,prediction,risk_pct,risk_level,confidence,features,shap_values) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            user_id,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["disease"],
            result["prediction"],
            round(result["risk_percent"], 2),
            result["risk_level"],
            round(result["confidence"], 2),
            json.dumps(result.get("features",    {})),
            json.dumps(result.get("shap_values", {})),
        )
    )
    conn.commit()
    conn.close()


def load_history(user_id: int = 0, disease: str = None, limit: int = 200) -> list:
    """
    Fetch predictions for one user, newest first.
    Optionally filter by disease name (e.g. "Diabetes").
    features and shap_values are returned as Python dicts (not JSON strings).
    """
    conn = _db()
    if disease:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id=? AND disease=? ORDER BY ts DESC LIMIT ?",
            (user_id, disease, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    conn.close()

    records = []
    for r in rows:
        d = dict(r)
        d["features"]    = json.loads(d["features"])
        d["shap_values"] = json.loads(d["shap_values"]) if d["shap_values"] else {}
        records.append(d)
    return records


def delete_record(record_id: int):
    """Delete a single prediction row by its id."""
    conn = _db()
    conn.execute("DELETE FROM predictions WHERE id=?", (record_id,))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────
# CHAT SESSIONS
# ─────────────────────────────────────────────────────────────────
def create_chat_session(user_id: int, title: str = "New Chat",
                        chat_type: str = "normal") -> int:
    """
    Start a new chat session for a user.
    chat_type = 'normal'  → regular AI conversation
    chat_type = 'report'  → triggered by OCR report upload
    Returns the new session's id.
    """
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _db()
    cur  = conn.execute(
        "INSERT INTO chat_sessions(user_id,title,chat_type,messages,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?)",
        (user_id, title, chat_type, "[]", now, now)
    )
    chat_id = cur.lastrowid
    conn.commit()
    conn.close()
    return chat_id


def load_chat_sessions(user_id: int) -> list:
    """Return all sessions for a user, newest first."""
    conn  = _db()
    rows  = conn.execute(
        "SELECT * FROM chat_sessions WHERE user_id=? ORDER BY updated_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()

    sessions = []
    for r in rows:
        msgs = json.loads(r["messages"]) if isinstance(r["messages"], str) else r["messages"]
        sessions.append({
            "id":         r["id"],
            "title":      r["title"],
            "chat_type":  r["chat_type"],
            "messages":   msgs,
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    return sessions


def save_chat_messages(chat_id: int, messages: list, title: str = None):
    """
    Overwrite the message list for a session.
    If title is provided, also rename the session (used for auto-titling).
    """
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _db()
    if title:
        conn.execute(
            "UPDATE chat_sessions SET messages=?, updated_at=?, title=? WHERE id=?",
            (json.dumps(messages), now, title, chat_id)
        )
    else:
        conn.execute(
            "UPDATE chat_sessions SET messages=?, updated_at=? WHERE id=?",
            (json.dumps(messages), now, chat_id)
        )
    conn.commit()
    conn.close()


def delete_chat_session(chat_id: int):
    """Permanently delete a chat session and all its messages."""
    conn = _db()
    conn.execute("DELETE FROM chat_sessions WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()
