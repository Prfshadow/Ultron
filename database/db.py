"""
Ultron - SQLite database layer.

A thin, dependency-free wrapper around sqlite3 that:
  * creates the schema on first run
  * exposes tiny helpers (query / query_one / execute / executemany)
  * stores app settings as a key/value table
"""
import os
import sqlite3
import threading

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(_BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "ultron.db")

# A lock is enough for a local single-user app running under a threaded dev
# server. sqlite connections are created per call to avoid cross-thread issues.
_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL,
    role        TEXT NOT NULL,                -- 'user' | 'assistant'
    content     TEXT NOT NULL DEFAULT '',
    files       TEXT NOT NULL DEFAULT '[]',   -- JSON list of attachment meta
    sources     TEXT NOT NULL DEFAULT '[]',   -- JSON list of RAG source docs/pages
    model       TEXT,                          -- provider / model that answered
    status      TEXT NOT NULL DEFAULT 'complete', -- 'complete' | 'stopped'
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'general',
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'doc',   -- 'doc' | 'image'
    path        TEXT NOT NULL,                 -- stored name inside uploads/
    doc_hash    TEXT,                          -- dedupe hash for caching
    filetype    TEXT,
    chunks      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    vector      TEXT NOT NULL,                 -- JSON list (float32 -> float)
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_chat    ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_doc   ON embeddings(doc_id);

INSERT OR IGNORE INTO users (id, name) VALUES (1, 'Local User');
"""

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "provider": "auto",        # 'auto' | 'cyfuture' | 'groq' | 'gemini'
    "cyfuture_key": "",
    "cyfuture_model": "gpt-4o-mini",
    "cyfuture_base_url": "https://api.openai.com/v1",
    "cyfuture_vision_model": "gpt-4o",
    "groq_key": "",
    "groq_model": "openai/gpt-oss-120b",
    "groq_vision_model": "openai/gpt-oss-120b",
    "gemini_key": "",
    "gemini_model": "gemini-flash-latest",
    "gemini_vision_model": "gemini-flash-latest",
    "tavily_key": "",
    "brave_key": "",
    "temperature": "0.0",
    "max_tokens": "1024",
    "rag_enabled": "true",
    "memory_enabled": "true",
    "memory_auto": "true",
    "tools_enabled": "true",
    "provider_fallback": "false",
    "default_city": "",
    "web_results": "4",
    "kb_namespace": "",
    "theme": "dark",
}


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def connect():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    """Create tables and seed default settings if they do not exist."""
    with _LOCK:
        conn = connect()
        try:
            conn.executescript(SCHEMA)
            # Migration for existing databases that predate the sources column.
            cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)")]
            if "sources" not in cols:
                conn.execute(
                    "ALTER TABLE messages ADD COLUMN sources TEXT NOT NULL DEFAULT '[]'"
                )
            for key, value in DEFAULT_SETTINGS.items():
                conn.execute(
                    "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                    (key, value),
                )
            # Web search cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS web_search_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_search_cache_expires ON web_search_cache(expires)")
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def query(sql, params=()):
    """Run a SELECT and return a list of dicts."""
    with _LOCK:
        conn = connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def query_one(sql, params=()):
    """Run a SELECT and return one dict or None."""
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql, params=()):
    """Run a write statement and return the last inserted row id."""
    with _LOCK:
        conn = connect()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def executemany(sql, seq_params):
    with _LOCK:
        conn = connect()
        try:
            conn.executemany(sql, seq_params)
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------
def get_setting(key, default=""):
    row = query_one("SELECT value FROM settings WHERE key = ?", (key,))
    return row["value"] if row else default


def set_setting(key, value):
    execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


# Keys that must never be writable or exposed via the API.
# They are provided exclusively through environment variables / .env.
SECRET_KEYS = {
    "cyfuture_key",
    "groq_key",
    "gemini_key",
    "tavily_key",
    "brave_key",
}


def all_settings():
    """Return all settings as a dict, merged with defaults and env keys."""
    rows = query("SELECT key, value FROM settings")
    merged = dict(DEFAULT_SETTINGS)
    merged.update({r["key"]: r["value"] for r in rows})
    # Environment variables override the DB (most secure option).
    import os as _os

    env_map = {
        "cyfuture_key": "CYFUTURE_API_KEY",
        "groq_key": "GROQ_API_KEY",
        "gemini_key": "GEMINI_API_KEY",
    }
    for cfg_key, env_key in env_map.items():
        if _os.getenv(env_key):
            merged[cfg_key] = _os.getenv(env_key)
    return merged


def save_settings(payload):
    """Persist a partial settings payload. Booleans/numbers are stored as text.

    Secret keys (SECRET_KEYS) can never be changed here - they come from
    environment variables (.env) only.
    """
    allowed = set(DEFAULT_SETTINGS.keys()) - SECRET_KEYS
    for key, value in payload.items():
        if key not in allowed:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        set_setting(key, value)


# ---------------------------------------------------------------------------
# Bootstrap on import
# ---------------------------------------------------------------------------
init_db()
