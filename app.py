"""
Ultron - AI Assistant
=====================

Entry point. Run with:

    python app.py

A lightweight, ChatGPT-style assistant built with Flask + vanilla HTML/CSS/JS — no Node.js, no React, no heavy frontend framework.

Features
--------
  * General conversation with short-term memory
  * Long-term memory stored in SQLite (view / add / edit / delete / auto-learn)
  * RAG over uploaded PDF / DOCX / TXT / MD / CSV files
  * Image understanding (vision models + optional OCR)
    * Multiple AI providers (Groq / Gemini) with auto-fallback
  * Chat history with create / rename / delete / search / export / import
  * Streaming (SSE) responses, model switching, premium glass UI
"""
import asyncio
import base64
import json
import os
import queue
import re
import shutil
import threading

import requests
from dotenv import load_dotenv

try:
    import edge_tts
except ImportError:
    edge_tts = None

# Load API keys from the .env file (keys in .env override anything else).
load_dotenv()

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from database import db
from memory import manager as memory
from models import providers
from models.providers import PERSONA
from rag import core as rag_core
from tools import create_tool_manager
from utils import security
from utils.logger import get_logger
from utils.ocr import ocr_image

log = get_logger("app")

# ---------------------------------------------------------------------------
# Response cache for deterministic queries (temp=0)
# ---------------------------------------------------------------------------
_response_cache = {}
_CACHE_MAX = 100

def _cache_key(messages, settings):
    """Generate cache key from last user message + key settings."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    model = settings.get("provider", "auto")
    temp = settings.get("temperature", "0")
    return f"{model}:{temp}:{last_user.strip()}"

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
KB_INBOX_DIR = os.path.join(BASE_DIR, "data", "kb_inbox")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KB_INBOX_DIR, exist_ok=True)

_KB_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = security.MAX_FILE_SIZE * 4  # a few files per request (800 MB)

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Disable caching for API endpoints
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Global error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": f"File too large. Maximum size is {security.MAX_FILE_SIZE // (1024*1024)} MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    log.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_file_meta(file_ids):
    """Turn file ids (or file dicts) into attachment meta for messages."""
    meta = []
    for fid in file_ids:
        if isinstance(fid, dict):
            fid = fid.get("id")
        row = db.query_one("SELECT * FROM documents WHERE id = ?", (fid,))
        if not row:
            continue
        meta.append(
            {
                "id": row["id"],
                "name": row["filename"],
                "kind": row["kind"],
                "url": f"/uploads/{row['path']}" if row["kind"] == "image" else None,
            }
        )
    return meta


def _image_resolver(file_meta):
    """For a file meta dict, return {mime, base64} if it is an image else None."""
    if isinstance(file_meta, dict):
        fid = file_meta.get("id")
    else:
        fid = file_meta
    if fid is None:
        return None
    row = db.query_one("SELECT * FROM documents WHERE id = ?", (fid,))
    if not row or row["kind"] != "image":
        return None
    try:
        with open(os.path.join(UPLOAD_DIR, row["path"]), "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
        mime = security.mime_for(row["filename"]) or "image/png"
        return {"mime": mime, "base64": data}
    except Exception:
        return None


def _image_paths(file_meta_list):
    """Return absolute paths of image attachments for local OCR."""
    paths = []
    for meta in file_meta_list or []:
        if meta.get("kind") != "image":
            continue
        row = db.query_one("SELECT * FROM documents WHERE id = ?", (meta.get("id"),))
        if row:
            paths.append(os.path.join(UPLOAD_DIR, row["path"]))
    return paths


def _rag_context(chat_id, query, settings, current_file_ids=None):
    """Build retrieved context from uploaded documents and the domain
    knowledge base (namespace from the kb_namespace setting).

    If current_file_ids is provided (non-empty), only search those specific documents.
    If empty or None, don't search chat documents (only knowledge base if configured).
    """
    if not settings.get("rag_enabled") in (True, "true", "1") or not query:
        return None, [], [], 0

    doc_ids = [fid for fid in (current_file_ids or []) if fid]
    kb_namespace = (settings.get("kb_namespace") or "").strip()
    if not doc_ids and not kb_namespace:
        return None, [], [], 0

    try:
        vector_store = rag_core.get_store()
    except Exception as e:
        log.error("Failed to initialize vector store: %s", e)
        return None, [], [], 0

    results = []
    # 1) Documents attached to the CURRENT message (if provided and non-empty).
    if doc_ids:
        results += vector_store.search(query, doc_ids=doc_ids, k=4)

    # 2) Global knowledge base (e.g. healthcare PDFs) - only if configured.
    if kb_namespace:
        results += vector_store.search(query, ns=kb_namespace, k=4)

    if not results:
        return None, [], [], 0

    # Dedupe on the chunk text and keep the strongest matches overall.
    seen, merged = set(), []
    for r in sorted(results, key=lambda r: r["score"], reverse=True):
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        merged.append(r)
        if len(merged) >= 4:
            break

    if not merged:
        return None, [], [], 0

    doc_info = {}
    for r in db.query("SELECT id, filename, path FROM documents"):
        doc_info[str(r["id"])] = {"filename": r["filename"], "path": r["path"]}

    def source_label(r):
        info = doc_info.get(str(r["doc_id"])) or {}
        name = info.get("filename", "doc")
        pages = r.get("pages")
        return f"{name}, p. {pages}" if pages else name

    blocks = [f"{source_label(r)}\n{r['text']}" for r in merged]
    raw_blocks = [r['text'] for r in merged]

    # Sources shown in the chat UI (dedupe on doc + page).
    sources, seen_src = [], set()
    for r in merged:
        key = (str(r["doc_id"]), r.get("pages"))
        if key in seen_src:
            continue
        seen_src.add(key)
        info = doc_info.get(str(r["doc_id"])) or {}
        sources.append(
            {
                "doc_id": r["doc_id"],
                "filename": info.get("filename", "doc"),
                "path": info.get("path"),
                "pages": r.get("pages"),
                "score": r["score"],
            }
        )

    context = (
        "Relevant context from the user's documents (you may not need all of it). "
        "Answer the user's question using this context when it applies; otherwise "
        "answer from general knowledge. Never name, cite, quote, or reference the "
        "documents in your answer - no document names, no brackets like [file.pdf], "
        "no page numbers, and never write the word 'Source'. At the very end of your "
        "reply add exactly one line containing just: USE_DOCS (if you actually used "
        "the context above) or NO_DOCS (if you answered without using it).\n\n"
        + "\n\n".join(blocks)
    )
    best_score = merged[0]["score"] if merged else 0
    return context, sources, raw_blocks, best_score


def _derive_title(text):
    words = text.split()
    return " ".join(words[:6]) if words else "New Chat"


def _save_assistant(chat_id, content, status, provider_meta, sources=None):
    if not content.strip():
        return None
    # Strip image markdown so it's not sent back to the LLM as image input
    content = re.sub(r'!\[.*?\]\(https?://[^)]+\)', '', content).strip()
    model = f"{provider_meta['provider']} ({provider_meta['model']})" if provider_meta else None
    return db.execute(
        "INSERT INTO messages (chat_id, role, content, model, status, sources) "
        "VALUES (?, 'assistant', ?, ?, ?, ?)",
        (chat_id, content, model, status, json.dumps(sources or [])),
    )


def _strip_trailing_source(answer):
    """Remove a trailing 'Source = None' / 'Source =' line the model sometimes
    adds even when it did not actually use the document context."""
    m = re.search(r"(?is)\s*source\s*[:=]\s*(?:none|n/?a)?\s*$", answer)
    if not m:
        return answer
    cleaned = answer[: m.start()].rstrip()
    lines = cleaned.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _source_line_from_sources(sources):
    """Build the server-controlled source line from the retrieved (score-gated)
    sources that were actually used by the model. Returns (line_or_'', sources).
    """
    order, pages = [], {}
    for s in sources or []:
        name = str(s.get("filename") or "").strip()
        if not name:
            continue
        if name not in pages:
            order.append(name)
            pages[name] = []
        pg = str(s.get("pages") or "").strip()
        if pg and pg not in pages[name]:
            pages[name].append(pg)
    rendered = []
    for name in order:
        pgs = pages.get(name) or []
        if len(pgs) == 1:
            rendered.append(name + ", p. " + pgs[0])
        elif pgs:
            rendered.append(name + ", pp. " + ", ".join(pgs))
        else:
            rendered.append(name)
    if not rendered:
        return "", []
    return "\n\nSource = " + ", ".join(rendered), list(sources)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/uploads/<path:name>")
def uploaded_file(name):
    safe = os.path.basename(name)  # block path traversal
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(UPLOAD_DIR, safe)


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------
@app.get("/api/chats")
def list_chats():
    chats = db.query(
        """
        SELECT c.id, c.title, c.created_at,
               (SELECT content FROM messages WHERE chat_id = c.id
                 AND role = 'user' ORDER BY id DESC LIMIT 1) AS preview
        FROM chats c ORDER BY c.updated_at DESC, c.id DESC
        """
    )
    return jsonify(chats)


@app.post("/api/chats")
def create_chat():
    chat_id = db.execute("INSERT INTO chats (title) VALUES (?)", ("New Chat",))
    return jsonify({"id": chat_id, "title": "New Chat"})


@app.get("/api/chats/<int:chat_id>")
def get_chat(chat_id):
    chat = db.query_one("SELECT * FROM chats WHERE id = ?", (chat_id,))
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    msgs = db.query(
        "SELECT * FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)
    )
    return jsonify({"chat": chat, "messages": msgs})


@app.patch("/api/chats/<int:chat_id>")
def rename_chat(chat_id):
    title = (request.get_json(force=True).get("title") or "New Chat").strip()[:80]
    db.execute(
        "UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (title, chat_id),
    )
    return jsonify({"ok": True, "title": title})


@app.delete("/api/chats/<int:chat_id>")
def delete_chat(chat_id):
    messages = db.query("SELECT files, content FROM messages WHERE chat_id = ?", (chat_id,))
    doc_ids = set()
    upload_files = set()
    for msg in messages:
        # Collect document IDs from file attachments
        for f in json.loads(msg["files"] or "[]"):
            if f.get("id"):
                doc_ids.add(f["id"])
        # Collect image paths from markdown in content (generated images)
        for m in re.finditer(r'!\[.*?\]\(/uploads/([^)]+)\)', msg["content"] or ""):
            upload_files.add(m.group(1))
    # Delete physical files from document records
    for doc_id in doc_ids:
        row = db.query_one("SELECT kind, path FROM documents WHERE id = ?", (doc_id,))
        if row:
            upload_files.add(row["path"])
            if row["kind"] == "doc":
                rag_core.get_store().delete_doc(doc_id)
            db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    # Delete all physical files from uploads/
    for filename in upload_files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    return jsonify({"ok": True})


@app.delete("/api/chats/all")
def delete_all_chats():
    """Delete all chats and associated files."""
    all_messages = db.query("SELECT files, content FROM messages")
    doc_ids = set()
    upload_files = set()
    for msg in all_messages:
        for f in json.loads(msg["files"] or "[]"):
            if f.get("id"):
                doc_ids.add(f["id"])
        for m in re.finditer(r'!\[.*?\]\(/uploads/([^)]+)\)', msg["content"] or ""):
            upload_files.add(m.group(1))
    for doc_id in doc_ids:
        row = db.query_one("SELECT kind, path FROM documents WHERE id = ?", (doc_id,))
        if row:
            upload_files.add(row["path"])
            if row["kind"] == "doc":
                rag_core.get_store().delete_doc(doc_id)
            db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    for filename in upload_files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    db.execute("DELETE FROM messages")
    db.execute("DELETE FROM chats")
    return jsonify({"ok": True})


@app.get("/api/search")
def search_chats():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify([])
    like = f"%{q}%"
    chat_ids = {
        r["chat_id"]
        for r in db.query(
            "SELECT DISTINCT chat_id FROM messages WHERE content LIKE ?",
            (like,),
        )
    }
    rows = db.query(
        "SELECT id, title FROM chats WHERE title LIKE ? %s ORDER BY updated_at DESC"
        % ("OR id IN ({})".format(",".join("?" * len(chat_ids))) if chat_ids else ""),
        (like, *tuple(chat_ids)) if chat_ids else (like,),
    )
    return jsonify(rows)


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
@app.post("/api/upload")
def upload_files():
    files = request.files.getlist("files")
    results = []
    for f in files:
        name = security.sanitize_display_name(f.filename or "")
        try:
            info = security.validate_upload(f, UPLOAD_DIR)
        except security.UploadError as exc:
            results.append({"error": str(exc), "name": name})
            continue

        if info["kind"] == "image":
            doc_id = db.execute(
                "INSERT INTO documents (filename, kind, path, filetype) VALUES (?, 'image', ?, ?)",
                (name, info["path"], info["filetype"]),
            )
            results.append(
                {"id": doc_id, "name": name, "kind": "image", "url": f"/uploads/{info['path']}"}
            )
            log.info("Uploaded image: %s", name)
        else:
            stored_path = os.path.join(UPLOAD_DIR, info["path"])
            doc_hash = rag_core.file_hash(stored_path)
            existing = db.query_one("SELECT id FROM documents WHERE doc_hash = ?", (doc_hash,))
            if existing:
                os.remove(stored_path)
                results.append({"id": existing["id"], "name": name, "kind": "doc"})
                log.info("Reused cached document: %s", name)
                continue

            chunks, pages = rag_core.load_and_split(stored_path, info["filetype"])
            if not chunks:
                os.remove(stored_path)
                results.append({"error": "No readable text found (scanned PDFs must be uploaded as images).", "name": name})
                continue

            doc_id = db.execute(
                "INSERT INTO documents (filename, kind, path, doc_hash, filetype, chunks) "
                "VALUES (?, 'doc', ?, ?, ?, ?)",
                (name, info["path"], doc_hash, info["filetype"], len(chunks)),
            )
            rag_core.get_store().add(str(doc_id), chunks, "", pages)
            results.append({"id": doc_id, "name": name, "kind": "doc"})
            log.info("Uploaded & indexed: %s (%d chunks)", name, len(chunks))

    return jsonify({"results": results})


@app.delete("/api/files/<int:file_id>")
def delete_file(file_id):
    """Delete an uploaded file: removes its chunks from the vector store,
    deletes the stored file, and drops the documents row."""
    row = db.query_one("SELECT * FROM documents WHERE id = ?", (file_id,))
    if not row:
        return jsonify({"error": "File not found"}), 404
    if row["kind"] == "doc":
        try:
            rag_core.delete_doc(str(file_id))
        except Exception as e:
            log.error("Vector store delete failed: %s", e)
    path = os.path.join(UPLOAD_DIR, row["path"])
    if os.path.exists(path):
        os.remove(path)
    db.execute("DELETE FROM documents WHERE id = ?", (file_id,))
    log.info("Deleted uploaded file #%s (%s)", file_id, row["filename"])
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
@app.post("/api/kb/ingest")
def kb_ingest():
    """Add PDFs/DOCX/TXT/MD/CSV to the domain knowledge base namespace.

    Runs inside the server process because Qdrant LocalMode allows only one
    process to hold the storage directory.
    """
    files = request.files.getlist("files")
    namespace = (request.form.get("namespace") or "").strip() or "healthcare"
    results = []
    total = 0
    for f in files:
        name = security.sanitize_display_name(f.filename or "")
        try:
            info = security.validate_upload(f, UPLOAD_DIR)
        except security.UploadError as exc:
            results.append({"name": name, "error": str(exc)})
            continue
        if info["kind"] != "doc":
            os.remove(os.path.join(UPLOAD_DIR, info["path"]))
            results.append({"name": name, "error": "Only documents (PDF/DOCX/TXT/MD/CSV) go in the knowledge base."})
            continue

        stored_path = os.path.join(UPLOAD_DIR, info["path"])
        doc_hash = rag_core.file_hash(stored_path)
        existing = db.query_one("SELECT id FROM documents WHERE doc_hash = ?", (doc_hash,))
        try:
            chunks, pages = rag_core.load_and_split(stored_path, info["filetype"])
        except Exception as exc:
            os.remove(stored_path)
            results.append({"name": name, "error": f"Could not read file: {exc}"})
            continue
        if not chunks:
            os.remove(stored_path)
            results.append({"name": name, "error": "No readable text found (scanned PDFs must be uploaded as images)."})
            continue

        if existing:
            doc_id = existing["id"]
        else:
            doc_id = db.execute(
                "INSERT INTO documents (filename, kind, path, doc_hash, filetype, chunks) "
                "VALUES (?, 'doc', ?, ?, ?, ?)",
                (name, info["path"], doc_hash, info["filetype"], len(chunks)),
            )

        try:
            added = rag_core.get_store().add(doc_id=str(doc_id), chunks=chunks, ns=namespace, pages=pages)
        except Exception as e:
            log.error("KB indexing failed: %s", e)
            added = 0

        db.execute("UPDATE documents SET chunks = ? WHERE id = ?", (len(chunks), doc_id))
        total += added
        results.append({"id": doc_id, "name": name, "chunks": added, "namespace": namespace})
        log.info("Ingested %s into KB '%s' (%d chunks)", name, namespace, added)

    return jsonify({"namespace": namespace, "total_chunks": total, "results": results})


@app.get("/api/kb/status")
def kb_status():
    try:
        return jsonify({"namespaces": rag_core.get_store().count_by_namespace()})
    except Exception as e:
        log.error("KB status failed: %s", e)
        return jsonify({"namespaces": {}})


@app.get("/api/kb/files")
def kb_files():
    """List documents that have points in a namespace (for per-file delete)."""
    namespace = (request.args.get("namespace") or "").strip()
    try:
        ids = rag_core.get_store().doc_ids_in_namespace(namespace) if namespace else set()
        if not ids:
            return jsonify({"files": []})
        placeholders = ",".join("?" * len(ids))
        rows = db.query(
            f"SELECT id, filename, filetype, chunks FROM documents WHERE id IN ({placeholders})",
            tuple(ids),
        )
        return jsonify({"files": rows})
    except Exception as e:
        log.error("KB files failed: %s", e)
        return jsonify({"files": []})


@app.delete("/api/kb/namespace/<namespace>")
def kb_clear_namespace(namespace):
    try:
        rag_core.get_store().delete_namespace(namespace)
        log.info("Cleared knowledge base namespace '%s'", namespace)
        return jsonify({"ok": True, "namespace": namespace})
    except Exception as e:
        log.error("KB clear namespace failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Chat (streaming)
# ---------------------------------------------------------------------------
@app.post("/api/chat")
def chat():
    data = request.get_json(force=True) or {}
    chat_id = data.get("chat_id")
    file_ids = data.get("files") or []
    regenerate = bool(data.get("regenerate"))
    rewrite_from = data.get("rewrite_from")
    text = (data.get("message") or "").strip()

    if not chat_id:
        chat_id = db.execute("INSERT INTO chats (title) VALUES (?)", ("New Chat",))
    if not db.query_one("SELECT id FROM chats WHERE id = ?", (chat_id,)):
        return jsonify({"error": "Chat not found"}), 404

    settings = db.all_settings()

    current_content = text
    current_files = _resolve_file_meta(file_ids)

    if rewrite_from:
        db.execute(
            "DELETE FROM messages WHERE chat_id = ? AND id >= ?", (chat_id, rewrite_from)
        )
        db.execute(
            "INSERT INTO messages (chat_id, role, content, files) VALUES (?, 'user', ?, ?)",
            (chat_id, text, json.dumps(current_files)),
        )
    elif regenerate:
        last_user = db.query_one(
            "SELECT * FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1",
            (chat_id,),
        )
        if last_user:
            db.execute(
                "DELETE FROM messages WHERE chat_id = ? AND id > ?", (chat_id, last_user["id"])
            )
            current_content = last_user["content"] or ""
            current_files = json.loads(last_user["files"] or "[]")
    else:
        db.execute(
            "INSERT INTO messages (chat_id, role, content, files) VALUES (?, 'user', ?, ?)",
            (chat_id, text, json.dumps(current_files)),
        )
        # Auto-name the chat from the first message.
        chat = db.query_one("SELECT title FROM chats WHERE id = ?", (chat_id,))
        if chat and chat["title"] == "New Chat" and text:
            db.execute("UPDATE chats SET title = ? WHERE id = ?", (_derive_title(text), chat_id))

        # Auto-learn simple facts (optional).
        if settings.get("memory_auto") in (True, "true", "1"):
            for content, category in memory.extract_facts(text):
                memory.add(content, category)

    # Build the provider conversation from stored messages.
    # Limit history to last 20 messages to avoid token waste and improve speed.
    MAX_HISTORY = 20
    rows = db.query(
        "SELECT role, content, files FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
        (chat_id, MAX_HISTORY)
    )
    # Reverse to get chronological order
    convo = [
        {"role": r["role"], "content": r["content"], "files": json.loads(r["files"] or "[]")}
        for r in reversed(rows)
    ]

    # --- System prompt: persona + long-term memory + RAG context + OCR ---
    system_parts = [PERSONA]
    if settings.get("memory_enabled") in (True, "true", "1"):
        ctx = memory.build_context(limit=20)  # Limit memory entries
        if ctx:
            system_parts.append(ctx)

    # RAG context for attached documents
    rag, sources, raw_blocks, rag_best_score = _rag_context(
        chat_id, current_content, settings,
        current_file_ids=[f["id"] for f in current_files if f.get("kind") == "doc"]
    )
    log.info("RAG: current_files=%s, doc_ids=%s", current_files, [f["id"] for f in current_files if f.get("kind") == "doc"])
    if rag:
        system_parts.append(rag)

    # OCR hint so Ultron can read text even if the vision provider struggles.
    ocr_texts = [ocr_image(p) for p in _image_paths(current_files)]
    ocr_texts = [t for t in ocr_texts if t]
    if ocr_texts:
        system_parts.append(
            "Optical character recognition of the attached image(s):\n" + "\n---\n".join(ocr_texts)
        )

    # Web search will be handled by ToolManager if needed (for queries needing live data)
    web_searched = False
    web_sources = []

    # Strip image markdown from assistant messages so the LLM doesn't re-send images.
    for m in convo:
        if m["role"] == "assistant":
            m["content"] = re.sub(r'!\[.*?\]\([^)]+\)', '', m["content"]).strip()

    provider_messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + convo

    def sse(obj):
        return f"data: {json.dumps(obj)}\n\n"

    def generate():
        full = ""
        tail = ""
        status = "complete"
        provider_meta = None
        ok = True
        tail_len = 500  # Larger buffer for fewer yields
        token_batch = ""
        try:
            # Check cache for identical deterministic queries
            cache_key = _cache_key(provider_messages, settings)
            if cache_key in _response_cache and settings.get("temperature") == "0":
                cached = _response_cache[cache_key]
                yield sse({"type": "token", "content": cached["content"]})
                yield sse({"type": "meta", "provider": cached["provider"], "model": cached["model"]})
                _save_assistant(chat_id, cached["content"], "complete", cached["provider_meta"], cached["sources"])
                yield sse({"type": "sources", "sources": cached["sources"]})
                yield sse({"type": "done"})
                return

            # --- Server-side tools (via ToolManager) ----------
            tool_result = None
            tool_plan = None
            if settings.get("tools_enabled") in (True, "true", "1", 1):
                tool_mgr = create_tool_manager(settings)
                has_images = any(_image_resolver(f) for f in current_files if f.get("kind") == "image")
                has_files = len(current_files) > 0
                decision = tool_mgr.classify_intent(current_content, has_images=has_images, has_files=has_files)
                if decision.primary_tool != "chat":
                    yield sse({"type": "tool", "message": "Working..."})
                    tool_plan = tool_mgr.execute_plan(current_content, has_images=has_images, has_files=has_files)
                    # Convert to legacy format for compatibility
                    if tool_plan.get("combined_direct"):
                        tool_result = {"direct": tool_plan["combined_direct"], "kind": "multi", "title": "Tool Manager"}
                    elif tool_plan.get("combined_context"):
                        tool_result = {"context": tool_plan["combined_context"], "kind": "multi", "title": "Tool Manager", "sources": tool_plan.get("sources", [])}

            # Web-search/scan results are fed to the model as extra context.
            prompt_messages = provider_messages
            if tool_result and tool_result.get("context"):
                yield sse({"type": "tool", "message": tool_result["title"], "kind": tool_result.get("kind"), "image_url": tool_result.get("image_url")})
                prompt_messages = provider_messages + [
                    {"role": "system", "content": tool_result["context"]}
                ]

            # A self-contained tool answer (time / weather / imagegen) streams directly.
            if tool_result and tool_result.get("direct"):
                answer = tool_result["direct"]
                full = answer
                provider_meta = {
                    "provider": "tool",
                    "model": tool_result.get("title") or tool_result["kind"],
                }
                yield sse({
                    "type": "tool",
                    "message": tool_result["title"],
                    "kind": tool_result.get("kind"),
                    "image_url": tool_result.get("image_url"),
                })
                yield sse({"type": "token", "content": answer})
                yield sse({"type": "meta", "provider": "tool", "model": provider_meta["model"]})
            else:
                # Include tool sources if available
                if tool_plan and tool_plan.get("sources"):
                    sources.extend(tool_plan["sources"])
                model_order = tool_plan.get("model_order") if tool_plan else None
                for ev in providers.stream_with_fallback(
                    settings, prompt_messages, _image_resolver, model_order=model_order
                ):
                    if ev["type"] == "token":
                        full += ev["content"]
                        token_batch += ev["content"]
                        tail += ev["content"]
                        # Batch tokens to reduce network overhead
                        if len(token_batch) >= 80:
                            yield sse({"type": "token", "content": token_batch})
                            token_batch = ""
                        elif len(tail) > tail_len:
                            safe, tail = tail[:-tail_len], tail[-tail_len:]
                            yield sse({"type": "token", "content": safe})
                    elif ev["type"] == "meta":
                        if token_batch:
                            yield sse({"type": "token", "content": token_batch})
                            token_batch = ""
                        provider_meta = ev
                        yield sse(ev)
                    elif ev["type"] == "fallback":
                        if token_batch:
                            yield sse({"type": "token", "content": token_batch})
                            token_batch = ""
                        yield sse(ev)
                    elif ev["type"] == "error":
                        if token_batch:
                            yield sse({"type": "token", "content": token_batch})
                            token_batch = ""
                        status = "stopped"
                        yield sse(ev)
                        break
        except GeneratorExit:
            status = "stopped"
            ok = False
        except Exception as exc:
            status = "stopped"
            ok = False
            log.error("Streaming error: %s", exc)
            yield sse({"type": "error", "message": str(exc)})
        finally:
            if token_batch:
                yield sse({"type": "token", "content": token_batch})
            if ok:
                # Model-forced usage token (USE_DOCS / NO_DOCS) at the very end.
                used_flag = None
                m_token = re.search(r"(?is)(use_docs|no_docs)\s*$", full)
                if m_token:
                    used_flag = m_token.group(1).strip().lower()
                    full = full[: m_token.start()].rstrip()

                # Remove any stray model-written source line/citation.
                full = _strip_trailing_source(full)

                tail = full[-tail_len:] if len(full) >= tail_len else full
                if tail:
                    yield sse({"type": "token", "content": tail})

                if used_flag == "use_docs":
                    # Use RAG sources
                    active_sources = sources
                    source_line, final_sources = _source_line_from_sources(active_sources)
                    if source_line:
                        full = full.rstrip() + source_line
                else:
                    final_sources = []
            else:
                final_sources = sources
            _save_assistant(chat_id, full, status, provider_meta, final_sources)
            # Cache successful deterministic responses
            if ok and status == "complete" and settings.get("temperature") == "0":
                if len(_response_cache) >= _CACHE_MAX:
                    _response_cache.pop(next(iter(_response_cache)))
                _response_cache[cache_key] = {
                    "content": full,
                    "provider": provider_meta.get("provider") if provider_meta else "unknown",
                    "model": provider_meta.get("model") if provider_meta else "unknown",
                    "provider_meta": provider_meta,
                    "sources": final_sources,
                }
        if status == "complete":
            yield sse({"type": "sources", "sources": final_sources})
            yield sse({"type": "done"})

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response
    data = request.get_json(force=True) or {}
    text = (data.get("message") or "").strip()
    chat_id = data.get("chat_id")
    file_ids = data.get("files") or []
    regenerate = bool(data.get("regenerate"))
    rewrite_from = data.get("rewrite_from")

    if not chat_id:
        chat_id = db.execute("INSERT INTO chats (title) VALUES (?)", ("New Chat",))
    if not db.query_one("SELECT id FROM chats WHERE id = ?", (chat_id,)):
        return jsonify({"error": "Chat not found"}), 404

    settings = db.all_settings()

    current_content = text
    current_files = _resolve_file_meta(file_ids)

    if rewrite_from:
        db.execute(
            "DELETE FROM messages WHERE chat_id = ? AND id >= ?", (chat_id, rewrite_from)
        )
        db.execute(
            "INSERT INTO messages (chat_id, role, content, files) VALUES (?, 'user', ?, ?)",
            (chat_id, text, json.dumps(current_files)),
        )
    elif regenerate:
        last_user = db.query_one(
            "SELECT * FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1",
            (chat_id,),
        )
        if last_user:
            db.execute(
                "DELETE FROM messages WHERE chat_id = ? AND id > ?", (chat_id, last_user["id"])
            )
            current_content = last_user["content"] or ""
            current_files = json.loads(last_user["files"] or "[]")
    else:
        db.execute(
            "INSERT INTO messages (chat_id, role, content, files) VALUES (?, 'user', ?, ?)",
            (chat_id, text, json.dumps(current_files)),
        )
        # Auto-name the chat from the first message.
        chat = db.query_one("SELECT title FROM chats WHERE id = ?", (chat_id,))
        if chat and chat["title"] == "New Chat" and text:
            db.execute("UPDATE chats SET title = ? WHERE id = ?", (_derive_title(text), chat_id))

        # Auto-learn simple facts (optional).
        if settings.get("memory_auto") in (True, "true", "1"):
            for content, category in memory.extract_facts(text):
                memory.add(content, category)

    # Build the provider conversation from stored messages.
    # Limit history to last 20 messages to avoid token waste and improve speed.
    MAX_HISTORY = 20
    rows = db.query(
        "SELECT role, content, files FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
        (chat_id, MAX_HISTORY)
    )
    # Reverse to get chronological order
    convo = [
        {"role": r["role"], "content": r["content"], "files": json.loads(r["files"] or "[]")}
        for r in reversed(rows)
    ]

    # --- System prompt: persona + long-term memory + RAG context + OCR ---
    system_parts = [PERSONA]
    if settings.get("memory_enabled") in (True, "true", "1"):
        ctx = memory.build_context(limit=20)  # Limit memory entries
        if ctx:
            system_parts.append(ctx)

    # RAG context for attached documents
    rag, sources, raw_blocks, rag_best_score = _rag_context(
        chat_id, current_content, settings,
        current_file_ids=[f["id"] for f in current_files if f.get("kind") == "doc"]
    )
    log.info("RAG: current_files=%s, doc_ids=%s", current_files, [f["id"] for f in current_files if f.get("kind") == "doc"])
    if rag:
        system_parts.append(rag)

    # OCR hint so Ultron can read text even if the vision provider struggles.
    ocr_texts = [ocr_image(p) for p in _image_paths(current_files)]
    ocr_texts = [t for t in ocr_texts if t]
    if ocr_texts:
        system_parts.append(
            "Optical character recognition of the attached image(s):\n" + "\n---\n".join(ocr_texts)
        )

    # Web search will be handled by ToolManager if needed (for queries needing live data)
    web_searched = False
    web_sources = []

    # Strip image markdown from assistant messages so the LLM doesn't re-send images.
    for m in convo:
        if m["role"] == "assistant":
            m["content"] = re.sub(r'!\[.*?\]\([^)]+\)', '', m["content"]).strip()

    provider_messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + convo

    def sse(obj):
        return f"data: {json.dumps(obj)}\n\n"

    def generate():
        with open("chat_debug.log", "a") as f:
            f.write("=== GENERATE FUNCTION CALLED ===\n")
        full = ""
        tail = ""
        status = "complete"
        provider_meta = None
        ok = True
        tail_len = 500  # Larger buffer for fewer yields
        token_batch = ""
        try:
            # Check cache for identical deterministic queries
            cache_key = _cache_key(provider_messages, settings)
            if cache_key in _response_cache and settings.get("temperature") == "0":
                cached = _response_cache[cache_key]
                yield sse({"type": "token", "content": cached["content"]})
                yield sse({"type": "meta", "provider": cached["provider"], "model": cached["model"]})
                _save_assistant(chat_id, cached["content"], "complete", cached["provider_meta"], cached["sources"])
                yield sse({"type": "sources", "sources": cached["sources"]})
                yield sse({"type": "done"})
                return

            # --- Server-side tools (via ToolManager) ----------
            tool_result = None
            tool_plan = None
            if settings.get("tools_enabled") in (True, "true", "1", 1):
                tool_mgr = create_tool_manager(settings)
                has_images = any(_image_resolver(f) for f in current_files if f.get("kind") == "image")
                has_files = len(current_files) > 0
                decision = tool_mgr.classify_intent(current_content, has_images=has_images, has_files=has_files)
                if decision.primary_tool != "chat":
                    yield sse({"type": "tool", "message": "Working..."})
                    tool_plan = tool_mgr.execute_plan(current_content, has_images=has_images, has_files=has_files)
                    # Convert to legacy format for compatibility
                    if tool_plan.get("combined_direct"):
                        tool_result = {"direct": tool_plan["combined_direct"], "kind": "multi", "title": "Tool Manager"}
                    elif tool_plan.get("combined_context"):
                        tool_result = {"context": tool_plan["combined_context"], "kind": "multi", "title": "Tool Manager", "sources": tool_plan.get("sources", [])}

            # Web-search/scan results are fed to the model as extra context.
            prompt_messages = provider_messages
            if tool_result and tool_result.get("context"):
                yield sse({"type": "tool", "message": tool_result["title"], "kind": tool_result.get("kind"), "image_url": tool_result.get("image_url")})
                prompt_messages = provider_messages + [
                    {"role": "system", "content": tool_result["context"]}
                ]

            # A self-contained tool answer (time / weather / imagegen) streams directly.
            if tool_result and tool_result.get("direct"):
                answer = tool_result["direct"]
                full = answer
                provider_meta = {
                    "provider": "tool",
                    "model": tool_result.get("title") or tool_result["kind"],
                }
                yield sse({
                    "type": "tool",
                    "message": tool_result["title"],
                    "kind": tool_result.get("kind"),
                    "image_url": tool_result.get("image_url"),
                })
                yield sse({"type": "token", "content": answer})
                yield sse({"type": "meta", "provider": "tool", "model": provider_meta["model"]})
            else:
                # Include tool sources if available
                if tool_plan and tool_plan.get("sources"):
                    sources.extend(tool_plan["sources"])
                model_order = tool_plan.get("model_order") if tool_plan else None
                for ev in providers.stream_with_fallback(
                    settings, prompt_messages, _image_resolver, model_order=model_order
                ):
                    if ev["type"] == "token":
                        full += ev["content"]
                        token_batch += ev["content"]
                        tail += ev["content"]
                        # Batch tokens to reduce network overhead
                        if len(token_batch) >= 80:
                            yield sse({"type": "token", "content": token_batch})
                            token_batch = ""
                        elif len(tail) > tail_len:
                            safe, tail = tail[:-tail_len], tail[-tail_len:]
                            yield sse({"type": "token", "content": safe})
                    elif ev["type"] == "meta":
                        if token_batch:
                            yield sse({"type": "token", "content": token_batch})
                            token_batch = ""
                        provider_meta = ev
                        yield sse(ev)
                    elif ev["type"] == "fallback":
                        if token_batch:
                            yield sse({"type": "token", "content": token_batch})
                            token_batch = ""
                        yield sse(ev)
                    elif ev["type"] == "error":
                        if token_batch:
                            yield sse({"type": "token", "content": token_batch})
                            token_batch = ""
                        status = "stopped"
                        yield sse(ev)
                        break
        except GeneratorExit:
            status = "stopped"
            ok = False
        except Exception as exc:
            status = "stopped"
            ok = False
            log.error("Streaming error: %s", exc)
            yield sse({"type": "error", "message": str(exc)})
        finally:
            if token_batch:
                yield sse({"type": "token", "content": token_batch})
            if ok:
                # Model-forced usage token (USE_DOCS / NO_DOCS) at the very end.
                used_flag = None
                m_token = re.search(r"(?is)(use_docs|no_docs)\s*$", full)
                if m_token:
                    used_flag = m_token.group(1).strip().lower()
                    full = full[: m_token.start()].rstrip()

                # Remove any stray model-written source line/citation.
                full = _strip_trailing_source(full)

                tail = full[-tail_len:] if len(full) >= tail_len else full
                if tail:
                    yield sse({"type": "token", "content": tail})

                if used_flag == "use_docs":
                    # Use RAG sources
                    active_sources = sources
                    source_line, final_sources = _source_line_from_sources(active_sources)
                    if source_line:
                        full = full.rstrip() + source_line
                else:
                    final_sources = []
            else:
                final_sources = sources
            _save_assistant(chat_id, full, status, provider_meta, final_sources)
            # Cache successful deterministic responses
            if ok and status == "complete" and settings.get("temperature") == "0":
                if len(_response_cache) >= _CACHE_MAX:
                    _response_cache.pop(next(iter(_response_cache)))
                _response_cache[cache_key] = {
                    "content": full,
                    "provider": provider_meta.get("provider") if provider_meta else "unknown",
                    "model": provider_meta.get("model") if provider_meta else "unknown",
                    "provider_meta": provider_meta,
                    "sources": final_sources,
                }
        if status == "complete":
            yield sse({"type": "sources", "sources": final_sources})
            yield sse({"type": "done"})

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
@app.get("/api/memory")
def list_memory():
    q = (request.args.get("q") or "").strip()
    return jsonify(memory.search(q) if q else memory.get_all())


@app.post("/api/memory")
def add_memory():
    content = (request.get_json(force=True).get("content") or "").strip()
    if not content:
        return jsonify({"error": "Empty memory"}), 400
    return jsonify({"id": memory.add(content, "general")})


@app.patch("/api/memory/<int:mem_id>")
def update_memory(mem_id):
    content = (request.get_json(force=True).get("content") or "").strip()
    ok = memory.update(mem_id, content)
    return jsonify({"ok": ok})


@app.delete("/api/memory/<int:mem_id>")
def delete_memory(mem_id):
    memory.delete(mem_id)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@app.get("/api/settings")
def get_settings():
    settings = db.all_settings()
    # Never expose API keys to the client - they are managed via .env only.
    for k in db.SECRET_KEYS:
        settings.pop(k, None)
    return jsonify(settings)


@app.post("/api/settings")
def save_settings():
    payload = request.get_json(force=True) or {}
    db.save_settings(payload)
    log.info("Settings updated")
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Edge neural TTS (free, no key) - Ultron-style deep voice
# ---------------------------------------------------------------------------
_EDGE_VOICES = [
    {"id": "en-US-ChristopherNeural", "name": "Christopher (deep, authoritative)"},
    {"id": "en-US-GuyNeural", "name": "Guy (deep, steady)"},
    {"id": "en-US-EricNeural", "name": "Eric (deep)"},
    {"id": "en-US-RogerNeural", "name": "Roger (deep)"},
    {"id": "en-US-SteffanNeural", "name": "Steffan (deep, US)"},
    {"id": "en-GB-RyanNeural", "name": "Ryan (deep, UK)"},
    {"id": "en-GB-ThomasNeural", "name": "Thomas (deep, UK)"},
]


@app.get("/api/tts/voices")
def tts_voices():
    """Curated list of deep Edge neural voices for the Settings dropdown."""
    if edge_tts is None:
        return jsonify({"voices": [], "engine": None})
    return jsonify({"voices": _EDGE_VOICES, "engine": "edge"})


@app.get("/api/tts")
def tts_speak():
    """Synthesize text with Edge neural voices (deep pitch, slow rate) as MP3."""
    if edge_tts is None:
        return jsonify({"error": "Edge TTS not installed"}), 501
    text = (request.args.get("text") or "").strip()[:2000]
    if not text:
        return jsonify({"error": "No text to speak"}), 400
    voice = (request.args.get("voice") or "").strip() or "en-US-ChristopherNeural"
    pitch = (request.args.get("pitch") or "").strip() or "-30Hz"
    rate = (request.args.get("rate") or "").strip() or "-8%"

    def _stream():
        q = queue.Queue(maxsize=8)

        def _producer():
            async def _run():
                com = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate, volume="+5%")
                async for chunk in com.stream():
                    if chunk["type"] == "audio":
                        q.put(chunk["data"])
            try:
                asyncio.run(_run())
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(None)

        threading.Thread(target=_producer, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                log.error("Edge TTS error: %s", item)
                break
            yield item

    try:
        return Response(
            _stream(),
            mimetype="audio/mpeg",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
    except Exception as exc:
        log.error("Edge TTS error: %s", exc)
        return jsonify({"error": str(exc)}), 502


# ---------------------------------------------------------------------------
# Voice chat (speech-to-text fallback via Groq Whisper)
# ---------------------------------------------------------------------------
@app.post("/api/voice/transcribe")
def voice_transcribe():
    """Fallback transcription for browsers without Web Speech API (Groq Whisper)."""
    key = (db.all_settings().get("groq_key") or "").strip()
    if not key:
        return jsonify({"error": "Groq API key is not configured"}), 400
    f = request.files.get("audio")
    if not f or not f.filename:
        return jsonify({"error": "No audio uploaded"}), 400
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (f.filename, f.stream, f.mimetype or "audio/webm")},
            data={"model": "whisper-large-v3-turbo", "language": "en", "temperature": 0.0},
            timeout=60,
        )
        if resp.status_code != 200:
            log.error("Groq transcribe error %s: %s", resp.status_code, resp.text[:300])
            return jsonify({"error": "Transcription failed. Check the Groq key."}), resp.status_code
        return jsonify({"text": (resp.json().get("text") or "").strip()})
    except requests.Timeout:
        return jsonify({"error": "Transcription timed out"}), 504
    except Exception as exc:
        log.error("Transcribe error: %s", exc)
        return jsonify({"error": str(exc)}), 502


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------
@app.get("/api/export")
def export_data():
    return jsonify(
        {
            "app": "ultron",
            "version": 1,
            "chats": db.query("SELECT id, title, created_at FROM chats"),
            "messages": db.query("SELECT id, chat_id, role, content, files, sources, model, status FROM messages"),
            "memory": db.query("SELECT id, content, category FROM memory"),
        }
    )


@app.post("/api/import")
def import_data():
    data = request.get_json(force=True) or {}
    if data.get("app") != "ultron":
        return jsonify({"error": "Invalid file"}), 400
    db.execute("DELETE FROM messages")
    db.execute("DELETE FROM chats")
    db.execute("DELETE FROM memory")
    for c in data.get("chats", []):
        db.execute(
            "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
            (c.get("id"), c.get("title", "New Chat"), c.get("created_at")),
        )
    for m in data.get("messages", []):
        db.execute(
            "INSERT INTO messages (id, chat_id, role, content, files, sources, model, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                m.get("id"),
                m.get("chat_id"),
                m.get("role", "user"),
                m.get("content", ""),
                m.get("files", "[]"),
                m.get("sources", "[]"),
                m.get("model"),
                m.get("status", "complete"),
            ),
        )
    for mem in data.get("memory", []):
        db.execute(
            "INSERT INTO memory (id, content, category) VALUES (?, ?, ?)",
            (mem.get("id"), mem.get("content", ""), mem.get("category", "general")),
        )
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Background services
# ---------------------------------------------------------------------------
def _kb_inbox_scan(inbox, done_dir):
    """Ingest every file dropped into data/kb_inbox into the knowledge base."""
    namespace = (db.get_setting("kb_namespace") or "healthcare").strip() or "healthcare"
    for fname in sorted(os.listdir(inbox)):
        path = os.path.join(inbox, fname)
        if os.path.isdir(path) or fname.startswith("_"):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in _KB_EXTENSIONS:
            continue
        try:
            doc_hash = rag_core.file_hash(path)
            existing = db.query_one("SELECT id FROM documents WHERE doc_hash = ?", (doc_hash,))
            chunks, pages = rag_core.load_and_split(path, ext)
            if not chunks:
                log.warning("Inbox file has no readable text: %s", fname)
            elif existing:
                doc_id = existing["id"]
                added = rag_core.get_store().add(doc_id=str(doc_id), chunks=chunks, ns=namespace)
                db.execute("UPDATE documents SET chunks = ? WHERE id = ?", (len(chunks), doc_id))
                log.info("Inbox re-ingested %s -> doc #%s (%d chunks)", fname, doc_id, len(chunks))
            else:
                doc_id = db.execute(
                    "INSERT INTO documents (filename, kind, path, doc_hash, filetype, chunks) "
                    "VALUES (?, 'doc', ?, ?, ?, ?)",
                    (fname, path, doc_hash, ext, len(chunks)),
                )
                added = rag_core.get_store().add(doc_id=str(doc_id), chunks=chunks, ns=namespace)
                log.info("Inbox ingested %s -> doc #%s (%d chunks) into '%s'", fname, doc_id, len(chunks), namespace)
        except Exception as exc:
            log.exception("Failed to ingest inbox file %s: %s", fname, exc)
        finally:
            try:
                shutil.move(path, os.path.join(done_dir, fname))
            except Exception:
                pass


def _kb_inbox_loop(stop_event):
    """Background thread: watch data/kb_inbox and auto-ingest new files."""
    done_dir = os.path.join(KB_INBOX_DIR, "_done")
    os.makedirs(done_dir, exist_ok=True)
    while not stop_event.is_set():
        try:
            _kb_inbox_scan(KB_INBOX_DIR, done_dir)
        except Exception:
            log.exception("Knowledge base inbox scan failed")
        stop_event.wait(10)


_STOP_EVENT = threading.Event()
_BG_THREADS = []


def start_background_services():
    """Start non-blocking background jobs (called once by the server entry point)."""
    if _BG_THREADS:
        return
    t = threading.Thread(target=_kb_inbox_loop, args=(_STOP_EVENT,), daemon=True, name="kb-inbox")
    t.start()
    _BG_THREADS.append(t)
    log.info("Background services started (kb inbox: %s)", KB_INBOX_DIR)

    # Pre-warm embeddings & vector store in background
    def _warmup():
        try:
            from rag.core import get_store, Embeddings
            Embeddings.get()  # Load HF model
            get_store()       # Init Qdrant/SQLite
            log.info("Warmup complete: embeddings + vector store ready")
        except Exception as e:
            log.warning("Warmup failed: %s", e)
    threading.Thread(target=_warmup, daemon=True, name="warmup").start()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    log.info("Ultron starting on http://%s:%s/", host, port)
    start_background_services()
    # use_reloader=False keeps a single worker process, which keeps in-memory
    # caches (embedding model, vector store) consistent and avoids surprises.
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)