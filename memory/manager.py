"""
Ultron - long-term memory manager.

Stores persistent facts ("My name is Rahul", "I work at ABC", ...) in
SQLite and injects them into the system prompt so Ultron remembers them
across chats and sessions. Users can view, add, edit and delete memories
from the UI, and Ultron can auto-save simple facts from conversation.

Short-term / conversation memory lives in the `messages` table and is
replayed into the model automatically (see app.py).
"""
import re

from database import db
from utils.logger import get_logger

log = get_logger("memory")

# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
def get_all():
    return db.query("SELECT * FROM memory ORDER BY id DESC")


def get(id_):
    return db.query_one("SELECT * FROM memory WHERE id = ?", (id_,))


def add(content, category="general"):
    content = (content or "").strip()
    if not content:
        return None
    if db.query_one("SELECT id FROM memory WHERE content = ?", (content,)):
        return None  # avoid duplicates
    row_id = db.execute(
        "INSERT INTO memory (content, category) VALUES (?, ?)",
        (content, category),
    )
    log.info("Memory added: %s", content[:60])
    return row_id


def update(id_, content):
    content = (content or "").strip()
    if not content:
        return False
    db.execute("UPDATE memory SET content = ? WHERE id = ?", (content, id_))
    log.info("Memory updated: #%s -> %s", id_, content[:60])
    return True


def delete(id_):
    db.execute("DELETE FROM memory WHERE id = ?", (id_,))
    log.info("Memory deleted: #%s", id_)


def search(q):
    if not q:
        return get_all()
    return db.query(
        "SELECT * FROM memory WHERE content LIKE ? ORDER BY id DESC",
        (f"%{q}%",),
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_context(memories=None, limit=40):
    """Format memories into a compact block for the system prompt."""
    memories = memories if memories is not None else get_all()[:limit]
    if not memories:
        return ""
    lines = [f"- {m['content']}" for m in memories]
    return "Long-term memory about the user:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-extraction of simple facts
# ---------------------------------------------------------------------------
# Patterns that capture "I am / my name is / I work at / I like ..." facts.
_FACT_PATTERNS = [
    (r"\bmy name is ([A-Z][\w.\-]+(?:\s+[\w.\-]+){0,3})", "name"),
    (r"\bi(?:'m| am) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\b(?!\s+(?:a|an|the|learning|working|using))", "name"),
    (r"\bi work at (.+?)(?:\.|$)", "work"),
    (r"\bi(?:'m| am)? (?:currently )?(?:learning|studying) (.+?)(?:\.|$)", "learning"),
    (r"\bi (?:know|know how to use) (.+?)(?:\.|$)", "skill"),
    (r"\bi (?:like|love) (.+?)(?:\.|$)", "preference"),
    (r"\bi (?:don't|do not) (?:like|love) (.+?)(?:\.|$)", "dislike"),
    (r"\bi live (?:in|at) (.+?)(?:\.|$)", "location"),
    (r"\bi (?:am|ap) from (.+?)(?:\.|$)", "location"),
]

_REMEMBER_PATTERNS = [
    r"\bremember (?:that |this:?\s*)?(.+?)(?:\.|$)",
    r"\bremember:\s*(.+?)(?:\.|$)",
    r"\bsave this(?: info(?:rmation)?)?:?\s*(.+?)(?:\.|$)",
]


def extract_facts(text):
    """
    Heuristically pull simple self-referential facts out of a user message.

    Returns a list of (content, category) tuples. Misses are fine - this is
    a best-effort helper that makes the demo feel alive; users can always
    manage memory manually from the UI.
    """
    if not text:
        return []
    facts = []
    for pattern, category in _FACT_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            value = m.group(1).strip()
            if len(value) > 1 and len(value) <= 80:
                facts.append((value.capitalize(), category))
    for pattern in _REMEMBER_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            value = m.group(1).strip()
            if value and len(value) <= 200:
                facts.append((value.capitalize(), "general"))

    # De-duplicate while keeping order.
    seen, unique = set(), []
    for content, category in facts:
        key = content.lower()
        if key not in seen:
            seen.add(key)
            unique.append((content, category))
    return unique
