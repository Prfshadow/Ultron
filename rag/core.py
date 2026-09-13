"""
Ultron - minimal RAG core (single file).

LangChain pieces actually used here:
- langchain_core.documents.Document — chunk metadata (page/source)
- langchain_huggingface.HuggingFaceEmbeddings — embeddings, with TF-IDF fallback
- Qdrant directly via qdrant_client (LangChain-compatible payloads)

Deliberately NOT using langchain_community loaders / langchain_text_splitters:
they pull in transformers->torch, which is broken on this Windows box.
Document text comes from small custom extractors (PyMuPDF -> pypdf -> OCR),
split with an equivalent recursive character splitter below.
"""
import hashlib
import os
import re
import threading
from typing import List, Dict, Any, Optional

import numpy as np

from database import db
from utils.logger import get_logger

log = get_logger("rag.core")

# Config
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
COLLECTION = "ultron_documents"
QDRANT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "qdrant")
MIN_SCORE = 0.2


# ─── Embeddings ──────────────────────────────────────────────────────
class _TfidfEncoder:
    """TF-IDF fallback (no torch needed)."""
    def __init__(self, dim=512):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.dim = dim
        self.vec = __import__("sklearn.feature_extraction.text", fromlist=["HashingVectorizer"]).HashingVectorizer(
            n_features=dim, analyzer="char_wb", ngram_range=(2,4), norm="l2", alternate_sign=False
        )

    def encode(self, texts):
        return np.asarray(self.vec.transform(texts).toarray(), dtype="float32") if texts else np.zeros((0,self.dim),"float32")
    
    def embed_documents(self, texts): return self.encode(texts).tolist()
    def embed_query(self, text): return self.encode([text])[0].tolist()


class Embeddings:
    _model = None
    _lock = threading.Lock()
    _query_cache = {}
    _cache_lock = threading.Lock()
    _max_cache_size = 1000
    
    @classmethod
    def get(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    try:
                        from langchain_huggingface import HuggingFaceEmbeddings
                        cls._model = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True}
                        )
                        cls._model.embed_query("test")
                        cls._backend = "transformer"
                    except Exception as e:
                        log.warning(f"HF embeddings failed: {e}; using TF-IDF")
                        cls._model = _TfidfEncoder()
                        cls._backend = "tfidf"
        return cls._model
    
    @classmethod
    def embed_documents(cls, texts): return cls.get().embed_documents(texts)
    @classmethod
    def embed_query(cls, text): return cls.get().embed_query(text)
    
    @classmethod
    def embed(cls, texts):
        m = cls.get()
        return np.asarray(m.embed_documents(texts), "float32") if hasattr(m, "embed_documents") else m.encode(texts)
    
    @classmethod
    def dim(cls): return 384 if getattr(cls, "_backend", None) == "transformer" else 512

    @classmethod
    def _get_cached_query(cls, text: str):
        """Get cached query embedding if available."""
        with cls._cache_lock:
            return cls._query_cache.get(text)

    @classmethod
    def _set_cached_query(cls, text: str, embedding):
        """Cache query embedding with LRU eviction."""
        with cls._cache_lock:
            if len(cls._query_cache) >= cls._max_cache_size:
                cls._query_cache.pop(next(iter(cls._query_cache)))
            cls._query_cache[text] = embedding

    @classmethod
    def cached_embed_query(cls, text: str):
        """Get query embedding with caching."""
        cached = cls._get_cached_query(text)
        if cached is not None:
            return cached
        embedding = cls.embed_query(text)
        cls._set_cached_query(text, embedding)
        return embedding


# ─── Vector Store ────────────────────────────────────────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    QDRANT_OK = True
except Exception:
    QDRANT_OK = False

import uuid
import json
from typing import List, Dict, Any, Optional

class VectorStore:
    def __init__(self):
        self.emb = Embeddings()
        if QDRANT_OK:
            try:
                self.client = QdrantClient(path=QDRANT_PATH)
                self._ensure_coll()
                self.backend = "qdrant"
                log.info("Vector store: Qdrant")
                return
            except Exception as e:
                log.warning(f"Qdrant failed: {e}")
        # Local fallback
        self._ensure_schema()
        self.reload()
        self.backend = "local"
        log.info("Vector store: local (SQLite)")

    def _ensure_coll(self):
        dim = self.emb.dim()
        try:
            info = self.client.get_collection(COLLECTION)
            if info.config.params.vectors.size != dim:
                self.client.recreate_collection(COLLECTION, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
        except Exception:
            self.client.recreate_collection(COLLECTION, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))

    def _ensure_schema(self):
        with threading.Lock():
            cols = [c["name"] for c in db.query("PRAGMA table_info(embeddings)")]
            if "namespace" not in cols:
                db.execute("ALTER TABLE embeddings ADD COLUMN namespace TEXT DEFAULT ''")
                db.execute("CREATE INDEX IF NOT EXISTS idx_emb_ns ON embeddings(namespace)")

    def reload(self):
        with threading.Lock():
            rows = db.query("SELECT id, doc_id, chunk_index, chunk_text, vector, namespace FROM embeddings")
            vecs, meta = [], []
            for r in rows:
                vecs.append(np.asarray(json.loads(r["vector"]), dtype="float32"))
                meta.append({"id": r["id"], "doc_id": str(r["doc_id"]), "chunk": r["chunk_index"], "text": r["chunk_text"], "ns": r["namespace"] or ""})
            self.vecs = np.vstack(vecs) if vecs else np.zeros((0,0),"float32")
            self.meta = meta

    def add(self, doc_id: str, chunks: List[str], ns: str = "", pages: List[str] = None) -> int:
        if not chunks: return 0
        vecs = self.emb.embed(chunks)
        ns = ns or ""
        if self.backend == "qdrant":
            pts = [qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}")),
                vector=v.tolist(),
                payload={"doc_id": doc_id, "chunk": i, "text": c, "namespace": ns, "pages": str(pages[i]) if pages and i < len(pages) and pages[i] else None}
            ) for i, (c, v) in enumerate(zip(chunks, vecs))]
            self.client.upsert(COLLECTION, points=pts)
        else:
            with threading.Lock():
                params = [(doc_id, i, c, json.dumps(v.tolist()), ns) for i, (c, v) in enumerate(zip(chunks, vecs))]
                db.executemany("INSERT INTO embeddings (doc_id, chunk_index, chunk_text, vector, namespace) VALUES (?,?,?,?,?)", params)
                self.reload()
        return len(chunks)

    def search(self, query: str, doc_ids: List[str] = None, k: int = 4, ns: str = None) -> List[Dict]:
        qv = np.array(Embeddings.cached_embed_query(query), "float32")
        if self.backend == "qdrant":
            must = []
            if doc_ids:
                must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=[str(d) for d in doc_ids])))
            if ns:
                must.append(qm.FieldCondition(key="namespace", match=qm.MatchValue(value=ns)))
            flt = qm.Filter(must=must) if must else None
            try:
                res = self.client.query_points(COLLECTION, query=qv.tolist(), query_filter=flt, limit=k, with_payload=True).points
            except:
                res = self.client.search(COLLECTION, query_vector=qv.tolist(), query_filter=flt, limit=k, with_payload=True)
            return [{"text": p.payload["text"], "doc_id": p.payload["doc_id"], "pages": p.payload.get("pages"), "score": round(p.score, 4)} for p in res if p.score >= MIN_SCORE]
        # Local fallback
        if self.vecs.size == 0: return []
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        scores = (self.vecs / (np.linalg.norm(self.vecs, axis=1, keepdims=True) + 1e-12)) @ qv
        if doc_ids: scores = np.where(np.array([m["doc_id"] in set(doc_ids) for m in self.meta]), scores, -np.inf)
        if ns: scores = np.where(np.array([m["ns"] == ns for m in self.meta]), scores, -np.inf)
        top = np.argpartition(scores, -min(k, len(scores)))[-min(k, len(scores)):]
        top = top[np.argsort(scores[top])[::-1]]
        return [{"text": self.meta[i]["text"], "doc_id": self.meta[i]["doc_id"], "score": round(float(scores[i]), 4)} for i in top if scores[i] >= MIN_SCORE]

    def delete_doc(self, doc_id: str):
        if self.backend == "qdrant":
            self.client.delete(COLLECTION, points_selector=qm.FilterSelector(filter=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=str(doc_id)))])))
        else:
            db.execute("DELETE FROM embeddings WHERE doc_id=?", (doc_id,))
            self.reload()

    def delete_namespace(self, namespace: str):
        if self.backend == "qdrant":
            self.client.delete(COLLECTION, points_selector=qm.FilterSelector(filter=qm.Filter(must=[qm.FieldCondition(key="namespace", match=qm.MatchValue(value=namespace))])))
        else:
            db.execute("DELETE FROM embeddings WHERE namespace=?", (namespace,))
            self.reload()

    def count_by_namespace(self):
        if self.backend == "qdrant":
            counts, offset = {}, None
            while True:
                res, offset = self.client.scroll(COLLECTION, limit=1000, with_payload=["namespace"], with_vectors=False, offset=offset)
                for p in res:
                    ns = (p.payload or {}).get("namespace") or ""
                    counts[ns or "default"] = counts.get(ns or "default", 0) + 1
                if not offset:
                    break
            return counts
        rows = db.query("SELECT namespace, COUNT(*) AS c FROM embeddings GROUP BY namespace")
        return {(r["namespace"] or "default"): r["c"] for r in rows}

    def doc_ids_in_namespace(self, namespace: str):
        if self.backend == "qdrant":
            ids, offset = set(), None
            while True:
                res, offset = self.client.scroll(
                    COLLECTION, limit=1000, with_payload=["doc_id"], with_vectors=False, offset=offset,
                    scroll_filter=qm.Filter(must=[qm.FieldCondition(key="namespace", match=qm.MatchValue(value=namespace))]),
                )
                for p in res:
                    doc = (p.payload or {}).get("doc_id")
                    if doc:
                        ids.add(int(doc))
                if not offset:
                    break
            return ids
        rows = db.query("SELECT DISTINCT doc_id FROM embeddings WHERE namespace=?", (namespace,))
        return {int(r["doc_id"]) for r in rows}


# ─── Document Pipeline (LangChain Loaders + Custom Splitter) ──────────
# Use custom splitter to avoid transformers dependency
def _split_text(text: str) -> List[str]:
    """Split text using custom recursive character splitter (no torch needed)."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    separators = ["\n\n", "\n", " "]
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            for sep in separators:
                cut = text.rfind(sep, start + int(CHUNK_SIZE * 0.5), end)
                if cut > start + 1:
                    end = cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks

def _split_documents(docs) -> List[str]:
    """Split LangChain documents preserving metadata."""
    all_chunks = []
    for doc in docs:
        chunks = _split_text(doc.page_content)
        all_chunks.extend(chunks)
    return all_chunks

def _load_pdf(path: str):
    """Load PDF using custom extractor (Windows-compatible)."""
    text, pages = _extract_pdf(path)
    if not text:
        return [], None
    from langchain_core.documents import Document
    docs = []
    for i, page_text in enumerate(pages):
        docs.append(Document(page_content=page_text, metadata={"page": i, "source": path}))
    log.info("Custom PDF extractor loaded %d pages from %s", len(docs), path)
    return docs, pages


def _load_docx(path: str):
    """Load DOCX using custom extractor."""
    text, _ = _extract_docx(path)
    if not text:
        return []
    from langchain_core.documents import Document
    return [Document(page_content=text, metadata={"source": path})]


def _load_text(path: str):
    """Load text/markdown files."""
    text, _ = _extract_plain(path)
    if not text:
        return []
    from langchain_core.documents import Document
    return [Document(page_content=text, metadata={"source": path})]


def _load_csv(path: str):
    """Load CSV using custom extractor."""
    text, _ = _extract_csv(path)
    if not text:
        return []
    from langchain_core.documents import Document
    return [Document(page_content=text, metadata={"source": path})]


def _load_markdown(path: str):
    """Load Markdown using custom extractor."""
    return _load_text(path)


# ─── Custom Extractors (Windows-compatible) ───
def _extract_pdf(path):
    """Extract text from a PDF using PyMuPDF (fitz), falling back to pypdf, then OCR."""
    # Try PyMuPDF first (best for text-based PDFs)
    try:
        import fitz
        log.info("PDF extraction: using PyMuPDF (fitz)")
        doc = fitz.open(path)
        try:
            # Check if PDF is password protected
            if doc.needs_pass:
                log.warning("PDF is password protected")
                doc.close()
                return "", []
            pages = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text") or ""
                if not text.strip():
                    blocks = page.get_text("blocks")
                    text = "\n".join([b[4] for b in blocks if b[4].strip()])
                tables = page.find_tables()
                if tables.tables:
                    for table in tables:
                        try:
                            table_data = table.extract()
                            if table_data:
                                text += "\n\n" + _format_table_as_markdown(table_data)
                        except Exception:
                            pass
                pages.append(text)
        finally:
            doc.close()
        full_text = "\n\n".join(pages).strip()
        if full_text:
            log.info("PyMuPDF extracted %d chars from %d pages", len(full_text), len(pages))
            return full_text, pages
        log.warning("PyMuPDF extracted no text, trying pypdf fallback")
    except Exception as e:
        log.warning("PyMuPDF failed: %s, trying pypdf fallback", e)

    # Try pypdf as fallback
    try:
        import pypdf
        log.info("PDF extraction: using pypdf fallback")
        reader = pypdf.PdfReader(path)
        # Check if PDF is password protected
        if reader.is_encrypted:
            log.warning("PDF is password protected (encrypted)")
            return "", []
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if not text.strip():
                text = page.extract_text(extraction_mode="layout") or ""
            pages.append(text)
        full_text = "\n\n".join(pages).strip()
        if full_text:
            log.info("pypdf extracted %d chars from %d pages", len(full_text), len(pages))
            return full_text, pages
        log.warning("pypdf also extracted no text")
    except Exception as e:
        log.error("pypdf extraction failed: %s", e)

    # Optional OCR fallback for scanned PDFs (requires tesseract binary)
    try:
        import pytesseract
        from PIL import Image
        import fitz as fitz_ocr
        log.info("PDF extraction: attempting OCR with pytesseract")
        doc = fitz_ocr.open(path)
        if doc.needs_pass:
            log.warning("PDF is password protected, cannot OCR")
            doc.close()
            return "", []
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, config='--psm 6')
            pages.append(text)
        doc.close()
        full_text = "\n\n".join(pages).strip()
        if full_text:
            log.info("OCR extracted %d chars from %d pages", len(full_text), len(pages))
            return full_text, pages
    except FileNotFoundError:
        log.warning("OCR skipped: tesseract binary not found. Install tesseract-ocr for scanned PDF support.")
    except Exception as e:
        log.warning("OCR extraction failed: %s", e)
    return "", []


def _format_table_as_markdown(table_data):
    """Convert table data to markdown format."""
    if not table_data:
        return ""
    header = table_data[0]
    rows = table_data[1:]
    if not header:
        return ""
    header = [str(c).replace("|", "\\|").strip() for c in header]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        cells = [str(c).replace("|", "\\|").strip() for c in row]
        while len(cells) < len(header):
            cells.append("")
        lines.append("| " + " | ".join(cells[:len(header)]) + " |")
    return "\n".join(lines)


def _extract_docx(path):
    """Extract paragraphs + tables from a DOCX using python-docx."""
    from docx import Document
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [c.text.replace("\n", " ").strip() for c in row.cells]
            rows.append(cells)
        if rows:
            parts.append(_format_table_as_markdown(rows))
    return "\n\n".join(parts).strip(), None


def _extract_plain(path):
    """Read TXT / Markdown with utf-8 (falling back to latin-1)."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read().strip(), None
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "", None


def _extract_csv(path):
    """Render a CSV as a Markdown table so the LLM reads it naturally."""
    import csv
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([cell.replace("|", "\\|") for cell in row])
    if not rows:
        return "", None
    def md_table(header, body):
        sep = ["---"] * len(header)
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
        for r in body:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)
    parts = []
    header, body = rows[0], rows[1:]
    for i in range(0, len(body), 50):
        parts.append(md_table(header, body[i : i + 50]))
    return "\n\n".join(parts), None


_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_text,
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".csv": _load_csv,
}


def load_and_split(path: str, ext: str):
    """Load document with custom extractor, split with custom recursive splitter.
    Returns (chunks, page_labels) where page_labels is a list of page numbers for each chunk (PDF only) or None.
    """
    loader = _LOADERS.get(ext.lower())
    if not loader:
        log.warning("No loader for extension: %s", ext)
        return [], None
    
    result = loader(path)
    if not result:
        return [], None
    
    # Handle both old format (just docs) and new format (docs, pages)
    if isinstance(result, tuple):
        docs, pages = result
    else:
        docs, pages = result, None
    
    if not docs:
        return [], None
    
    # Split documents - preserves metadata (page numbers, etc.)
    chunks = _split_documents(docs)
    log.info("Split %d docs into %d chunks from %s", len(docs), len(chunks), path)
    
    # Generate page labels for PDFs if we have page metadata
    page_labels = None
    if ext.lower() == ".pdf" and docs:
        page_labels = _generate_page_labels(chunks, docs)
    
    return chunks, page_labels


def _generate_page_labels(chunks, docs):
    """Generate page labels for each chunk based on source document page metadata."""
    page_labels = []
    for chunk in chunks:
        best_page = "1"
        best_overlap = 0
        for doc in docs:
            source_text = doc.page_content
            page = doc.metadata.get("page", 0) + 1  # 1-indexed
            # Find longest common substring (first 100 chars of chunk)
            test_text = chunk[:100] if len(chunk) > 100 else chunk
            if test_text in source_text and len(test_text) > best_overlap:
                best_overlap = len(test_text)
                best_page = str(page)
        page_labels.append(best_page)
    return page_labels


# ─── Global Instance ─────────────────────────────────────────────────
_store = None
def get_store():
    global _store
    if _store is None: _store = VectorStore()
    return _store


# ─── Public API ──────────────────────────────────────────────────────
def index_doc(path: str, ext: str, doc_id: str, ns: str = "") -> int:
    """Load, split, embed, and index a document."""
    chunks, pages = load_and_split(path, ext)
    if not chunks:
        return 0
    return get_store().add(doc_id, chunks, ns, pages)


def file_hash(path: str, chunk_size: int = 1_000_000) -> str:
    """A stable hash used to skip re-processing of already-uploaded files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def search(query: str, doc_ids: List[str] = None, k=4, ns: str = None) -> List[Dict]:
    return get_store().search(query, doc_ids, k, ns)


def delete_doc(doc_id: str):
    get_store().delete_doc(doc_id)