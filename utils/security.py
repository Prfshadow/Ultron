"""Ultron - security utilities."""
import hashlib
import mimetypes
import os
import re
import secrets
from typing import Optional

MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB per file

ALLOWED_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
ALLOWED_DOC = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv"}
ALLOWED_EXT = ALLOWED_IMAGE | ALLOWED_DOC


class UploadError(Exception):
    """Raised when file validation fails."""

def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, bytes]:
    """Hash a password with salt using PBKDF2."""
    if salt is None:
        salt = secrets.token_bytes(16)
    hash_obj = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return hash_obj.hex(), salt

def verify_password(password: str, password_hash: str, salt: bytes) -> bool:
    """Verify a password against its hash."""
    test_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(test_hash, password_hash)

def generate_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe storage."""
    import re
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    return filename[:255]

def validate_api_key(key: str, prefix: str = "") -> bool:
    """Validate API key format."""
    if not key:
        return False
    if prefix and not key.startswith(prefix):
        return False
    return len(key) >= 20


def sanitize_display_name(filename: str) -> str:
    """Sanitize a filename for display (strip path components, dangerous chars)."""
    name = os.path.basename(filename or "").strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    return name[:255] or "upload"


def mime_for(filename: str) -> Optional[str]:
    """Return the MIME type for a given filename, or None."""
    mime, _ = mimetypes.guess_type(filename)
    return mime


def validate_upload(file_storage, upload_dir: str) -> dict:
    """Validate an uploaded file and return info dict.

    Returns {"kind": "image"|"doc", "path": stored_filename, "filetype": ext}.
    Raises UploadError on validation failure.
    """
    filename = sanitize_display_name(file_storage.filename or "")
    if not filename:
        raise UploadError("No filename provided.")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise UploadError(f"File type '{ext}' is not allowed.")

    file_storage.seek(0, os.SEEK_END)
    size = file_storage.tell()
    file_storage.seek(0)
    if size > MAX_FILE_SIZE:
        raise UploadError(f"File exceeds the {MAX_FILE_SIZE // (1024*1024)} MB limit.")

    kind = "image" if ext in ALLOWED_IMAGE else "doc"

    # Unique stored name to avoid collisions
    stored = f"{secrets.token_hex(8)}{ext}"
    file_storage.save(os.path.join(upload_dir, stored))

    return {"kind": kind, "path": stored, "filetype": ext}