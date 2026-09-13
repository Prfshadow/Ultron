"""
Ultron - image generation tool.
Primary: Hugging Face Inference API (FLUX.1-dev via fal-ai)
Fallback: Hugging Face Inference API (FLUX.1-schnell via together)
"""
import re
import os
import uuid
import io
import time
from pathlib import Path
from utils.logger import get_logger

log = get_logger("tools.imagegen")

_IMAGE_PATTERNS = [
    r"\b(?:generate|create|make|draw|paint|render)\b.*\b(?:image|picture|photo|art|illustration)\b",
    r"\b(?:image|picture|photo|art|illustration)\b.*\b(?:of|showing|with)\b",
    r"\bcan you (?:generate|create|make|draw)\b.*\b(?:image|picture|photo)\b",
    r"\bgenerate (?:an?|the) (?:image|picture|photo)\b",
    r"\bmake (?:an?|the) (?:image|picture|photo)\b",
    r"\bdraw (?:an?|the) (?:image|picture|photo)\b",
]

_DEFAULT_WIDTH = 768
_DEFAULT_HEIGHT = 768
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 3  # seconds


def _extract_dimensions(text):
    """Extract width/height from text like '512x512'."""
    for m in re.finditer(r"(\d{2,4})\s*[xxX]\s*(\d{2,4})", text, re.IGNORECASE):
        w, h = int(m.group(1)), int(m.group(2))
        w = max(64, min(1024, w))
        h = max(64, min(1024, h))
        return w, h
    return _DEFAULT_WIDTH, _DEFAULT_HEIGHT


def _sanitize_prompt(text):
    """Extract the image prompt from the user message."""
    cleaned = text.strip()
    triggers = [
        r"^\s*(?:generate|create|make|draw|paint|render)\s+(?:an?|the)?\s*(?:image|picture|photo|art|illustration)\s+(?:of|showing|with|that\s+)?",
        r"^\s*(?:can you|could you)\s+(?:generate|create|make|draw|paint|render)\s+(?:an?|the)?\s*(?:image|picture|photo|art|illustration)\s+(?:of|showing|with|that\s+)?",
        r"^\s*(?:image|picture|photo|art|illustration)\s+(?:of|showing|with)\s+",
    ]
    for t in triggers:
        cleaned = re.sub(t, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\b\d{2,4}\s*[xxX]\s*\d{2,4}\b", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .,;")
    return cleaned


def _save_image(image_bytes):
    """Save image bytes to uploads/ and return the local URL."""
    base_dir = Path(__file__).resolve().parent.parent
    upload_dir = base_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.png"
    filepath = upload_dir / filename

    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return f"/uploads/{filename}"


def _get_hf_client(provider="fal-ai"):
    """Create an InferenceClient with the given provider."""
    from huggingface_hub import InferenceClient

    api_key = os.environ.get("HF_API_KEY") or os.getenv("HF_API_KEY")
    if not api_key:
        raise Exception("HF_API_KEY not set. Add it to .env or environment variables.")

    return InferenceClient(provider=provider, api_key=api_key)


def _call_with_retry(func, label):
    """Call func() with exponential backoff on 429 rate limits."""
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            return func()
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            is_rate_limit = "429" in msg or "Too Many Requests" in msg or "rate_limit" in msg
            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                log.warning("%s rate limited (attempt %d/%d), retrying in %ds...", label, attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)
            else:
                raise
    raise last_err


def _generate_primary(prompt, width, height):
    """Primary: FLUX.1-dev via fal-ai (higher quality)."""
    client = _get_hf_client(provider="fal-ai")

    log.info("Generating image via FLUX.1-dev (fal-ai): %s...", prompt[:50])

    def _call():
        return client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-dev",
            width=width,
            height=height,
        )

    image = _call_with_retry(_call, "FLUX.1-dev")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    return _save_image(buf.read())


def _generate_fallback(prompt, width, height):
    """Fallback: FLUX.1-schnell via together (faster, lower quality)."""
    client = _get_hf_client(provider="together")

    log.info("Generating image via FLUX.1-schnell (together): %s...", prompt[:50])

    def _call():
        return client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-schnell",
            width=width,
            height=height,
        )

    image = _call_with_retry(_call, "FLUX.1-schnell")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    return _save_image(buf.read())


def generate_image(prompt, width=None, height=None):
    """Generate an image - primary: FLUX.1-dev, fallback: FLUX.1-schnell."""
    if not prompt:
        raise ValueError("Empty prompt")

    w, h = width or _DEFAULT_WIDTH, height or _DEFAULT_HEIGHT

    # Primary: FLUX.1-dev via fal-ai
    try:
        image_url = _generate_primary(prompt, w, h)
        return {"url": image_url, "prompt": prompt, "width": w, "height": h, "source": "hf_fal"}
    except Exception as exc:
        log.warning("FLUX.1-dev failed: %s. Trying FLUX.1-schnell...", exc)

    # Fallback: FLUX.1-schnell via together
    try:
        image_url = _generate_fallback(prompt, w, h)
        return {"url": image_url, "prompt": prompt, "width": w, "height": h, "source": "hf_together"}
    except Exception as exc2:
        log.error("Both generators failed: %s", exc2)
        raise Exception(f"Both generators failed: {exc2}")


def imagegen_answer(text):
    """Generate an image based on the prompt in text."""
    prompt = _sanitize_prompt(text)
    if not prompt:
        return {"kind": "imagegen", "direct": "Please describe what you'd like me to generate.", "title": "Image generation"}

    w, h = _extract_dimensions(text)

    try:
        result = generate_image(prompt, w, h)
    except Exception as exc:
        return {"kind": "imagegen", "direct": f"Image generation failed: {exc}", "title": "Image generation"}

    return {
        "kind": "imagegen",
        "direct": f"[Image generated: {prompt}]\n\n![{prompt}]({result['url']})",
        "image_url": result["url"],
        "prompt": prompt,
        "width": w,
        "height": h,
        "title": "Image generation",
        "source": result.get("source"),
    }


def detect_image(text):
    """Check if the query is an image generation request."""
    q = (text or "").strip()
    if not q:
        return False
    return any(re.search(p, q, re.IGNORECASE) for p in _IMAGE_PATTERNS)
