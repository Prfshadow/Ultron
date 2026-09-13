"""Ultron - OCR utilities."""
import os
import io

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

def ocr_image(image_path: str) -> str:
    """Extract text from an image using Tesseract OCR."""
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        img = Image.open(image_path)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Use better OCR config for documents
        text = pytesseract.image_to_string(img, config='--psm 6')
        return text.strip()
    except Exception:
        return ""

def ocr_image_from_bytes(image_bytes: bytes) -> str:
    """Extract text from image bytes."""
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        text = pytesseract.image_to_string(img, config='--psm 6')
        return text.strip()
    except Exception:
        return ""