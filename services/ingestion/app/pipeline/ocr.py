import os
from typing import Any

import fitz
import pytesseract
from PIL import Image
from shared.config import get_settings


def process_page_ocr(page: fitz.Page, dpi: int = 300) -> tuple[str, float]:
    """
    Runs real Tesseract OCR on a PyMuPDF page rendered to an image bitmap.
    Computes per-page confidence by aggregating valid word-level confidence scores (0 to 100)
    and normalizing to a 0.0 to 1.0 range.
    Returns (extracted_text, average_page_confidence).
    """
    settings = get_settings()
    if settings.TESSERACT_CMD and os.path.exists(settings.TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

    pix = page.get_pixmap(dpi=dpi)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

    data: dict[str, Any] = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    confs = data.get("conf", [])
    words = data.get("text", [])

    valid_confs: list[float] = []
    extracted_words: list[str] = []

    for conf_val, word_str in zip(confs, words, strict=False):
        try:
            c = float(conf_val)
            if c >= 0.0:
                valid_confs.append(c)
                if word_str and str(word_str).strip():
                    extracted_words.append(str(word_str))
        except (ValueError, TypeError):
            continue

    page_confidence = (sum(valid_confs) / len(valid_confs)) / 100.0 if valid_confs else 0.0
    extracted_text = " ".join(extracted_words).strip()

    return extracted_text, page_confidence
