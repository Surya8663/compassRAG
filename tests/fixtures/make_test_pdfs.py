import os
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


def create_native_pdf(output_path: Path | str, text: str | None = None) -> str:
    """
    Creates a native text PDF where PyMuPDF can directly extract characters.
    If custom `text` is provided, inserts it into a bounding box onto page 1.
    """
    path_str = str(output_path)
    with fitz.open() as doc:
        page1 = doc.new_page()
        content = text if text else (
            "Compass RAG Native Text Page 1. This page contains extractable text "
            "for the document classifier to verify native text layer extraction."
        )
        page1.insert_textbox(fitz.Rect(50, 50, 550, 750), content, fontsize=12)
        if not text:
            page2 = doc.new_page()
            page2.insert_textbox(
                fitz.Rect(50, 50, 550, 750),
                "Compass RAG Native Text Page 2. Multi-page tracking verifies Batch State Manager "
                "increments received_pages correctly.",
                fontsize=12,
            )
        doc.save(path_str)
    return path_str


def create_scanned_pdf(output_path: Path | str) -> str:
    """
    Creates a 1-page scanned/image-based PDF where text is rendered into bitmap pixels
    and no text layer exists. Forces PyMuPDF to classify as SCANNED_IMAGE and invoke Tesseract OCR.
    """
    path_str = str(output_path)
    tmp_img_path = str(output_path) + "_temp.png"

    # Create image with clear black text on white background so Tesseract reads it reliably
    img = Image.new("RGB", (1200, 1600), color="white")
    draw = ImageDraw.Draw(img)

    try:
        # Try default font or basic scalable font if available
        font = ImageFont.load_default()
    except Exception:
        font = None

    text_lines = [
        "COMPASS RAG REAL OCR TEST",
        "SCANNED DOCUMENT INGESTION PIPELINE",
        "PER PAGE CONFIDENCE SCORING VERIFICATION",
        "MANUAL REVIEW QUEUE ROUTING THRESHOLD CHECK",
    ]
    y = 200
    for line in text_lines:
        draw.text((100, y), line, fill="black", font=font)
        y += 150

    img.save(tmp_img_path)

    try:
        with fitz.open() as doc:
            page = doc.new_page(width=1200, height=1600)
            page.insert_image(fitz.Rect(0, 0, 1200, 1600), filename=tmp_img_path)
            doc.save(path_str)
    finally:
        if os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)

    return path_str
