import fitz


def classify_and_extract_page(page: fitz.Page, min_text_length: int = 20) -> tuple[str, str]:
    """
    Classifies a PDF page as 'NATIVE_TEXT' vs 'SCANNED_IMAGE' using PyMuPDF.
    If extracted text length is above `min_text_length`, returns ('NATIVE_TEXT', text).
    If near-zero or empty, returns ('SCANNED_IMAGE', '').
    """
    raw_text = page.get_text("text") or ""
    stripped_text = raw_text.strip()
    if len(stripped_text) > min_text_length:
        return "NATIVE_TEXT", raw_text
    return "SCANNED_IMAGE", ""
