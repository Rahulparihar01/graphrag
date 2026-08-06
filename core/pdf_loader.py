import fitz  # PyMuPDF


def load_pdf(pdf_path: str) -> str:
    """Load a PDF file and return its full text content."""
    doc = fitz.open(pdf_path)
    page_count = doc.page_count  # save BEFORE closing
    full_text = ""
    for page_num, page in enumerate(doc):
        full_text += f"\n[Page {page_num + 1}]\n"
        full_text += page.get_text()
    doc.close()
    print(f"✅ PDF loaded: {len(full_text)} characters, {page_count} pages")
    return full_text
