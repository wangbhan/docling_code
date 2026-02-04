import fitz  # PyMuPDF
from pymupdf import Document

doc = Document("../demo4.pdf")
toc = doc.get_toc()

print(toc)
for level, title, page in toc:
    print(f"{'#' * level} {title} (page {page})")