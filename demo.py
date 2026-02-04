from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.base import  ImageRefMode
converter = DocumentConverter()
result = converter.convert(Path("./demo4.docx"))
result.document.save_as_markdown('./test.md',image_mode=ImageRefMode.EMBEDDED)