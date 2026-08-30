import os
from pathlib import Path
from io import BytesIO
from langchain_core.documents import Document

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

def load_pdf(file_source: str | Path | bytes | BytesIO, file_name: str = "document.pdf") -> list[Document]:
    if PdfReader is None:
        raise ImportError("pypdf is required to load PDF documents.")
    if isinstance(file_source, (str, Path)):
        reader = PdfReader(str(Path(file_source)))
    elif isinstance(file_source, bytes):
        reader = PdfReader(BytesIO(file_source))
    elif isinstance(file_source, BytesIO):
        reader = PdfReader(file_source)
    else:
        raise ValueError(f"Unsupported file source type: {type(file_source)}")
    documents = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(Document(page_content=text.strip(), metadata={"source": file_name, "page": idx + 1}))
    return documents
