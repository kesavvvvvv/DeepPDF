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
        raise ImportError("pypdf is required to load PDF documents. Install via `pip install pypdf`.")
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_source}")
        file_name = path.name
        reader = PdfReader(str(path))
    elif isinstance(file_source, bytes):
        reader = PdfReader(BytesIO(file_source))
    elif isinstance(file_source, BytesIO):
        reader = PdfReader(file_source)
    else:
        raise ValueError(f"Unsupported file source type: {type(file_source)}")

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"The PDF '{file_name}' contains no pages.")

    documents: list[Document] = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)
        if not cleaned_text:
            continue
        metadata = {
            "source": file_name,
            "file_name": file_name,
            "page": idx + 1,
            "total_pages": total_pages,
            "char_count": len(cleaned_text),
        }
        documents.append(Document(page_content=cleaned_text, metadata=metadata))

    if not documents:
        raise ValueError(f"No readable text could be extracted from '{file_name}'.")
    return documents
