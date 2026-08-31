from pathlib import Path
from io import BytesIO
from langchain_core.documents import Document
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.text_splitter import split_documents, split_hierarchical_documents

def ingest_pdf(pdf_source: str | Path | bytes | BytesIO, file_name: str = "document.pdf", chunk_size: int = 1000, chunk_overlap: int = 200, hierarchical: bool = False, parent_chunk_size: int = 1200, child_chunk_size: int = 300) -> tuple[list[Document], dict[str, Document] | None]:
    raw_docs = load_pdf(pdf_source, file_name=file_name)
    if hierarchical:
        child_chunks, parent_docstore = split_hierarchical_documents(raw_docs, parent_chunk_size=parent_chunk_size, child_chunk_size=child_chunk_size)
        return child_chunks, parent_docstore
    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks, None
