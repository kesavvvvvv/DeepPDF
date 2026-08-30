try:
    import pytest
except ImportError:
    pytest = None
from pathlib import Path
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.text_splitter import split_documents, split_hierarchical_documents
from app.ingestion.pipeline import ingest_pdf

SAMPLE_PDF_PATH = Path("data/sample_docs/sample_research_paper.pdf")

def test_pdf_loading():
    if not SAMPLE_PDF_PATH.exists():
        if pytest: pytest.skip("Sample PDF not found")
        return
    docs = load_pdf(SAMPLE_PDF_PATH)
    assert len(docs) > 0
    assert docs[0].metadata["page"] == 1

def main():
    print("Testing PDF Ingestion...")
    if SAMPLE_PDF_PATH.exists():
        docs = load_pdf(SAMPLE_PDF_PATH)
        print(f"Loaded {len(docs)} pages.")

if __name__ == "__main__":
    main()
