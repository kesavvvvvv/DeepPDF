from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.parent_retriever import ParentDocumentRetriever

def test_parent():
    child_chunks, parent_docstore = ingest_pdf("data/sample_docs/sample_research_paper.pdf", hierarchical=True)
    vs = create_vectorstore(child_chunks, persist_directory="data/test_parent", recreate=True)
    p = ParentDocumentRetriever(vs, parent_docstore)
    assert len(p.retrieve("test", k=2)) > 0
