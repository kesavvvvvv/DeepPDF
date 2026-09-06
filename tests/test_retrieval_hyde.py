from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.hyde_retriever import HyDERetriever

def test_hyde():
    chunks, _ = ingest_pdf("data/sample_docs/sample_research_paper.pdf")
    vs = create_vectorstore(chunks, persist_directory="data/test_hyde", recreate=True)
    h = HyDERetriever(vs)
    assert len(h.retrieve("test", k=2)) > 0
