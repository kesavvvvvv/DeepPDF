from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.hybrid_rrf import HybridRRFRetriever

def test_hybrid():
    chunks, _ = ingest_pdf("data/sample_docs/sample_research_paper.pdf")
    vs = create_vectorstore(chunks, persist_directory="data/test_hyb", recreate=True)
    h = HybridRRFRetriever(vs, chunks)
    assert len(h.retrieve("RRF", k=2)) > 0
