from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.rerank_retriever import CrossEncoderRerankRetriever

def test_rerank():
    chunks, _ = ingest_pdf("data/sample_docs/sample_research_paper.pdf")
    vs = create_vectorstore(chunks, persist_directory="data/test_rerank", recreate=True)
    r = CrossEncoderRerankRetriever(vs)
    assert len(r.retrieve("test", k=2)) > 0
