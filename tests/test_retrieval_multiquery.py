from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.multiquery_retriever import MultiQueryRetriever

def test_mq():
    chunks, _ = ingest_pdf("data/sample_docs/sample_research_paper.pdf")
    vs = create_vectorstore(chunks, persist_directory="data/test_mq", recreate=True)
    m = MultiQueryRetriever(vs)
    assert len(m.retrieve("test", k=2)) > 0
