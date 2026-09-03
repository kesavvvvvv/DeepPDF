from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.dense_retriever import DenseVectorRetriever

def test_dense():
    chunks, _ = ingest_pdf("data/sample_docs/sample_research_paper.pdf")
    vs = create_vectorstore(chunks, persist_directory="data/test_dense", recreate=True)
    d = DenseVectorRetriever(vs)
    assert len(d.retrieve("RRF", k=2)) > 0
