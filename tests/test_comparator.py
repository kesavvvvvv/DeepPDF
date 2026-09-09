from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.dense_retriever import DenseVectorRetriever
from app.retrieval.comparator import RetrievalComparator

def test_comp():
    chunks, _ = ingest_pdf("data/sample_docs/sample_research_paper.pdf")
    vs = create_vectorstore(chunks, persist_directory="data/test_comp", recreate=True)
    c = RetrievalComparator({"dense": DenseVectorRetriever(vs)})
    res = c.compare("test", k=2)
    assert "dense" in res["strategies"]
