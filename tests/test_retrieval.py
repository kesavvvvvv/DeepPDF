from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.dense_retriever import DenseVectorRetriever
from app.retrieval.hybrid_rrf import HybridRRFRetriever

def main():
    chunks, _ = ingest_pdf("data/sample_docs/sample_research_paper.pdf")
    vs = create_vectorstore(chunks, persist_directory="data/test_ret_cli", recreate=True)
    h = HybridRRFRetriever(vs, chunks)
    print("Retrieved:", len(h.retrieve("RRF", k=2)))

if __name__ == "__main__":
    main()
