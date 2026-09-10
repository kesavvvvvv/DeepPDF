from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.hybrid_rrf import HybridRRFRetriever

def main():
    print("Testing End-to-End RAG...")
    chunks, _ = ingest_pdf("data/sample_docs/sample_research_paper.pdf")
    vs = create_vectorstore(chunks, persist_directory="data/test_e2e", recreate=True)
    h = HybridRRFRetriever(vs, chunks)
    docs = h.retrieve("RRF", k=2)
    print(f"Retrieved {len(docs)} documents.")

if __name__ == "__main__":
    main()
