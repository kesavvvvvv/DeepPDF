from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.hybrid_rrf import HybridRRFRetriever
from app.chains.rag_chain import create_rag_chain

def main():
    chunks, _ = ingest_pdf("data/sample_docs/sample_research_paper.pdf")
    vs = create_vectorstore(chunks, persist_directory="data/test_rag_cli", recreate=True)
    h = HybridRRFRetriever(vs, chunks)
    c = create_rag_chain()
    print("Ready!")

if __name__ == "__main__":
    main()
