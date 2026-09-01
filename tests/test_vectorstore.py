from dotenv import load_dotenv
load_dotenv()
from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore

def main():
    print("Testing Vectorstore...")
    chunks, _ = ingest_pdf("data/sample_docs/sample_research_paper.pdf")
    vs = create_vectorstore(chunks, persist_directory="data/test_vs", recreate=True)
    print("Created:", vs)

if __name__ == "__main__":
    main()
