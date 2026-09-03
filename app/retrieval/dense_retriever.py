from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.retrieval.base import BaseRetrievalStrategy

class DenseVectorRetriever(BaseRetrievalStrategy):
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
    @property
    def name(self) -> str:
        return "Dense Vector Retrieval (Baseline)"
    @property
    def description(self) -> str:
        return "Standard cosine similarity vector search over ChromaDB embeddings."
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        return self.vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)
