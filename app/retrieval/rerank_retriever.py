# Cross-Encoder Re-ranking Stage 2
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.retrieval.base import BaseRetrievalStrategy
from app.retrieval.dense_retriever import DenseVectorRetriever

class CrossEncoderRerankRetriever(BaseRetrievalStrategy):
    def __init__(self, vectorstore: Chroma, initial_k: int = 20):
        self.dense = DenseVectorRetriever(vectorstore)
        self.initial_k = initial_k
        self.ce = None
        try:
            from sentence_transformers import CrossEncoder
            self.ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            pass
    @property
    def name(self) -> str:
        return "Cross-Encoder Re-ranking"
    @property
    def description(self) -> str:
        return "Two-stage retrieval with cross-encoder precision re-ranking."
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        cand = self.dense.retrieve(query, k=self.initial_k)
        if not cand: return []
        if self.ce:
            scores = self.ce.predict([[query, d.page_content] for d in cand])
            ranked = sorted(zip(scores, cand), key=lambda x: x[0], reverse=True)[:k]
            res = []
            for sc, doc in ranked:
                d = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                d.metadata["rerank_score"] = float(sc)
                res.append(d)
            return res
        return cand[:k]
