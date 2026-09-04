# Hybrid RRF Retriever with tuning support
from collections import defaultdict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.retrieval.base import BaseRetrievalStrategy
from app.retrieval.dense_retriever import DenseVectorRetriever
from app.retrieval.bm25_retriever import BM25LexicalRetriever

class HybridRRFRetriever(BaseRetrievalStrategy):
    def __init__(self, vectorstore: Chroma, documents: list[Document], k_rrf: int = 60, dense_weight: float = 0.5, bm25_weight: float = 0.5):
        self.dense = DenseVectorRetriever(vectorstore)
        self.bm25 = BM25LexicalRetriever(documents)
        self.k_rrf = k_rrf
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
    @property
    def name(self) -> str:
        return "Hybrid Search (BM25 + Dense RRF)"
    @property
    def description(self) -> str:
        return "Reciprocal Rank Fusion fusing BM25 sparse keyword and dense semantic vector search."
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        dense_docs = self.dense.retrieve(query, k=k*2)
        bm25_docs = self.bm25.retrieve(query, k=k*2)
        scores = defaultdict(float)
        doc_map = {}
        for r, d in enumerate(dense_docs, 1):
            key = d.page_content.strip()
            doc_map[key] = d
            scores[key] += self.dense_weight / (self.k_rrf + r)
        for r, d in enumerate(bm25_docs, 1):
            key = d.page_content.strip()
            doc_map[key] = d
            scores[key] += self.bm25_weight / (self.k_rrf + r)
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        res = []
        for key in sorted_keys:
            d = Document(page_content=doc_map[key].page_content, metadata=dict(doc_map[key].metadata))
            d.metadata["rrf_score"] = round(scores[key], 5)
            res.append(d)
        return res
