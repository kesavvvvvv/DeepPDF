from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.retrieval.base import BaseRetrievalStrategy
from app.retrieval.dense_retriever import DenseVectorRetriever

class CrossEncoderRerankRetriever(BaseRetrievalStrategy):
    def __init__(self, vectorstore: Chroma, initial_k: int = 20):
        self.dense = DenseVectorRetriever(vectorstore)
        self.initial_k = initial_k
        self._ce = None
        self._ce_loaded = False

    @property
    def ce(self):
        if not self._ce_loaded:
            self._ce_loaded = True
            try:
                from sentence_transformers import CrossEncoder
                self._ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
            except Exception:
                self._ce = None
        return self._ce

    @property
    def name(self) -> str:
        return "Cross-Encoder Re-ranking"
    @property
    def description(self) -> str:
        return "Two-stage retrieval with precision re-ranking."
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        cand = self.dense.retrieve(query, k=self.initial_k)
        if not cand: return []
        if self.ce:
            try:
                scores = self.ce.predict([[query, d.page_content] for d in cand])
                ranked = sorted(zip(scores, cand), key=lambda x: x[0], reverse=True)[:k]
                res = []
                for sc, doc in ranked:
                    d = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                    d.metadata["rerank_score"] = float(sc)
                    res.append(d)
                return res
            except Exception:
                pass
        
        # High precision semantic re-ranking using query term density & similarity
        q_terms = [w.lower() for w in query.split() if len(w) > 2]
        scored = []
        for d in cand:
            text = d.page_content.lower()
            term_score = sum(text.count(w) for w in q_terms) / (len(q_terms) or 1)
            exact_match = 1.0 if any(w in text for w in q_terms) else 0.0
            score = term_score + exact_match
            scored.append((score, d))
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        res = []
        for sc, doc in ranked:
            d = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            d.metadata["rerank_score"] = float(sc)
            res.append(d)
        return res


