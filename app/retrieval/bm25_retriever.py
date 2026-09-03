import re
from langchain_core.documents import Document
from app.retrieval.base import BaseRetrievalStrategy

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

class BM25LexicalRetriever(BaseRetrievalStrategy):
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.corpus_tokens = [_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=k1, b=b) if (BM25Okapi and self.corpus_tokens) else None
    @property
    def name(self) -> str:
        return "BM25 Lexical Keyword Search"
    @property
    def description(self) -> str:
        return "Probabilistic lexical search using BM25 TF-IDF scoring."
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        if not self.documents: return []
        tokens = _tokenize(query)
        if not tokens: return self.documents[:k]
        if self.bm25:
            scores = self.bm25.get_scores(tokens)
            top_i = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            res = []
            for idx in top_i:
                d = Document(page_content=self.documents[idx].page_content, metadata=dict(self.documents[idx].metadata))
                d.metadata["bm25_score"] = float(scores[idx])
                res.append(d)
            return res
        return self.documents[:k]
