from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.retrieval.base import BaseRetrievalStrategy
from app.chains.prompts import MULTI_QUERY_PROMPT
from app.models.llm import get_llm

class MultiQueryRetriever(BaseRetrievalStrategy):
    def __init__(self, vectorstore: Chroma, llm=None):
        self.vectorstore = vectorstore
        self.llm = llm
    @property
    def name(self) -> str:
        return "Multi-Query Expansion Retrieval"
    @property
    def description(self) -> str:
        return "Generates multiple query angles to maximize recall."
    def _gen_queries(self, q: str) -> list[str]:
        if not self.llm:
            try: self.llm = get_llm()
            except Exception: return [q]
        try:
            res = (MULTI_QUERY_PROMPT | self.llm).invoke({"question": q})
            lines = [l.strip() for l in (res.content if hasattr(res, "content") else str(res)).splitlines() if l.strip()]
            return lines[:4] or [q]
        except Exception:
            return [q]
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        sub_qs = self._gen_queries(query)
        ret = self.vectorstore.as_retriever(search_kwargs={"k": k})
        counts = {}
        docs = {}
        for sq in sub_qs:
            for d in ret.invoke(sq):
                key = d.page_content.strip()
                docs[key] = d
                counts[key] = counts.get(key, 0) + 1
        sorted_keys = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)[:k]
        return [docs[k_] for k_ in sorted_keys]
