# HyDE Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.retrieval.base import BaseRetrievalStrategy
from app.chains.prompts import HYDE_PROMPT
from app.models.llm import get_llm

class HyDERetriever(BaseRetrievalStrategy):
    def __init__(self, vectorstore: Chroma, llm=None):
        self.vectorstore = vectorstore
        self.llm = llm
    @property
    def name(self) -> str:
        return "Hypothetical Document Embeddings (HyDE)"
    @property
    def description(self) -> str:
        return "Generates a hypothetical document passage before embedding lookup."
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        if not self.llm:
            try: self.llm = get_llm()
            except Exception: pass
        hypo = query
        if self.llm:
            try:
                res = (HYDE_PROMPT | self.llm).invoke({"question": query})
                hypo = res.content if hasattr(res, "content") else str(res)
            except Exception: pass
        return self.vectorstore.as_retriever(search_kwargs={"k": k}).invoke(hypo)
