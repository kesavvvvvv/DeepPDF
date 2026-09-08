from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.retrieval.base import BaseRetrievalStrategy

class ParentDocumentRetriever(BaseRetrievalStrategy):
    def __init__(self, child_vectorstore: Chroma, parent_docstore: dict[str, Document]):
        self.child_vs = child_vectorstore
        self.parent_docstore = parent_docstore
    @property
    def name(self) -> str:
        return "Parent-Document (Hierarchical)"
    @property
    def description(self) -> str:
        return "Searches fine child vectors but returns rich parent context."
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        child_docs = self.child_vs.as_retriever(search_kwargs={"k": k*2}).invoke(query)
        seen = set()
        res = []
        for c in child_docs:
            p_id = c.metadata.get("parent_id")
            if p_id and p_id in self.parent_docstore:
                if p_id not in seen:
                    seen.add(p_id)
                    res.append(self.parent_docstore[p_id])
            else:
                if c.page_content not in seen:
                    seen.add(c.page_content)
                    res.append(c)
            if len(res) >= k: break
        return res[:k]
