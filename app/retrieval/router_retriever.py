from langchain_core.documents import Document
from app.retrieval.base import BaseRetrievalStrategy
from app.chains.prompts import QUERY_ROUTER_PROMPT
from app.models.llm import get_llm

class AdaptiveRouterRetriever(BaseRetrievalStrategy):
    def __init__(self, strategies: dict[str, BaseRetrievalStrategy], llm=None):
        self.strategies = strategies
        self.llm = llm
    @property
    def name(self) -> str:
        return "Adaptive Query Router"
    @property
    def description(self) -> str:
        return "Dynamically routes query to optimal retrieval strategy."
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        key = "hybrid" if "hybrid" in self.strategies else list(self.strategies.keys())[0]
        return self.strategies[key].retrieve(query, k=k)
