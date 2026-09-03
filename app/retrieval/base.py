from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseRetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        pass
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    @property
    @abstractmethod
    def description(self) -> str:
        pass
