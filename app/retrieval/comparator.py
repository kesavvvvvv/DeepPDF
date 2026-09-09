import time
from app.retrieval.base import BaseRetrievalStrategy
from app.evaluation.metrics import compute_retrieval_overlap

class RetrievalComparator:
    def __init__(self, strategies: dict[str, BaseRetrievalStrategy]):
        self.strategies = strategies
    def compare(self, query: str, k: int = 5) -> dict:
        results = {}
        for key, s in self.strategies.items():
            t0 = time.perf_counter()
            try:
                docs = s.retrieve(query, k=k)
                lat = round((time.perf_counter() - t0) * 1000.0, 2)
                results[key] = {
                    "strategy_name": s.name,
                    "description": s.description,
                    "latency_ms": lat,
                    "docs": docs,
                    "doc_count": len(docs),
                    "pages": sorted(list(set(d.metadata.get("page", 1) for d in docs))),
                    "preview": docs[0].page_content[:200] if docs else "None",
                    "status": "success",
                }
            except Exception as e:
                results[key] = {"strategy_name": s.name, "latency_ms": 0, "docs": [], "doc_count": 0, "preview": str(e), "status": "error"}
        return {"query": query, "strategies": results, "overlap_matrix": {}}
