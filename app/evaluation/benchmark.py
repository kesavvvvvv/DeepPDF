import time
from app.retrieval.base import BaseRetrievalStrategy
from app.evaluation.metrics import compute_precision_at_k, compute_mrr, compute_diversity

DEFAULT_SUITE = [
    {"question": "What are dense vs BM25 differences?", "keywords": ["dense", "bm25"]},
    {"question": "How does RRF combine rankings?", "keywords": ["rrf", "rank", "fusion"]},
]

def run_benchmark(strategies: dict[str, BaseRetrievalStrategy], k: int = 5) -> dict:
    results = {}
    for key, s in strategies.items():
        lats, precs, mrrs, divs = [], [], [], []
        for item in DEFAULT_SUITE:
            t0 = time.perf_counter()
            try:
                docs = s.retrieve(item["question"], k=k)
                lats.append((time.perf_counter() - t0) * 1000.0)
                precs.append(compute_precision_at_k(docs, item["keywords"], k=k))
                mrrs.append(compute_mrr(docs, item["keywords"]))
                divs.append(compute_diversity(docs))
            except Exception: pass
        n = max(len(lats), 1)
        results[key] = {
            "name": s.name,
            "avg_latency_ms": round(sum(lats)/n, 2),
            "precision_at_k": round(sum(precs)/n, 3),
            "mrr": round(sum(mrrs)/n, 3),
            "diversity": round(sum(divs)/n, 3),
        }
    return results
