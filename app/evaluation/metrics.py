def compute_retrieval_overlap(docs1: list, docs2: list) -> float:
    if not docs1 or not docs2: return 0.0
    s1 = set(d.page_content.strip() for d in docs1)
    s2 = set(d.page_content.strip() for d in docs2)
    return round(len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0, 3)

def compute_precision_at_k(docs: list, keywords: list[str], k: int = 5) -> float:
    if not docs or not keywords: return 0.0
    hits = sum(1 for d in docs[:k] if any(kw.lower() in d.page_content.lower() for kw in keywords))
    return round(hits / min(k, len(docs)), 3)

def compute_mrr(docs: list, keywords: list[str]) -> float:
    for rank, d in enumerate(docs, 1):
        if any(kw.lower() in d.page_content.lower() for kw in keywords):
            return round(1.0 / rank, 3)
    return 0.0

def compute_diversity(docs: list) -> float:
    tokens = [t for d in docs for t in d.page_content.lower().split()]
    return round(len(set(tokens)) / len(tokens), 3) if tokens else 0.0
