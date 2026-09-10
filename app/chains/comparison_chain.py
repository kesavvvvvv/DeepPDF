from app.models.llm import get_llm
from app.chains.rag_chain import create_rag_chain
from app.retrieval.comparator import RetrievalComparator

class ComparisonRAGChain:
    def __init__(self, comparator: RetrievalComparator, llm=None):
        self.comparator = comparator
        self.rag_chain = create_rag_chain(llm)
    def run_comparison(self, question: str, k: int = 4) -> dict:
        comp_data = self.comparator.compare(question, k=k)
        for key, s_res in comp_data["strategies"].items():
            docs = s_res["docs"]
            if docs:
                ctx = "\n\n".join(f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs)
                try:
                    s_res["answer"] = self.rag_chain.invoke({"context": ctx, "question": question})
                except Exception as e:
                    s_res["answer"] = f"Error: {e}"
            else:
                s_res["answer"] = "No docs retrieved."
        return comp_data
