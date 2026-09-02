from langchain_core.output_parsers import StrOutputParser
from app.models.llm import get_llm
from app.chains.prompts import RAG_PROMPT

def create_rag_chain(llm=None):
    model = llm or get_llm()
    return RAG_PROMPT | model | StrOutputParser()
