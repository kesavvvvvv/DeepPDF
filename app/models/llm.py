import os
from dotenv import load_dotenv
load_dotenv()

def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
