import os
from dotenv import load_dotenv
load_dotenv()

def get_llm(provider: str = "groq", model: str | None = None, temperature: float = 0.0):
    provider = provider.lower().strip()
    if provider == "groq" or (provider == "auto" and os.getenv("GROQ_API_KEY")):
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "llama-3.3-70b-versatile", temperature=temperature)
    elif provider in ("google", "gemini") or (provider == "auto" and os.getenv("GOOGLE_API_KEY")):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-1.5-pro", temperature=temperature)
    elif provider == "openai" or (provider == "auto" and os.getenv("OPENAI_API_KEY")):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature)
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)
    except Exception:
        raise ValueError("No LLM provider configured.")
