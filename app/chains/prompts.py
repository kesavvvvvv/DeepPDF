from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
# ROLE
You are DeepPDF, an elite AI research assistant specialized in answering questions about PDF documents with absolute precision.
# CONTEXT
{context}
# USER QUESTION
{question}
# INSTRUCTIONS
1. Use strictly the provided context.
2. Ground all answers with page citations [Page X].
3. If not found, reply: "I could not find sufficient information in the provided PDF document."
# ANSWER
""")

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template("""
Generate 3 distinct search variations of this query to maximize vector retrieval recall:
Original: {question}
Variations:
""")

HYDE_PROMPT = ChatPromptTemplate.from_template("""
Write a hypothetical scientific document excerpt that answers: {question}
Hypothetical passage:
""")

QUERY_ROUTER_PROMPT = ChatPromptTemplate.from_template("""
Classify query intent into: 'hybrid', 'multiquery', 'hyde', 'rerank', or 'dense'.
Query: {question}
Choice:
""")

SUMMARIZATION_PROMPT = ChatPromptTemplate.from_template("""
Summarize this PDF content:
{context}
""")
