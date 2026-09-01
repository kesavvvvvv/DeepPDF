import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.embeddings.embedder import get_embeddings

def create_vectorstore(documents: list[Document], persist_directory: str = "data/vectorstore", collection_name: str = "deeppdf_collection", recreate: bool = False) -> Chroma:
    embeddings = get_embeddings()
    if recreate and os.path.exists(persist_directory):
        try: shutil.rmtree(persist_directory)
        except Exception: pass
    os.makedirs(persist_directory, exist_ok=True)
    return Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory, collection_name=collection_name)

def load_vectorstore(persist_directory: str = "data/vectorstore", collection_name: str = "deeppdf_collection") -> Chroma:
    embeddings = get_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
