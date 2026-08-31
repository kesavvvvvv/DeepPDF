import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, separators=["\n\n", "\n", ". ", " ", ""])
    return splitter.split_documents(documents)

def split_hierarchical_documents(documents: list[Document], parent_chunk_size: int = 1200, parent_overlap: int = 150, child_chunk_size: int = 300, child_overlap: int = 60) -> tuple[list[Document], dict[str, Document]]:
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_overlap, length_function=len)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_overlap, length_function=len)
    parent_docs = parent_splitter.split_documents(documents)
    parent_docstore = {}
    child_chunks = []
    for p_idx, p_doc in enumerate(parent_docs):
        parent_id = f"parent_{uuid.uuid4().hex[:8]}_{p_idx}"
        parent_docstore[parent_id] = p_doc
        children = child_splitter.split_text(p_doc.page_content)
        for c_idx, c_text in enumerate(children):
            c_meta = dict(p_doc.metadata)
            c_meta["parent_id"] = parent_id
            c_meta["child_index"] = c_idx
            child_chunks.append(Document(page_content=c_text, metadata=c_meta))
    return child_chunks, parent_docstore
