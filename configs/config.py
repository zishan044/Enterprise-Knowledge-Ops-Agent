from dataclasses import dataclass

@dataclass
class Config:
    """Configuration settings for the Enterprise Knowledge Ops Agent."""
    docs_path: str = "./rag_docs"
    vectorstore_path: str = "./vectorstore"
    chunk_size: int = 400
    chunk_overlap: int = 100