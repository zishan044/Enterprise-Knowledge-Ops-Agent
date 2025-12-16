
from dataclasses import dataclass
from typing import Optional, List
from dataclasses import field

@dataclass
class Config:
    docs_path: str = "./source_documents"
    loader_type: str = "pdf"
    splitter_type: str = "recursive"
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    loader_kwargs: Optional[dict] = None
    splitter_kwargs: Optional[dict] = None

    embedding_model: str = "ollama"
    embedding_params: dict = field(default_factory=lambda: {
        "model": "nomic-embed-text"
    })
    dense_k: int = 4
    sparse_k: int = 4
    retriever_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    
    def __post_init__(self):
        if self.loader_kwargs is None:
            self.loader_kwargs = {}
        if self.splitter_kwargs is None:
            self.splitter_kwargs = {}