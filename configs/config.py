
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    docs_path: str = "./source_documents"
    loader_type: str = "pdf"
    splitter_type: str = "recursive"
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    loader_kwargs: Optional[dict] = None
    splitter_kwargs: Optional[dict] = None
    
    def __post_init__(self):
        if self.loader_kwargs is None:
            self.loader_kwargs = {}
        if self.splitter_kwargs is None:
            self.splitter_kwargs = {}