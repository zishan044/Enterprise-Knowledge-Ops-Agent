from pathlib import Path
from typing import List
from langchain_classic.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IngestionPipeline:
    """Pipeline to load and chunk PDF documents with metadata."""

    def __init__(self, dir_path: str | Path, loader: PyMuPDFLoader, splitter: RecursiveCharacterTextSplitter):
        """
        Args:
            dir_path: Directory containing PDF files.
            loader: PDF loader class (e.g., PyMuPDFLoader).
            splitter: Text splitter for chunking documents.
        """
        self.dir_path = Path(dir_path)
        self.loader = loader
        self.splitter = splitter

    def load_documents(self) -> List[Document]:
        """Load all PDFs in directory, adding file name and page metadata."""
        docs = []

        for pdf_file in self.dir_path.rglob("*.pdf"):
            loader_instance = self.loader(str(pdf_file))
            loaded_docs = loader_instance.load()

            for doc in loaded_docs:
                doc.metadata.update({
                    "file_name": pdf_file.name,
                    "page_no": doc.metadata.get("page_number"),
                })
                docs.append(doc)
        return docs

    def chunk_documents(self) -> List[Document]:
        """Chunk loaded documents into smaller pieces using the splitter."""
        docs = self.load_documents()
        return self.splitter.split_documents(docs)
    
    def create_ingestion_pipeline(docs_path: Path, chunk_size: int = 400, chunk_overlap: int = 100) -> List[Document]:
        """Load and chunk PDF documents from the given path."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ingestion = IngestionPipeline(dir_path=docs_path, loader=PyMuPDFLoader, splitter=splitter)
        return ingestion.chunk_documents()