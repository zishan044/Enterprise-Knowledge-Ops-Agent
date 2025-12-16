from pathlib import Path
from typing import List
from langchain_classic.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IngestionPipeline:
    """Pipeline to load and chunk PDF documents with metadata."""

    def __init__(self, docs_path: str | Path, loader: PyMuPDFLoader, splitter: RecursiveCharacterTextSplitter):
        """
        Args:
            docs_path: Directory containing PDF files.
            loader: PDF loader class (e.g., PyMuPDFLoader).
            splitter: Text splitter for chunking documents.
        """
        self.docs_path = Path(docs_path)
        self.loader = loader
        self.splitter = splitter

    def load_documents(self) -> List[Document]:
        """Load all PDFs in directory, adding file name and page metadata."""
        docs = []

        for pdf_file in self.docs_path.rglob("*.pdf"):
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