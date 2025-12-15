from pathlib import Path
from retrievers.ingestion import create_ingestion_pipeline
from configs.config import Config

def main():
    # Load configuration
    config = Config()
    docs_path = Path(config.docs_path)
    vectorstore_path = Path(config.vectorstore_path)
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap

    # Ingest and chunk documents
    documents = create_ingestion_pipeline(docs_path, chunk_size, chunk_overlap)


if __name__ == "__main__":
    main()