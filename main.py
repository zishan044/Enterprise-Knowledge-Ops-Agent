from pathlib import Path
from retrievers.ingestion import create_ingestion_pipeline
from configs.config import Config

def main():
    config = Config()
    docs_path = Path(config.docs_path)
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap

    documents = create_ingestion_pipeline(docs_path, chunk_size, chunk_overlap)


if __name__ == "__main__":
    main()