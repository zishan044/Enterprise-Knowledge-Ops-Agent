from pathlib import Path
from retrievers.ingestion import IngestionPipeline
from configs.config import Config
from factories.loader_factory import LoaderFactory
from factories.splitter_factory import SplitterFactory

def main():
    config = Config()
    docs_path = Path(config.docs_path)
    
    loader = LoaderFactory.create_loader(
        loader_type=config.loader_type,
        **config.loader_kwargs
    )
    
    splitter = SplitterFactory.create_splitter(
        splitter_type=config.splitter_type,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        **config.splitter_kwargs
    )
    
    ingestion_pipeline = IngestionPipeline(
        docs_path=docs_path, 
        loader=loader, 
        splitter=splitter
    )
    
    ingested_docs = ingestion_pipeline.load_documents()
    chunked_docs = ingestion_pipeline.chunk_documents()

    print(f"Loaded {len(ingested_docs)} documents.")
    print(f"Chunked into {len(chunked_docs)} pieces.")
    print(f"Loader type: {config.loader_type}")
    print(f"Splitter type: {config.splitter_type}")

if __name__ == "__main__":
    main()