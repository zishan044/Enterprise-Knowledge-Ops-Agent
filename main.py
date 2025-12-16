from pathlib import Path
from retrievers.ingestion import IngestionPipeline
from configs.config import Config
from factories.loader_factory import LoaderFactory
from factories.splitter_factory import SplitterFactory
from factories.vectorstore_factory import VectorStoreFactory

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

    
    print("\n" + "="*50)
    print("CREATING HYBRID RETRIEVER (FAISS + BM25)")
    print("="*50)
    
    retriever = VectorStoreFactory.create_retriever(
        documents=chunked_docs,
        embedding_model=config.embedding_model,
        embedding_params=config.embedding_params,
        dense_k=config.dense_k,
        sparse_k=config.sparse_k,
        retriever_weights=config.retriever_weights,
    )
    
    print(f"Hybrid retriever created!")
    print(f"Dense (FAISS): k={config.dense_k}")
    print(f"Sparse (BM25): k={config.sparse_k}")
    print(f"Weights: {config.retriever_weights}")
    
    print("\n" + "="*50)
    print("TESTING RETRIEVER")
    print("="*50)
    
    test_queries = [
        "enterprise knowledge operations",
        "vector store hybrid retrieval",
        "document chunking strategy"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        docs = retriever.invoke(query)
        print(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs[:3], 1):
            print(f"  {i}. {doc.page_content[:100]}...")
            if doc.metadata:
                print(f"     Source: {doc.metadata.get('source', 'N/A')}")
    
    print("\n" + "="*50)
    print("SAVING RETRIEVER")
    print("="*50)
    
    retriever_path = Path("artifacts") / "hybrid_retriever"
    retriever_path.mkdir(parents=True, exist_ok=True)
    
    faiss_vectorstore = retriever.retrievers[0].vectorstore
    faiss_vectorstore.save_local(str(retriever_path / "faiss_index"))
    
    print(f"Retriever saved to {retriever_path}")
    print("Ready for RAG pipeline!")

if __name__ == "__main__":
    main()