from pathlib import Path
from retrievers.ingestion import IngestionPipeline
from configs.config import Config
from factories.loader_factory import LoaderFactory
from factories.splitter_factory import SplitterFactory
from factories.vectorstore_factory import VectorStoreFactory
from agents.router import RouterAgent

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

    
    retriever = VectorStoreFactory.create_retriever(
        documents=chunked_docs,
        embedding_model=config.embedding_model,
        embedding_params=config.embedding_params,
        dense_k=config.dense_k,
        sparse_k=config.sparse_k,
        retriever_weights=config.retriever_weights,
    )
    
    
    router = RouterAgent(model=config.router_model)
    
    test_queries = [
        "enterprise knowledge operations",
        "vector store hybrid retrieval", 
        "document chunking strategy"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        docs = retriever.invoke(query)
        print(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs[:2], 1):
            print(f"  {i}. {doc.page_content[:100]}...")
    
    routing_test_queries = [
        "What is the definition of enterprise knowledge operations? (factual)",
        "Compare different vector store retrieval strategies (analytical)", 
        "What is our company's policy on data retention? (policy)",
        "How does hybrid retrieval improve accuracy? (analytical)",
        "When was the knowledge ops agent first deployed? (factual)"
    ]
    
    for query in routing_test_queries:
        print(f"\n📝 Query: '{query}'")
        
        route_decision = router.route(query)
        print(f"   Route: {route_decision.route.upper()}")
        
        docs = retriever.invoke(query)
        print(f"   Retrieved {len(docs)} documents:")
        
        if route_decision.route == "factual":
            print(f"   🎯 FACTUAL: Using top {config.dense_k} semantic matches")
        elif route_decision.route == "analytical":
            print(f"   📊 ANALYTICAL: Combining dense+sparse for comparison")
        else:
            print(f"   📜 POLICY: Prioritizing exact keyword matches")
            
        if docs:
            top_doc = docs[0]
            snippet = top_doc.page_content[:120] + "..." if len(top_doc.page_content) > 120 else top_doc.page_content
            print(f"   📄 Top result: {snippet}")

    
    retriever_path = Path("artifacts") / "hybrid_retriever"
    retriever_path.mkdir(parents=True, exist_ok=True)
    
    try:
        faiss_vectorstore = retriever.retrievers[0].vectorstore
        faiss_vectorstore.save_local(str(retriever_path / "faiss_index"))
    except Exception as e:
        print(f"⚠️  Could not save FAISS (MergeRetriever issue): {e}")

    print("🎉 Pipeline ready! Router → Retriever → RAG")

if __name__ == "__main__":
    main()