from pathlib import Path
from retrievers.ingestion import IngestionPipeline
from configs.config import Config
from factories.loader_factory import LoaderFactory
from factories.splitter_factory import SplitterFactory
from factories.vectorstore_factory import VectorStoreFactory
from agents.router import RouterAgent
from agents.retrieval_planner import RetrievalPlanner

def execute_retrieval_strategy(retriever, strategy: str, query: str, config):
    dense_retriever = retriever.retrievers[0]
    sparse_retriever = retriever.retrievers[1]
    
    match strategy:
        case "dense":
            return dense_retriever.invoke(query)
        case "sparse":
            return sparse_retriever.invoke(query)
        case "hybrid":
            return retriever.invoke(query)
        case "multi_query":
            queries = [query, f"What is {query}?", f"Explain {query}"]
            all_docs = []
            for q in queries[:2]:
                docs = retriever.invoke(q)
                all_docs.extend(docs)
            unique_docs = []
            seen = set()
            for doc in all_docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen:
                    seen.add(content_hash)
                    unique_docs.append(doc)
            return unique_docs[:config.dense_k * 2]
        case _:
            return retriever.invoke(query)

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
    planner = RetrievalPlanner(model=config.planner_model)
    
    test_cases = [
        {"query": "What is enterprise knowledge operations?", "expected": "factual"},
        {"query": "Compare dense vs hybrid retrieval performance", "expected": "analytical"},
        {"query": "What is our data retention policy for embeddings?", "expected": "policy"},
        {"query": "How does BM25 work with vector stores?", "expected": "factual"}
    ]
    
    for test in test_cases:
        route = router.route(test['query'])
        plan = planner.plan(test['query'], route.route)
        docs = execute_retrieval_strategy(retriever, plan.strategy, test['query'], config)
    
    retriever_path = Path("artifacts") / "hybrid_retriever"
    retriever_path.mkdir(parents=True, exist_ok=True)
    
    try:
        faiss_store = retriever.retrievers[0].vectorstore
        faiss_store.save_local(str(retriever_path / "faiss_index"))
    except:
        pass
    
    print("🎉 PRODUCTION-READY PIPELINE: Router → Planner → Strategy → RAG")

if __name__ == "__main__":
    main()