from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from typing import List
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

class VectorStoreFactory:
    """Factory for creating hybrid retrievers (FAISS + BM25) based on configuration."""
    
    @staticmethod
    def create_retriever(
        documents: List[Document], 
        embedding_model: str, 
        **kwargs
    ):
        """
        Create a hybrid retriever combining FAISS (dense) and BM25 (sparse).
        
        Args:
            documents: List of documents to index
            embedding_model: Type of embedding model (e.g., 'ollama')
            **kwargs: Additional arguments for embeddings and retrievers
            
        Returns:
            EnsembleRetriever combining FAISS and BM25
        """
        embedding_map = {'ollama': OllamaEmbeddings}
        
        if embedding_model not in embedding_map:
            raise ValueError(
                f"Unknown embedding model: {embedding_model}. "
                f"Available: {list(embedding_map.keys())}"
            )
        
        embedding_params = kwargs.get('embedding_params', {})
        embeddings = embedding_map[embedding_model](**embedding_params)
        
        vectorstore = FAISS.from_documents(documents, embeddings)
        faiss_retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": kwargs.get("dense_k", 4),
                **kwargs.get("faiss_search_kwargs", {})
            }
        )
        
        bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=kwargs.get("sparse_k", 4)
        )
        bm25_retriever.k = kwargs.get("sparse_k", 4)
        
        weights = kwargs.get("retriever_weights", [0.5, 0.5])
        hybrid_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=weights
        )
        
        return hybrid_retriever
    
    @staticmethod
    def create_vector_store(embedding_model: str, **kwargs):
        """Legacy method - returns FAISS only (use create_retriever for hybrid)."""
        embedding_map = {'ollama': OllamaEmbeddings}
        
        if embedding_model not in embedding_map:
            raise ValueError(
                f"Unknown embedding model: {embedding_model}. "
                f"Available: {list(embedding_map.keys())}"
            )
        
        embedding_params = kwargs.get('embedding_params', {})
        embeddings = embedding_map[embedding_model](**embedding_params)
        
        return FAISS(embeddings, **kwargs.get('vectorstore_params', {}))