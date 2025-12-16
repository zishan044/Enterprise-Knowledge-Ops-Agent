from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)

class LoaderFactory:
    """Factory for creating document loaders based on configuration."""
    
    @staticmethod
    def create_loader(loader_type: str, **kwargs):
        """
        Create a document loader based on the specified type.
        
        Args:
            loader_type: Type of loader to create (e.g., 'pdf', 'text', 'docx', 'csv', 'markdown')
            **kwargs: Additional arguments to pass to the loader constructor
            
        Returns:
            An instance of the specified document loader
        """
        loader_map = {
            'pdf': PyMuPDFLoader,
            'text': TextLoader,
            'docx': Docx2txtLoader,
            'csv': CSVLoader,
            'markdown': UnstructuredMarkdownLoader,
        }
        
        if loader_type not in loader_map:
            raise ValueError(f"Unknown loader type: {loader_type}. "
                           f"Available types: {list(loader_map.keys())}")
        
        return loader_map[loader_type]