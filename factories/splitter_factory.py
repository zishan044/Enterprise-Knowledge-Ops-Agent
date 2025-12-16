from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter
)

class SplitterFactory:
    """Factory for creating text splitters based on configuration."""
    
    @staticmethod
    def create_splitter(splitter_type: str, **kwargs):
        """
        Create a text splitter based on the specified type.
        
        Args:
            splitter_type: Type of splitter to create (e.g., 'recursive', 'character', 'token', 'spacy')
            **kwargs: Additional arguments to pass to the splitter constructor
            
        Returns:
            An instance of the specified text splitter
        """
        # Default configurations for each splitter type
        default_configs = {
            'recursive': {
                'chunk_size': 200,
                'chunk_overlap': 10,
                'separators': ["\n\n", "\n", " ", ""]
            },
            'character': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'separator': "\n\n"
            },
            'token': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
            },
            'spacy': {
                'chunk_size': 1000,
            }
        }
        
        splitter_map = {
            'recursive': RecursiveCharacterTextSplitter,
            'character': CharacterTextSplitter,
            'token': TokenTextSplitter,
            'spacy': SpacyTextSplitter,
        }
        
        if splitter_type not in splitter_map:
            raise ValueError(f"Unknown splitter type: {splitter_type}. "
                           f"Available types: {list(splitter_map.keys())}")
        
        config = default_configs.get(splitter_type, {}).copy()
        config.update(kwargs)
        
        return splitter_map[splitter_type](**config)