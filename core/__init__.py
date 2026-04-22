"""
Core module for the Academic Assistant.
Contains document processing, vector store, and RAG chain functionality.
"""

# import and re-export document processor functions
from core.document_processor import (
    process_uploaded_file,           # processes a single uploaded file into chunks
    process_multiple_files,          # processes multiple files at once
    is_supported_file,               # checks if file extension is supported
    validate_document_content,       # validates that chunks contain meaningful content
    get_document_stats,              # calculates statistics about processed documents
    extract_document_metadata,       # extracts metadata without full processing
    clean_document_text,             # cleans and normalizes extracted text
    get_loader_for_file,             # gets appropriate loader for file type
    ProcessedDocument                # dataclass for processed document info
)

# import and re-export vector store functions
from core.vector_store import (
    VectorStoreManager,              # class for managing faiss vector store
    get_vector_store_manager,        # singleton accessor for vector store manager
    create_vectorstore_from_chunks,  # convenience function to create vector store
    add_chunks_to_vectorstore,       # convenience function to add chunks
    get_retriever                    # convenience function to get retriever
)

# import and re-export rag chain functions
from core.rag_chain import (
    RAGChainManager,                 # class for managing rag chains
    ChainMode,                       # enum for chain operation modes
    get_rag_chain_manager,           # singleton accessor for rag chain manager
    ask_question,                    # convenience function to ask questions
    get_api_usage_stats,             # function to get api key usage statistics
)

# define what gets exported when someone does "from core import *"
__all__ = [
    # document processor
    "process_uploaded_file",
    "process_multiple_files", 
    "is_supported_file",
    "validate_document_content",
    "get_document_stats",
    "extract_document_metadata",
    "clean_document_text",
    "get_loader_for_file",
    "ProcessedDocument",
    
    # vector store
    "VectorStoreManager",
    "get_vector_store_manager",
    "create_vectorstore_from_chunks",
    "add_chunks_to_vectorstore", 
    "get_retriever",
    
    # rag chain
    "RAGChainManager",
    "ChainMode",
    "get_rag_chain_manager",
    "ask_question",
    "get_api_usage_stats",
]