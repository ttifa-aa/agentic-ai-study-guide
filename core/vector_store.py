"""
Vector store management module for document embeddings and similarity search.
Handles creation, persistence, and retrieval operations for FAISS vector stores.
"""

import os  # for file path operations and directory management
import pickle  # for serializing and deserializing vector store metadata
from typing import List, Optional, Dict, Any, Tuple  # type hints for function signatures
from pathlib import Path  # object-oriented file path handling

# langchain imports for vector store operations
from langchain_community.vectorstores import FAISS  # facebook ai similarity search for vector storage
from langchain_huggingface import HuggingFaceEmbeddings  # embedding model for text to vector conversion
from langchain_core.documents import Document  # base document class for langchain
from langchain_core.vectorstores import VectorStoreRetriever  # retriever interface for vector stores

from config.settings import config  # import system configuration


class VectorStoreManager:
    """
    Manages FAISS vector store operations including creation, updates, and retrieval.
    provides a clean interface for vector store lifecycle management.
    """
    
    def __init__(self, embedding_model_name: str = None):
        """
        Initialize the vector store manager with embedding model.
        
        Args:
            embedding_model_name (str, optional): Name of huggingface embedding model
        """
        # use default model from config if not specified
        self.embedding_model_name = embedding_model_name or config.EMBEDDING_MODEL
        
        # initialize embeddings model
        # huggingfaceembeddings provides free, local embeddings without api calls
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name  # using all-minilm-l6-v2 for speed/quality balance
        )
        
        # initialize vector store as none - will be created when documents are added
        self.vectorstore: Optional[FAISS] = None
        
        # track document metadata for persistence
        self.document_metadata: Dict[str, Any] = {}
        # stores information about indexed documents for management
    
    def create_from_documents(
        self,
        documents: List[Document],
        save_path: Optional[str] = None
    ) -> FAISS:
        """
        Create a new vector store from documents.
        
        Args:
            documents (List[Document]): List of document chunks to index
            save_path (str, optional): Path to save the vector store
        
        Returns:
            FAISS: Created vector store instance
        """
        # validate that we have documents to index
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        # create vector store from documents
        # faiss.from_documents converts each document to embedding and builds index
        self.vectorstore = FAISS.from_documents(
            documents,           # document chunks to index
            self.embeddings      # embedding model to use
        )
        
        # update metadata with document information
        self._update_metadata(documents)
        
        # save to disk if path provided
        if save_path:
            self.save_vectorstore(save_path)
        
        return self.vectorstore  # return created vector store
    
    def add_documents(
        self,
        documents: List[Document],
        save_path: Optional[str] = None
    ) -> None:
        """
        Add new documents to existing vector store.
        
        Args:
            documents (List[Document]): New document chunks to add
            save_path (str, optional): Path to save updated vector store
        """
        if self.vectorstore is None:
            # if no vector store exists, create new one
            self.create_from_documents(documents, save_path)
        else:
            # add documents to existing vector store
            # this updates the faiss index with new embeddings
            self.vectorstore.add_documents(documents)
            
            # update metadata with new document information
            self._update_metadata(documents)
            
            # save updated vector store if path provided
            if save_path:
                self.save_vectorstore(save_path)
    
    def _update_metadata(self, documents: List[Document]) -> None:
        """
        Update internal metadata tracking with document information.
        
        Args:
            documents (List[Document]): Documents to track in metadata
        """
        # count documents by source file
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            content_type = doc.metadata.get("content_type", "unknown")
            
            # track unique sources
            if "sources" not in self.document_metadata:
                self.document_metadata["sources"] = {}
            
            if source not in self.document_metadata["sources"]:
                self.document_metadata["sources"][source] = {
                    "content_type": content_type,
                    "chunk_count": 0,
                    "total_chars": 0
                }
            
            # update source statistics
            self.document_metadata["sources"][source]["chunk_count"] += 1
            self.document_metadata["sources"][source]["total_chars"] += len(doc.page_content)
        
        # update total document count
        self.document_metadata["total_documents"] = len(self.document_metadata.get("sources", {}))
        self.document_metadata["total_chunks"] = self.vectorstore.index.ntotal if self.vectorstore else 0
    
    def save_vectorstore(self, path: str) -> None:
        """
        Save vector store to disk for persistence.
        
        Args:
            path (str): Directory path to save vector store
        """
        if self.vectorstore is None:
            raise ValueError("No vector store to save")
        
        # create directory if it doesn't exist
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # save faiss index and document store
        # faiss.save_local saves both the vector index and document mappings
        self.vectorstore.save_local(str(save_dir))
        
        # save additional metadata using pickle
        metadata_path = save_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.document_metadata, f)
    
    def load_vectorstore(self, path: str) -> FAISS:
        """
        Load vector store from disk.
        
        Args:
            path (str): Directory path where vector store is saved
        
        Returns:
            FAISS: Loaded vector store instance
        """
        load_path = Path(path)
        
        # check if vector store exists at path
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        # load faiss vector store
        # embeddings must be the same model used for creation
        self.vectorstore = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True  # required for pickle deserialization
            # note: in production, ensure the source is trusted
        )
        
        # load saved metadata if exists
        metadata_path = load_path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.document_metadata = pickle.load(f)
        
        return self.vectorstore  # return loaded vector store
    
    def get_retriever(
        self,
        k: int = None,
        similarity_threshold: float = None,
        search_type: str = "similarity"
    ) -> VectorStoreRetriever:
        """
        Get a retriever instance for the vector store.
        
        Args:
            k (int, optional): Number of documents to retrieve
            similarity_threshold (float, optional): Minimum similarity score
            search_type (str): Type of search ("similarity" or "mmr")
        
        Returns:
            VectorStoreRetriever: Configured retriever instance
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        # use config defaults if not specified
        k = k or config.RETRIEVAL_K
        similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        
        # create retriever with search parameters
        # use "similarity_score_threshold" so FAISS actually applies the threshold filter;
        # with plain "similarity" the score_threshold key is silently ignored
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,                              # number of documents to retrieve
                "score_threshold": similarity_threshold  # minimum similarity score
            }
        )
        
        return retriever  # return configured retriever
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with optional filtering.
        
        Args:
            query (str): Search query text
            k (int, optional): Number of results to return
            filter_dict (Dict, optional): Metadata filters for search
        
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        k = k or config.RETRIEVAL_K
        
        # perform similarity search with optional filters
        if filter_dict:
            # search with metadata filtering
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict  # filter by metadata fields
            )
        else:
            # standard similarity search without filters
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                k=k
            )
        
        return results  # return search results with scores
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current vector store index.
        
        Returns:
            Dict[str, Any]: Statistics about the vector store
        """
        stats = {
            "initialized": self.vectorstore is not None,
            "embedding_model": self.embedding_model_name,
            **self.document_metadata  # include tracked metadata
        }
        
        if self.vectorstore:
            # add faiss-specific statistics
            stats["total_vectors"] = self.vectorstore.index.ntotal
            stats["vector_dimension"] = self.vectorstore.index.d
        
        return stats  # return statistics dictionary
    
    def clear_vectorstore(self) -> None:
        """
        Clear the current vector store and metadata.
        """
        self.vectorstore = None  # remove vector store reference
        self.document_metadata = {}  # clear metadata
        # note: this doesn't delete saved files, only clears in-memory state
    
    def filter_by_content_type(self, content_type: str) -> Dict[str, Any]:
        """
        Create filter dictionary for specific content type.
        
        Args:
            content_type (str): Content type to filter by
        
        Returns:
            Dict[str, Any]: Filter dictionary for search operations
        """
        # return filter that matches the specified content type
        return {"content_type": content_type}
    
    def filter_by_source(self, source_filename: str) -> Dict[str, Any]:
        """
        Create filter dictionary for specific source file.
        
        Args:
            source_filename (str): Source filename to filter by
        
        Returns:
            Dict[str, Any]: Filter dictionary for search operations
        """
        # return filter that matches the specified source file
        return {"source": source_filename}
    
    def get_unique_sources(self) -> List[str]:
        """
        Get list of unique document sources in the vector store.
        
        Returns:
            List[str]: List of unique source filenames
        """
        if "sources" not in self.document_metadata:
            return []
        
        # return list of all unique source filenames
        return list(self.document_metadata["sources"].keys())


# singleton instance for application-wide use
# using singleton pattern ensures one vector store manager across the app
_vector_store_manager: Optional[VectorStoreManager] = None


def get_vector_store_manager() -> VectorStoreManager:
    """
    Get or create singleton vector store manager instance.
    
    Returns:
        VectorStoreManager: Singleton instance
    """
    global _vector_store_manager
    
    if _vector_store_manager is None:
        # create new instance if not exists
        _vector_store_manager = VectorStoreManager()
    
    return _vector_store_manager  # return singleton instance


def create_vectorstore_from_chunks(
    chunks: List[Document],
    persist_directory: Optional[str] = None
) -> FAISS:
    """
    Convenience function to create vector store from document chunks.
    
    Args:
        chunks (List[Document]): Document chunks to index
        persist_directory (str, optional): Directory to persist vector store
    
    Returns:
        FAISS: Created vector store
    """
    manager = get_vector_store_manager()  # get singleton manager
    return manager.create_from_documents(chunks, persist_directory)


def add_chunks_to_vectorstore(
    chunks: List[Document],
    persist_directory: Optional[str] = None
) -> None:
    """
    Convenience function to add chunks to existing vector store.
    
    Args:
        chunks (List[Document]): Document chunks to add
        persist_directory (str, optional): Directory to persist vector store
    """
    manager = get_vector_store_manager()  # get singleton manager
    manager.add_documents(chunks, persist_directory)


def get_retriever(
    k: int = None,
    similarity_threshold: float = None
) -> VectorStoreRetriever:
    """
    Convenience function to get retriever from vector store.
    
    Args:
        k (int, optional): Number of documents to retrieve
        similarity_threshold (float, optional): Minimum similarity score
    
    Returns:
        VectorStoreRetriever: Configured retriever
    """
    manager = get_vector_store_manager()  # get singleton manager
    return manager.get_retriever(k, similarity_threshold)