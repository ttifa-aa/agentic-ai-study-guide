"""
base track class for academic assistant tracks.
provides common interface and shared functionality for all tracks.
"""

from abc import ABC, abstractmethod  # for creating abstract base classes
from typing import Dict, List, Optional, Any, Tuple  # type hints for function signatures
from dataclasses import dataclass, field  # for structured data classes
from enum import Enum  # for enumerated types

from config.settings import TrackType, ContentType  # import track and content type definitions
from core.vector_store import get_vector_store_manager  # vector store access
from core.rag_chain import RAGChainManager, ChainMode  # rag chain management


@dataclass
class TrackFeatures:
    """Features and capabilities of a track."""
    name: str                                    # display name of the track
    description: str                             # detailed description of track features
    supported_content_types: List[str]           # content types this track works with
    special_prompts: Dict[str, str]              # track-specific prompt templates
    analytics_enabled: bool = False              # whether track supports learning analytics
    progress_tracking: bool = False              # whether track tracks learning progress
    export_formats: List[str] = field(default_factory=list)  # supported export formats


class BaseTrack(ABC):
    """
    Abstract base class for all academic assistant tracks.
    defines the interface that all tracks must implement.
    """
    
    def __init__(self):
        """Initialize the base track with common components."""
        # get vector store manager for document retrieval
        self.vector_store_manager = get_vector_store_manager()
        
        # get rag chain manager for question answering
        self.rag_manager = RAGChainManager()
        
        # track metadata
        self.track_type: Optional[TrackType] = None
        self.features: Optional[TrackFeatures] = None
        
        # session data storage
        self.session_data: Dict[str, Any] = {}
    
    @abstractmethod
    def get_features(self) -> TrackFeatures:
        """
        Get the features and capabilities of this track.
        must be implemented by each concrete track class.
        
        Returns:
            TrackFeatures: Track features and capabilities
        """
        pass
    
    @abstractmethod
    def process_query(
        self,
        query: str,
        query_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user query with track-specific handling.
        must be implemented by each concrete track class.
        
        Args:
            query (str): User's query or question
            query_type (str): Type of query (explain, solve, analyze, etc.)
            **kwargs: Additional track-specific parameters
        
        Returns:
            Dict[str, Any]: Response with metadata
        """
        pass
    
    @abstractmethod
    def get_specialized_prompt(self, prompt_type: str) -> str:
        """
        Get a track-specific specialized prompt template.
        
        Args:
            prompt_type (str): Type of prompt needed
        
        Returns:
            str: Specialized prompt template
        """
        pass
    
    def get_retriever(self, k: int = None, **filters):
        """
        Get a retriever configured for this track.
        
        Args:
            k (int, optional): Number of documents to retrieve
            **filters: Additional metadata filters
        
        Returns:
            Retriever: Configured retriever instance
        """
        # get base retriever from vector store
        return self.vector_store_manager.get_retriever(k=k)
    
    def format_response(
        self,
        answer: str,
        sources: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Format a response with track-specific styling.
        
        Args:
            answer (str): Generated answer text
            sources (List[str], optional): Source citations
            metadata (Dict, optional): Additional metadata
        
        Returns:
            Dict[str, Any]: Formatted response dictionary
        """
        response = {
            "answer": answer,
            "track_type": self.track_type.value if self.track_type else "unknown",
            "sources": sources or [],
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    def get_welcome_message(self) -> str:
        """
        Get a track-specific welcome message.
        
        Returns:
            str: Welcome message for the track
        """
        if self.features:
            return f"Welcome to {self.features.name}! {self.features.description[:100]}..."
        return "Welcome to the Academic Assistant! Upload documents to get started."
    
    def get_capabilities_list(self) -> List[str]:
        """
        Get a list of track capabilities for display.
        
        Returns:
            List[str]: List of capability descriptions
        """
        if self.features:
            capabilities = []
            
            # add content type support information
            if self.features.supported_content_types:
                capabilities.append(f"Supports: {', '.join(self.features.supported_content_types)}")
            
            # add analytics capability
            if self.features.analytics_enabled:
                capabilities.append("Learning analytics enabled")
            
            # add progress tracking capability
            if self.features.progress_tracking:
                capabilities.append("Progress tracking available")
            
            # add export formats
            if self.features.export_formats:
                capabilities.append(f"Export formats: {', '.join(self.features.export_formats)}")
            
            return capabilities
        
        return ["Basic academic assistance"]
    
    def validate_content_type(self, content_type: str) -> bool:
        """
        Check if a content type is supported by this track.
        
        Args:
            content_type (str): Content type to validate
        
        Returns:
            bool: True if content type is supported
        """
        if self.features and self.features.supported_content_types:
            return content_type in self.features.supported_content_types
        return True  # default to accepting all content types
    
    def clear_session(self) -> None:
        """
        Clear track-specific session data.
        """
        self.session_data.clear()