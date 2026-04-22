"""
RAG (Retrieval-Augmented Generation) chain module.
Builds and manages the pipeline for answering questions using retrieved context.
Includes multi-API-key support with automatic rotation on rate limits.
"""

import time  # for delays and rate limit handling
from typing import Callable, Optional, Dict, Any, List, Tuple  # type hints for function signatures
from enum import Enum  # for chain type enumeration

# langchain core imports for building rag pipelines
from langchain_core.prompts import ChatPromptTemplate  # for creating prompt templates
from langchain_core.runnables import RunnablePassthrough, RunnableParallel  # for chain composition
from langchain_core.output_parsers import StrOutputParser  # parses llm output to string
from langchain_core.documents import Document  # base document class

# langchain groq for llm integration
from langchain_groq import ChatGroq  # groq's fast llm inference

# local imports
from config.settings import (  # system configuration
    config, 
    api_key_manager, 
    get_current_api_key, 
    handle_api_failure
)
from core.vector_store import get_vector_store_manager  # vector store access
from prompts.base_prompts import (  # prompt templates
    STUDY_MODE_PROMPT,
    EXAM_MODE_PROMPT,
    QUICK_REVISION_PROMPT
)


class ChainMode(str, Enum):
    """enumeration of rag chain operation modes."""
    STUDY = "study"          # comprehensive study mode with detailed explanations
    EXAM = "exam"            # exam mode with step-by-step solutions
    QUICK = "quick"          # quick revision mode with concise answers


class RateLimitError(Exception):
    """custom exception for rate limit errors."""
    pass


class TokenLimitError(Exception):
    """custom exception for token limit errors."""
    pass


class RAGChainManager:
    """
    manages RAG chain creation and execution for different modes.
    provides a unified interface for different types of question answering.
    includes multi-api-key support with automatic rotation.
    """
    
    def __init__(self):
        """
        initialize the RAG chain manager.
        uses api_key_manager for automatic key rotation.
        """
        # initialize llm with current api key
        self.current_api_key = get_current_api_key()
        self.llm = self._create_llm(self.current_api_key)
        
        # cache for storing created chains by mode
        # avoids recreating chains for each query
        self._chain_cache: Dict[ChainMode, Any] = {}
        
        # reference to vector store manager for retrieval
        self.vector_store_manager = get_vector_store_manager()
        
        # track key rotation statistics
        self.key_rotations: int = 0
        self.last_key_rotation_time: Optional[float] = None
        
        # track rate limit errors for backoff
        self.rate_limit_backoff: float = 1.0  # initial backoff in seconds
        self.max_backoff: float = 60.0  # maximum backoff in seconds
    
    def _create_llm(self, api_key: str) -> ChatGroq:
        """
        create a new LLM instance with the given API key.
        
        args:
            api_key (str): Groq API key to use
        
        returns:
            ChatGroq: Configured LLM instance
        """
        return ChatGroq(
            model=config.DEFAULT_MODEL,           # model name from config
            api_key=api_key,                      # api key for authentication
            temperature=config.DEFAULT_TEMPERATURE,  # temperature for response randomness
            max_tokens=config.MAX_TOKENS          # maximum response length
        )
    
    def _rotate_api_key(self) -> None:
        """
        rotate to the next working API key and recreate the LLM.
        """
        old_key_masked = api_key_manager._mask_key(self.current_api_key)
        
        # mark current key as failed and get next key
        self.current_api_key = handle_api_failure(self.current_api_key)
        
        # recreate llm with new key
        self.llm = self._create_llm(self.current_api_key)
        
        # update rotation statistics
        self.key_rotations += 1
        self.last_key_rotation_time = time.time()
        
        # clear chain cache since llm has changed
        self._chain_cache.clear()
        
        new_key_masked = api_key_manager._mask_key(self.current_api_key)
        print(f"[RAGChainManager] Rotated API key: {old_key_masked} -> {new_key_masked}")
    
    def _handle_rate_limit_error(self, error: Exception) -> None:
        """
        handle rate limit errors with exponential backoff.
        
        args:
            error (Exception): The rate limit error
        """
        # increase backoff exponentially (capped at max_backoff)
        self.rate_limit_backoff = min(self.rate_limit_backoff * 2, self.max_backoff)
        
        print(f"[RAGChainManager] Rate limit hit. Waiting {self.rate_limit_backoff:.1f}s...")
        time.sleep(self.rate_limit_backoff)
        
        # rotate to next key
        self._rotate_api_key()
    
    def _reset_backoff(self) -> None:
        """
        reset rate limit backoff after successful request.
        """
        self.rate_limit_backoff = 1.0
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        check if an error is due to rate limiting.
        
        args:
            error (Exception): The error to check
        
        returns:
            bool: True if it's a rate limit error
        """
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit",
            "rate_limit",
            "too many requests",
            "429",
            "quota exceeded",
            "limit exceeded"
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def _is_token_limit_error(self, error: Exception) -> bool:
        """
        check if an error is due to token limit exceeded.
        
        args:
            error (Exception): The error to check
        
        returns:
            bool: True if it's a token limit error
        """
        error_str = str(error).lower()
        token_limit_indicators = [
            "token limit",
            "context length",
            "maximum context",
            "too many tokens"
        ]
        return any(indicator in error_str for indicator in token_limit_indicators)
    
    def _format_docs_for_context(self, docs: List[Document]) -> str:
        """
        format retrieved documents into context string for prompt.
        this function is used in the rag chain to prepare context.
        
        args:
            docs (List[Document]): Retrieved document chunks
        
        returns:
            str: Formatted context string with source tags
        """
        formatted_docs = []  # list to store formatted document strings
        
        for doc in docs:
            # get metadata for source attribution
            content_type = doc.metadata.get("content_type", "Unknown")
            source = doc.metadata.get("source", "Unknown")
            
            # format with content type tag for attribution
            formatted_doc = f"[{content_type}] {doc.page_content}"
            formatted_docs.append(formatted_doc)
        
        # join all formatted documents with double newline separator
        return "\n\n".join(formatted_docs)
    
    def _get_prompt_for_mode(self, mode: ChainMode) -> ChatPromptTemplate:
        """
        get appropriate prompt template for the selected mode.
        
        args:
            mode (ChainMode): Operation mode (study, exam, or quick)
        
        returns:
            ChatPromptTemplate: Prompt template for the mode
        """
        if mode == ChainMode.STUDY:
            return ChatPromptTemplate.from_template(STUDY_MODE_PROMPT)
        elif mode == ChainMode.EXAM:
            return ChatPromptTemplate.from_template(EXAM_MODE_PROMPT)
        elif mode == ChainMode.QUICK:
            return ChatPromptTemplate.from_template(QUICK_REVISION_PROMPT)
        else:
            return ChatPromptTemplate.from_template(STUDY_MODE_PROMPT)
    
    def create_chain(
        self,
        mode: ChainMode = ChainMode.STUDY,
        k: int = None,
        force_recreate: bool = False
    ):
        """
        create a RAG chain for the specified mode.
        
        args:
            mode (ChainMode): Operation mode for the chain
            k (int, optional): Number of documents to retrieve
            force_recreate (bool): Force recreation even if cached
        
        returns:
            Runnable: Configured RAG chain ready for invocation
        """
        # return cached chain if available and not forced to recreate
        if mode in self._chain_cache and not force_recreate:
            return self._chain_cache[mode]
        
        # get retriever from vector store
        retriever = self.vector_store_manager.get_retriever(k=k)
        
        # get appropriate prompt for the mode
        prompt = self._get_prompt_for_mode(mode)
        
        # build the rag chain using langchain's runnable composition
        rag_chain = (
            RunnableParallel(
                context=retriever | self._format_docs_for_context,
                question=RunnablePassthrough()
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # cache the created chain for future use
        self._chain_cache[mode] = rag_chain
        
        return rag_chain  # return the configured chain
    
    def invoke_with_retry(
        self,
        question: str,
        mode: ChainMode = ChainMode.STUDY,
        k: int = None,
        max_retries: int = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        invoke the RAG chain with automatic retry and key rotation on failure.
        
        args:
            question (str): User's question
            mode (ChainMode): Operation mode
            k (int, optional): Number of documents to retrieve
            max_retries (int, optional): Maximum retry attempts (default from config)
        
        returns:
            Tuple[str, Dict[str, Any]]: (answer, metadata) including key usage info
        """
        max_retries = max_retries or (len(api_key_manager.keys) * 2)
        # allow up to 2 attempts per key
        
        attempts = 0
        keys_tried = set()  # track which keys we've tried
        metadata = {
            "key_rotations": 0,
            "keys_used": [],
            "attempts": 0,
            "backoff_seconds": 0
        }
        
        while attempts < max_retries:
            attempts += 1
            current_key_masked = api_key_manager._mask_key(self.current_api_key)
            keys_tried.add(current_key_masked)
            
            try:
                # create or get cached chain
                chain = self.create_chain(mode=mode, k=k)
                
                # invoke chain with question
                answer = chain.invoke(question)
                
                # mark key as successful (resets failure count)
                api_key_manager.mark_key_success(self.current_api_key)
                
                # reset backoff on success
                self._reset_backoff()
                
                # update metadata
                metadata.update({
                    "attempts": attempts,
                    "keys_used": list(keys_tried),
                    "key_rotations": self.key_rotations,
                    "final_key": current_key_masked
                })
                
                return answer, metadata
                
            except Exception as e:
                print(f"[RAGChainManager] Attempt {attempts} failed: {type(e).__name__}: {str(e)[:100]}")
                
                if self._is_rate_limit_error(e):
                    # rate limit - rotate key and wait
                    self._handle_rate_limit_error(e)
                    metadata["key_rotations"] = self.key_rotations
                    metadata["backoff_seconds"] = self.rate_limit_backoff
                    
                elif self._is_token_limit_error(e):
                    # token limit - try with reduced context
                    print("[RAGChainManager] Token limit exceeded. Reducing context size...")
                    k = max(2, (k or config.RETRIEVAL_K) - 1)  # reduce retrieved chunks
                    # don't rotate key for token limit (key is fine, just context too large)
                    
                else:
                    # other error - might be key-related, try rotating
                    print(f"[RAGChainManager] Unknown error. Rotating key...")
                    self._rotate_api_key()
                    metadata["key_rotations"] = self.key_rotations
                
                # if we've tried all keys and still failing, wait longer
                if len(keys_tried) >= api_key_manager.get_working_key_count():
                    wait_time = min(self.rate_limit_backoff * 3, 120)
                    print(f"[RAGChainManager] All keys attempted. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    # reset key statuses and try again
                    api_key_manager.reset_all_keys()
                    keys_tried.clear()
        
        # if we exhaust all retries, raise error
        raise RuntimeError(
            f"Failed to get response after {attempts} attempts with {len(keys_tried)} keys. "
            "All API keys may be rate-limited. Please try again later."
        )
    
    def invoke(
        self,
        question: str,
        mode: ChainMode = ChainMode.STUDY,
        k: int = None
    ) -> str:
        """
        invoke the RAG chain with a question (simplified interface).
        
        args:
            question (str): User's question
            mode (ChainMode): Operation mode
            k (int, optional): Number of documents to retrieve
        
        returns:
            str: Generated answer
        """
        answer, _ = self.invoke_with_retry(question, mode, k)
        return answer
    
    def stream_response(
        self,
        question: str,
        mode: ChainMode = ChainMode.STUDY,
        k: int = None
    ):
        """
        stream the response token by token with automatic retry.
        useful for real-time display of generated answers.
        
        Args:
            question (str): User's question
            mode (ChainMode): Operation mode
            k (int, optional): Number of documents to retrieve
        
        Yields:
            str: Response tokens as they are generated
        """
        try:
            chain = self.create_chain(mode=mode, k=k)
            for chunk in chain.stream(question):
                yield chunk
            # mark success after successful stream
            api_key_manager.mark_key_success(self.current_api_key)
            self._reset_backoff()
        except Exception as e:
            if self._is_rate_limit_error(e):
                self._handle_rate_limit_error(e)
                # retry with new key
                chain = self.create_chain(mode=mode, k=k, force_recreate=True)
                for chunk in chain.stream(question):
                    yield chunk
            else:
                raise e
    
    def get_api_key_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API key usage.
        
        Returns:
            Dict: API key statistics
        """
        stats = api_key_manager.get_key_stats()
        stats.update({
            "rag_chain_rotations": self.key_rotations,
            "last_rotation_time": self.last_key_rotation_time,
            "current_backoff": self.rate_limit_backoff
        })
        return stats
    
    def clear_cache(self) -> None:
        """
        clear the chain cache.
        useful when switching between different document sets.
        """
        self._chain_cache.clear()
        # also reset backoff when clearing cache
        self._reset_backoff()


# =============================================================================
# Format Citations Function
# =============================================================================

def format_citations(docs: List[Document]) -> str:
    """
    Format document citations for display.
    
    Args:
        docs (List[Document]): Retrieved documents used in answer
    
    Returns:
        str: Formatted citation string
    """
    if not docs:
        return "No sources cited."
    
    citations = []  # list for citation strings
    seen_sources = set()  # track unique sources to avoid duplicates
    
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        content_type = doc.metadata.get("content_type", "Unknown")
        
        # create unique identifier for source
        source_id = f"{source} ({content_type})"
        
        if source_id not in seen_sources:
            citations.append(f"- {source_id}")
            seen_sources.add(source_id)
    
    if citations:
        return "**Sources:**\n" + "\n".join(citations)
    else:
        return "**Sources:** No specific sources cited."


# =============================================================================
# Singleton Instance
# =============================================================================

# singleton instance for application-wide use
_rag_chain_manager: Optional[RAGChainManager] = None


def get_rag_chain_manager() -> RAGChainManager:
    """
    get or create singleton RAG chain manager instance.
    
    returns:
        RAGChainManager: Singleton instance
    """
    global _rag_chain_manager
    
    if _rag_chain_manager is None:
        _rag_chain_manager = RAGChainManager()
    
    return _rag_chain_manager


def ask_question(
    question: str,
    mode: str = "study",
    k: int = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to ask a question using the rag chain.
    returns both answer and metadata including key usage.
    
    Args:
        question (str): User's question
        mode (str): Operation mode ("study", "exam", or "quick")
        k (int, optional): Number of documents to retrieve
    
    Returns:
        Tuple[str, Dict]: (answer, metadata)
    """
    manager = get_rag_chain_manager()
    
    # convert string mode to enum
    mode_map = {
        "study": ChainMode.STUDY,
        "exam": ChainMode.EXAM,
        "quick": ChainMode.QUICK
    }
    
    chain_mode = mode_map.get(mode.lower(), ChainMode.STUDY)
    
    # invoke and return answer with metadata
    return manager.invoke_with_retry(question, mode=chain_mode, k=k)


def get_api_usage_stats() -> Dict[str, Any]:
    """
    get API key usage statistics for display.
    
    returns:
        Dict: API usage statistics
    """
    manager = get_rag_chain_manager()
    return manager.get_api_key_stats()