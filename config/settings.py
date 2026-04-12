"""
Configuration settings and constants for the Academic Assistant.
Contains track definitions, subject mappings, and system configurations.
Includes multi-API-key support for cycling through keys when limits are reached.
"""

import os  # to access environment variables like api keys and file paths
from enum import Enum  # to create enumerated constants for track types and content types
from dataclasses import dataclass, field  # to create data classes for configuration objects
from typing import Any, Dict, List, Set, Optional  # for type hints to make code more readable and maintainable

# Track Definitions - used to identify which version of the assistant the user wants to use
class TrackType(str, Enum):
    """Enumeration of available academic assistant tracks."""
    TRACK_A1_CS = "track_a1_cs"           # computer science subject guide track with cs-specific features
    TRACK_A2_EXAM = "track_a2_exam"       # comprehensive exam preparation track with study planning

# content types - used as metadata tags for documents uploaded by users
# helps the assistant understand what kind of content it's working with
class ContentType(str, Enum):
    """Enumeration of academic content types for metadata tagging."""
    LECTURE_NOTES = "Lecture Notes"        # notes taken during lectures or provided by professors
    TEXTBOOK = "Textbook"                  # standard textbook content with theory and examples
    LAB_MANUAL = "Lab Manual"              # practical laboratory instructions and experiments
    PAST_PAPER = "Past Paper"              # previous year examination question papers
    ASSIGNMENT = "Assignment"              # homework assignments and problem sets
    REFERENCE = "Reference Material"       # additional reference books and supplementary materials

# CS subject ares for track A1 - used to categorize content and detect topics
# maps subject names to keywords that help identify which subject a document belongs to
CS_SUBJECTS: Dict[str, List[str]] = {
    "Database Systems": ["sql", "normalization", "dbms", "query", "transaction", "indexing", "acid"],
    "Data Structures": ["array", "linked list", "stack", "queue", "tree", "graph", "hash", "heap"],
    "Algorithms": ["sorting", "searching", "dynamic programming", "greedy", "divide and conquer", "complexity"],
    "Operating Systems": ["process", "thread", "scheduling", "memory", "deadlock", "file system", "kernel"],
    "Computer Networks": ["tcp/ip", "osi", "routing", "protocol", "dns", "http", "socket"],
    "Programming": ["python", "java", "function", "class", "inheritance", "polymorphism", "exception"],
    "Software Engineering": ["sdlc", "agile", "testing", "design pattern", "uml", "requirement"],
    "Machine Learning": ["supervised", "unsupervised", "neural network", "classification", "regression", "clustering"]
}

# programming language detection patterns - used to identify code snippets and their languages
CODE_PATTERNS: Dict[str, List[str]] = {
    "python": ["def ", "import ", "class ", "print(", "if __name__", "self", "lambda"],
    "java": ["public class", "public static void", "System.out", "new ", "extends", "implements"],
    "cpp": ["#include", "std::", "cout", "cin", "->", "::", "template<"],
    "javascript": ["function", "const ", "let ", "=>", "console.log", "document."],
    "sql": ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE TABLE", "JOIN", "WHERE"]
}

# algorithm complexity patterns - used to detect time complexity from algorithm descriptions
COMPLEXITY_PATTERNS: Dict[str, str] = {
    "O(1)": ["constant", "array access", "hash lookup"],
    "O(log n)": ["binary search", "balanced tree", "divide"],
    "O(n)": ["linear", "traversal", "single loop"],
    "O(n log n)": ["merge sort", "quick sort", "heap sort", "efficient sorting"],
    "O(n²)": ["nested loop", "bubble sort", "insertion sort", "quadratic"],
    "O(2ⁿ)": ["exponential", "recursive fibonacci", "subset", "brute force"]
}

# exam patterns for track A2 - used to generate practice questions and identify question types
EXAM_PATTERNS: Dict[str, Dict] = {
    "internal": {
        "marks_distribution": {"theory": 70, "practical": 30},
        "question_types": ["short_answer", "long_answer", "problem_solving"],
        "duration_minutes": 180
    },
    "external": {
        "marks_distribution": {"theory": 80, "practical": 20},
        "question_types": ["mcq", "short_answer", "long_answer", "case_study"],
        "duration_minutes": 180
    },
    "practical": {
        "marks_distribution": {"viva": 40, "experiment": 40, "record": 20},
        "question_types": ["procedure", "implementation", "analysis"],
        "duration_minutes": 180
    }
}

# System Configuration - central configuration class for all system-wide settings
@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    
    # Vector Store Settings - controls how documents are chunked and retrieved
    CHUNK_SIZE: int = 500
    # size of text chunks in characters - balances context preservation with retrieval precision
    
    CHUNK_OVERLAP: int = 50
    # overlap between consecutive chunks in characters - prevents information loss at boundaries
    
    RETRIEVAL_K: int = 5
    # number of relevant chunks to retrieve per query - more chunks means more context but slower
    
    SIMILARITY_THRESHOLD: float = 0.7
    # minimum similarity score for retrieved chunks - filters out irrelevant content
    
    # LLM Settings - controls language model behavior and performance
    DEFAULT_MODEL: str = "llama-3.1-8b-instant"
    # using groq's llama model - good balance of speed and quality
    
    DEFAULT_TEMPERATURE: float = 0.3
    # temperature controls response randomness - lower means more focused and deterministic
    
    MAX_TOKENS: int = 2048
    # maximum tokens for llm response - enough for detailed explanations without hitting limits
    
    # API Key Management - supports multiple keys with automatic cycling
    MAX_RETRIES_PER_KEY: int = 3
    # number of retries per api key before switching to next key
    
    RATE_LIMIT_WAIT_SECONDS: int = 5
    # seconds to wait when rate limited before retrying
    
    # document Processing - controls which file types are accepted
    SUPPORTED_EXTENSIONS: Set[str] = field(default_factory=lambda: {".pdf", ".docx", ".txt", ".pptx"})
    # set of supported file extensions
    
    MAX_FILE_SIZE_MB: int = 50
    # maximum file size in megabytes - prevents memory issues with very large documents
    
    # Progress Tracking - database and caching settings
    DATABASE_PATH: str = "data/progress.db"
    # path to sqlite database file for storing learning progress data
    
    CACHE_TTL_SECONDS: int = 3600
    # time-to-live for cached embeddings in seconds - 1 hour


# API key manager - handles multiple API keys with automatic cycling
class APIKeyManager:
    """
    Manages multiple API keys with automatic cycling when limits are reached.
    supports loading keys from environment variables in format:
    GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3, etc.
    """
    
    def __init__(self, key_prefix: str = "GROQ_API_KEY"):
        """
        Initialize the API key manager.
        
        Args:
            key_prefix (str): Prefix for environment variable keys
        """
        self.key_prefix = key_prefix  # prefix for environment variables (e.g., "GROQ_API_KEY")
        self.keys: List[str] = []  # list to store loaded api keys
        self.current_index: int = 0  # current index in the keys list
        self.key_status: Dict[str, Dict] = {}  # track status of each key
        # status format: {"key": "sk-...", "working": True, "failures": 0, "last_used": timestamp}
        
        # load all available keys
        self._load_keys_from_environment()
        
        # validate that at least one key is available
        if not self.keys:
            raise ValueError(f"No API keys found with prefix '{key_prefix}'. "
                           f"Please set at least {key_prefix}_1 in your .env file.")
    
    def _load_keys_from_environment(self) -> None:
        """
        Load all API keys from environment variables.
        looks for GROQ_API_KEY, GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
        """
        # first, try the base key without number (for backward compatibility)
        base_key = os.getenv(self.key_prefix)
        if base_key:
            self.keys.append(base_key)
            self.key_status[base_key] = {
                "working": True,
                "failures": 0,
                "last_used": None,
                "index": 0
            }
        
        # then try numbered keys: GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
        index = 1
        while True:
            key_name = f"{self.key_prefix}_{index}"
            key_value = os.getenv(key_name)
            
            if not key_value:
                break  # stop when no more keys are found
            
            self.keys.append(key_value)
            self.key_status[key_value] = {
                "working": True,
                "failures": 0,
                "last_used": None,
                "index": index
            }
            index += 1
        
        # also check for comma-separated keys in a single variable (alternative format)
        # format: GROQ_API_KEYS="key1,key2,key3"
        keys_string = os.getenv("GROQ_API_KEYS")
        if keys_string:
            for key in keys_string.split(","):
                key = key.strip()
                if key and key not in self.keys:
                    self.keys.append(key)
                    self.key_status[key] = {
                        "working": True,
                        "failures": 0,
                        "last_used": None,
                        "index": len(self.keys)
                    }
        
        print(f"[APIKeyManager] Loaded {len(self.keys)} API key(s)")
    
    def get_current_key(self) -> str:
        """
        Get the current active API key.
        
        Returns:
            str: Current API key
        """
        if not self.keys:
            raise ValueError("No API keys available")
        
        key = self.keys[self.current_index]
        
        # update last used timestamp
        import time
        self.key_status[key]["last_used"] = time.time()
        
        return key
    
    def mark_key_failure(self, key: str = None) -> None:
        """
        Mark a key as having failed (rate limit or other error).
        
        Args:
            key (str, optional): The key that failed. If None, uses current key.
        """
        if key is None:
            key = self.get_current_key()
        
        if key in self.key_status:
            self.key_status[key]["failures"] += 1
            
            # if too many failures, mark as not working
            if self.key_status[key]["failures"] >= config.MAX_RETRIES_PER_KEY:
                self.key_status[key]["working"] = False
                print(f"[APIKeyManager] Key {self._mask_key(key)} marked as failed "
                      f"after {self.key_status[key]['failures']} failures")
    
    def mark_key_success(self, key: str = None) -> None:
        """
        Mark a key as working successfully (resets failure count).
        
        Args:
            key (str, optional): The key that succeeded. If None, uses current key.
        """
        if key is None:
            key = self.get_current_key()
        
        if key in self.key_status:
            self.key_status[key]["failures"] = 0
            self.key_status[key]["working"] = True
    
    def rotate_to_next_key(self) -> str:
        """
        Rotate to the next working API key.
        
        Returns:
            str: The new current API key
        """
        original_index = self.current_index
        rotations = 0
        
        while rotations < len(self.keys):
            # move to next index (wrap around)
            self.current_index = (self.current_index + 1) % len(self.keys)
            rotations += 1
            
            # check if this key is working
            current_key = self.keys[self.current_index]
            if self.key_status.get(current_key, {}).get("working", True):
                print(f"[APIKeyManager] Rotated to key {self._mask_key(current_key)} "
                      f"(index {self.current_index + 1}/{len(self.keys)})")
                return current_key
        
        # if we get here, all keys are marked as not working
        # reset all keys to working and try again
        print("[APIKeyManager] All keys marked as failed. Resetting status...")
        self.reset_all_keys()
        self.current_index = 0
        return self.keys[0]
    
    def reset_all_keys(self) -> None:
        """
        Reset all keys to working status.
        useful after cooldown period or when limits reset.
        """
        for key in self.key_status:
            self.key_status[key]["working"] = True
            self.key_status[key]["failures"] = 0
        print(f"[APIKeyManager] Reset {len(self.key_status)} keys to working status")
    
    def _mask_key(self, key: str) -> str:
        """
        Mask an API key for safe logging.
        
        Args:
            key (str): The full API key
        
        Returns:
            str: Masked key (e.g., "gsk_abc...xyz")
        """
        if len(key) <= 8:
            return "***"
        return f"{key[:8]}...{key[-4:]}"
    
    def get_key_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all API keys.
        
        Returns:
            Dict: Statistics about key usage and status
        """
        return {
            "total_keys": len(self.keys),
            "current_index": self.current_index,
            "current_key_masked": self._mask_key(self.get_current_key()),
            "keys": [
                {
                    "masked": self._mask_key(k),
                    "working": self.key_status.get(k, {}).get("working", True),
                    "failures": self.key_status.get(k, {}).get("failures", 0),
                    "last_used": self.key_status.get(k, {}).get("last_used")
                }
                for k in self.keys
            ]
        }
    
    def has_working_key(self) -> bool:
        """
        Check if there is at least one working key.
        
        Returns:
            bool: True if at least one key is working
        """
        return any(
            self.key_status.get(k, {}).get("working", True)
            for k in self.keys
        )
    
    def get_working_key_count(self) -> int:
        """
        Get the number of working keys.
        
        Returns:
            int: Number of working keys
        """
        return sum(
            1 for k in self.keys
            if self.key_status.get(k, {}).get("working", True)
        )


# create global config instance - singleton pattern for configuration access
config = SystemConfig()
# creates a single instance of systemconfig that can be imported and used throughout the application

# create API key manager instance - singleton for key management
api_key_manager = APIKeyManager()
# manages multiple groq api keys with automatic cycling

# environment variables
# GROQ_API_KEY is kept for backward compatibility
GROQ_API_KEY: str = api_key_manager.get_current_key() if api_key_manager.keys else os.getenv("GROQ_API_KEY", "")

# function to get current API key (with optional rotation)
def get_current_api_key() -> str:
    """
    Get the current active API key from the manager.
    
    Returns:
        str: Current API key
    """
    return api_key_manager.get_current_key()

# function to handle API key failure and rotation
def handle_api_failure(current_key: str = None) -> str:
    """
    Handle an API key failure by marking it and rotating to next key.
    
    Args:
        current_key (str, optional): The key that failed
    
    Returns:
        str: The new API key to use
    """
    if current_key:
        api_key_manager.mark_key_failure(current_key)
    else:
        api_key_manager.mark_key_failure()
    
    # wait a bit before switching (rate limit cooldown)
    import time
    time.sleep(config.RATE_LIMIT_WAIT_SECONDS)
    
    return api_key_manager.rotate_to_next_key()

# embedding model configuration
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
# huggingface embedding model name - small but effective for semantic search

# track-specific display names - user-friendly names shown in the ui
TRACK_DISPLAY_NAMES: Dict[TrackType, str] = {
    TrackType.TRACK_A1_CS: "Track A1: Computer Science Subject Guide",
    TrackType.TRACK_A2_EXAM: "Track A2: Comprehensive Exam Preparation Assistant"
}

# track-specific descriptions - detailed explanations shown when user selects a track
TRACK_DESCRIPTIONS: Dict[TrackType, str] = {
    TrackType.TRACK_A1_CS: """
    **Computer Science Subject Guide**
    - CS-specific content processing with code examples and algorithms
    - Programming language detection and syntax highlighting
    - Algorithm explanation with step-by-step breakdowns
    - Database, Networking, OS specific content understanding
    - Indian university CS curriculum patterns
    """,
    
    TrackType.TRACK_A2_EXAM: """
    **Comprehensive Exam Preparation Assistant**
    - Complete exam preparation workflows combining all materials
    - Topic-wise question practice with solutions from study materials
    - Weak area identification and targeted content recommendations
    - Custom study plans based on syllabus and available content
    - Progress tracking across topics and question practice
    """
}