"""
Configuration module for the Academic Assistant.
Exports settings, constants, and API key management utilities.
"""

# import and re-export commonly used configuration items
# this allows other modules to import directly from config
from config.settings import (
    # system configuration
    config,                    # singleton systemconfig instance with all settings
    api_key_manager,           # singleton apikeymanager for multi-key rotation
    
    # track type enumeration
    TrackType,                 # enum for track types (track_a1_cs, track_a2_exam)
    
    # content type enumeration  
    ContentType,               # enum for document content types
    
    # display names and descriptions
    TRACK_DISPLAY_NAMES,       # dictionary mapping tracktype to user-friendly name
    TRACK_DESCRIPTIONS,        # dictionary mapping tracktype to detailed description
    
    # cs-specific configuration
    CS_SUBJECTS,               # dictionary mapping cs subjects to keywords
    CODE_PATTERNS,             # dictionary for programming language detection
    COMPLEXITY_PATTERNS,       # dictionary for algorithm complexity detection
    EXAM_PATTERNS,             # dictionary for exam pattern configurations
    
    # api key utilities
    get_current_api_key,       # function to get current active api key
    handle_api_failure,        # function to handle api key failure and rotation
    
    # environment variables
    GROQ_API_KEY,              # current api key (for backward compatibility)
    EMBEDDING_MODEL            # name of huggingface embedding model
)

# define what gets exported when someone does "from config import *"
__all__ = [
    "config",
    "api_key_manager", 
    "TrackType",
    "ContentType",
    "TRACK_DISPLAY_NAMES",
    "TRACK_DESCRIPTIONS",
    "CS_SUBJECTS",
    "CODE_PATTERNS",
    "COMPLEXITY_PATTERNS",
    "EXAM_PATTERNS",
    "get_current_api_key",
    "handle_api_failure",
    "GROQ_API_KEY",
    "EMBEDDING_MODEL"
]