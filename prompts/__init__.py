"""
Prompts module for the Academic Assistant.
Contains all prompt templates used by the RAG chain.
"""

# import and re-export all prompt templates
from prompts.base_prompts import (
    STUDY_MODE_PROMPT,               # prompt for comprehensive study mode explanations
    EXAM_MODE_PROMPT,                # prompt for step-by-step exam solutions
    QUICK_REVISION_PROMPT,           # prompt for concise quick revision answers
    CS_SPECIFIC_PROMPT,              # prompt for cs-specific topics with code
    EXAM_PREPARATION_PROMPT,         # prompt for exam preparation guidance
    TOPIC_SYNTHESIS_PROMPT           # prompt for combining info from multiple sources
)

# define what gets exported when someone does "from prompts import *"
__all__ = [
    "STUDY_MODE_PROMPT",
    "EXAM_MODE_PROMPT",
    "QUICK_REVISION_PROMPT",
    "CS_SPECIFIC_PROMPT",
    "EXAM_PREPARATION_PROMPT",
    "TOPIC_SYNTHESIS_PROMPT"
]