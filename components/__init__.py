"""
UI Components module for the Academic Assistant.
Contains reusable Streamlit components for the user interface.
"""

# import and re-export sidebar components
from components.sidebar import (
    render_sidebar,                  # main sidebar render function
    render_api_key_status,           # renders api key status section
    render_document_upload_section,  # renders document upload section
    render_document_library_section, # renders document library section
    process_uploaded_documents,      # processes uploaded documents
    clear_all_documents             # clears all documents from vector store
)

# import and re-export chat interface components
from components.chat_interface import (
    render_chat_interface,           # main chat interface render function
    render_chat_history,             # renders chat history
    render_mode_selector,            # renders study/exam mode selector
    render_welcome_message,          # renders welcome message
    generate_response,               # generates response for user query
    clear_chat_history               # clears chat history
)

# import and re-export progress tracker components
from components.progress_tracker import (
    render_progress_dashboard,       # main progress dashboard router
    render_exam_progress_dashboard,  # renders exam track progress dashboard
    render_cs_dashboard              # renders cs track dashboard
)

# define what gets exported when someone does "from components import *"
__all__ = [
    # sidebar
    "render_sidebar",
    "render_api_key_status",
    "render_document_upload_section",
    "render_document_library_section",
    "process_uploaded_documents",
    "clear_all_documents",
    
    # chat interface
    "render_chat_interface",
    "render_chat_history",
    "render_mode_selector",
    "render_welcome_message",
    "generate_response",
    "clear_chat_history",
    
    # progress tracker
    "render_progress_dashboard",
    "render_exam_progress_dashboard",
    "render_cs_dashboard"
]