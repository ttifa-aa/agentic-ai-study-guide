"""
academic assistant - main streamlit application
a comprehensive AI-powered academic assistant with multiple specialized tracks.
students can upload lecture notes, textbooks, lab manuals, and question papers
to create a complete learning companion for exam preparation and subject mastery.
"""

import streamlit as st  # streamlit for building the web interface
import os  # for accessing environment variables like api keys
from dotenv import load_dotenv  # to load environment variables from .env file
from datetime import datetime  # for timestamp handling in chat history

# local imports - our custom modules
from config.settings import (  # configuration constants and track definitions
    TrackType,
    TRACK_DISPLAY_NAMES,
    TRACK_DESCRIPTIONS
)
from core.rag_chain import (  # rag chain management
    get_rag_chain_manager,
    ChainMode
)
from tracks.track_a1_cs import TrackA1CS  # computer science track
from tracks.track_a2_exam import TrackA2Exam  # exam preparation track

# component imports - ui components
from components.sidebar import render_sidebar  # sidebar with document management
from components.chat_interface import render_chat_interface  # chat interface
from components.progress_tracker import render_progress_dashboard, render_cs_dashboard  # dashboards

# load environment variables from .env file
# this includes groq_api_key needed for llm access
load_dotenv()

# verify that at least one groq api key is set
# the api key manager handles multiple keys (GROQ_API_KEY_1, GROQ_API_KEY_2, etc.)
# so we check for any key starting with GROQ_API_KEY
def check_api_keys() -> bool:
    """
    check if any groq api key is available in environment variables.
    looks for GROQ_API_KEY, GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
    
    Returns:
        bool: True if at least one api key is found, False otherwise
    """
    # check all environment variables for keys starting with GROQ_API_KEY
    for key, value in os.environ.items():
        if key.startswith("GROQ_API_KEY") and value and value != "":
            # found at least one valid key
            return True
    
    # also check for comma-separated keys in GROQ_API_KEYS
    keys_string = os.getenv("GROQ_API_KEYS")
    if keys_string:
        valid_keys = [k.strip() for k in keys_string.split(",") if k.strip()]
        if valid_keys:
            return True
    
    return False


# check if any api keys are available
if not check_api_keys():
    st.error("No GROQ API keys found in environment variables. Please check your .env file.")
    st.info("""
    **How to fix:**
    1. Create a `.env` file in the project root
    2. Add at least one of these:
       - `GROQ_API_KEY_1=gsk_your_key_here`
       - `GROQ_API_KEY=gsk_your_key_here`
    3. Get free API keys from: https://console.groq.com
    """)
    st.stop()  # stop execution if no api keys are found


def initialize_session_state():
    """
    Initialize all session state variables for the streamlit app.
    session state persists across reruns and stores user data.
    """
    
    # track selection and initialization
    if "track_selected" not in st.session_state:
        st.session_state.track_selected = False
        # flag to indicate if user has selected a track
    
    if "current_track" not in st.session_state:
        st.session_state.current_track = None
        # stores the currently active track instance (tracka1cs or tracka2exam)
    
    if "track_type" not in st.session_state:
        st.session_state.track_type = None
        # stores the track type enum value
    
    # vector store and document management
    if "vector_store_manager" not in st.session_state:
        from core.vector_store import get_vector_store_manager
        st.session_state.vector_store_manager = get_vector_store_manager()
        # singleton vector store manager instance
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
        # flag indicating if at least one document has been processed
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()
        # set of filenames that have been uploaded and processed
    
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = {}
        # statistics about processed documents (chunk count, sources, etc.)
    
    # rag chain management
    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = get_rag_chain_manager()
        # singleton rag chain manager instance
    
    # chat and conversation history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # list of message dictionaries with role and content
        # format: [{"role": "user/assistant", "content": "message", "timestamp": "...", "metadata": {...}}]
    
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "Study Mode"
        # current operation mode (study mode or exam mode)
    
    # progress tracking (for exam track)
    if "progress_data" not in st.session_state:
        st.session_state.progress_data = {}
        # stores learning progress data for exam track
    
    # ui state management
    if "processing_in_progress" not in st.session_state:
        st.session_state.processing_in_progress = False
        # flag to prevent multiple simultaneous processing operations


def render_track_selection():
    """
    Render the track selection interface.
    allows user to choose between track a1 (cs) and track a2 (exam).
    """
    st.title("Academic Assistant")
    st.markdown("---")
    
    st.header("Choose Your Learning Track")
    st.markdown("""
    Select the track that best matches your learning goals.
    You can switch tracks later, but your uploaded documents will be cleared.
    """)
    
    # create two columns for track selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Track A1")
        st.markdown(TRACK_DESCRIPTIONS[TrackType.TRACK_A1_CS])
        
        if st.button("Select Track A1", type="primary", use_container_width=True):
            # initialize computer science track
            st.session_state.current_track = TrackA1CS()
            st.session_state.track_type = TrackType.TRACK_A1_CS
            st.session_state.track_selected = True
            st.rerun()  # rerun to show main interface
    
    with col2:
        st.subheader("Track A2")
        st.markdown(TRACK_DESCRIPTIONS[TrackType.TRACK_A2_EXAM])
        
        if st.button("Select Track A2", type="primary", use_container_width=True):
            # initialize exam preparation track
            st.session_state.current_track = TrackA2Exam()
            st.session_state.track_type = TrackType.TRACK_A2_EXAM
            st.session_state.track_selected = True
            st.rerun()  # rerun to show main interface


def main():
    """
    Main application entry point.
    sets up the streamlit interface and handles routing.
    """
    # page configuration
    st.set_page_config(
        page_title="Academic Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # initialize session state
    initialize_session_state()
    
    # main routing logic
    if not st.session_state.track_selected:
        # show track selection screen
        render_track_selection()
    else:
        # show main application interface
        render_sidebar()
        
        # main content area
        render_chat_interface()
        
        # track-specific dashboards (only show when documents are processed)
        if st.session_state.documents_processed:
            if st.session_state.track_type == TrackType.TRACK_A1_CS:
                render_cs_dashboard()
            elif st.session_state.track_type == TrackType.TRACK_A2_EXAM:
                render_progress_dashboard()


# application entry point
if __name__ == "__main__":
    main()