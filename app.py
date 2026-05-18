"""
academic assistant - main streamlit application

a comprehensive AI-powered academic assistant with multiple specialized
tracks.

students can upload lecture notes, textbooks, lab manuals, and question
papers

to create a complete learning companion for exam preparation and subject
mastery.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

from config.settings import (
    TrackType,
    TRACK_DISPLAY_NAMES,
    TRACK_DESCRIPTIONS
)
from core.rag_chain import (
    get_rag_chain_manager,
    ChainMode
)
from tracks.track_a1_cs import TrackA1CS
from tracks.track_a2_exam import TrackA2Exam
from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface
from components.progress_tracker import render_progress_dashboard, render_cs_dashboard

load_dotenv()

def check_api_keys() -> bool:
    """
    check if any groq api key is available in environment variables.
    looks for GROQ_API_KEY, GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
    
    Returns:
        bool: True if at least one api key is found, False otherwise
    """
    for key, value in os.environ.items():
        if key.startswith("GROQ_API_KEY") and value and value != "":
            return True
    
    keys_string = os.getenv("GROQ_API_KEYS")
    if keys_string:
        valid_keys = [k.strip() for k in keys_string.split(",") if k.strip()]
        if valid_keys:
            return True
    
    return False

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
    st.stop()

def initialize_session_state():
    """
    Initialize all session state variables for the streamlit app.
    session state persists across reruns and stores user data.
    """
    if "track_selected" not in st.session_state:
        st.session_state.track_selected = False
    
    if "current_track" not in st.session_state:
        st.session_state.current_track = None
    
    if "track_type" not in st.session_state:
        st.session_state.track_type = None
    
    if "vector_store_manager" not in st.session_state:
        from core.vector_store import get_vector_store_manager
        st.session_state.vector_store_manager = get_vector_store_manager()
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()
    
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = {}
    
    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = get_rag_chain_manager()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "Study Mode"
    
    if "progress_data" not in st.session_state:
        st.session_state.progress_data = {}
    
    if "processing_in_progress" not in st.session_state:
        st.session_state.processing_in_progress = False

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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Track A1")
        st.markdown(TRACK_DESCRIPTIONS[TrackType.TRACK_A1_CS])
        if st.button("Select Track A1", type="primary", use_container_width=True):
            st.session_state.current_track = TrackA1CS()
            st.session_state.track_type = TrackType.TRACK_A1_CS
            st.session_state.track_selected = True
            st.rerun()
    
    with col2:
        st.subheader("Track A2")
        st.markdown(TRACK_DESCRIPTIONS[TrackType.TRACK_A2_EXAM])
        if st.button("Select Track A2", type="primary", use_container_width=True):
            st.session_state.current_track = TrackA2Exam()
            st.session_state.track_type = TrackType.TRACK_A2_EXAM
            st.session_state.track_selected = True
            st.rerun()

def main():
    """
    Main application entry point.
    sets up the streamlit interface and handles routing.
    """
    st.set_page_config(
        page_title="Academic Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    if not st.session_state.track_selected:
        render_track_selection()
    else:
        render_sidebar()
        render_chat_interface()
        
        if st.session_state.documents_processed:
            if st.session_state.track_type == TrackType.TRACK_A1_CS:
                render_cs_dashboard()
            elif st.session_state.track_type == TrackType.TRACK_A2_EXAM:
                render_progress_dashboard()

if __name__ == "__main__":
    main()