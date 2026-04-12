"""
academic assistant - main streamlit application
a comprehensive AI-powered academic assistant with multiple specialized tracks.
students can upload lecture notes, textbooks, lab manuals, and question papers
to create a complete learning companion for exam preparation and subject mastery.
"""

import streamlit as st  # streamlit for building the web interface
import os  # for accessing environment variables like api keys
from dotenv import load_dotenv  # to load environment variables from .env file
import time  # for adding delays and measuring processing time
from datetime import datetime  # for timestamp handling in chat history
import tempfile  # for creating temporary files during document processing
from pathlib import Path  # object-oriented file path handling

# langchain imports for document processing and rag
from langchain_text_splitters import RecursiveCharacterTextSplitter  # for splitting documents into chunks

# local imports - our custom modules
from config.settings import (  # configuration constants and track definitions
    config,
    TrackType,
    ContentType,
    TRACK_DISPLAY_NAMES,
    TRACK_DESCRIPTIONS
)
from core.document_processor import (  # document processing utilities
    process_uploaded_file,
    is_supported_file,
    validate_document_content,
    get_document_stats
)
from core.vector_store import (  # vector store management
    get_vector_store_manager,
    create_vectorstore_from_chunks,
    add_chunks_to_vectorstore
)
from core.rag_chain import (  # rag chain management
    get_rag_chain_manager,
    ChainMode,
    format_citations
)
from tracks.track_a1_cs import TrackA1CS  # computer science track
from tracks.track_a2_exam import TrackA2Exam  # exam preparation track

# load environment variables from .env file
# this includes groq_api_key needed for llm access
load_dotenv()

# verify that groq api key is set
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()  # stop execution if api key is missing


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
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False
        # track sidebar collapse state
    
    if "processing_in_progress" not in st.session_state:
        st.session_state.processing_in_progress = False
        # flag to prevent multiple simultaneous processing operations


def render_track_selection():
    """
    Render the track selection interface.
    allows user to choose between track a1 (cs) and track a2 (exam).
    """
    st.title("🎓 Academic Assistant")
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


def render_sidebar():
    """
    Render the sidebar with document upload and management options.
    this is the main control panel for document processing.
    """
    with st.sidebar:
        st.header("📚 Document Management")
        
        # show current track information
        if st.session_state.track_type:
            track_name = TRACK_DISPLAY_NAMES[st.session_state.track_type]
            st.caption(f"Active Track: {track_name}")
        
        st.markdown("---")
        
        # document upload section
        st.subheader("Upload Documents")
        
        # file uploader with multiple file support
        uploaded_files = st.file_uploader(
            "Upload academic documents",
            type=list(config.SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, PPTX"
        )
        
        # content type selection for each uploaded file
        if uploaded_files:
            st.markdown("**Select content type for each file:**")
            
            content_types = {}
            for file in uploaded_files:
                # create selectbox for each uploaded file
                content_types[file.name] = st.selectbox(
                    f"{file.name}",
                    options=[ct.value for ct in ContentType],
                    key=f"ct_{file.name}"
                )
            
            # process documents button
            if st.button("Process Documents", type="primary", use_container_width=True):
                process_uploaded_documents(uploaded_files, content_types)
        render_api_key_status() # show api key status in sidebar
        
        st.markdown("---")
        
        # document library section (shown only when documents are processed)
        if st.session_state.documents_processed:
            st.subheader("📖 Document Library")
            
            # show document statistics
            stats = st.session_state.document_stats
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            st.metric("Unique Sources", len(stats.get("sources", {})))
            
            # list uploaded files
            if st.session_state.uploaded_files:
                st.markdown("**Uploaded Files:**")
                for filename in st.session_state.uploaded_files:
                    st.caption(f"✓ {filename}")
            
            # clear all documents button
            if st.button("Clear All Documents", type="secondary", use_container_width=True):
                clear_all_documents()
        
        st.markdown("---")
        
        # track switching option
        if st.button("Switch Track", use_container_width=True):
            # reset track selection to show track selection screen
            st.session_state.track_selected = False
            st.session_state.current_track = None
            st.session_state.documents_processed = False
            st.session_state.uploaded_files = set()
            st.session_state.chat_history = []
            st.rerun()


def process_uploaded_documents(uploaded_files, content_types):
    """
    Process uploaded documents and add them to the vector store.
    
    Args:
        uploaded_files: List of uploaded file objects
        content_types: Dictionary mapping filenames to content types
    """
    if not uploaded_files:
        st.warning("Please upload at least one file.")
        return
    
    # set processing flag to show progress
    st.session_state.processing_in_progress = True
    
    all_chunks = []  # list to store all document chunks
    processed_files = []  # track successfully processed files
    failed_files = []  # track files that failed to process
    
    # create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # check if file type is supported
        if not is_supported_file(uploaded_file.name):
            failed_files.append((uploaded_file.name, "Unsupported file type"))
            continue
        
        try:
            # process the file with selected content type
            content_type = content_types[uploaded_file.name]
            chunks = process_uploaded_file(
                uploaded_file,
                content_type,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            
            # validate that we got meaningful content
            if validate_document_content(chunks):
                all_chunks.extend(chunks)
                processed_files.append(uploaded_file.name)
                st.session_state.uploaded_files.add(uploaded_file.name)
            else:
                failed_files.append((uploaded_file.name, "No meaningful content extracted"))
        
        except Exception as e:
            failed_files.append((uploaded_file.name, str(e)))
        
        # update progress bar
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    # clear status text
    status_text.empty()
    progress_bar.empty()
    
    # add chunks to vector store if any were processed successfully
    if all_chunks:
        with st.spinner("Creating vector embeddings..."):
            # add chunks to vector store
            add_chunks_to_vectorstore(all_chunks)
            
            # update document statistics
            st.session_state.document_stats = get_document_stats(all_chunks)
            st.session_state.documents_processed = True
            
            # clear rag chain cache to use new documents
            st.session_state.rag_manager.clear_cache()
        
        # show success message
        st.success(f"✅ Processed {len(processed_files)} files successfully!")
        
        # analyze documents with current track if applicable
        if st.session_state.current_track:
            track = st.session_state.current_track
            if hasattr(track, 'analyze_document_for_cs_content'):
                # for cs track, analyze documents for cs content
                for chunk in all_chunks[:10]:  # sample first 10 chunks
                    track.analyze_document_for_cs_content(chunk.page_content)
            elif hasattr(track, 'analyze_exam_paper'):
                # for exam track, check if any files are past papers
                for uploaded_file in uploaded_files:
                    content_type = content_types[uploaded_file.name]
                    if content_type == ContentType.PAST_PAPER.value:
                        # extract text from chunks for this file
                        paper_text = "\n".join([
                            chunk.page_content for chunk in all_chunks
                            if chunk.metadata.get("source") == uploaded_file.name
                        ])
                        if paper_text:
                            track.analyze_exam_paper(paper_text)
    
    # show failed files if any
    if failed_files:
        with st.expander(f"⚠️ {len(failed_files)} files failed to process"):
            for filename, error in failed_files:
                st.error(f"{filename}: {error}")
    
    st.session_state.processing_in_progress = False


def clear_all_documents():
    """
    Clear all documents from the vector store and reset state.
    """
    # clear vector store
    st.session_state.vector_store_manager.clear_vectorstore()
    
    # reset document-related state
    st.session_state.documents_processed = False
    st.session_state.uploaded_files = set()
    st.session_state.document_stats = {}
    
    # clear chat history
    st.session_state.chat_history = []
    
    # clear rag chain cache
    st.session_state.rag_manager.clear_cache()
    
    # show success message
    st.success("All documents cleared!")
    time.sleep(1)
    st.rerun()


def render_chat_interface():
    """
    Render the main chat interface for asking questions.
    """
    st.header("Ask Questions About Your Documents")
    
    # mode selection (study/exam)
    mode_col1, mode_col2 = st.columns([1, 4])
    with mode_col1:
        st.session_state.current_mode = st.radio(
            "Mode:",
            options=["Study Mode", "Exam Mode"],
            horizontal=True,
            help="Study Mode: Comprehensive explanations | Exam Mode: Step-by-step solutions"
        )
    
    # display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # show metadata if available (in expander)
                if "metadata" in message and message["metadata"]:
                    with st.expander("View Details"):
                        st.json(message["metadata"])
    
    # initial greeting if no history
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            welcome_msg = st.session_state.current_track.get_welcome_message() if st.session_state.current_track else "Hello! Upload documents to get started."
            st.write(welcome_msg)
    
    # chat input
    if st.session_state.documents_processed:
        user_input = st.chat_input("Ask a question about your documents...")
        
        if user_input:
            # add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(user_input)
                    st.write(response["answer"])
                    
                    # show sources in expander
                    if response.get("sources"):
                        with st.expander("View Sources"):
                            for source in response["sources"]:
                                st.caption(f"📄 {source}")
            
            # add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "timestamp": datetime.now().isoformat(),
                "metadata": response.get("metadata", {})
            })
    else:
        # prompt user to upload documents
        st.info("👆 Please upload and process documents using the sidebar to start asking questions.")


def generate_response(user_input: str) -> dict:
    """
    Generate a response using the current track and mode.
    
    Args:
        user_input (str): User's question
    
    Returns:
        dict: Response dictionary with answer and metadata
    """
    # determine chain mode based on ui selection
    chain_mode = ChainMode.EXAM if st.session_state.current_mode == "Exam Mode" else ChainMode.STUDY
    
    # use track-specific processing if available
    if st.session_state.current_track:
        track = st.session_state.current_track
        
        # detect query type for track-specific handling
        query_type = "general"
        if st.session_state.current_mode == "Exam Mode":
            query_type = "solve"
        
        # process with track
        response = track.process_query(user_input, query_type=query_type)
    else:
        # fallback to standard rag chain
        answer = st.session_state.rag_manager.invoke(user_input, mode=chain_mode)
        
        # format response
        response = {
            "answer": answer,
            "sources": [],  # would extract from retrieved docs
            "metadata": {
                "mode": st.session_state.current_mode,
                "track": "default"
            }
        }
    
    return response


def render_progress_dashboard():
    """
    Render progress tracking dashboard for exam track.
    only shown when track a2 (exam) is active.
    """
    if st.session_state.track_type != TrackType.TRACK_A2_EXAM:
        return
    
    track = st.session_state.current_track
    if not hasattr(track, 'get_progress_summary'):
        return
    
    st.markdown("---")
    st.header("📊 Progress Dashboard")
    
    # get progress summary
    summary = track.get_progress_summary()
    
    # display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = summary.get("metrics", {})
    with col1:
        st.metric(
            "Overall Accuracy",
            f"{metrics.get('overall_accuracy', 0):.1f}%"
        )
    
    with col2:
        st.metric(
            "Questions Attempted",
            metrics.get("total_questions", 0)
        )
    
    with col3:
        st.metric(
            "Topics Covered",
            metrics.get("topics_covered", 0)
        )
    
    with col4:
        st.metric(
            "Study Time",
            f"{metrics.get('total_time_hours', 0):.1f} hrs"
        )
    
    # weak areas section
    weak_areas = summary.get("weak_areas", [])
    if weak_areas:
        st.subheader("🎯 Focus Areas")
        for topic, score in weak_areas[:3]:
            st.progress(score, text=f"{topic} (Weakness: {score:.2f})")
    
    # export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Progress Report", use_container_width=True):
            report = track.export_progress_report()
            st.download_button(
                "Download Report",
                report,
                file_name=f"progress_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    with col2:
        if st.button("Generate Study Plan", use_container_width=True):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Generate a study plan for me"
            })
            st.rerun()


def render_cs_dashboard():
    """
    Render cs-specific dashboard for track a1.
    shows detected code blocks, algorithms, and subjects.
    """
    if st.session_state.track_type != TrackType.TRACK_A1_CS:
        return
    
    track = st.session_state.current_track
    if not hasattr(track, 'get_cs_subject_summary'):
        return
    
    st.markdown("---")
    st.header("💻 CS Subject Analysis")
    
    # get cs subject summary
    summary = track.get_cs_subject_summary()
    
    # display detected subjects
    subjects = summary.get("identified_subjects", {})
    if subjects:
        st.subheader("Detected Subjects")
        for subject, confidence in subjects.items():
            st.progress(confidence, text=f"{subject}")
    
    # code and algorithm stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Code Blocks Found", summary.get("code_blocks_found", 0))
    with col2:
        st.metric("Algorithms Detected", summary.get("algorithms_detected", 0))


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
        
        # track-specific dashboards
        if st.session_state.documents_processed:
            if st.session_state.track_type == TrackType.TRACK_A1_CS:
                render_cs_dashboard()
            elif st.session_state.track_type == TrackType.TRACK_A2_EXAM:
                render_progress_dashboard()

# add this function to app.py

def render_api_key_status():
    """
    Render API key status in the sidebar.
    shows how many keys are available and working.
    """
    from core.rag_chain import get_api_usage_stats
    
    stats = get_api_usage_stats()
    
    st.markdown("---")
    st.subheader("🔑 API Key Status")
    
    total_keys = stats.get("total_keys", 0)
    working_keys = sum(1 for k in stats.get("keys", []) if k.get("working", True))
    
    # show key status
    if working_keys == total_keys:
        st.success(f"✅ {working_keys}/{total_keys} keys working")
    elif working_keys > 0:
        st.warning(f"⚠️ {working_keys}/{total_keys} keys working")
    else:
        st.error(f"❌ No working keys!")
    
    # show key details in expander
    with st.expander("View Key Details"):
        st.metric("Total Keys", total_keys)
        st.metric("Working Keys", working_keys)
        st.metric("Key Rotations", stats.get("rag_chain_rotations", 0))
        
        # list each key's status
        for i, key_info in enumerate(stats.get("keys", []), 1):
            status = "✅" if key_info.get("working") else "❌"
            failures = key_info.get("failures", 0)
            st.caption(f"{status} Key {i}: {key_info.get('masked', '***')} (failures: {failures})")
    
    # reset keys button (if all keys are failing)
    if working_keys == 0:
        if st.button("🔄 Reset All Keys", use_container_width=True):
            from config.settings import api_key_manager
            api_key_manager.reset_all_keys()
            st.rerun()

# application entry point
if __name__ == "__main__":
    main()