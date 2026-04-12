"""
Sidebar component for the Academic Assistant.
Handles document upload, processing, and API key status display.
This is the main control panel for the application.
"""

import streamlit as st  # streamlit for building ui components
import time  # for adding delays and timing operations
from typing import List, Dict, Any  # type hints for function signatures

# local imports from our custom modules
from config.settings import config, ContentType, TRACK_DISPLAY_NAMES  # configuration constants and enums
from core.document_processor import (  # document processing utilities
    process_uploaded_file,            # processes a single uploaded file into chunks
    is_supported_file,                # checks if file extension is supported
    validate_document_content,        # validates that chunks have meaningful content
    get_document_stats                # calculates statistics about processed documents
)
from core.vector_store import add_chunks_to_vectorstore  # adds document chunks to faiss index
from core.rag_chain import get_api_usage_stats  # gets api key usage statistics


def render_api_key_status():
    """
    Render API key status in the sidebar.
    displays how many api keys are available, which are working, and usage statistics.
    includes option to reset keys if all are failing.
    """
    try:
        # get current api key statistics from the rag chain manager
        stats = get_api_usage_stats()  # returns dict with total_keys, working_keys, rotations, etc.
        
        st.markdown("---")  # visual separator
        st.subheader("API Key Status")  # section header
        
        total_keys = stats.get("total_keys", 0)  # total number of configured api keys
        # count how many keys are currently marked as working
        working_keys = sum(1 for k in stats.get("keys", []) if k.get("working", True))
        
        # display overall key status with appropriate color coding
        if working_keys == total_keys and total_keys > 0:
            st.success(f"{working_keys}/{total_keys} keys working")  # all keys working - green
        elif working_keys > 0:
            st.warning(f"{working_keys}/{total_keys} keys working")  # some keys working - yellow
        else:
            st.error(f"No working keys!")  # no keys working - red
        
        # create expandable section for detailed key information
        with st.expander("View Key Details"):
            # display summary metrics
            st.metric("Total Keys", total_keys)  # total number of keys
            st.metric("Working Keys", working_keys)  # number of working keys
            st.metric("Key Rotations", stats.get("rag_chain_rotations", 0))  # times keys were rotated
            st.metric("Current Backoff", f"{stats.get('current_backoff', 1.0):.1f}s")  # rate limit backoff
            
            # display individual key status
            # each key is shown with masked value for security
            for i, key_info in enumerate(stats.get("keys", []), 1):
                status = "Working" if key_info.get("working") else "Failed"  # status text
                status_icon = "✅" if key_info.get("working") else "❌"  # visual indicator
                failures = key_info.get("failures", 0)  # number of failures for this key
                masked_key = key_info.get("masked", "***")  # masked key (e.g., "gsk_abc...xyz")
                
                # display key info in a compact caption
                st.caption(f"{status_icon} Key {i}: {masked_key} - {status} (failures: {failures})")
        
        # if all keys are failing, provide a reset button
        # this clears failure counts and marks all keys as working again
        if working_keys == 0 and total_keys > 0:
            if st.button("Reset All Keys", use_container_width=True):
                # import here to avoid circular imports
                from config.settings import api_key_manager
                api_key_manager.reset_all_keys()  # reset all key statuses
                st.rerun()  # rerun to update the display
                
    except Exception as e:
        # gracefully handle case where api key manager isn't initialized yet
        # this can happen on first run before documents are processed
        st.markdown("---")
        st.subheader("API Key Status")
        st.warning("API key manager not initialized")  # show warning instead of error


def render_document_upload_section():
    """
    Render the document upload section of the sidebar.
    includes file uploader, content type selection, and process button.
    """
    st.subheader("Upload Documents")  # section header
    
    # file uploader widget that accepts multiple files
    # converts config.SUPPORTED_EXTENSIONS set to list for streamlit
    uploaded_files = st.file_uploader(
        "Upload academic documents",  # label for uploader
        type=list(config.SUPPORTED_EXTENSIONS),  # allowed file extensions (pdf, docx, txt, pptx)
        accept_multiple_files=True,  # allow selecting multiple files at once
        help="Supported formats: PDF, DOCX, TXT, PPTX"  # tooltip help text
    )
    
    # if user has uploaded files, show content type selection for each file
    if uploaded_files:
        st.markdown("**Select content type for each file:**")  # instruction text
        
        content_types = {}  # dictionary to store filename -> content_type mapping
        
        # create a selectbox for each uploaded file
        for file in uploaded_files:
            # each selectbox needs a unique key for streamlit to track state
            content_types[file.name] = st.selectbox(
                f"{file.name}",  # label shows filename
                options=[ct.value for ct in ContentType],  # all content type enum values
                key=f"ct_{file.name}"  # unique key per file
            )
        
        # process button to start document processing
        if st.button("Process Documents", type="primary", use_container_width=True):
            # call function to process all uploaded files
            process_uploaded_documents(uploaded_files, content_types)


def process_uploaded_documents(uploaded_files: List, content_types: Dict[str, str]):
    """
    Process uploaded documents and add them to the vector store.
    this function handles the entire document ingestion pipeline:
    load -> chunk -> embed -> store.
    
    Args:
        uploaded_files: List of streamlit UploadedFile objects
        content_types: Dictionary mapping filenames to their content types
    """
    # validate that files were actually uploaded
    if not uploaded_files:
        st.warning("Please upload at least one file.")
        return
    
    all_chunks = []  # master list to store all document chunks from all files
    processed_files = []  # track files that were successfully processed
    failed_files = []  # track files that failed with error messages
    
    # create progress indicators for visual feedback
    progress_bar = st.progress(0)  # progress bar from 0 to 1
    status_text = st.empty()  # placeholder for status text that we can update
    
    # process each file individually
    for i, uploaded_file in enumerate(uploaded_files):
        # update status text to show current file being processed
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # verify file extension is supported before attempting to process
        if not is_supported_file(uploaded_file.name):
            failed_files.append((uploaded_file.name, "Unsupported file type"))
            continue  # skip this file and move to next
        
        try:
            # get the content type selected by user for this file
            content_type = content_types[uploaded_file.name]
            
            # process the file into document chunks
            # this loads the file, splits into chunks, and adds metadata
            chunks = process_uploaded_file(
                uploaded_file,                     # the uploaded file object
                content_type,                      # selected content type
                chunk_size=config.CHUNK_SIZE,      # characters per chunk from config
                chunk_overlap=config.CHUNK_OVERLAP # overlap between chunks from config
            )
            
            # validate that we extracted meaningful content
            # filters out empty or near-empty chunks
            if validate_document_content(chunks):
                all_chunks.extend(chunks)  # add valid chunks to master list
                processed_files.append(uploaded_file.name)  # mark as processed
                
                # add to session state for tracking uploaded files
                if "uploaded_files" not in st.session_state:
                    st.session_state.uploaded_files = set()
                st.session_state.uploaded_files.add(uploaded_file.name)
            else:
                failed_files.append((uploaded_file.name, "No meaningful content extracted"))
        
        except Exception as e:
            # catch any errors during processing and record them
            failed_files.append((uploaded_file.name, str(e)))
        
        # update progress bar based on number of files processed
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    # clean up progress indicators
    status_text.empty()  # remove status text
    progress_bar.empty()  # remove progress bar
    
    # if any files were processed successfully, add them to vector store
    if all_chunks:
        with st.spinner("Creating vector embeddings..."):
            # add all chunks to faiss vector store
            # this creates embeddings and builds the search index
            add_chunks_to_vectorstore(all_chunks)
            
            # calculate and store document statistics in session state
            st.session_state.document_stats = get_document_stats(all_chunks)
            st.session_state.documents_processed = True  # mark docs as processed
            
            # clear rag chain cache so new documents are used for future queries
            if "rag_manager" in st.session_state:
                st.session_state.rag_manager.clear_cache()
        
        # show success message with count of processed files
        st.success(f"Processed {len(processed_files)} files successfully!")
        
        # track-specific document analysis
        # different tracks analyze documents differently
        if "current_track" in st.session_state and st.session_state.current_track:
            track = st.session_state.current_track
            
            # check if track has cs content analysis capability (track a1)
            if hasattr(track, 'analyze_document_for_cs_content'):
                # for cs track, analyze first 10 chunks for cs content
                # sampling to avoid processing overhead
                for chunk in all_chunks[:10]:
                    track.analyze_document_for_cs_content(chunk.page_content)
            
            # check if track has exam paper analysis capability (track a2)
            elif hasattr(track, 'analyze_exam_paper'):
                # for exam track, check if any uploaded files are past papers
                for uploaded_file in uploaded_files:
                    content_type = content_types[uploaded_file.name]
                    
                    # only analyze files marked as past papers
                    if content_type == ContentType.PAST_PAPER.value:
                        # extract text from chunks belonging to this file
                        paper_text = "\n".join([
                            chunk.page_content for chunk in all_chunks
                            if chunk.metadata.get("source") == uploaded_file.name
                        ])
                        
                        # if we got text, analyze the exam paper
                        if paper_text:
                            track.analyze_exam_paper(paper_text)
    
    # show failed files in an expandable section if any failures occurred
    if failed_files:
        with st.expander(f"{len(failed_files)} files failed to process"):
            for filename, error in failed_files:
                st.error(f"{filename}: {error}")


def render_document_library_section():
    """
    Render the document library section showing processed documents.
    displays statistics about indexed documents and list of uploaded files.
    only shown when documents have been processed.
    """
    # only show if documents have been processed
    if not st.session_state.get("documents_processed", False):
        return
    
    st.markdown("---")  # visual separator
    st.subheader("Document Library")  # section header
    
    # get document statistics from session state
    stats = st.session_state.get("document_stats", {})
    
    # display metrics in two columns
    col1, col2 = st.columns(2)
    with col1:
        # total chunks - each chunk is a retrievable piece of text
        st.metric("Total Chunks", stats.get("total_chunks", 0))
    with col2:
        # unique sources - number of distinct files uploaded
        st.metric("Unique Sources", len(stats.get("sources", {})))
    
    # list all uploaded files that were successfully processed
    uploaded_files = st.session_state.get("uploaded_files", set())
    if uploaded_files:
        st.markdown("**Uploaded Files:**")
        # sort files alphabetically for consistent display
        for filename in sorted(uploaded_files):
            st.caption(f"📄 {filename}")  # document icon + filename
    
    # button to clear all documents from the system
    if st.button("Clear All Documents", type="secondary", use_container_width=True):
        clear_all_documents()


def clear_all_documents():
    """
    Clear all documents from the vector store and reset related state.
    this removes all indexed content and resets the application state.
    """
    # import here to avoid circular imports
    from core.vector_store import get_vector_store_manager
    
    # get vector store manager and clear the index
    manager = get_vector_store_manager()
    manager.clear_vectorstore()  # removes all vectors and documents from faiss
    
    # reset all document-related session state variables
    st.session_state.documents_processed = False  # no documents are processed
    st.session_state.uploaded_files = set()  # clear list of uploaded files
    st.session_state.document_stats = {}  # clear document statistics
    
    # clear chat history since answers depended on now-deleted documents
    st.session_state.chat_history = []
    
    # clear rag chain cache to force recreation with empty vector store
    if "rag_manager" in st.session_state:
        st.session_state.rag_manager.clear_cache()
    
    # show success message and wait briefly for user to see it
    st.success("All documents cleared!")
    time.sleep(1)  # brief pause before rerun
    st.rerun()  # rerun the app to refresh the interface


def render_sidebar():
    """
    Main sidebar render function.
    combines all sidebar components into a cohesive interface.
    this is the entry point called from the main app.
    """
    with st.sidebar:  # everything inside this block goes in the sidebar
        st.header("Document Management")  # main sidebar header
        
        # show which track is currently active
        if st.session_state.get("track_type"):
            track_name = TRACK_DISPLAY_NAMES.get(st.session_state.track_type, "Unknown Track")
            st.caption(f"Active Track: {track_name}")  # subtle text showing active track
        
        st.markdown("---")  # visual separator
        
        # document upload section - primary way to add content
        render_document_upload_section()
        
        # document library section - shows what's been uploaded
        render_document_library_section()
        
        # track switching option
        st.markdown("---")
        if st.button("Switch Track", use_container_width=True):
            # reset track selection to show track selection screen again
            st.session_state.track_selected = False  # go back to track selection
            st.session_state.current_track = None  # clear current track
            st.session_state.documents_processed = False  # reset document state
            st.session_state.uploaded_files = set()  # clear uploaded files
            st.session_state.chat_history = []  # clear chat history
            st.rerun()  # rerun to show track selection screen
        
        # api key status section - shows at bottom of sidebar
        render_api_key_status()