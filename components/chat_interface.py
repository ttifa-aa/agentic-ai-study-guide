"""
Chat interface component for the Academic Assistant.
Handles user queries, response display, and chat history.
This is the main interaction area where users ask questions.
"""

import streamlit as st  # streamlit for building ui components
from datetime import datetime  # for adding timestamps to messages
from typing import Dict, Any, List  # type hints for function signatures

# local imports from our custom modules
from core.rag_chain import ChainMode, format_citations  # rag chain utilities


def render_mode_selector() -> str:
    """
    Render the mode selector radio button (Study Mode / Exam Mode).
    study mode provides comprehensive explanations.
    exam mode provides step-by-step solutions optimized for exams.
    
    Returns:
        str: Selected mode ("Study Mode" or "Exam Mode")
    """
    # create two columns - one narrow for the label, one wide for the radio
    mode_col1, mode_col2 = st.columns([1, 4])
    
    with mode_col1:
        # radio button for mode selection
        selected_mode = st.radio(
            "Mode:",  # label
            options=["Study Mode", "Exam Mode"],  # available options
            horizontal=True,  # display options side by side
            help="Study Mode: Comprehensive explanations | Exam Mode: Step-by-step solutions",
            key="mode_selector"  # unique key for streamlit state
        )
    
    # update session state with current mode for other components to access
    st.session_state.current_mode = selected_mode
    return selected_mode  # return the selected mode


def render_chat_history():
    """
    Render the chat history from session state.
    displays all previous messages in chronological order with their metadata.
    """
    # get chat history from session state (default to empty list if not exists)
    chat_history = st.session_state.get("chat_history", [])
    
    # iterate through each message in the history
    for message in chat_history:
        # create a chat message container with appropriate role (user/assistant)
        with st.chat_message(message["role"]):
            # display the main message content
            st.write(message["content"])
            
            # if message has metadata, show it in an expandable section
            if "metadata" in message and message["metadata"]:
                with st.expander("View Details"):
                    # display metadata as formatted json
                    st.json(message["metadata"])
                    
                    # if sources are included in metadata, display them
                    if "sources" in message["metadata"]:
                        st.caption("Sources:")
                        for source in message["metadata"]["sources"]:
                            st.caption(f"📄 {source}")


def render_welcome_message():
    """
    Render the welcome message when no chat history exists.
    shows track-specific welcome message if a track is selected.
    """
    # only show welcome if chat history is empty
    if st.session_state.get("chat_history"):
        return
    
    # create assistant message container
    with st.chat_message("assistant"):
        # check if a track is selected
        if st.session_state.get("current_track"):
            # get track-specific welcome message
            welcome_msg = st.session_state.current_track.get_welcome_message()
        else:
            # default welcome message
            welcome_msg = "Hello! Upload documents to get started."
        
        st.write(welcome_msg)  # display the welcome message


def generate_response(user_input: str) -> Dict[str, Any]:
    """
    Generate a response using the current track and mode.
    this function routes the query to the appropriate processing pipeline.
    
    Args:
        user_input (str): User's question or query text
    
    Returns:
        Dict: Response dictionary containing answer, sources, and metadata
    """
    # determine chain mode based on ui selection
    current_mode = st.session_state.get("current_mode", "Study Mode")
    chain_mode = ChainMode.EXAM if current_mode == "Exam Mode" else ChainMode.STUDY
    
    # check if we have a track selected for specialized processing
    if st.session_state.get("current_track"):
        track = st.session_state.current_track
        
        # detect query type for track-specific handling
        # default is "general", exam mode uses "solve"
        query_type = "general"
        if current_mode == "Exam Mode":
            query_type = "solve"
        
        # process query using the track's specialized pipeline
        response = track.process_query(user_input, query_type=query_type)
        
        # ensure response has expected structure
        # handle case where track returns string instead of dict
        if isinstance(response, dict):
            if "answer" not in response:
                response = {"answer": str(response), "metadata": {}}
        else:
            response = {"answer": str(response), "metadata": {}}
            
        return response
    else:
        # fallback to standard rag chain if no track is selected
        if "rag_manager" in st.session_state:
            # use the generic rag chain manager
            answer = st.session_state.rag_manager.invoke(user_input, mode=chain_mode)
        else:
            # no documents processed yet
            answer = "Please upload documents first to enable question answering."
        
        # return response in standard format
        return {
            "answer": answer,
            "sources": [],  # no source tracking in fallback mode
            "metadata": {
                "mode": current_mode,
                "track": "default"
            }
        }


def handle_user_input():
    """
    Handle user input from chat input widget.
    processes the question, generates a response, and updates chat history.
    this is the core interaction loop.
    """
    # create chat input widget at the bottom of the chat area
    user_input = st.chat_input("Ask a question about your documents...")
    
    # if no input, just return
    if not user_input:
        return
    
    # add user message to chat history with timestamp
    st.session_state.chat_history.append({
        "role": "user",  # message from user
        "content": user_input,  # the question text
        "timestamp": datetime.now().isoformat()  # iso format timestamp
    })
    
    # display user message immediately in the chat
    with st.chat_message("user"):
        st.write(user_input)
    
    # generate and display assistant response
    with st.chat_message("assistant"):
        # show spinner while generating response
        with st.spinner("Thinking..."):
            # call response generation function
            response = generate_response(user_input)
            
            # display the generated answer
            st.write(response["answer"])
            
            # if response includes sources, show them in expander
            sources = response.get("sources", [])
            if sources:
                with st.expander("View Sources"):
                    for source in sources:
                        st.caption(f"📄 {source}")
            
            # if metadata includes api key rotation info, show it
            metadata = response.get("metadata", {})
            if metadata:
                # show key rotation count if keys were rotated during this request
                if "key_rotations" in metadata and metadata["key_rotations"] > 0:
                    st.caption(f"🔄 API key rotations: {metadata['key_rotations']}")
    
    # add assistant response to chat history for future display
    st.session_state.chat_history.append({
        "role": "assistant",  # message from assistant
        "content": response["answer"],  # the generated answer
        "timestamp": datetime.now().isoformat(),  # iso format timestamp
        "metadata": response.get("metadata", {}),  # any additional metadata
        "sources": response.get("sources", [])  # list of source citations
    })


def render_chat_interface():
    """
    Main chat interface render function.
    combines chat history display and user input handling.
    this is the entry point called from the main app.
    """
    st.header("Ask Questions About Your Documents")  # main header
    
    # render the mode selector (study/exam)
    render_mode_selector()
    
    # container for chat messages - keeps everything aligned
    chat_container = st.container()
    
    with chat_container:
        # render existing chat history (all previous messages)
        render_chat_history()
        
        # render welcome message if this is the first interaction
        render_welcome_message()
    
    # handle user input (only if documents have been processed)
    # otherwise show a prompt to upload documents first
    if st.session_state.get("documents_processed", False):
        handle_user_input()  # show chat input and process queries
    else:
        # info message prompting user to upload documents
        st.info("👆 Please upload and process documents using the sidebar to start asking questions.")


def clear_chat_history():
    """
    Clear the chat history from session state.
    useful when switching documents or resetting the conversation.
    """
    st.session_state.chat_history = []  # empty the chat history list
    st.rerun()  # rerun to refresh the display