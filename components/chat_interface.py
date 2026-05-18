"""
Chat interface component for the Academic Assistant.
Handles user queries, chat history, and track-specific response generation.
"""

import streamlit as st
from datetime import datetime
from typing import Any, Dict


def render_mode_selector():
    """
    Render the study/exam mode selector.
    """
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Study Mode", type="secondary", use_container_width=True):
            st.session_state.current_mode = "Study Mode"
            st.experimental_rerun()
    with col2:
        if st.button("Exam Mode", type="secondary", use_container_width=True):
            st.session_state.current_mode = "Exam Mode"
            st.experimental_rerun()

    mode = st.session_state.get("current_mode", "Study Mode")
    st.caption(f"Current Mode: {mode}")


def render_welcome_message():
    """
    Render a welcome message for the active track.
    """
    track = st.session_state.get("current_track")
    if track:
        st.markdown(f"### {track.get_welcome_message()}")
    else:
        st.info("Select a track and upload documents to start your session.")


def render_chat_history():
    """
    Render the conversation history in the main chat area.
    """
    history = st.session_state.get("chat_history", [])
    if not history:
        st.info("No messages yet. Ask a question to get started.")
        return

    for message in history:
        role = message.get("role", "user")
        content = message.get("content", "")
        timestamp = message.get("timestamp", "")

        if role == "assistant":
            st.markdown(f"**Assistant**  \n{timestamp}")
            st.write(content)
        else:
            st.markdown(f"**You**  \n{timestamp}")
            st.write(content)


def generate_response(query: str) -> Dict[str, Any]:
    """
    Generate a response from the active track.
    """
    track = st.session_state.get("current_track")
    if not track:
        return {
            "answer": "Please select a track first.",
            "track_type": "none",
            "sources": []
        }

    try:
        response = track.process_query(query)
        if isinstance(response, dict):
            return response
        return {"answer": str(response), "track_type": track.track_type.value if hasattr(track, 'track_type') else 'unknown', "sources": []}
    except Exception as exc:
        return {"answer": f"Error generating response: {exc}", "track_type": track.track_type.value if hasattr(track, 'track_type') else 'unknown', "sources": []}


def clear_chat_history():
    """
    Clear the conversation history.
    """
    st.session_state.chat_history = []
    st.experimental_rerun()


def render_chat_interface():
    """
    Render the main chat interface.
    """
    st.markdown("---")
    st.header("Academic Assistant Chat")
    render_mode_selector()

    if not st.session_state.get("chat_history"):
        render_welcome_message()

    if st.session_state.get("documents_processed"):
        st.success("Document context is available for your questions.")
    else:
        st.info("Upload documents to enable context-aware answers.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your question",
            placeholder="Ask a question about your uploaded documents or course content...",
            height=120
        )
        submitted = st.form_submit_button("Send")

        if submitted and user_input.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input.strip(),
                "timestamp": timestamp
            })

            response = generate_response(user_input.strip())
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.get("answer", "I could not generate a response."),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    if st.button("Clear Chat", type="secondary", use_container_width=True):
        clear_chat_history()

    render_chat_history()
