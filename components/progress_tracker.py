"""
Progress tracker component for the Academic Assistant.
Displays learning progress, weak areas, and track-specific dashboards.
Different dashboards are shown based on the active track.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, List

from config.settings import TrackType

def render_exam_progress_dashboard():
    """
    Render progress tracking dashboard for exam track (Track A2).
    shows comprehensive metrics, weak areas, study plans, and export options.
    this dashboard helps students track their exam preparation progress.
    """
    track = st.session_state.get("current_track")
    
    if not track or not hasattr(track, 'get_progress_summary'):
        return
    
    st.markdown("---")
    st.header("Progress Dashboard")
    
    summary = track.get_progress_summary()
    metrics = summary.get("metrics", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Accuracy",
            f"{metrics.get('overall_accuracy', 0):.1f}%",
            help="Percentage of questions answered correctly"
        )
    
    with col2:
        st.metric(
            "Questions Attempted",
            metrics.get("total_questions", 0),
            help="Total number of questions practiced"
        )
    
    with col3:
        st.metric(
            "Topics Covered",
            metrics.get("topics_covered", 0),
            help="Number of unique topics practiced"
        )
    
    with col4:
        st.metric(
            "Study Time",
            f"{metrics.get('total_time_hours', 0):.1f} hrs",
            help="Total time spent practicing"
        )
    
    weak_areas = summary.get("weak_areas", [])
    if weak_areas:
        st.subheader("Focus Areas (Weak Topics)")
        for topic, weakness_score in weak_areas[:5]:
            st.progress(
                1.0 - weakness_score,
                text=f"{topic} (Weakness: {weakness_score:.2f})"
            )
    else:
        st.info("No significant weak areas identified yet. Keep practicing!")
    
    topics_mastered = summary.get("topics_mastered", [])
    topics_needing_work = summary.get("topics_needing_work", [])
    
    if topics_mastered or topics_needing_work:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Mastered Topics")
            if topics_mastered:
                for topic in topics_mastered:
                    st.caption(f"• {topic}")
            else:
                st.caption("No topics mastered yet")
        
        with col2:
            st.subheader("📚 Topics Needing Work")
            if topics_needing_work:
                for topic in topics_needing_work:
                    st.caption(f"• {topic}")
            else:
                st.caption("All topics on track!")
    
    progress_trend = summary.get("progress_trend", {})
    if progress_trend:
        trend = progress_trend.get("trend", "insufficient_data")
        improvement = progress_trend.get("improvement_rate", 0) * 100
        
        st.subheader("Progress Trend")
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            if trend == "improving":
                st.success(f"📈 {trend.title()} ({improvement:.1f}% improvement)")
            elif trend == "declining":
                st.warning(f"📉 {trend.title()}")
            else:
                st.info(f"📊 {trend.replace('_', ' ').title()}")
        
        with trend_col2:
            st.metric("Average Score", f"{progress_trend.get('average_score', 0):.1%}")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Progress Report", use_container_width=True):
            report = track.export_progress_report()
            st.download_button(
                "Download Report",
                report,
                file_name=f"progress_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                key="download_report"
            )
    
    with col2:
        if st.button("📅 Generate Study Plan", use_container_width=True):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Generate a study plan for me based on my progress"
            })
            st.rerun()
    
    with col3:
        if st.button("🎯 Recommend Practice", use_container_width=True):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "What topics should I focus on? Recommend practice questions for my weak areas"
            })
            st.rerun()

def _render_cs_dashboard_content():
    """
    Internal function to render CS dashboard content for Track A1.
    shows detected code blocks, algorithms, and subject analysis.
    """
    track = st.session_state.get("current_track")
    if not track:
        return
    
    st.markdown("---")
    st.header("CS Subject Analysis")
    
    summary = track.get_cs_subject_summary()
    subjects = summary.get("identified_subjects", {})
    
    if subjects:
        st.subheader("Detected CS Subjects")
        sorted_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)
        for subject, confidence in sorted_subjects:
            st.progress(
                confidence,
                text=f"{subject} (Confidence: {confidence:.1%})"
            )
    else:
        st.info("No CS subjects detected yet. Upload more CS-related documents.")
    
    primary_subject = summary.get("primary_subject")
    if primary_subject:
        st.success(f"📌 Primary Subject: **{primary_subject}**")
    
    st.subheader("Content Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Code Blocks",
            summary.get("code_blocks_found", 0),
            help="Number of code snippets detected in documents"
        )
    
    with col2:
        st.metric(
            "Algorithms",
            summary.get("algorithms_detected", 0),
            help="Number of algorithms identified"
        )
    
    with col3:
        st.metric(
            "Total Subjects",
            summary.get("total_subjects", 0),
            help="Number of distinct CS subjects identified"
        )
    
    if hasattr(track, 'detected_algorithms') and track.detected_algorithms:
        st.subheader("Detected Algorithms")
        for algo in track.detected_algorithms[:5]:
            with st.expander(f"🔍 {algo.name}"):
                st.caption(f"**Type:** {algo.algorithm_type.value}")
                st.caption(f"**Time Complexity:** {algo.complexity_time}")
                st.caption(f"**Space Complexity:** {algo.complexity_space}")
                if algo.steps:
                    st.caption("**Steps:**")
                    for i, step in enumerate(algo.steps[:5], 1):
                        st.caption(f" {i}. {step}")
    
    if hasattr(track, 'detected_code_blocks') and track.detected_code_blocks:
        st.subheader("Code Language Distribution")
        lang_counts = {}
        for block in track.detected_code_blocks:
            lang = block.language
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
            st.caption(f"• {lang}: {count} block(s)")

def render_cs_dashboard():
    """
    Wrapper for CS dashboard that checks track type before rendering.
    called from the main app for Track A1.
    """
    if st.session_state.get("track_type") == TrackType.TRACK_A1_CS:
        track = st.session_state.get("current_track")
        if track and hasattr(track, 'get_cs_subject_summary'):
            _render_cs_dashboard_content()

def render_progress_dashboard():
    """
    Main progress dashboard router.
    renders the appropriate dashboard based on active track type.
    this is the entry point called from the main app.
    """
    track_type = st.session_state.get("track_type")
    
    if track_type == TrackType.TRACK_A2_EXAM:
        render_exam_progress_dashboard()
    elif track_type == TrackType.TRACK_A1_CS:
        render_cs_dashboard()