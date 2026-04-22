"""
Progress tracker component for the Academic Assistant.
Displays learning progress, weak areas, and track-specific dashboards.
Different dashboards are shown based on the active track.
"""

import streamlit as st  # streamlit for building ui components
from datetime import datetime  # for timestamp formatting in exports
from typing import Dict, Any, List  # type hints for function signatures

# local imports from our custom modules
from config.settings import TrackType  # track type enumeration for routing


def render_exam_progress_dashboard():
    """
    Render progress tracking dashboard for exam track (Track A2).
    shows comprehensive metrics, weak areas, study plans, and export options.
    this dashboard helps students track their exam preparation progress.
    """
    # get the current track from session state
    track = st.session_state.get("current_track")
    
    # verify track exists and has the required progress summary method
    if not track or not hasattr(track, 'get_progress_summary'):
        return
    
    st.markdown("---")  # visual separator from chat interface
    st.header("Progress Dashboard")  # main dashboard header
    
    # get comprehensive progress summary from the track
    summary = track.get_progress_summary()
    
    # display key metrics in a row of 4 columns
    metrics = summary.get("metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # overall accuracy - percentage of correct answers
        st.metric(
            "Overall Accuracy",
            f"{metrics.get('overall_accuracy', 0):.1f}%",
            help="Percentage of questions answered correctly"
        )
    
    with col2:
        # total questions attempted across all topics
        st.metric(
            "Questions Attempted",
            metrics.get("total_questions", 0),
            help="Total number of questions practiced"
        )
    
    with col3:
        # number of unique topics covered
        st.metric(
            "Topics Covered",
            metrics.get("topics_covered", 0),
            help="Number of unique topics practiced"
        )
    
    with col4:
        # total study time in hours
        st.metric(
            "Study Time",
            f"{metrics.get('total_time_hours', 0):.1f} hrs",
            help="Total time spent practicing"
        )
    
    # weak areas section - topics that need more attention
    weak_areas = summary.get("weak_areas", [])
    if weak_areas:
        st.subheader("Focus Areas (Weak Topics)")  # section header
        
        # display top 5 weakest areas with progress bars
        # higher weakness score = more attention needed
        for topic, weakness_score in weak_areas[:5]:
            # invert: high weakness = low bar fill, so bar reads as "completion" not "weakness"
            st.progress(
                1.0 - weakness_score,
                text=f"{topic} (Weakness: {weakness_score:.2f})"
            )
    else:
        # no weak areas identified yet
        st.info("No significant weak areas identified yet. Keep practicing!")
    
    # topic mastery section - split into mastered and needing work
    topics_mastered = summary.get("topics_mastered", [])
    topics_needing_work = summary.get("topics_needing_work", [])
    
    if topics_mastered or topics_needing_work:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Mastered Topics")
            if topics_mastered:
                # list all topics with mastery level >= 80%
                for topic in topics_mastered:
                    st.caption(f"• {topic}")
            else:
                st.caption("No topics mastered yet")
        
        with col2:
            st.subheader("📚 Topics Needing Work")
            if topics_needing_work:
                # list all topics with mastery level < 60%
                for topic in topics_needing_work:
                    st.caption(f"• {topic}")
            else:
                st.caption("All topics on track!")
    
    # progress trend analysis - shows improvement over time
    progress_trend = summary.get("progress_trend", {})
    if progress_trend:
        trend = progress_trend.get("trend", "insufficient_data")  # improving/declining/insufficient_data
        improvement = progress_trend.get("improvement_rate", 0) * 100  # convert to percentage
        
        st.subheader("Progress Trend")
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            # show trend with appropriate icon and color
            if trend == "improving":
                st.success(f"📈 {trend.title()} ({improvement:.1f}% improvement)")
            elif trend == "declining":
                st.warning(f"📉 {trend.title()}")
            else:
                st.info(f"📊 {trend.replace('_', ' ').title()}")
        
        with trend_col2:
            # average mastery score across all topics
            st.metric("Average Score", f"{progress_trend.get('average_score', 0):.1%}")
    
    # action buttons - export, study plan, recommendations
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # export progress report button
        if st.button("📄 Export Progress Report", use_container_width=True):
            # generate report from track
            report = track.export_progress_report()
            
            # provide download button for the report
            st.download_button(
                "Download Report",
                report,
                file_name=f"progress_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                key="download_report"  # unique key for button
            )
    
    with col2:
        # generate study plan button
        if st.button("📅 Generate Study Plan", use_container_width=True):
            # trigger study plan generation through chat interface
            # adds a message to chat history that will be processed
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Generate a study plan for me based on my progress"
            })
            st.rerun()  # rerun to process the message
    
    with col3:
        # recommend practice button
        if st.button("🎯 Recommend Practice", use_container_width=True):
            # trigger recommendation through chat interface
            st.session_state.chat_history.append({
                "role": "user",
                "content": "What topics should I focus on? Recommend practice questions for my weak areas"
            })
            st.rerun()  # rerun to process the message


def _render_cs_dashboard_content():
    """
    Internal function to render CS dashboard content for Track A1.
    shows detected code blocks, algorithms, and subject analysis.
    """
    # get the current track from session state
    track = st.session_state.get("current_track")
    if not track:
        return
    
    st.markdown("---")  # visual separator from chat interface
    st.header("CS Subject Analysis")  # main dashboard header
    
    # get cs subject summary from track
    summary = track.get_cs_subject_summary()
    
    # detected subjects section
    subjects = summary.get("identified_subjects", {})
    if subjects:
        st.subheader("Detected CS Subjects")  # section header
        
        # sort subjects by confidence score (highest first)
        sorted_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)
        
        # display each subject with confidence progress bar
        for subject, confidence in sorted_subjects:
            # confidence ranges from 0.0 to 1.0
            st.progress(
                confidence,
                text=f"{subject} (Confidence: {confidence:.1%})"
            )
    else:
        # no cs subjects detected in documents yet
        st.info("No CS subjects detected yet. Upload more CS-related documents.")
    
    # highlight the primary subject (highest confidence)
    primary_subject = summary.get("primary_subject")
    if primary_subject:
        st.success(f"📌 Primary Subject: **{primary_subject}**")
    
    # content analysis statistics
    st.subheader("Content Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # number of code blocks found in documents
        st.metric(
            "Code Blocks",
            summary.get("code_blocks_found", 0),
            help="Number of code snippets detected in documents"
        )
    
    with col2:
        # number of algorithms identified
        st.metric(
            "Algorithms",
            summary.get("algorithms_detected", 0),
            help="Number of algorithms identified"
        )
    
    with col3:
        # number of distinct cs subjects identified
        st.metric(
            "Total Subjects",
            summary.get("total_subjects", 0),
            help="Number of distinct CS subjects identified"
        )
    
    # detected algorithms section - show if any algorithms were found
    if hasattr(track, 'detected_algorithms') and track.detected_algorithms:
        st.subheader("Detected Algorithms")
        
        # display first 5 algorithms in expandable sections
        for algo in track.detected_algorithms[:5]:
            with st.expander(f"🔍 {algo.name}"):
                st.caption(f"**Type:** {algo.algorithm_type.value}")
                st.caption(f"**Time Complexity:** {algo.complexity_time}")
                st.caption(f"**Space Complexity:** {algo.complexity_space}")
                
                # show algorithm steps if available
                if algo.steps:
                    st.caption("**Steps:**")
                    for i, step in enumerate(algo.steps[:5], 1):
                        st.caption(f"  {i}. {step}")
    
    # code language distribution section
    if hasattr(track, 'detected_code_blocks') and track.detected_code_blocks:
        st.subheader("Code Language Distribution")
        
        # count occurrences of each programming language
        lang_counts = {}
        for block in track.detected_code_blocks:
            lang = block.language
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # display language distribution
        for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
            st.caption(f"• {lang}: {count} block(s)")


def render_cs_dashboard():
    """
    Wrapper for CS dashboard that checks track type before rendering.
    called from the main app for Track A1.
    """
    # only render if current track is A1 (CS track)
    if st.session_state.get("track_type") == TrackType.TRACK_A1_CS:
        track = st.session_state.get("current_track")
        # verify track has required method
        if track and hasattr(track, 'get_cs_subject_summary'):
            _render_cs_dashboard_content()  # call the actual render function


def render_progress_dashboard():
    """
    Main progress dashboard router.
    renders the appropriate dashboard based on active track type.
    this is the entry point called from the main app.
    """
    track_type = st.session_state.get("track_type")
    
    # route to appropriate dashboard based on track
    if track_type == TrackType.TRACK_A2_EXAM:
        render_exam_progress_dashboard()  # exam preparation dashboard
    elif track_type == TrackType.TRACK_A1_CS:
        render_cs_dashboard()  # cs subject analysis dashboard
    # if no track or unknown track, don't render anything