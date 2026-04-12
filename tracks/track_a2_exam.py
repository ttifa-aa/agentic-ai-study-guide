"""
track A2: comprehensive exam preparation assistant.
specialized track for exam preparation with study plans and progress tracking.
"""

import json  # for serializing progress data
from typing import Dict, List, Optional, Any, Tuple  # type hints for function signatures
from datetime import datetime, timedelta  # for date calculations
from pathlib import Path

from sympy import re

from sympy import re  # for file path handling

# local imports
from core.rag_chain import ChainMode
from tracks.base_track import BaseTrack, TrackFeatures  # base track class
from config.settings import TrackType, ContentType, EXAM_PATTERNS  # exam configuration
from utils.exam_utils import (  # exam utility functions
    analyze_exam_pattern,
    extract_questions_from_paper,
    identify_weak_areas,
    generate_study_plan,
    calculate_topic_mastery,
    recommend_practice_questions,
    track_progress_over_time,
    format_study_plan_display,
    analyze_performance_metrics,
    generate_exam_tips,
    TopicProgress,
    StudyPlan,
    ExamQuestion
)
from prompts.base_prompts import EXAM_PREPARATION_PROMPT, TOPIC_SYNTHESIS_PROMPT  # prompt templates


class TrackA2Exam(BaseTrack):
    """
    Track A2: Comprehensive Exam Preparation Assistant.
    provides exam-focused features with study planning and progress tracking.
    """
    
    def __init__(self):
        """Initialize the exam preparation track."""
        # call parent class initializer
        super().__init__()
        
        # set track type
        self.track_type = TrackType.TRACK_A2_EXAM
        
        # initialize track features
        self.features = self.get_features()
        
        # progress tracking data
        self.topic_progress: Dict[str, TopicProgress] = {}
        self.progress_history: List[Dict] = []
        self.study_plans: List[StudyPlan] = []
        self.extracted_questions: List[ExamQuestion] = []
        
        # exam pattern analysis
        self.detected_exam_pattern: Optional[Dict] = None
        
        # session metrics
        self.session_start_time = datetime.now()
        self.questions_answered = 0
    
    def get_features(self) -> TrackFeatures:
        """
        Get the features and capabilities of the exam track.
        
        Returns:
            TrackFeatures: Exam track features
        """
        return TrackFeatures(
            name="Comprehensive Exam Preparation Assistant",
            description="""
            Exam-focused track with:
            - Complete exam preparation workflows combining all materials
            - Topic-wise question practice with solutions from study materials
            - Weak area identification and targeted content recommendations
            - Custom study plans based on syllabus and available content
            - Progress tracking across topics and question practice
            """,
            supported_content_types=[
                ContentType.LECTURE_NOTES.value,
                ContentType.TEXTBOOK.value,
                ContentType.PAST_PAPER.value,
                ContentType.ASSIGNMENT.value
            ],
            special_prompts={
                "exam_solve": EXAM_PREPARATION_PROMPT,
                "topic_synthesis": TOPIC_SYNTHESIS_PROMPT,
                "study_plan": "Generate a study plan for..."
            },
            analytics_enabled=True,
            progress_tracking=True,
            export_formats=["study_guide", "practice_test", "progress_report", "pdf"]
        )
    
    def get_specialized_prompt(self, prompt_type: str) -> str:
        """
        Get an exam-specific specialized prompt template.
        
        Args:
            prompt_type (str): Type of prompt needed
        
        Returns:
            str: Specialized prompt template
        """
        if prompt_type in self.features.special_prompts:
            return self.features.special_prompts[prompt_type]
        
        return EXAM_PREPARATION_PROMPT
    
    def process_query(
        self,
        query: str,
        query_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user query with exam-specific handling.
        
        Args:
            query (str): User's query or question
            query_type (str): Type of query (solve, plan, analyze, etc.)
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Response with exam-specific metadata
        """
        # detect query category
        detected_category = self._detect_query_category(query)
        
        # increment questions answered counter
        self.questions_answered += 1
        
        # process based on query category
        if detected_category == "solve":
            return self._handle_solve_query(query, **kwargs)
        elif detected_category == "plan":
            return self._handle_plan_query(query, **kwargs)
        elif detected_category == "analyze":
            return self._handle_analyze_query(query, **kwargs)
        elif detected_category == "recommend":
            return self._handle_recommend_query(query, **kwargs)
        else:
            return self._handle_general_exam_query(query, **kwargs)
    
    def _detect_query_category(self, query: str) -> str:
        """
        Detect the category of an exam query.
        
        Args:
            query (str): User's query
        
        Returns:
            str: Query category
        """
        query_lower = query.lower()
        
        # check for solve/explain queries
        solve_keywords = ["solve", "explain", "answer", "how to", "what is", "calculate"]
        if any(keyword in query_lower for keyword in solve_keywords):
            return "solve"
        
        # check for study plan queries
        plan_keywords = ["study plan", "schedule", "prepare for", "revision", "timeline"]
        if any(keyword in query_lower for keyword in plan_keywords):
            return "plan"
        
        # check for analysis queries
        analyze_keywords = ["analyze", "weak", "strength", "progress", "performance", "gap"]
        if any(keyword in query_lower for keyword in analyze_keywords):
            return "analyze"
        
        # check for recommendation queries
        recommend_keywords = ["recommend", "suggest", "what should i", "which topic", "focus on"]
        if any(keyword in query_lower for keyword in recommend_keywords):
            return "recommend"
        
        return "general"
    
    def _handle_solve_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle exam question solving queries.
        
        Args:
            query (str): Question to solve
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Solution with exam-focused explanation
        """
        # detect if this is from an exam paper
        is_exam_question = self._is_exam_question(query)
        
        # use exam preparation prompt
        prompt = self.get_specialized_prompt("exam_solve")
        
        # get answer with exam focus
        answer = self.rag_manager.invoke(
            query,
            mode=ChainMode.EXAM  # use exam mode for step-by-step solutions
        )
        
        # estimate marks for this question
        estimated_marks = self._estimate_question_marks(query)
        
        # track topic progress
        detected_topic = self._detect_question_topic(query)
        self._update_topic_progress(detected_topic, is_correct=None)  # none means attempted but not graded
        
        # prepare metadata
        metadata = {
            "query_category": "solve",
            "is_exam_question": is_exam_question,
            "estimated_marks": estimated_marks,
            "detected_topic": detected_topic,
            "suggested_time_minutes": estimated_marks  # 1 minute per mark guideline
        }
        
        return self.format_response(answer, metadata=metadata)
    
    def _handle_plan_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle study plan generation queries.
        
        Args:
            query (str): Study plan request
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Generated study plan
        """
        # extract parameters from query
        days_available = self._extract_days_available(query) or 14  # default 14 days
        hours_per_day = self._extract_hours_per_day(query) or 3  # default 3 hours
        
        # get topics from progress data
        topics = list(self.topic_progress.keys())
        if not topics:
            # fallback to default topics if no progress data
            topics = ["General Concepts", "Core Topics", "Advanced Topics"]
        
        # detect exam pattern
        exam_pattern = "internal"  # default
        if self.detected_exam_pattern:
            exam_pattern = self.detected_exam_pattern.get("exam_type", "internal")
        
        # generate study plan
        study_plan = generate_study_plan(
            subject=self._extract_subject(query) or "Current Subject",
            topics=topics,
            available_days=days_available,
            hours_per_day=hours_per_day,
            exam_pattern=exam_pattern
        )
        
        # store study plan
        self.study_plans.append(study_plan)
        
        # format for display
        plan_display = format_study_plan_display(study_plan)
        
        # prepare metadata
        metadata = {
            "query_category": "plan",
            "days_available": days_available,
            "hours_per_day": hours_per_day,
            "total_hours": study_plan.total_hours,
            "exam_pattern": exam_pattern,
            "topics_count": len(topics)
        }
        
        return self.format_response(plan_display, metadata=metadata)
    
    def _handle_analyze_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle progress analysis queries.
        
        Args:
            query (str): Analysis request
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Progress analysis
        """
        # identify weak areas
        weak_areas = identify_weak_areas(self.topic_progress)
        
        # track progress over time
        progress_analytics = track_progress_over_time(self.progress_history)
        
        # calculate overall metrics
        total_questions = sum(p.questions_attempted for p in self.topic_progress.values())
        total_correct = sum(p.questions_correct for p in self.topic_progress.values())
        total_time = sum(p.time_spent_minutes for p in self.topic_progress.values())
        
        # build analysis response
        analysis = f"""
## Progress Analysis

### Overall Statistics
- Total Questions Attempted: {total_questions}
- Overall Accuracy: {(total_correct / total_questions * 100) if total_questions > 0 else 0:.1f}%
- Total Study Time: {total_time / 60:.1f} hours
- Progress Trend: {progress_analytics.get('trend', 'N/A')}

### Weak Areas (Need Focus)
"""
        
        if weak_areas:
            for topic, weakness_score in weak_areas[:5]:
                analysis += f"- **{topic}**: Weakness Score {weakness_score:.2f}\n"
        else:
            analysis += "- No significant weak areas identified yet.\n"
        
        analysis += f"\n### Improvement Rate: {progress_analytics.get('improvement_rate', 0) * 100:.1f}%"
        
        # prepare metadata
        metadata = {
            "query_category": "analyze",
            "weak_areas_count": len(weak_areas),
            "total_questions": total_questions,
            "overall_accuracy": (total_correct / total_questions * 100) if total_questions > 0 else 0,
            "trend": progress_analytics.get('trend')
        }
        
        return self.format_response(analysis, metadata=metadata)
    
    def _handle_recommend_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle topic/question recommendation queries.
        
        Args:
            query (str): Recommendation request
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Recommendations
        """
        # identify weak areas
        weak_areas = identify_weak_areas(self.topic_progress)
        
        # organize questions by topic
        questions_by_topic = self._organize_questions_by_topic()
        
        # get recommendations
        recommended_questions = recommend_practice_questions(
            weak_areas,
            questions_by_topic,
            count=5
        )
        
        # generate exam tips
        exam_tips = generate_exam_tips(
            self.detected_exam_pattern.get("exam_type", "internal") if self.detected_exam_pattern else "internal",
            list(self.topic_progress.keys())
        )
        
        # build recommendation response
        response = "## Recommended Focus Areas\n\n"
        
        if weak_areas:
            response += "### Priority Topics (Weak Areas)\n"
            for topic, score in weak_areas[:3]:
                response += f"- **{topic}** (Weakness: {score:.2f})\n"
        
        if recommended_questions:
            response += "\n### Suggested Practice Questions\n"
            for i, q in enumerate(recommended_questions, 1):
                response += f"{i}. {q.question_text} [{q.marks} marks]\n"
        
        if exam_tips:
            response += "\n### Exam Preparation Tips\n"
            for tip in exam_tips[:5]:
                response += f"- {tip}\n"
        
        # prepare metadata
        metadata = {
            "query_category": "recommend",
            "recommendations_count": len(recommended_questions),
            "tips_count": len(exam_tips)
        }
        
        return self.format_response(response, metadata=metadata)
    
    def _handle_general_exam_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle general exam-related queries.
        
        Args:
            query (str): General exam query
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Response with exam context
        """
        # use standard exam mode
        answer = self.rag_manager.invoke(
            query,
            mode=ChainMode.EXAM
        )
        
        # prepare metadata
        metadata = {
            "query_category": "general",
            "questions_answered_session": self.questions_answered,
            "session_duration_minutes": (datetime.now() - self.session_start_time).total_seconds() / 60
        }
        
        return self.format_response(answer, metadata=metadata)
    
    def analyze_exam_paper(self, paper_text: str) -> Dict[str, Any]:
        """
        Analyze an uploaded exam paper for patterns and questions.
        
        Args:
            paper_text (str): Exam paper text content
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        # analyze exam pattern
        self.detected_exam_pattern = analyze_exam_pattern(paper_text)
        
        # extract questions
        questions = extract_questions_from_paper(paper_text)
        self.extracted_questions.extend(questions)
        
        # analyze topic distribution
        topic_distribution = {}
        for q in questions:
            topic_distribution[q.topic] = topic_distribution.get(q.topic, 0) + q.marks
        
        # organize questions by topic
        questions_by_topic = self._organize_questions_by_topic()
        
        return {
            "exam_pattern": self.detected_exam_pattern,
            "total_questions": len(questions),
            "total_marks": self.detected_exam_pattern.get("total_marks", 0),
            "topic_distribution": topic_distribution,
            "questions_by_topic": {
                topic: len(qs) for topic, qs in questions_by_topic.items()
            }
        }
    
    def _update_topic_progress(
        self,
        topic: str,
        is_correct: Optional[bool] = None,
        time_spent: int = 0
    ) -> None:
        """
        Update progress tracking for a topic.
        
        Args:
            topic (str): Topic name
            is_correct (bool, optional): Whether answer was correct
            time_spent (int): Time spent in minutes
        """
        if topic not in self.topic_progress:
            self.topic_progress[topic] = TopicProgress(topic_name=topic)
        
        progress = self.topic_progress[topic]
        progress.questions_attempted += 1
        
        if is_correct is not None and is_correct:
            progress.questions_correct += 1
        
        progress.time_spent_minutes += time_spent
        progress.last_practiced = datetime.now()
        
        # recalculate mastery
        consistency_days = (datetime.now() - self.session_start_time).days
        progress.mastery_level = calculate_topic_mastery(
            progress.questions_attempted,
            progress.questions_correct,
            progress.time_spent_minutes,
            consistency_days
        )
        
        # add to history
        self.progress_history.append({
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "is_correct": is_correct,
            "mastery_score": progress.mastery_level
        })
    
    def _is_exam_question(self, query: str) -> bool:
        """
        Check if query appears to be from an exam paper.
        
        Args:
            query (str): User query
        
        Returns:
            bool: True if likely an exam question
        """
        exam_indicators = [
            r'^\d+[\.\)]',  # starts with number followed by . or )
            r'\[\d+\]',     # contains [marks] format
            r'\(\d+\)',     # contains (marks) format
            r'explain|describe|discuss|solve|calculate',  # exam command words
        ]
        
        query_lower = query.lower()
        for indicator in exam_indicators:
            if re.search(indicator, query, re.IGNORECASE):
                return True
        
        return False
    
    def _estimate_question_marks(self, query: str) -> int:
        """
        Estimate marks allocation for a question.
        
        Args:
            query (str): Question text
        
        Returns:
            int: Estimated marks
        """
        # look for explicit marks indication
        marks_pattern = r'[\[\(](\d+)[\]\)]\s*(?:marks?)?'
        match = re.search(marks_pattern, query.lower())
        if match:
            return int(match.group(1))
        
        # estimate based on question length and complexity
        word_count = len(query.split())
        if word_count < 10:
            return 2  # short answer
        elif word_count < 30:
            return 5  # medium answer
        else:
            return 10  # long answer
    
    def _detect_question_topic(self, query: str) -> str:
        """
        Detect topic from question text.
        
        Args:
            query (str): Question text
        
        Returns:
            str: Detected topic
        """
        from config.settings import CS_SUBJECTS
        
        query_lower = query.lower()
        
        for subject, keywords in CS_SUBJECTS.items():
            if any(keyword in query_lower for keyword in keywords):
                return subject
        
        # check for topics in existing progress data
        for topic in self.topic_progress.keys():
            if topic.lower() in query_lower:
                return topic
        
        return "General"
    
    def _extract_days_available(self, query: str) -> Optional[int]:
        """
        Extract days available from query.
        
        Args:
            query (str): User query
        
        Returns:
            Optional[int]: Days available or None
        """
        patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?',  # will convert to days
            r'in\s+(\d+)\s+days?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                days = int(match.group(1))
                if 'week' in query.lower():
                    days *= 7
                return days
        
        return None
    
    def _extract_hours_per_day(self, query: str) -> Optional[int]:
        """
        Extract hours per day from query.
        
        Args:
            query (str): User query
        
        Returns:
            Optional[int]: Hours per day or None
        """
        patterns = [
            r'(\d+)\s*hours?\s*(?:per|a|each)?\s*day',
            r'(\d+)\s*hrs?\/day'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_subject(self, query: str) -> Optional[str]:
        """
        Extract subject name from query.
        
        Args:
            query (str): User query
        
        Returns:
            Optional[str]: Subject name or None
        """
        from config.settings import CS_SUBJECTS
        
        query_lower = query.lower()
        
        for subject in CS_SUBJECTS.keys():
            if subject.lower() in query_lower:
                return subject
        
        return None
    
    def _organize_questions_by_topic(self) -> Dict[str, List[ExamQuestion]]:
        """
        Organize extracted questions by topic.
        
        Returns:
            Dict[str, List[ExamQuestion]]: Questions grouped by topic
        """
        organized = {}
        
        for question in self.extracted_questions:
            if question.topic not in organized:
                organized[question.topic] = []
            organized[question.topic].append(question)
        
        return organized
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive progress summary.
        
        Returns:
            Dict[str, Any]: Progress summary
        """
        # calculate overall metrics
        metrics = analyze_performance_metrics(
            questions_attempted=sum(p.questions_attempted for p in self.topic_progress.values()),
            questions_correct=sum(p.questions_correct for p in self.topic_progress.values()),
            time_spent_minutes=sum(p.time_spent_minutes for p in self.topic_progress.values()),
            topic_coverage={t: p.questions_attempted for t, p in self.topic_progress.items()}
        )
        
        # identify weak areas
        weak_areas = identify_weak_areas(self.topic_progress)
        
        # track progress trend
        progress_analytics = track_progress_over_time(self.progress_history)
        
        return {
            "metrics": metrics,
            "weak_areas": weak_areas,
            "progress_trend": progress_analytics,
            "topics_mastered": [
                t for t, p in self.topic_progress.items() if p.mastery_level >= 0.8
            ],
            "topics_needing_work": [
                t for t, p in self.topic_progress.items() if p.mastery_level < 0.6
            ],
            "total_study_plans": len(self.study_plans),
            "questions_extracted": len(self.extracted_questions)
        }
    
    def export_progress_report(self) -> str:
        """
        Export progress report as formatted string.
        
        Returns:
            str: Formatted progress report
        """
        summary = self.get_progress_summary()
        
        report = f"""
# Exam Preparation Progress Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance
- Overall Accuracy: {summary['metrics']['overall_accuracy']:.1f}%
- Total Questions: {summary['metrics']['total_questions']}
- Total Study Time: {summary['metrics']['total_time_hours']:.1f} hours

## Topic Mastery
"""
        
        for topic, progress in self.topic_progress.items():
            mastery_percent = progress.mastery_level * 100
            status = "✅ Mastered" if mastery_percent >= 80 else "🟡 In Progress" if mastery_percent >= 60 else "🔴 Needs Work"
            report += f"\n### {topic}\n"
            report += f"- Status: {status} ({mastery_percent:.1f}%)\n"
            report += f"- Questions: {progress.questions_correct}/{progress.questions_attempted} correct\n"
            report += f"- Time Spent: {progress.time_spent_minutes} minutes\n"
        
        if summary['weak_areas']:
            report += "\n## Weak Areas (Priority Focus)\n"
            for topic, score in summary['weak_areas']:
                report += f"- {topic} (Weakness: {score:.2f})\n"
        
        return report