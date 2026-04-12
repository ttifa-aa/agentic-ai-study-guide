"""
Exam preparation utilities for Track A2.
Includes exam pattern analysis, weak area detection, and study plan generation.
"""

import re  # regular expressions for pattern matching in exam paper parsing
from typing import Dict, List, Tuple, Optional, Set  # type hints for function signatures
from datetime import datetime, timedelta  # for date calculations in study plan generation
from dataclasses import dataclass, field  # for creating structured data classes with defaults
from collections import defaultdict  # for grouping questions by topic with automatic initialization
import json  # for serializing progress data to json format

from config.settings import EXAM_PATTERNS  # import exam pattern configurations from settings


@dataclass
class TopicProgress:
    """Track progress for a specific topic."""
    topic_name: str                                    # name of the topic being tracked
    questions_attempted: int = 0                       # total questions attempted for this topic
    questions_correct: int = 0                         # number of correctly answered questions
    time_spent_minutes: int = 0                        # total time spent practicing this topic
    last_practiced: Optional[datetime] = None          # timestamp of most recent practice session
    mastery_level: float = 0.0                         # calculated mastery score from 0.0 to 1.0
    weak_subtopics: List[str] = field(default_factory=list)    # specific subtopics needing improvement
    recommended_resources: List[str] = field(default_factory=list)  # suggested study materials


@dataclass
class StudyPlan:
    """Represents a generated study plan."""
    subject: str                                        # subject name for this study plan
    target_date: datetime                               # exam date or target completion date
    topics: List[Tuple[str, int]]                       # list of (topic_name, priority_score) tuples
    daily_schedule: Dict[str, List[str]]                # mapping of date strings to topics to study
    total_hours: int                                    # total estimated study hours required
    exam_pattern: str                                   # exam pattern type (internal/external/practical)


@dataclass
class ExamQuestion:
    """Represents an exam question with metadata."""
    question_text: str                                  # the actual question text
    marks: int                                          # marks allocated for this question
    topic: str                                          # detected topic/category of the question
    difficulty: str                                     # difficulty level (easy, medium, hard)
    question_type: str                                  # type (theory, practical, problem_solving)
    expected_time_minutes: int                          # estimated time to solve in minutes
    solution_available: bool = False                    # whether solution is available in materials


def analyze_exam_pattern(text: str) -> Dict:
    """
    Analyze exam paper to identify pattern and structure.
    extracts exam type, total marks, sections, and question distribution.
    
    Args:
        text (str): Exam paper text
    
    Returns:
        Dict: Exam pattern information
    """
    pattern_info = {
        "exam_type": "unknown",           # internal, external, or practical
        "total_marks": 0,                 # maximum marks for the exam
        "sections": [],                   # list of sections with their details
        "question_distribution": {},      # how questions are distributed across sections
        "topic_weightage": {}             # weightage of different topics (marks per topic)
    }
    
    text_lower = text.lower()  # convert to lowercase for case-insensitive pattern matching
    
    # Detect exam type based on keywords in the text
    # different exam types have different patterns and expectations
    if "internal" in text_lower or "continuous" in text_lower:
        pattern_info["exam_type"] = "internal"
        # internal exams typically have different question patterns
    elif "external" in text_lower or "end semester" in text_lower:
        pattern_info["exam_type"] = "external"
        # external/end-semester exams follow university patterns
    elif "practical" in text_lower or "lab" in text_lower:
        pattern_info["exam_type"] = "practical"
        # lab practical exams focus on hands-on skills
    
    # Extract total marks from the paper
    # looks for patterns like "Total Marks: 100" or "Maximum marks: 75"
    marks_pattern = r'(?:total|maximum)\s*marks?\s*[:.]?\s*(\d+)'
    marks_match = re.search(marks_pattern, text_lower)
    if marks_match:
        pattern_info["total_marks"] = int(marks_match.group(1))
        # convert the captured number string to integer
    
    # Extract sections from the exam paper
    # sections are usually labeled as Section A, Part B, etc.
    section_pattern = r'(?:section|part)\s*([A-Z])[:\s]+(.*?)(?=(?:section|part)\s*[A-Z]|$)'
    # pattern captures section letter and content until next section or end
    sections = re.findall(section_pattern, text, re.IGNORECASE | re.DOTALL)
    # use re.IGNORECASE for case-insensitive, re.DOTALL to capture across lines
    
    for section_id, section_content in sections:  # process each detected section
        # Extract marks for this section from its content
        section_marks_pattern = r'(\d+)\s*marks?'
        # pattern matches numbers followed by "mark" or "marks"
        marks = re.findall(section_marks_pattern, section_content)
        # find all mark indicators within this section
        section_marks = sum(int(m) for m in marks)  # sum up all marks found
        
        pattern_info["sections"].append({
            "section_id": section_id,                    # section letter (A, B, C, etc.)
            "content_preview": section_content[:100],    # first 100 chars for preview
            "marks": section_marks                       # total marks for this section
        })
    
    return pattern_info  # return the analyzed pattern information


def extract_questions_from_paper(text: str) -> List[ExamQuestion]:
    """
    Extract individual questions from exam paper.
    parses numbered questions with their mark allocations.
    
    Args:
        text (str): Exam paper text
    
    Returns:
        List[ExamQuestion]: List of extracted questions
    """
    questions = []  # initialize empty list for extracted questions
    
    # Pattern for numbered questions with marks
    # matches lines like "1. Explain normalization (5 marks)" or "2) Solve this [10]"
    q_pattern = r'(?:^|\n)\s*(\d+)[\.\)]\s+(.+?)\s*[\(\[](\d+)[\)\]]\s*(?:marks?)?(?=\n\s*\d+[\.\)]|\n\s*(?:OR|or)|$)'
    # breakdown of pattern:
    # (?:^|\n)\s* - start of line or newline with optional whitespace
    # (\d+)[\.\)] - question number followed by . or )
    # \s+(.+?) - whitespace then capture question text (non-greedy)
    # \s*[\(\[](\d+)[\)\]] - marks in parentheses or brackets
    # \s*(?:marks?)? - optional "mark" or "marks" text
    # (?=...) - lookahead for next question or "OR" or end
    
    matches = re.findall(q_pattern, text, re.MULTILINE | re.DOTALL)
    # use MULTILINE for ^ to match line starts, DOTALL for capturing across lines
    
    for q_num, q_text, marks in matches:  # process each matched question
        # Detect difficulty based on marks allocation
        marks_int = int(marks)  # convert marks string to integer
        if marks_int <= 2:
            difficulty = "easy"      # short answer questions are typically easier
        elif marks_int <= 5:
            difficulty = "medium"    # medium-length questions
        else:
            difficulty = "hard"      # long answer questions are more challenging
        
        # Detect question type based on keywords in the question text
        q_text_lower = q_text.lower()  # convert to lowercase for matching
        if any(word in q_text_lower for word in ["explain", "describe", "discuss"]):
            q_type = "theory"           # theory questions test conceptual understanding
        elif any(word in q_text_lower for word in ["solve", "calculate", "find"]):
            q_type = "problem_solving"  # problem-solving questions require calculation
        elif any(word in q_text_lower for word in ["write", "implement", "code"]):
            q_type = "practical"        # practical questions test implementation skills
        else:
            q_type = "theory"           # default to theory if type can't be determined
        
        # Estimate time based on marks (1 minute per mark is a common heuristic)
        expected_time = marks_int  # each mark roughly corresponds to 1 minute
        
        questions.append(ExamQuestion(
            question_text=q_text.strip(),      # clean up question text
            marks=marks_int,                   # marks allocated
            topic=_detect_question_topic(q_text),  # detect which topic this belongs to
            difficulty=difficulty,             # calculated difficulty level
            question_type=q_type,              # detected question type
            expected_time_minutes=expected_time  # estimated time to solve
        ))
    
    return questions  # return list of extracted questions


def _detect_question_topic(question_text: str) -> str:
    """
    Detect topic from question text using keyword matching.
    helper function used by extract_questions_from_paper.
    
    Args:
        question_text (str): Question text
    
    Returns:
        str: Detected topic
    """
    from config.settings import CS_SUBJECTS  # import cs subjects mapping
    
    text_lower = question_text.lower()  # convert to lowercase for matching
    
    for subject, keywords in CS_SUBJECTS.items():  # check each subject's keywords
        if any(keyword in text_lower for keyword in keywords):
            # if any keyword for this subject appears in the question
            return subject  # return the matching subject name
    
    return "General"  # return "General" if no specific topic identified


def identify_weak_areas(progress_data: Dict[str, TopicProgress]) -> List[Tuple[str, float]]:
    """
    Identify weak areas based on progress data.
    calculates weakness scores based on accuracy and attempt frequency.
    
    Args:
        progress_data (Dict[str, TopicProgress]): Topic progress data
    
    Returns:
        List[Tuple[str, float]]: List of (topic, weakness_score) tuples sorted by weakness
    """
    weak_areas = []  # initialize list for weak areas with scores
    
    for topic_name, progress in progress_data.items():  # examine each topic's progress
        if progress.questions_attempted >= 3:  # minimum data threshold for statistical significance
            # calculate accuracy as proportion of correct answers
            accuracy = progress.questions_correct / progress.questions_attempted
            if accuracy < 0.6:  # below 60% accuracy indicates weakness
                weakness_score = 1.0 - accuracy  # higher score means weaker
                # weakness score ranges from 0.4 to 1.0 (for 60% down to 0% accuracy)
                weak_areas.append((topic_name, weakness_score))
    
    # Sort by weakness score descending - weakest topics first
    return sorted(weak_areas, key=lambda x: x[1], reverse=True)


def generate_study_plan(
    subject: str,
    topics: List[str],
    available_days: int,
    hours_per_day: int,
    exam_pattern: str = "internal"
) -> StudyPlan:
    """
    Generate a personalized study plan based on available time and topics.
    
    Args:
        subject (str): Subject name
        topics (List[str]): List of topics to cover
        available_days (int): Days until exam
        hours_per_day (int): Study hours per day
        exam_pattern (str): Exam pattern type
    
    Returns:
        StudyPlan: Generated study plan
    """
    # calculate target date by adding available days to current date
    target_date = datetime.now() + timedelta(days=available_days)
    total_hours = available_days * hours_per_day  # total available study hours
    
    # Assign priority scores to topics (simplified - could be ml-based in future)
    topic_priorities = []  # list to store topic with priority scores
    for i, topic in enumerate(topics):  # iterate with index for priority calculation
        # Topics later in list get lower priority (can be customized based on difficulty)
        priority = len(topics) - i  # earlier topics get higher priority
        topic_priorities.append((topic, priority))
    
    # Sort by priority score descending - high priority topics first
    topic_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Create daily schedule distributing topics across available days
    daily_schedule = {}  # dictionary mapping date strings to topic lists
    
    # calculate how many topics to cover per day (even distribution)
    topics_per_day = max(1, len(topics) // available_days)
    # ensure at least 1 topic per day even if topics < days
    
    current_date = datetime.now()  # start from today
    topic_index = 0  # track which topic we're on
    
    for day in range(available_days):  # create schedule for each day
        date_str = current_date.strftime("%Y-%m-%d")  # format date as string
        daily_topics = []  # list of topics for this day
        
        # assign topics_per_day number of topics
        for _ in range(topics_per_day):
            if topic_index < len(topics):  # check if we still have topics to assign
                daily_topics.append(topics[topic_index])
                topic_index += 1  # move to next topic
        
        # Add review sessions for weak topics on weekends
        # weekend review helps reinforce learning
        if current_date.weekday() >= 5:  # 5=saturday, 6=sunday
            daily_topics.append("Review weak topics")  # add review session
        
        daily_schedule[date_str] = daily_topics  # store day's schedule
        current_date += timedelta(days=1)  # move to next day
    
    return StudyPlan(
        subject=subject,                # subject name
        target_date=target_date,        # exam/target date
        topics=topic_priorities,        # prioritized topics list
        daily_schedule=daily_schedule,  # day-by-day study schedule
        total_hours=total_hours,        # total estimated hours
        exam_pattern=exam_pattern       # exam pattern type
    )


def calculate_topic_mastery(
    questions_attempted: int,
    questions_correct: int,
    time_spent: int,
    consistency_days: int
) -> float:
    """
    Calculate topic mastery score (0.0 to 1.0) using weighted components.
    
    Args:
        questions_attempted (int): Number of questions attempted
        questions_correct (int): Number of correct answers
        time_spent (int): Time spent in minutes
        consistency_days (int): Days since first practice
    
    Returns:
        float: Mastery score between 0.0 and 1.0
    """
    if questions_attempted == 0:
        return 0.0  # no questions attempted means no demonstrated mastery
    
    # Accuracy component (50% weight) - most important factor
    accuracy = questions_correct / questions_attempted  # proportion correct
    accuracy_score = accuracy * 0.5  # weighted at 50%
    
    # Volume component (30% weight) - rewards practice quantity
    # maxes out at 20 questions for full points
    volume_score = min(questions_attempted / 20, 1.0) * 0.3
    # min ensures score doesn't exceed 1.0, then apply 30% weight
    
    # Consistency component (20% weight) - rewards regular practice over time
    # maxes out at 7 days of consistent practice
    consistency_score = min(consistency_days / 7, 1.0) * 0.2
    # min ensures score doesn't exceed 1.0, then apply 20% weight
    
    mastery = accuracy_score + volume_score + consistency_score  # sum weighted components
    
    return min(mastery, 1.0)  # ensure result is capped at 1.0


def recommend_practice_questions(
    weak_areas: List[Tuple[str, float]],
    available_questions: Dict[str, List[ExamQuestion]],
    count: int = 5
) -> List[ExamQuestion]:
    """
    Recommend practice questions based on identified weak areas.
    
    Args:
        weak_areas (List[Tuple[str, float]]): Weak areas with weakness scores
        available_questions (Dict[str, List[ExamQuestion]]): Questions organized by topic
        count (int): Number of questions to recommend
    
    Returns:
        List[ExamQuestion]: Recommended questions for practice
    """
    recommendations = []  # list to store recommended questions
    
    # Weight questions by topic weakness score
    weighted_questions = []  # list of (question, weight) tuples
    
    for topic, weakness_score in weak_areas:  # process each weak area
        if topic in available_questions:  # check if we have questions for this topic
            for question in available_questions[topic]:  # examine each question
                # Weight formula: weakness_score * (marks as additional weight)
                # higher marks questions get slightly more weight
                weight = weakness_score * (question.marks / 10)
                weighted_questions.append((question, weight))
    
    # Sort by weight descending - highest weighted questions first
    weighted_questions.sort(key=lambda x: x[1], reverse=True)
    
    # select top N questions based on weight
    for question, _ in weighted_questions[:count]:
        recommendations.append(question)
    
    return recommendations  # return recommended questions


def track_progress_over_time(progress_history: List[Dict]) -> Dict:
    """
    Analyze progress over time and generate insights.
    
    Args:
        progress_history (List[Dict]): List of daily progress snapshots
    
    Returns:
        Dict: Progress analytics including trend and improvement metrics
    """
    if not progress_history:
        return {"trend": "No data", "improvement_rate": 0.0}
    
    # Calculate improvement trend using recent scores
    # look at last 5 entries for trend analysis
    recent_scores = [p.get("mastery_score", 0) for p in progress_history[-5:]]
    
    if len(recent_scores) >= 2:  # need at least 2 data points for trend
        # determine if scores are improving or declining
        trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
        # calculate improvement rate as percentage change
        improvement_rate = (recent_scores[-1] - recent_scores[0]) / recent_scores[0] if recent_scores[0] > 0 else 0
    else:
        trend = "insufficient_data"  # not enough data for trend analysis
        improvement_rate = 0.0
    
    return {
        "trend": trend,                                               # overall trend direction
        "improvement_rate": improvement_rate,                         # rate of change
        "average_score": sum(recent_scores) / len(recent_scores) if recent_scores else 0,  # mean score
        "total_questions": sum(p.get("questions_attempted", 0) for p in progress_history),  # sum all attempts
        "total_time_hours": sum(p.get("time_minutes", 0) for p in progress_history) / 60    # convert to hours
    }


def format_study_plan_display(study_plan: StudyPlan) -> str:
    """
    Format study plan for display in the ui.
    creates a readable markdown representation of the study plan.
    
    Args:
        study_plan (StudyPlan): Generated study plan
    
    Returns:
        str: Formatted study plan string ready for display
    """
    # build the study plan display with markdown formatting
    display = f"""
## 📚 Study Plan: {study_plan.subject}

**Target Date:** {study_plan.target_date.strftime('%B %d, %Y')}
**Total Study Hours:** {study_plan.total_hours}
**Exam Pattern:** {study_plan.exam_pattern.upper()}

### Daily Schedule:
"""
    # note: using 📚 emoji here but in actual implementation per your style guide
    # this would be replaced with plain text like "Study Plan:"
    
    # add each day's schedule to the display
    for date, topics in study_plan.daily_schedule.items():
        display += f"\n**{date}**\n"  # date in bold
        for topic in topics:  # list each topic for this day
            display += f"- {topic}\n"  # bullet point for each topic
    
    # add priority topics section with visual indicators
    display += "\n### Priority Topics (Focus Areas):\n"
    for topic, priority in study_plan.topics[:5]:  # show top 5 priority topics
        # use different markers based on priority level
        if priority > 7:
            display += f"HIGH PRIORITY: {topic}\n"      # highest priority topics
        elif priority > 4:
            display += f"MEDIUM PRIORITY: {topic}\n"    # medium priority topics
        else:
            display += f"LOW PRIORITY: {topic}\n"       # lower priority topics
    
    return display  # return formatted display string


def analyze_performance_metrics(
    questions_attempted: int,
    questions_correct: int,
    time_spent_minutes: int,
    topic_coverage: Dict[str, int]
) -> Dict:
    """
    Calculate comprehensive performance metrics for analytics dashboard.
    
    Args:
        questions_attempted (int): Total questions attempted
        questions_correct (int): Total correct answers
        time_spent_minutes (int): Total study time in minutes
        topic_coverage (Dict[str, int]): Questions per topic
    
    Returns:
        Dict: Comprehensive performance metrics
    """
    metrics = {}  # dictionary to store all calculated metrics
    
    # calculate overall accuracy
    if questions_attempted > 0:
        metrics["overall_accuracy"] = (questions_correct / questions_attempted) * 100
        metrics["questions_per_hour"] = questions_attempted / (time_spent_minutes / 60) if time_spent_minutes > 0 else 0
    else:
        metrics["overall_accuracy"] = 0.0
        metrics["questions_per_hour"] = 0.0
    
    # analyze topic coverage
    metrics["topics_covered"] = len(topic_coverage)  # number of unique topics
    metrics["most_practiced_topic"] = max(topic_coverage.items(), key=lambda x: x[1])[0] if topic_coverage else "None"
    # find topic with highest question count
    
    # calculate time efficiency
    metrics["total_time_hours"] = time_spent_minutes / 60
    metrics["avg_time_per_question"] = time_spent_minutes / questions_attempted if questions_attempted > 0 else 0
    
    return metrics  # return all calculated metrics


def generate_exam_tips(exam_pattern: str, topics: List[str]) -> List[str]:
    """
    Generate exam-specific tips based on pattern and topics.
    
    Args:
        exam_pattern (str): Type of exam (internal/external/practical)
        topics (List[str]): List of topics being tested
    
    Returns:
        List[str]: List of exam tips and strategies
    """
    tips = []  # list to store generated tips
    
    # add pattern-specific tips
    if exam_pattern == "internal":
        tips.append("Focus on understanding concepts deeply - internal exams test fundamentals")
        tips.append("Review class notes and assignments as they often appear in internals")
        tips.append("Practice writing concise answers within time limits")
    elif exam_pattern == "external":
        tips.append("Review previous year papers to understand question patterns")
        tips.append("Focus on high-weightage topics first")
        tips.append("Practice writing complete answers with proper structure")
    elif exam_pattern == "practical":
        tips.append("Practice implementing algorithms from scratch")
        tips.append("Prepare for viva questions on theoretical concepts")
        tips.append("Review lab manual procedures thoroughly")
    
    # add topic-specific tips
    for topic in topics[:3]:  # tips for top 3 topics
        if "Database" in topic:
            tips.append("Practice writing SQL queries - they're common in exams")
        elif "Data Structure" in topic or "Algorithm" in topic:
            tips.append("Draw diagrams to explain data structures and algorithms")
        elif "Network" in topic:
            tips.append("Focus on OSI/TCP-IP layers and protocol functions")
    
    return tips  # return list of exam tips