"""
Database manager for the Academic Assistant.
Handles SQLite database operations for progress tracking and data persistence.
stores learning progress, study plans, exam questions, and session data.
"""

import sqlite3  # built-in sqlite3 module for database operations
import json  # for serializing complex data structures to json format
import os  # for file path operations and directory creation
from datetime import datetime  # for timestamp handling in database records
from typing import Dict, List, Optional, Any, Tuple  # type hints for function signatures
from pathlib import Path  # object-oriented file path handling
from contextlib import contextmanager  # for creating context managers to manage connections


class DatabaseManager:
    """
    Manages SQLite database operations for the Academic Assistant.
    provides persistent storage for learning progress and user data.
    uses sqlite for lightweight, file-based storage without external dependencies.
    """
    
    def __init__(self, db_path: str = "data/progress.db"):
        """
        Initialize the database manager with path to sqlite file.
        
        Args:
            db_path (str): Path to the SQLite database file (default: data/progress.db)
        """
        self.db_path = db_path  # store database path for connection creation
        
        # ensure the data directory exists before creating database
        # pathlib.Path handles cross-platform path operations
        db_dir = os.path.dirname(db_path)  # extract directory portion from path
        if db_dir:  # if there is a directory specified (not just filename)
            Path(db_dir).mkdir(parents=True, exist_ok=True)
            # creates data/ directory and any parent directories if they don't exist
            # exist_ok=True prevents error if directory already exists
        
        # initialize database tables on first run
        # this creates all necessary tables if they don't exist
        self._initialize_database()
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.
        ensures connections are properly closed after use, even if errors occur.
        automatically handles commits and rollbacks.
        
        Yields:
            sqlite3.Connection: Active database connection with row factory enabled
        """
        conn = sqlite3.connect(self.db_path)  # create connection to sqlite file
        # row_factory allows accessing columns by name like row['column_name']
        # instead of by index like row[0]
        conn.row_factory = sqlite3.Row  # enable dictionary-like row access
        
        try:
            yield conn  # provide connection to the with block
            conn.commit()  # commit any changes made during the block
        except Exception:
            conn.rollback()  # rollback changes if an error occurred
            raise  # re-raise the exception for caller to handle
        finally:
            conn.close()  # always close the connection to free resources
    
    def _initialize_database(self):
        """
        Create database tables if they don't already exist.
        defines the complete schema for all persistent data storage.
        this is idempotent - safe to call multiple times.
        """
        with self._get_connection() as conn:  # use context manager for safe connection
            cursor = conn.cursor()  # get cursor for executing sql statements
            
            # =================================================================
            # topic_progress table
            # stores current progress for each topic being studied
            # one row per unique topic
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topic_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- unique identifier for each progress record
                    
                    topic_name TEXT UNIQUE NOT NULL,
                    -- name of the topic (unique constraint prevents duplicates)
                    
                    questions_attempted INTEGER DEFAULT 0,
                    -- total number of questions attempted for this topic
                    
                    questions_correct INTEGER DEFAULT 0,
                    -- number of correctly answered questions
                    
                    time_spent_minutes INTEGER DEFAULT 0,
                    -- total time spent practicing this topic in minutes
                    
                    last_practiced TIMESTAMP,
                    -- timestamp of the most recent practice session
                    
                    mastery_level REAL DEFAULT 0.0,
                    -- calculated mastery score from 0.0 to 1.0
                    
                    weak_subtopics TEXT,
                    -- JSON serialized list of subtopics needing improvement
                    
                    recommended_resources TEXT,
                    -- JSON serialized list of recommended study resources
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    -- when this record was first created
                    
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    -- when this record was last updated
                )
            """)
            
            # =================================================================
            # progress_history table
            # stores historical snapshots for trend analysis
            # each practice attempt adds a row here
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS progress_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- unique identifier for each history entry
                    
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    -- when this practice attempt occurred
                    
                    topic TEXT NOT NULL,
                    -- which topic was being practiced
                    
                    is_correct INTEGER,
                    -- 1 for correct answer, 0 for incorrect, NULL for not graded
                    
                    mastery_score REAL,
                    -- mastery score at the time of this attempt
                    
                    time_spent INTEGER,
                    -- time spent on this specific question in seconds
                    
                    session_id TEXT
                    -- identifier linking to a user session
                )
            """)
            
            # =================================================================
            # study_plans table
            # stores generated study plans for exam preparation
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS study_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- unique identifier for each study plan
                    
                    subject TEXT NOT NULL,
                    -- subject name for this study plan
                    
                    target_date TIMESTAMP NOT NULL,
                    -- target completion or exam date
                    
                    total_hours INTEGER DEFAULT 0,
                    -- total estimated study hours required
                    
                    exam_pattern TEXT,
                    -- type of exam (internal, external, practical)
                    
                    plan_data TEXT NOT NULL,
                    -- JSON serialized StudyPlan object with full schedule
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    -- when this plan was created
                    
                    is_active INTEGER DEFAULT 1
                    -- 1 for active plans, 0 for archived/completed plans
                )
            """)
            
            # =================================================================
            # exam_questions table
            # stores questions extracted from past papers
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exam_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- unique identifier for each question
                    
                    question_text TEXT NOT NULL,
                    -- the actual question text
                    
                    marks INTEGER DEFAULT 0,
                    -- marks allocated for this question
                    
                    topic TEXT,
                    -- detected topic/category of the question
                    
                    difficulty TEXT,
                    -- difficulty level (easy, medium, hard)
                    
                    question_type TEXT,
                    -- type (theory, practical, problem_solving)
                    
                    expected_time_minutes INTEGER,
                    -- estimated time to solve in minutes
                    
                    source_file TEXT,
                    -- original file this question was extracted from
                    
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    -- when this question was extracted
                )
            """)
            
            # =================================================================
            # user_sessions table
            # tracks study sessions for analytics
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- unique identifier for each session
                    
                    session_id TEXT UNIQUE NOT NULL,
                    -- unique session identifier (uuid or timestamp-based)
                    
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    -- when the session started
                    
                    end_time TIMESTAMP,
                    -- when the session ended (NULL for active sessions)
                    
                    duration_minutes INTEGER,
                    -- calculated session duration in minutes
                    
                    questions_asked INTEGER DEFAULT 0,
                    -- total questions asked during this session
                    
                    track_type TEXT
                    -- which track was active (track_a1_cs or track_a2_exam)
                )
            """)
            
            # =================================================================
            # create indexes for frequently queried columns
            # indexes improve query performance for large datasets
            # =================================================================
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_progress_topic 
                ON topic_progress(topic_name)
            """)
            # speeds up lookups by topic name
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_timestamp 
                ON progress_history(timestamp)
            """)
            # speeds up time-based queries for trend analysis
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_topic 
                ON progress_history(topic)
            """)
            # speeds up queries filtering by topic
    
    # =========================================================================
    # Topic Progress Operations
    # methods for saving, retrieving, and managing topic progress data
    # =========================================================================
    
    def save_topic_progress(
        self,
        topic_name: str,
        questions_attempted: int = 0,
        questions_correct: int = 0,
        time_spent_minutes: int = 0,
        last_practiced: Optional[datetime] = None,
        mastery_level: float = 0.0,
        weak_subtopics: List[str] = None,
        recommended_resources: List[str] = None
    ) -> None:
        """
        Save or update topic progress in the database.
        uses upsert (insert or update) pattern to handle both new and existing topics.
        
        Args:
            topic_name (str): Name of the topic (unique identifier)
            questions_attempted (int): Total questions attempted for this topic
            questions_correct (int): Number of correctly answered questions
            time_spent_minutes (int): Total time spent practicing in minutes
            last_practiced (datetime, optional): Timestamp of last practice session
            mastery_level (float): Calculated mastery score from 0.0 to 1.0
            weak_subtopics (List[str], optional): List of weak subtopics needing work
            recommended_resources (List[str], optional): Suggested study resources
        """
        with self._get_connection() as conn:  # use context manager for safe connection
            cursor = conn.cursor()  # get cursor for executing sql
            
            # serialize list data to json strings for storage
            # sqlite doesn't natively support lists, so we use json serialization
            weak_json = json.dumps(weak_subtopics) if weak_subtopics else None
            # converts list to json string like '["subtopic1", "subtopic2"]'
            
            resources_json = json.dumps(recommended_resources) if recommended_resources else None
            # converts resource list to json string
            
            # use provided timestamp or default to current time
            practice_time = last_practiced.isoformat() if last_practiced else datetime.now().isoformat()
            # isoformat() creates string like '2024-01-15T14:30:00'
            
            # upsert operation (update or insert)
            # ON CONFLICT clause handles the case where topic already exists
            cursor.execute("""
                INSERT INTO topic_progress 
                    (topic_name, questions_attempted, questions_correct, time_spent_minutes, 
                     last_practiced, mastery_level, weak_subtopics, recommended_resources, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(topic_name) DO UPDATE SET
                    questions_attempted = excluded.questions_attempted,
                    questions_correct = excluded.questions_correct,
                    time_spent_minutes = excluded.time_spent_minutes,
                    last_practiced = excluded.last_practiced,
                    mastery_level = excluded.mastery_level,
                    weak_subtopics = excluded.weak_subtopics,
                    recommended_resources = excluded.recommended_resources,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                topic_name,           # ? placeholder 1
                questions_attempted,  # ? placeholder 2
                questions_correct,    # ? placeholder 3
                time_spent_minutes,   # ? placeholder 4
                practice_time,        # ? placeholder 5
                mastery_level,        # ? placeholder 6
                weak_json,           # ? placeholder 7
                resources_json       # ? placeholder 8
            ))
            # excluded.column_name refers to the value that would have been inserted
            # this allows updating existing rows with new values
    
    def get_topic_progress(self, topic_name: str) -> Optional[Dict[str, Any]]:
        """
        Get progress data for a specific topic from the database.
        
        Args:
            topic_name (str): Name of the topic to retrieve
        
        Returns:
            Optional[Dict]: Topic progress data as dictionary, or None if not found
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            cursor.execute("""
                SELECT * FROM topic_progress WHERE topic_name = ?
            """, (topic_name,))  # parameterized query prevents sql injection
            
            row = cursor.fetchone()  # get first row (should be at most one due to unique constraint)
            if row:
                # convert sqlite row object to regular dictionary
                data = dict(row)  # dict() converts row to {column: value} mapping
                
                # deserialize json fields back to python lists
                if data.get("weak_subtopics"):
                    data["weak_subtopics"] = json.loads(data["weak_subtopics"])
                    # json.loads() converts json string back to list
                
                if data.get("recommended_resources"):
                    data["recommended_resources"] = json.loads(data["recommended_resources"])
                    # convert json string back to resource list
                
                return data  # return the deserialized data
            
            return None  # topic not found in database
    
    def get_all_topic_progress(self) -> Dict[str, Dict[str, Any]]:
        """
        Get progress data for all topics in the database.
        returns a dictionary mapping topic names to their progress data.
        
        Returns:
            Dict[str, Dict]: Dictionary with topic_name as key and progress data as value
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            cursor.execute("SELECT * FROM topic_progress ORDER BY topic_name")
            # order by topic name for consistent output
            
            result = {}  # initialize empty dictionary for results
            for row in cursor.fetchall():  # iterate through all rows
                data = dict(row)  # convert row to dictionary
                
                # deserialize json fields back to python lists
                if data.get("weak_subtopics"):
                    data["weak_subtopics"] = json.loads(data["weak_subtopics"])
                
                if data.get("recommended_resources"):
                    data["recommended_resources"] = json.loads(data["recommended_resources"])
                
                result[data["topic_name"]] = data  # add to result dictionary
            
            return result  # return all topic progress data
    
    def delete_topic_progress(self, topic_name: str) -> bool:
        """
        Delete progress data for a specific topic from the database.
        
        Args:
            topic_name (str): Name of the topic to delete
        
        Returns:
            bool: True if topic was found and deleted, False if topic not found
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            cursor.execute("DELETE FROM topic_progress WHERE topic_name = ?", (topic_name,))
            # parameterized delete query
            
            return cursor.rowcount > 0  # returns True if at least one row was deleted
            # rowcount contains number of affected rows
    
    # =========================================================================
    # Progress History Operations
    # methods for recording and retrieving historical progress data
    # =========================================================================
    
    def add_progress_history(
        self,
        topic: str,
        is_correct: Optional[bool] = None,
        mastery_score: Optional[float] = None,
        time_spent: int = 0,
        session_id: Optional[str] = None
    ) -> None:
        """
        Add a progress history entry to track learning over time.
        each practice attempt should call this method.
        
        Args:
            topic (str): Name of the topic being practiced
            is_correct (bool, optional): Whether the answer was correct
            mastery_score (float, optional): Mastery score at this time
            time_spent (int): Time spent on this question in seconds
            session_id (str, optional): Identifier for the current session
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            # convert boolean to integer for sqlite storage
            # sqlite doesn't have native boolean type, uses 0/1 integers
            is_correct_int = None  # default to NULL
            if is_correct is not None:
                is_correct_int = 1 if is_correct else 0  # 1 for True, 0 for False
            
            cursor.execute("""
                INSERT INTO progress_history 
                    (topic, is_correct, mastery_score, time_spent, session_id)
                VALUES (?, ?, ?, ?, ?)
            """, (topic, is_correct_int, mastery_score, time_spent, session_id))
            # parameterized insert prevents sql injection
    
    def get_progress_history(
        self,
        topic: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get progress history entries from the database.
        can filter by topic and limit number of results.
        
        Args:
            topic (str, optional): Filter results to specific topic
            limit (int): Maximum number of entries to return (default 100)
        
        Returns:
            List[Dict]: List of history entries as dictionaries
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            if topic:
                # query with topic filter
                cursor.execute("""
                    SELECT * FROM progress_history 
                    WHERE topic = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (topic, limit))  # most recent entries first
            else:
                # query without topic filter
                cursor.execute("""
                    SELECT * FROM progress_history 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            # convert all rows to dictionaries and return as list
            return [dict(row) for row in cursor.fetchall()]
            # list comprehension creates list of dictionaries
    
    def get_progress_trend(
        self,
        topic: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get progress trend data for a topic over a specified time period.
        useful for visualizing improvement over time.
        
        Args:
            topic (str): Topic to analyze
            days (int): Number of days to look back (default 30)
        
        Returns:
            List[Dict]: List of daily progress snapshots
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            # query to get daily aggregates
            cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as attempts,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(mastery_score) as avg_mastery
                FROM progress_history
                WHERE topic = ? 
                    AND timestamp >= datetime('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (topic, f'-{days} days'))
            # datetime('now', '-30 days') gets date from 30 days ago
            
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Study Plan Operations
    # methods for saving and retrieving study plans
    # =========================================================================
    
    def save_study_plan(
        self,
        subject: str,
        target_date: datetime,
        plan_data: Dict[str, Any],
        total_hours: int = 0,
        exam_pattern: str = "internal"
    ) -> int:
        """
        Save a study plan to the database.
        
        Args:
            subject (str): Subject name for this study plan
            target_date (datetime): Target completion or exam date
            plan_data (Dict): Full serialized plan data with schedule
            total_hours (int): Total estimated study hours required
            exam_pattern (str): Type of exam (internal/external/practical)
        
        Returns:
            int: ID of the newly inserted study plan
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            cursor.execute("""
                INSERT INTO study_plans 
                    (subject, target_date, total_hours, exam_pattern, plan_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                subject,                        # subject name
                target_date.isoformat(),        # convert datetime to iso string
                total_hours,                    # estimated hours
                exam_pattern,                   # exam type
                json.dumps(plan_data)          # serialize plan data to json
            ))
            
            return cursor.lastrowid  # return the auto-generated primary key id
            # lastrowid contains the rowid of the most recent insert
    
    def get_study_plans(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get saved study plans from the database.
        
        Args:
            active_only (bool): If True, only return active plans (default True)
        
        Returns:
            List[Dict]: List of study plans with deserialized plan_data
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            query = "SELECT * FROM study_plans"
            if active_only:
                query += " WHERE is_active = 1"  # filter for active plans only
            
            query += " ORDER BY created_at DESC"  # newest plans first
            
            cursor.execute(query)
            
            plans = []  # initialize empty list for results
            for row in cursor.fetchall():  # iterate through all rows
                data = dict(row)  # convert row to dictionary
                
                # deserialize plan data from json string back to dictionary
                data["plan_data"] = json.loads(data["plan_data"])
                
                plans.append(data)  # add to results list
            
            return plans  # return list of study plans
    
    def deactivate_study_plan(self, plan_id: int) -> bool:
        """
        Mark a study plan as inactive (completed or abandoned).
        
        Args:
            plan_id (int): ID of the study plan to deactivate
        
        Returns:
            bool: True if plan was found and deactivated, False otherwise
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            cursor.execute("""
                UPDATE study_plans 
                SET is_active = 0 
                WHERE id = ?
            """, (plan_id,))
            
            return cursor.rowcount > 0  # True if a row was updated
    
    # =========================================================================
    # Exam Questions Operations
    # methods for storing and retrieving extracted exam questions
    # =========================================================================
    
    def save_exam_questions(self, questions: List[Dict[str, Any]]) -> int:
        """
        Save extracted exam questions to the database.
        processes a list of questions and inserts them.
        
        Args:
            questions (List[Dict]): List of question dictionaries with required fields
        
        Returns:
            int: Number of questions successfully saved
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            count = 0  # counter for successfully inserted questions
            for q in questions:  # iterate through each question
                cursor.execute("""
                    INSERT INTO exam_questions 
                        (question_text, marks, topic, difficulty, question_type, 
                         expected_time_minutes, source_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    q.get("question_text", ""),        # required field
                    q.get("marks", 0),                 # default 0 marks
                    q.get("topic", "General"),         # default "General" topic
                    q.get("difficulty", "medium"),     # default medium difficulty
                    q.get("question_type", "theory"),  # default theory type
                    q.get("expected_time_minutes", 5), # default 5 minutes
                    q.get("source_file", None)         # optional source file
                ))
                count += 1  # increment success counter
            
            return count  # return number of questions saved
    
    def get_exam_questions(
        self,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get exam questions from database with optional filtering.
        
        Args:
            topic (str, optional): Filter questions by topic
            difficulty (str, optional): Filter by difficulty (easy/medium/hard)
            limit (int): Maximum number of questions to return (default 50)
        
        Returns:
            List[Dict]: List of exam questions matching the filters
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            # build query dynamically based on provided filters
            query = "SELECT * FROM exam_questions WHERE 1=1"  # 1=1 is a trick to simplify AND conditions
            params = []  # list for parameterized query values
            
            if topic:
                query += " AND topic = ?"  # add topic filter
                params.append(topic)
            
            if difficulty:
                query += " AND difficulty = ?"  # add difficulty filter
                params.append(difficulty)
            
            query += " ORDER BY extracted_at DESC LIMIT ?"  # newest first, with limit
            params.append(limit)
            
            cursor.execute(query, params)  # execute with parameters
            
            return [dict(row) for row in cursor.fetchall()]  # convert to list of dicts
    
    def get_questions_by_topic(self) -> Dict[str, int]:
        """
        Get count of questions grouped by topic.
        useful for showing topic coverage statistics.
        
        Returns:
            Dict[str, int]: Dictionary mapping topic names to question counts
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            cursor.execute("""
                SELECT topic, COUNT(*) as count
                FROM exam_questions
                GROUP BY topic
                ORDER BY count DESC
            """)
            
            # convert results to simple dictionary
            return {row["topic"]: row["count"] for row in cursor.fetchall()}
            # dictionary comprehension creates {topic: count} mapping
    
    # =========================================================================
    # Session Operations
    # methods for tracking user study sessions
    # =========================================================================
    
    def start_session(self, session_id: str, track_type: Optional[str] = None) -> None:
        """
        Record the start of a user session in the database.
        
        Args:
            session_id (str): Unique identifier for this session
            track_type (str, optional): Which track is active (track_a1_cs or track_a2_exam)
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            cursor.execute("""
                INSERT INTO user_sessions (session_id, track_type)
                VALUES (?, ?)
            """, (session_id, track_type))
    
    def end_session(self, session_id: str, questions_asked: int = 0) -> None:
        """
        Record the end of a user session and calculate duration.
        
        Args:
            session_id (str): Session identifier to end
            questions_asked (int): Total questions asked during this session
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            # update session with end time and calculated duration
            # JULIANDAY function calculates difference between two timestamps
            cursor.execute("""
                UPDATE user_sessions 
                SET end_time = CURRENT_TIMESTAMP,
                    duration_minutes = ROUND((JULIANDAY(CURRENT_TIMESTAMP) - JULIANDAY(start_time)) * 24 * 60),
                    questions_asked = ?
                WHERE session_id = ?
            """, (questions_asked, session_id))
            # JULIANDAY returns fractional days, multiply by 24*60 for minutes
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get overall session statistics.
        
        Returns:
            Dict: Session statistics including total sessions, total time, etc.
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(duration_minutes) as total_minutes,
                    SUM(questions_asked) as total_questions,
                    AVG(duration_minutes) as avg_session_minutes
                FROM user_sessions
                WHERE end_time IS NOT NULL
            """)
            
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    # =========================================================================
    # Utility Operations
    # helper methods for database management
    # =========================================================================
    
    def clear_all_data(self) -> None:
        """
        Clear all data from all tables.
        WARNING: This is irreversible and deletes all user progress!
        use with caution - typically only for testing or reset.
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            # delete all rows from each table
            cursor.execute("DELETE FROM topic_progress")     # clear topic progress
            cursor.execute("DELETE FROM progress_history")   # clear history
            cursor.execute("DELETE FROM study_plans")        # clear study plans
            cursor.execute("DELETE FROM exam_questions")     # clear questions
            cursor.execute("DELETE FROM user_sessions")      # clear sessions
            
            # reset auto-increment counters so ids start from 1 again
            cursor.execute("DELETE FROM sqlite_sequence")
            # sqlite_sequence table tracks auto-increment values
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database for monitoring.
        
        Returns:
            Dict: Database statistics including row counts and file size
        """
        with self._get_connection() as conn:  # use context manager
            cursor = conn.cursor()  # get cursor
            
            stats = {}  # initialize stats dictionary
            
            # count rows in each table for statistics
            tables = ["topic_progress", "progress_history", "study_plans", "exam_questions", "user_sessions"]
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")  # count rows in table
                stats[f"{table}_count"] = cursor.fetchone()[0]  # store count
            
            # get database file size if it exists
            if os.path.exists(self.db_path):
                stats["file_size_kb"] = os.path.getsize(self.db_path) / 1024  # bytes to kb
                stats["file_size_mb"] = stats["file_size_kb"] / 1024  # kb to mb
            
            return stats  # return statistics dictionary
    
    def vacuum_database(self) -> None:
        """
        Optimize the database by running VACUUM command.
        reclaims unused space and defragments the database file.
        should be called periodically for maintenance.
        """
        with self._get_connection() as conn:  # use context manager
            conn.execute("VACUUM")  # sqlite vacuum command rebuilds database file
            # this can significantly reduce file size after many deletions
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_path (str, optional): Path for backup file
        
        Returns:
            str: Path to the created backup file
        """
        if backup_path is None:
            # generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/progress_backup_{timestamp}.db"
        
        # ensure backup directory exists
        backup_dir = os.path.dirname(backup_path)
        if backup_dir:
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
        
        # use sqlite's built-in backup capability
        source = sqlite3.connect(self.db_path)
        destination = sqlite3.connect(backup_path)
        
        source.backup(destination)  # copy entire database
        
        source.close()
        destination.close()
        
        return backup_path  # return path to backup file


# =============================================================================
# Singleton Instance
# provides a single shared database manager instance across the application
# =============================================================================

_db_manager: Optional[DatabaseManager] = None  # global variable for singleton
# using Optional type hint since it starts as None


def get_db_manager() -> DatabaseManager:
    """
    Get or create singleton database manager instance.
    ensures only one database connection manager exists.
    this is the recommended way to access the database.
    
    Returns:
        DatabaseManager: Singleton database manager instance
    """
    global _db_manager  # access the global variable
    
    if _db_manager is None:  # check if instance already exists
        _db_manager = DatabaseManager()  # create new instance if needed
        # uses default path "data/progress.db"
    
    return _db_manager  # return the singleton instance