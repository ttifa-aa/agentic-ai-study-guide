# Data Directory

This directory stores persistent data for the Academic Assistant application.
all user progress, study plans, and extracted questions are stored here.

## Directory Structure

data/
├── README.md # this file - documentation
├── .gitkeep # ensures directory is tracked in git
├── progress.db # sqlite database (created on first run)
├── vector_store/ # faiss vector store index files (if persisted locally)
├── exports/ # user-exported study guides and reports
└── backups/ # database backup files
text


## Contents

### progress.db (SQLite Database)
the main database file containing all persistent data.
created automatically when the application first runs.

**Tables:**
- `topic_progress` - current progress for each topic being studied
- `progress_history` - historical snapshots for trend analysis
- `study_plans` - generated study plans for exam preparation
- `exam_questions` - questions extracted from past papers
- `user_sessions` - study session tracking for analytics

### vector_store/ (Directory)
contains faiss index files when vector store persistence is enabled.
files include:
- `index.faiss` - the faiss vector index
- `index.pkl` - serialized document metadata

### exports/ (Directory)
stores user-exported files:
- study guides (markdown format)
- progress reports (markdown/pdf format)
- practice tests (text format)

### backups/ (Directory)
contains database backup files created by the backup function.
files are named: `progress_backup_YYYYMMDD_HHMMSS.db`

## Database Schema Details

### topic_progress Table
```sql
CREATE TABLE topic_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,    -- unique record id
    topic_name TEXT UNIQUE NOT NULL,         -- topic identifier
    questions_attempted INTEGER DEFAULT 0,   -- total attempts
    questions_correct INTEGER DEFAULT 0,     -- correct answers
    time_spent_minutes INTEGER DEFAULT 0,    -- study time in minutes
    last_practiced TIMESTAMP,                -- last practice session
    mastery_level REAL DEFAULT 0.0,          -- 0.0 to 1.0 mastery score
    weak_subtopics TEXT,                     -- json array of weak areas
    recommended_resources TEXT,              -- json array of resources
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

progress_history Table
sql

CREATE TABLE progress_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- when recorded
    topic TEXT NOT NULL,                           -- topic practiced
    is_correct INTEGER,                            -- 1=correct, 0=incorrect, NULL=ungraded
    mastery_score REAL,                            -- score at this time
    time_spent INTEGER,                            -- seconds on this question
    session_id TEXT                                -- links to user_sessions
);

study_plans Table
sql

CREATE TABLE study_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,                    -- subject name
    target_date TIMESTAMP NOT NULL,           -- exam/completion date
    total_hours INTEGER DEFAULT 0,            -- estimated hours needed
    exam_pattern TEXT,                        -- internal/external/practical
    plan_data TEXT NOT NULL,                  -- json serialized plan
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1               -- 1=active, 0=archived
);

exam_questions Table
sql

CREATE TABLE exam_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_text TEXT NOT NULL,              -- the question
    marks INTEGER DEFAULT 0,                  -- marks allocated
    topic TEXT,                               -- subject topic
    difficulty TEXT,                          -- easy/medium/hard
    question_type TEXT,                       -- theory/practical/problem_solving
    expected_time_minutes INTEGER,            -- estimated time to solve
    source_file TEXT,                         -- original file name
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

user_sessions Table
sql

CREATE TABLE user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,          -- uuid or timestamp-based id
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,                       -- null for active sessions
    duration_minutes INTEGER,                 -- calculated when ended
    questions_asked INTEGER DEFAULT 0,        -- total questions in session
    track_type TEXT                           -- track_a1_cs or track_a2_exam
);

Indexes

the following indexes are created for query performance:

    idx_progress_topic on topic_progress(topic_name)

    idx_history_timestamp on progress_history(timestamp)

    idx_history_topic on progress_history(topic)
    