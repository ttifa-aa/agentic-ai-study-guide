"""
Database module for the Academic Assistant.
Exports database manager and utilities for progress tracking persistence.
"""

# import and re-export database manager and utilities
from database.db_manager import (
    DatabaseManager,           # main class for sqlite database operations
    get_db_manager            # singleton accessor for database manager
)

# define what gets exported when someone does "from database import *"
__all__ = [
    "DatabaseManager",
    "get_db_manager"
]