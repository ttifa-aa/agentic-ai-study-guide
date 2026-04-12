"""
Tracks module for the Academic Assistant.
Contains base track class and specialized track implementations.
"""

# import and re-export base track
from tracks.base_track import (
    BaseTrack,                       # abstract base class for all tracks
    TrackFeatures                    # dataclass for track feature definitions
)

# import and re-export track implementations
from tracks.track_a1_cs import TrackA1CS        # computer science subject guide track
from tracks.track_a2_exam import TrackA2Exam    # exam preparation assistant track

# define what gets exported when someone does "from tracks import *"
__all__ = [
    "BaseTrack",
    "TrackFeatures",
    "TrackA1CS",
    "TrackA2Exam"
]