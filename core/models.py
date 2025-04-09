from typing import TypedDict, List, Optional

class Caption(TypedDict):
    start: str
    end: str
    start_seconds: float
    end_seconds: float
    duration: float
    text: str
    sentiment_compound: Optional[float]
    sentiment: Optional[str]

class HeatmapSegment(TypedDict):
    start_time: float
    end_time: float 
    value: float

class PodcastSegment(TypedDict):
    start_time: float
    end_time: float
    start: str
    end: str
    duration: float
    text: str
    viral_score: float
    heatmap_value: float