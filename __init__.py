"""
Podcast Analyzer - A tool for identifying viral moments in YouTube podcasts
"""

__version__ = "1.0.0"
__all__ = ['cli', 'core']

# Import key classes for easier access
from .core.youtube import YouTubeProcessor
from .core.analyzer import PodcastAnalyzer

# Initialize NLTK data
try:
    nltk.data.find('vader_lexicon')
except (LookupError, ImportError):
    import nltk
    nltk.download('vader_lexicon', quiet=True)