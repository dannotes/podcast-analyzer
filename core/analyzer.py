import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class PodcastAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def parse_webvtt(self, file_path: str) -> List[Dict]:
        """Parse WebVTT file and extract captions with timestamps."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if content.startswith('WEBVTT'):
            content = content.split('\n', 1)[1]
        
        pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})(.*?)(?=\n\n|\Z)'
        captions = re.findall(pattern, content, re.DOTALL)
        
        return [{
            'start': start,
            'end': end,
            'start_seconds': self._timestamp_to_seconds(start),
            'end_seconds': self._timestamp_to_seconds(end),
            'duration': self._timestamp_to_seconds(end) - self._timestamp_to_seconds(start),
            'text': self._clean_text(text)
        } for start, end, text in captions]

    def _clean_text(self, text: str) -> str:
        """Clean subtitle text by removing metadata and duplicates."""
        text = re.sub(r'<.*?>|align:start position:\d+%', '', text)
        text = re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp (HH:MM:SS.mmm) to seconds."""
        h, m, s = timestamp.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to timestamp (HH:MM:SS)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def analyze_sentiment(self, captions: List[Dict]) -> List[Dict]:
        """Analyze sentiment for each caption."""
        for caption in captions:
            sentiment = self.sia.polarity_scores(caption['text'])
            caption.update({
                'sentiment_compound': sentiment['compound'],
                'sentiment': 'positive' if sentiment['compound'] > 0.05 
                            else 'negative' if sentiment['compound'] < -0.05 
                            else 'neutral',
                'sentiment_intensity': abs(sentiment['compound'])
            })
        return captions

    def identify_topics(self, captions: List[Dict], top_n: int = 10) -> List[Tuple[str, int]]:
        """Identify main topics using word frequency analysis."""
        all_text = ' '.join([c['text'] for c in captions]).lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        
        stop_words = {'the', 'and', 'that', 'this', 'with', 'for', 'you', 'have', 'are'}
        filtered = [w for w in words if w not in stop_words]
        
        counts = Counter(filtered) + Counter(bigrams)
        return counts.most_common(top_n)

    def detect_questions(self, text: str) -> bool:
        """Check if text contains questions."""
        if '?' in text:
            return True
        sentences = re.split(r'[.!?]+', text.lower())
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'do', 'does', 'did', 'is', 'are', 'can'}
        return any(s.strip().split()[0] in question_words for s in sentences if s.strip())

    def detect_debate(self, text: str) -> bool:
        """Check for debate/disagreement indicators."""
        debate_phrases = {
            'disagree', "don't think", 'but actually', 'not true', 
            'incorrect', "don't agree", 'on the contrary', "that's not"
        }
        return any(phrase in text.lower() for phrase in debate_phrases)

    def calculate_speech_density(self, text: str, duration: float) -> float:
        """Calculate words per minute."""
        words = len(re.findall(r'\b\w+\b', text))
        return words / (duration / 60) if duration > 0 else 0

    def segment_podcast(self, captions: List[Dict], window_size: int = 90) -> List[Dict]:
        """Create analysis segments with 50% overlap."""
        segments = []
        max_time = max(c['end_seconds'] for c in captions)
        start_time = 0
        
        while start_time < max_time - window_size/2:
            end_time = start_time + window_size
            segment_captions = [
                c for c in captions 
                if c['start_seconds'] >= start_time and c['end_seconds'] <= end_time
            ]
            
            if segment_captions:
                sentiment = [c['sentiment_compound'] for c in segment_captions]
                intensity = [c['sentiment_intensity'] for c in segment_captions]
                segment_text = ' '.join(c['text'] for c in segment_captions)
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'start': self._seconds_to_timestamp(start_time),
                    'end': self._seconds_to_timestamp(end_time),
                    'duration': end_time - start_time,
                    'text': segment_text,
                    'avg_sentiment': sum(sentiment)/len(sentiment) if sentiment else 0,
                    'sentiment_variance': sum((x - (sum(sentiment)/len(sentiment)))**2 for x in sentiment)/len(sentiment) if len(sentiment) > 1 else 0,
                    'max_intensity': max(intensity) if intensity else 0,
                    'has_question': self.detect_questions(segment_text),
                    'has_debate': self.detect_debate(segment_text),
                    'speech_density': self.calculate_speech_density(segment_text, end_time - start_time),
                    'heatmap_value': 0  # To be populated later
                })
            
            start_time += window_size / 2
        
        return segments

    def apply_heatmap(self, segments: List[Dict], heatmap_data: List[Dict]) -> List[Dict]:
        """Apply heatmap values to segments."""
        if not heatmap_data:
            return segments
            
        for segment in segments:
            overlapping = [
                h for h in heatmap_data 
                if h['start_time'] <= segment['start_time'] <= h['end_time']
            ]
            if overlapping:
                segment['heatmap_value'] = max(h['value'] for h in overlapping)
            else:
                closest = min(
                    heatmap_data, 
                    key=lambda x: abs(x['start_time'] - segment['start_time'])
                )
                segment['heatmap_value'] = closest['value']
        return segments

    def score_segments(self, segments: List[Dict], topics: List[Tuple[str, int]], algorithm: str = 'viral') -> List[Dict]:
        """Score segments based on selected algorithm."""
        for segment in segments:
            content_score = 0
            heatmap_score = segment.get('heatmap_value', 0) * 100
            
            if algorithm in ['viral', 'combined']:
                topic_score = sum(
                    count/2 for word, count in topics 
                    if word in segment['text'].lower()
                )
                
                engagement = (
                    abs(segment['avg_sentiment']) * 20 +
                    (25 if segment['has_question'] else 0) +
                    (30 if segment['has_debate'] else 0)
                )
                
                tech = (
                    10 - min(abs(segment['speech_density'] - 150)/15, 10) +
                    10 - min(abs(segment['duration'] - 60)/6, 10) +
                    segment['sentiment_variance'] * 15
                )
                
                content_score = topic_score + engagement + tech
            
            if algorithm == 'viral':
                final_score = content_score
            elif algorithm == 'heatmap':
                final_score = heatmap_score
            else:  # combined
                final_score = (content_score * 0.3) + (heatmap_score * 0.7)
            
            segment.update({
                'viral_score': round(final_score, 2),
                'content_score': round(content_score, 2),
                'heatmap_score': round(heatmap_score, 2),
                'topic_matches': sum(1 for word,_ in topics if word in segment['text'].lower())
            })
        
        return sorted(segments, key=lambda x: (-x['viral_score'], -x['heatmap_score']))

    def analyze(self, vtt_path: str, heatmap_data: Optional[List[Dict]] = None, 
               window_size: int = 90, top_n: int = 5, algorithm: str = 'viral') -> Tuple[List[Dict], List[Dict], List[Tuple[str, int]]]:
        """Complete analysis pipeline."""
        captions = self.parse_webvtt(vtt_path)
        captions = self.analyze_sentiment(captions)
        topics = self.identify_topics(captions)
        segments = self.segment_podcast(captions, window_size)
        
        if heatmap_data:
            segments = self.apply_heatmap(segments, heatmap_data)
        
        scored = self.score_segments(segments, topics, algorithm)
        return scored[:top_n], captions, topics