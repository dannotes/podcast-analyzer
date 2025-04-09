import re
import os
import argparse
import subprocess
import json
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import urllib.parse
import math

# Download necessary NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class YouTubeProcessor:
    """Class to handle YouTube video processing tasks."""
    
    @staticmethod
    def extract_video_id(youtube_url):
        """Extract YouTube video ID from URL."""
        parsed_url = urllib.parse.urlparse(youtube_url)
        
        # Handle youtu.be format
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path[1:]
        
        # Handle regular youtube.com links
        if parsed_url.netloc in ('www.youtube.com', 'youtube.com','youtu.be'):
            if parsed_url.path == '/watch':
                query = urllib.parse.parse_qs(parsed_url.query)
                return query.get('v', [''])[0]
            
            # Handle shortened URLs
            if parsed_url.path.startswith('/v/'):
                return parsed_url.path.split('/')[2]
                
        return None
    
    @staticmethod
    def download_subtitles(youtube_url, output_dir='.'):
        """Download WebVTT subtitles using yt-dlp."""
        video_id = YouTubeProcessor.extract_video_id(youtube_url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {youtube_url}")
        
        output_file = os.path.join(output_dir, f"{video_id}.vtt")
        
        # Check if file already exists
        if os.path.exists(output_file):
            print(f"Subtitle file already exists: {output_file}")
            return video_id, output_file
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download subtitles with yt-dlp
        command = [
            'yt-dlp',
            '--skip-download',
            '--write-auto-sub',
            '--sub-format', 'vtt',
            '--output', os.path.join(output_dir, f"{video_id}"),
            youtube_url
        ]
        
        try:
            print(f"Downloading subtitles for video: {video_id}")
            subprocess.run(command, check=True, capture_output=True)
            
            # yt-dlp may append language code to the filename
            potential_files = [
                f for f in os.listdir(output_dir) 
                if f.startswith(video_id) and f.endswith('.vtt')
            ]
            
            if potential_files:
                subtitle_file = os.path.join(output_dir, potential_files[0])
                
                # Rename file to just the video_id if needed
                if subtitle_file != output_file:
                    os.rename(subtitle_file, output_file)
                
                print(f"Subtitles downloaded and saved to: {output_file}")
                return video_id, output_file
            else:
                raise FileNotFoundError(f"Subtitle file not found after download for {video_id}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error downloading subtitles: {e}")
            print(f"yt-dlp output: {e.output.decode('utf-8')}")
            print(f"yt-dlp error: {e.stderr.decode('utf-8')}")
            raise
    
    @staticmethod
    def get_video_json(youtube_url, output_dir='.'):
        """Fetch video metadata as JSON using yt-dlp."""
        video_id = YouTubeProcessor.extract_video_id(youtube_url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {youtube_url}")
        
        output_file = os.path.join(output_dir, f"{video_id}.json")
        
        # Check if file already exists
        if os.path.exists(output_file):
            print(f"Video JSON file already exists: {output_file}")
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download JSON with yt-dlp
        command = [
            'yt-dlp',
            '--dump-json',
            '--skip-download',
            '--output', os.path.join(output_dir, f"{video_id}"),
            youtube_url
        ]
        
        try:
            print(f"Fetching video JSON for: {video_id}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            # Save the JSON to file
            json_data = json.loads(result.stdout)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f)
                
            return json_data
                
        except subprocess.CalledProcessError as e:
            print(f"Error fetching video JSON: {e}")
            print(f"yt-dlp output: {e.output.decode('utf-8')}")
            print(f"yt-dlp error: {e.stderr.decode('utf-8')}")
            raise
    
    @staticmethod
    def create_timestamp_url(video_id, seconds):
        """Create a clickable YouTube URL with timestamp."""
        return f"https://youtu.be/{video_id}?t={int(seconds)}"


class PodcastAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def parse_webvtt(self, file_path):
        """Parse and clean WebVTT subtitles with duplicate removal."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Skip header and notes
        if content.startswith('WEBVTT'):
            content = content.split('\n\n', 1)[1]
        
        # Match caption blocks
        pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})(.*?)(?=\n\n|\Z)'
        captions = re.findall(pattern, content, re.DOTALL)
        
        parsed = []
        for start, end, text in captions:
            # 1. Remove all HTML tags and alignment markers
            clean_text = re.sub(r'<.*?>|align:start position:\d+%', '', text)
            # 2. Remove duplicate phrases (like "we we" -> "we")
            clean_text = re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', clean_text)
            # 3. Normalize whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            # Convert timestamps
            start_sec = self._timestamp_to_seconds(start)
            end_sec = self._timestamp_to_seconds(end)
            
            parsed.append({
                'start': start,
                'end': end,
                'start_seconds': start_sec,
                'end_seconds': end_sec,
                'duration': end_sec - start_sec,
                'text': clean_text
            })
        return parsed
    
    def _timestamp_to_seconds(self, timestamp):
        """Convert timestamp (HH:MM:SS.mmm) to seconds."""
        h, m, s = timestamp.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    
    def _seconds_to_timestamp(self, seconds):
        """Convert seconds to timestamp (HH:MM:SS)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def analyze_sentiment(self, captions):
        """Analyze sentiment for each caption."""
        for caption in captions:
            sentiment = self.sia.polarity_scores(caption['text'])
            caption['sentiment_compound'] = sentiment['compound']
            caption['sentiment'] = 'positive' if sentiment['compound'] > 0.05 else 'negative' if sentiment['compound'] < -0.05 else 'neutral'
            # Calculate absolute sentiment intensity (how emotional the text is, regardless of direction)
            caption['sentiment_intensity'] = abs(sentiment['compound'])
        return captions

    def identify_topics(self, captions, top_n=10):
        """Identify main topics in the podcast using simpler word-based approach."""
        all_text = ' '.join([c['text'] for c in captions])
        
        # Simple word extraction using regex instead of nltk tokenization
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Create potential phrases (adjacent words)
        bigrams = []
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]} {words[i+1]}")
            
        # Filter common words and phrases
        stop_words = {'the', 'and', 'that', 'this', 'with', 'for', 'from', 'but', 'not', 'you', 'have', 'are', 'was', 
                     'were', 'they', 'will', 'what', 'when', 'why', 'how', 'your', 'who', 'all', 'can', 'just', 'some', 
                     'like', 'very', 'know', 'think', 'going', 'yeah', 'okay', 'right', 'well', 'sure', 'thing', 'things'}
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        bigram_counts = Counter(bigrams)
        
        # Combine counts with more weight for bigrams
        all_counts = word_counts + bigram_counts
        
        return all_counts.most_common(top_n)
    
    def detect_questions(self, text):
        """Detect if text contains questions, which are common in podcast discussions."""
        # Look for question marks
        has_question_mark = '?' in text
        
        # Look for question words at the beginning of sentences
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'do', 'does', 'did', 'is', 'are', 'can', 'could']
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text.lower())
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                first_word = sentence.split()[0] if sentence.split() else ""
                if first_word in question_words:
                    return True
        
        return has_question_mark
    
    def calculate_speech_density(self, segment_text, duration):
        """Calculate how dense the speech is (words per minute)."""
        words = len(re.findall(r'\b\w+\b', segment_text))
        minutes = duration / 60
        return words / minutes if minutes > 0 else 0

    def detect_debate_or_disagreement(self, text):
        """Detect indicators of debate or disagreement in text."""
        debate_phrases = [
            'disagree', 'i don\'t think', 'but actually', 'not true', 'incorrect', 
            'i don\'t agree', 'counter', 'on the contrary', 'that\'s not', 'no, ', 
            'however,', 'actually,', 'to be fair', 'i would argue', 'the problem is'
        ]
        
        text_lower = text.lower()
        for phrase in debate_phrases:
            if phrase in text_lower:
                return True
        return False
    
    def segment_podcast(self, captions, window_size=90):
        """Segment the podcast into potential clips, optimized for podcast content."""
        max_video_length = max([c['end_seconds'] for c in captions])
        segments = []
        
        # Create overlapping segments - using larger window size for podcasts
        start_time = 0
        while start_time < max_video_length - window_size/2:
            end_time = start_time + window_size
            
            # Get captions in this segment
            segment_captions = [c for c in captions if c['start_seconds'] >= start_time and c['end_seconds'] <= end_time]
            
            if segment_captions:
                # Calculate segment metrics
                sentiment_values = [c['sentiment_compound'] for c in segment_captions]
                intensity_values = [c['sentiment_intensity'] for c in segment_captions]
                segment_text = ' '.join([c['text'] for c in segment_captions])
                
                segment = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'start': self._seconds_to_timestamp(start_time),
                    'end': self._seconds_to_timestamp(end_time),
                    'duration': end_time - start_time,
                    'caption_count': len(segment_captions),
                    'avg_sentiment': sum(sentiment_values) / len(sentiment_values) if sentiment_values else 0,
                    'sentiment_variance': sum((x - (sum(sentiment_values) / len(sentiment_values)))**2 for x in sentiment_values) / len(sentiment_values) if len(sentiment_values) > 1 else 0,
                    'max_intensity': max(intensity_values) if intensity_values else 0,
                    'text': segment_text,
                    'has_question': self.detect_questions(segment_text),
                    'has_debate': self.detect_debate_or_disagreement(segment_text),
                    'speech_density': self.calculate_speech_density(segment_text, end_time - start_time),
                    'heatmap_value': 0  # Will be populated later if heatmap is available
                }
                segments.append(segment)
            
            # Advance with 50% overlap
            start_time += window_size / 2
            
        return segments
    
    def apply_heatmap(self, segments, heatmap_data):
        """Apply heatmap values to segments."""
        if not heatmap_data:
            return segments
            
        for segment in segments:
            total_value = 0
            count = 0
            
            for heatmap_segment in heatmap_data:
                # Calculate overlap between our segment and heatmap segment
                overlap_start = max(segment['start_time'], heatmap_segment['start_time'])
                overlap_end = min(segment['end_time'], heatmap_segment['end_time'])
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    total_value += heatmap_segment['value'] * overlap_duration
                    count += overlap_duration
            
            if count > 0:
                segment['heatmap_value'] = total_value / count
            else:
                # If no overlap, use nearest heatmap value
                closest = min(heatmap_data, key=lambda x: abs(x['start_time'] - segment['start_time']))
                segment['heatmap_value'] = closest['value']
                
        return segments
    
    def score_podcast_segments(self, segments, topics, algorithm='viral', heatmap_weight=0.7):
        """Score segments with accurate heatmap value passthrough."""
        scored_segments = []
        
        for segment in segments:
            # Initialize score components
            scores = {
                'content': 0,
                'heatmap': segment.get('heatmap_value', 0) * 100,  # Direct 0-100 scaling
                'final': 0
            }
            
            # CONTENT SCORING (only for viral/combined)
            if algorithm in ['viral', 'combined']:
                # 1. Topic relevance
                topic_matches = sum(
                    1 for word, _ in topics 
                    if word in segment['text'].lower()
                )
                
                # 2. Engagement signals
                engagement = (
                    abs(segment['avg_sentiment']) * 20 +
                    (25 if segment['has_question'] else 0) +
                    (30 if segment['has_debate'] else 0)
                )
                
                # 3. Technical quality
                tech = (
                    10 - min(abs(segment['speech_density'] - 150)/15, 10) +
                    10 - min(abs(segment['duration'] - 60)/6, 10) +
                    segment['sentiment_variance'] * 15
                )
                
                scores['content'] = topic_matches*10 + engagement + tech
            
            # FINAL SCORE
            if algorithm == 'viral':
                scores['final'] = scores['content']
            elif algorithm == 'heatmap':
                scores['final'] = scores['heatmap']  # Pure heatmap value
            else:  # combined
                scores['final'] = (scores['content']*(1-heatmap_weight) + 
                                scores['heatmap']*heatmap_weight)
            
            # Add debug info
            segment.update({
                'viral_score': round(scores['final'], 2),
                'content_score': round(scores['content'], 2),
                'heatmap_score': round(scores['heatmap'], 2),
                'topic_matches': sum(1 for word,_ in topics if word in segment['text'].lower()),
                'raw_heatmap': segment.get('heatmap_value', 0)  # Original 0-1 value
            })
            scored_segments.append(segment)
        
        # Sort with heatmap as primary tiebreaker
        return sorted(scored_segments,
                    key=lambda x: (-x['viral_score'], 
                                -x['heatmap_score'],
                                -x['content_score']))
    
    def analyze(self, file_path, window_size=90, top_segments=5, heatmap_data=None, algorithm='viral'):
        """Main analysis function optimized for podcasts."""
        captions = self.parse_webvtt(file_path)
        print(f"Parsed {len(captions)} captions from podcast transcript")
        
        captions = self.analyze_sentiment(captions)
        print("Sentiment analysis complete")
        
        topics = self.identify_topics(captions, top_n=20)
        print(f"Identified main podcast topics: {', '.join([t[0] for t in topics[:5]])}")
        
        segments = self.segment_podcast(captions, window_size)
        print(f"Created {len(segments)} podcast segments")
        
        if heatmap_data:
            segments = self.apply_heatmap(segments, heatmap_data)
            print("Applied heatmap data to segments")
        
        scored_segments = self.score_podcast_segments(segments, topics, algorithm)
        print(f"Scored all segments using {algorithm} algorithm")
        
        # Return top segments
        return scored_segments[:top_segments], captions, topics
    
    def visualize_results(self, captions, top_segments, video_id=None, output_dir='.', heatmap_data=None):
        """Create podcast-specific visualizations of sentiment over time and highlight top segments."""
        # Convert captions to DataFrame
        df = pd.DataFrame(captions)
        
        # Create figure with subplots
        if heatmap_data:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot 1: Sentiment over time with highlighted segments
        ax1.plot(df['start_seconds'], df['sentiment_compound'], 'b-', alpha=0.7)
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Calculate minutes for x-axis
        max_time = df['end_seconds'].max()
        x_ticks = [i*600 for i in range(int(max_time/600) + 1)]  # Every 10 minutes
        x_labels = [f"{i*10}m" for i in range(int(max_time/600) + 1)]
        
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels)
        
        # Highlight top segments
        colors = plt.cm.tab10(range(min(len(top_segments), 10)))  # Get colors from a colormap
        for i, segment in enumerate(top_segments):
            color = colors[i % len(colors)]
            ax1.axvspan(segment['start_time'], segment['end_time'], color=color, alpha=0.3)
            ax1.text(segment['start_time'], 1.1, f"{i+1}: {segment['start']}", 
                     ha='left', va='center', backgroundcolor='white', fontsize=8)
        
        title = 'Podcast Sentiment Analysis and Top Viral Moments'
        if video_id:
            title += f" for Video: {video_id}"
            
        ax1.set_title(title)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Sentiment Score')
        ax1.set_ylim(-1.1, 1.3)
        
        # Plot 2: Timeline of top moments
        ax2.set_xlim(0, max_time)
        ax2.set_ylim(0, 1)
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels)
        ax2.set_yticks([])
        ax2.set_title("Top Viral Moments Timeline")
        
        # Add segments as colored blocks in timeline
        for i, segment in enumerate(top_segments):
            color = colors[i % len(colors)]
            ax2.add_patch(plt.Rectangle((segment['start_time'], 0.1), 
                                        segment['duration'], 0.8, 
                                        alpha=0.7, color=color))
            ax2.text(segment['start_time'] + segment['duration']/2, 0.5, f"{i+1}", 
                    ha='center', va='center', fontweight='bold')
        
        # Plot 3: Heatmap visualization if available
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            ax3.fill_between(heatmap_df['start_time'], heatmap_df['value'], color='red', alpha=0.3)
            ax3.plot(heatmap_df['start_time'], heatmap_df['value'], 'r-', alpha=0.7)
            ax3.set_title("YouTube Heatmap Data")
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Heatmap Value')
            ax3.set_xticks(x_ticks)
            ax3.set_xticklabels(x_labels)
            ax3.set_xlim(0, max_time)
            
        plt.tight_layout()
        
        # Save the plot
        if video_id:
            plot_file = os.path.join(output_dir, f"{video_id}_podcast_analysis.png")
        else:
            plot_file = os.path.join(output_dir, 'podcast_analysis.png')
            
        plt.savefig(plot_file)
        print(f"Visualization saved to {plot_file}")
        
        return plot_file


def main():
    parser = argparse.ArgumentParser(description='Analyze podcast YouTube videos to find viral moments for short-form content')
    parser.add_argument('--url', help='YouTube podcast video URL')
    parser.add_argument('--file', help='Path to WebVTT file (if already downloaded)')
    parser.add_argument('--window', type=int, default=90, help='Segment window size in seconds (default: 90)')
    parser.add_argument('--top', type=int, default=5, help='Number of top segments to return (default: 5)')
    parser.add_argument('--output-dir', default='output', help='Directory for output files (default: output)')
    parser.add_argument('--heatmap', action='store_true', help='Include YouTube heatmap data in analysis')
    parser.add_argument('--algorithm', choices=['viral', 'heatmap', 'combined'], default='viral',
                       help='Scoring algorithm to use (viral: our analysis, heatmap: YouTube data, combined: both)')
    
    args = parser.parse_args()
    
    if not args.url and not args.file:
        parser.error("Either --url or --file must be provided")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process input source
    video_id = None
    subtitle_file = None
    heatmap_data = None
    
    if args.url:
        try:
            video_id, subtitle_file = YouTubeProcessor.download_subtitles(args.url, args.output_dir)
            
            # Fetch heatmap data if requested
            if args.heatmap:
                try:
                    video_json = YouTubeProcessor.get_video_json(args.url, args.output_dir)
                    heatmap_data = video_json.get('heatmap', None)
                    if heatmap_data:
                        print(f"Found heatmap data with {len(heatmap_data)} segments")
                    else:
                        print("No heatmap data available for this video")
                except Exception as e:
                    print(f"Error fetching heatmap data: {e}")
        except Exception as e:
            print(f"Error processing YouTube URL: {e}")
            return
    else:
        subtitle_file = args.file
        # Try to extract video ID from filename if possible
        file_basename = os.path.basename(subtitle_file)
        if file_basename.endswith('.vtt'):
            potential_video_id = file_basename[:-4]  # Remove .vtt extension
            if re.match(r'^[a-zA-Z0-9_-]{11}$', potential_video_id):  # YouTube IDs are 11 characters
                video_id = potential_video_id
    
    # Determine algorithm parameters
    heatmap_weight = 0.3 if args.algorithm == 'combined' else (1.0 if args.algorithm == 'heatmap' else 0.0)
    
    # Run the analysis
    analyzer = PodcastAnalyzer()
    top_segments, captions, topics = analyzer.analyze(
        subtitle_file, 
        args.window, 
        args.top,
        heatmap_data,
        args.algorithm
    )
    
    # Generate detailed output file
    timestamp_file = os.path.join(args.output_dir, f"{video_id}_podcast_moments.txt" if video_id else "podcast_moments.txt")
    
    with open(timestamp_file, 'w', encoding='utf-8') as f:
        f.write("===== TOP POTENTIAL VIRAL MOMENTS FROM PODCAST =====\n\n")
        
        # Write podcast summary first
        f.write("PODCAST SUMMARY:\n")
        f.write(f"- Total duration: {analyzer._seconds_to_timestamp(max([c['end_seconds'] for c in captions]))}\n")
        f.write(f"- Main topics: {', '.join([t[0] for t in topics[:5]])}\n")
        if heatmap_data:
            f.write(f"- Heatmap segments: {len(heatmap_data)}\n")
        f.write(f"- Algorithm used: {args.algorithm}\n\n")
        
        for i, segment in enumerate(top_segments):
            start_seconds = int(segment['start_time'])
            url = YouTubeProcessor.create_timestamp_url(video_id, start_seconds) if video_id else None
            
            f.write(f"CLIP {i+1}:\n")
            f.write(f"- Time: {segment['start']} to {segment['end']} (Duration: {int(segment['duration'])}s)\n")
            if url:
                f.write(f"- Watch: {url}\n")
            
            # Add more detailed analysis for each segment
            f.write(f"- Viral Score: {segment['viral_score']:.2f}\n")
            if args.algorithm != 'heatmap':
                f.write(f"- Topics: {segment['topic_matches']} main topics mentioned\n")
            if heatmap_data:
                f.write(f"- Heatmap Value: {segment['heatmap_value']:.2f}\n")
            f.write(f"- Features: ")
            features = []
            if args.algorithm != 'heatmap':
                if segment['has_question']:
                    features.append("Contains questions")
                if segment['has_debate']:
                    features.append("Contains debate/disagreement")
                if abs(segment['avg_sentiment']) > 0.2:
                    sentiment_label = 'positive' if segment['avg_sentiment'] > 0.2 else 'negative' if segment['avg_sentiment'] < -0.2 else 'neutral'
                    features.append(f"Emotional content ({sentiment_label})")
                if segment['speech_density'] > 160:
                    features.append("Fast-paced discussion")
                elif segment['speech_density'] < 120:
                    features.append("Measured, thoughtful speech")
            if heatmap_data and segment['heatmap_value'] > 0.5:
                features.append("High engagement (heatmap)")
            f.write(", ".join(features) + "\n")
            
            # Add full transcript
            f.write(f"- Transcript: {segment['text']}\n\n")
            f.write("---\n\n")
    
    print(f"\nDetailed results saved to {timestamp_file}")
    
    # Print condensed results to console
    print("\n===== TOP POTENTIAL VIRAL MOMENTS FROM PODCAST =====")
    print(f"Main podcast topics: {', '.join([t[0] for t in topics[:5]])}")
    if heatmap_data:
        print(f"Heatmap segments: {len(heatmap_data)}")
    print(f"Algorithm used: {args.algorithm}")
    
    for i, segment in enumerate(top_segments):
        start_seconds = int(segment['start_time'])
        url = YouTubeProcessor.create_timestamp_url(video_id, start_seconds) if video_id else None
        
        print(f"\n{i+1}. Time: {segment['start']} to {segment['end']} (Duration: {int(segment['duration'])}s)")
        if url:
            print(f"   Link: {url}")
        print(f"   Viral Score: {segment['viral_score']:.2f}")
        if heatmap_data:
            print(f"   Heatmap Value: {segment['heatmap_value']:.2f}")
        
        # Print features
        features = []
        if args.algorithm != 'heatmap':
            if segment['has_question']:
                features.append("Questions")
            if segment['has_debate']:
                features.append("Debate")
            if abs(segment['avg_sentiment']) > 0.2:
                features.append("Emotional")
        if heatmap_data and segment['heatmap_value'] > 0.5:
            features.append("High Engagement")
        print(f"   Features: {', '.join(features)}")
        
        # Print preview (first 80 chars)
        print(f"   Preview: {segment['text'][:80]}...")
    
    # Create visualization
    analyzer.visualize_results(captions, top_segments, video_id, args.output_dir, heatmap_data)


if __name__ == "__main__":
    main()