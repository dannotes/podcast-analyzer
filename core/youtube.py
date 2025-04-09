import os
import re
import json
import urllib.parse
import subprocess
from typing import Tuple, Optional

class YouTubeProcessor:
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc == 'youtu.be':
            return parsed.path[1:]
        if parsed.netloc in ('www.youtube.com', 'youtube.com', 'youtu.be'):
            if parsed.path == '/watch':
                query = urllib.parse.parse_qs(parsed.query)
                return query.get('v', [None])[0]
            if parsed.path.startswith('/v/'):
                return parsed.path.split('/')[2]
        return None

    @staticmethod
    def download_subtitles(url: str, output_dir: str = 'output') -> Tuple[str, str]:
        video_id = YouTubeProcessor.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {url}")

        os.makedirs(output_dir, exist_ok=True)
        
        # Run yt-dlp with default subtitle naming
        subprocess.run([
            'yt-dlp',
            '--skip-download',
            '--write-auto-sub',
            '--sub-format', 'vtt',
            '--output', os.path.join(output_dir, f"{video_id}"),  # Base name, let yt-dlp append language
            url
        ], check=True)

        # Find the generated subtitle file
        vtt_files = [
            f for f in os.listdir(output_dir)
            if f.startswith(video_id) and f.endswith('.vtt')
        ]
        
        if not vtt_files:
            raise FileNotFoundError(f"No subtitle file found for {video_id}")
        
        # Use the first matching file as the final path (e.g., BlN7RIHu03I.en.vtt)
        final_path = os.path.join(output_dir, vtt_files[0])
        
        return video_id, final_path

    @staticmethod
    def get_video_json(url: str, output_dir: str = 'output') -> dict:
        """Fetch video metadata including heatmap."""
        video_id = YouTubeProcessor.extract_video_id(url)
        json_path = os.path.join(output_dir, f"{video_id}.json")

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)

        result = subprocess.run([
            'yt-dlp',
            '--dump-json',
            '--skip-download',
            '--output', os.path.join(output_dir, f"{video_id}"),
            url
        ], capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)
        with open(json_path, 'w') as f:
            json.dump(data, f)
        return data