#!/usr/bin/env python3
import argparse
import os
import json
import re
from typing import Optional
from core.youtube import YouTubeProcessor
from core.analyzer import PodcastAnalyzer
from core.visualization import PodcastVisualizer

def ensure_output_dir(output_dir: str):
    """Create output directory if needed."""
    os.makedirs(output_dir, exist_ok=True)

def save_results(results: dict, video_id: Optional[str], output_dir: str):
    """Save analysis results to files."""
    base_name = f"{video_id}_" if video_id else ""
    
    # Save text report
    report_path = os.path.join(output_dir, f"{base_name}report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(generate_text_report(results))
    
    print(f"\nAnalysis saved to {report_path}")

def generate_text_report(results: dict) -> str:
    """Generate formatted text report."""
    report = [
        "===== PODCAST ANALYSIS REPORT =====",
        f"Top {len(results['top_segments'])} Segments:",
        ""
    ]
    
    for i, seg in enumerate(results['top_segments'], 1):
        report.extend([
            f"{i}. {seg['start']} - {seg['end']} (Score: {seg['viral_score']})",
            f"   Heatmap: {seg.get('heatmap_value', 0):.2f}",
            f"   Features: {get_segment_features(seg)}",
            f"   Preview: {seg['text'][:100]}...",
            ""
        ])
    
    report.extend([
        "\nMain Topics:",
        ", ".join([t[0] for t in results['topics'][:5]]),
        f"\nAlgorithm Used: {results['algorithm']}"
    ])
    
    return "\n".join(report)

def get_segment_features(segment: dict) -> str:
    """Generate feature labels for segment."""
    features = []
    if segment.get('has_question'):
        features.append("Questions")
    if segment.get('has_debate'):
        features.append("Debate")
    if abs(segment.get('avg_sentiment', 0)) > 0.2:
        features.append("Emotional")
    if segment.get('heatmap_value', 0) > 0.5:
        features.append("High Engagement")
    return ", ".join(features) if features else "None"

def main():
    parser = argparse.ArgumentParser(
        description='Analyze YouTube podcasts for viral moments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--url', help='YouTube video URL')
    parser.add_argument('--file', help='Path to existing VTT file')
    parser.add_argument('--window', type=int, default=90, 
                       help='Analysis window size in seconds')
    parser.add_argument('--top', type=int, default=5, 
                       help='Number of top segments to return')
    parser.add_argument('--output-dir', default='output', 
                       help='Output directory')
    parser.add_argument('--heatmap', action='store_true', 
                       help='Include YouTube engagement data')
    parser.add_argument('--algorithm', choices=['viral','heatmap','combined'], 
                       default='viral', help='Scoring algorithm')
    
    args = parser.parse_args()
    ensure_output_dir(args.output_dir)
    
    youtube = YouTubeProcessor()
    analyzer = PodcastAnalyzer()
    video_id = None
    heatmap_data = None

    try:
        # Process input source
        if args.url:
            video_id, vtt_path = youtube.download_subtitles(args.url, args.output_dir)
            if args.heatmap:
                video_json = youtube.get_video_json(args.url, args.output_dir)
                heatmap_data = video_json.get('heatmap', [])
                print(f"Found {len(heatmap_data)} heatmap segments")
        elif args.file:
            vtt_path = args.file
            if os.path.basename(args.file).endswith('.vtt'):
                potential_id = os.path.basename(args.file)[:-4]
                if re.match(r'^[a-zA-Z0-9_-]{11}$', potential_id):
                    video_id = potential_id
        else:
            parser.error("Either --url or --file must be provided")

        # Run analysis
        print(f"Analyzing {vtt_path}...")
        top_segments, captions, topics = analyzer.analyze(
            vtt_path=vtt_path,
            heatmap_data=heatmap_data,
            window_size=args.window,
            top_n=args.top,
            algorithm=args.algorithm
        )

        plot_path = PodcastVisualizer.generate_analysis_plot(
            captions=captions,
            top_segments=top_segments,
            video_id=video_id,
            heatmap_data=heatmap_data,
            output_dir=args.output_dir
        )
        print(f"Visualization saved to {plot_path}")

        # Prepare and save results
        results = {
            'top_segments': top_segments,
            'topics': topics,
            'algorithm': args.algorithm,
            'heatmap_segments': len(heatmap_data) if heatmap_data else 0
        }
        
        save_results(results, video_id, args.output_dir)
        
        # Print summary
        print("\n===== TOP SEGMENTS =====")
        for i, seg in enumerate(top_segments, 1):
            print(f"{i}. {seg['start']} - {seg['end']} (Score: {seg['viral_score']})")
            print(f"   {seg['text'][:80]}...\n")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()