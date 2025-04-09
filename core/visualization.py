import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns  # Import Seaborn
import os
from typing import List, Dict, Optional

class PodcastVisualizer:
    @staticmethod
    def generate_analysis_plot(
        captions: List[Dict],
        top_segments: List[Dict],
        video_id: Optional[str] = None,
        heatmap_data: Optional[List[Dict]] = None,
        output_dir: str = "output"
    ) -> str:
        """Generate the combined analysis plot with original styling."""
        # Use Seaborn's default style
        sns.set_style("darkgrid")  # You can use "whitegrid", "darkgrid", etc.
        fig = plt.figure(figsize=(14, 10))
        
        # Create gridspec for custom layout
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        ax1 = fig.add_subplot(gs[0])  # Sentiment
        ax2 = fig.add_subplot(gs[1])  # Segments timeline
        ax3 = fig.add_subplot(gs[2])  # Heatmap
        
        # Convert to DataFrame
        df = pd.DataFrame(captions)
        max_time = df['end_seconds'].max()
        
        # 1. Sentiment Plot (Top)
        ax1.plot(df['start_seconds'], df['sentiment_compound'], 
                color='#1f77b4', linewidth=1.5, alpha=0.8)
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Sentiment Score', fontsize=10)
        ax1.set_ylim(-1.1, 1.3)
        
        # Highlight top segments
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_segments)))
        for i, seg in enumerate(top_segments):
            ax1.axvspan(seg['start_time'], seg['end_time'], 
                       color=colors[i], alpha=0.3)
            ax1.text(seg['start_time'], 1.1, f"{i+1}", 
                    ha='left', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # 2. Segments Timeline (Middle)
        ax2.set_xlim(0, max_time)
        ax2.set_ylim(0, 1)
        for i, seg in enumerate(top_segments):
            ax2.add_patch(plt.Rectangle(
                (seg['start_time'], 0.1),
                seg['duration'], 0.8,
                color=colors[i], alpha=0.7
            ))
            ax2.text(seg['start_time'] + seg['duration']/2, 0.5,
                   f"{i+1}", ha='center', va='center', 
                   fontweight='bold', color='white')
        ax2.set_ylabel('Top Moments', fontsize=10)
        ax2.set_yticks([])
        
        # 3. Heatmap Plot (Bottom)
        if heatmap_data:
            heat_df = pd.DataFrame(heatmap_data)
            ax3.fill_between(heat_df['start_time'], heat_df['value'],
                           color='#ff7f0e', alpha=0.3)
            ax3.plot(heat_df['start_time'], heat_df['value'],
                    color='#ff7f0e', linewidth=1.5)
            ax3.set_ylim(0, 1)
            ax3.set_ylabel('Engagement', fontsize=10)
        
        # Common X-axis settings
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(0, max_time)
            ax.set_xticks(np.arange(0, max_time + 600, 600))
            ax.set_xticklabels([f"{int(x/60)}m" for x in ax.get_xticks()])
            ax.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Time', fontsize=10)
        
        # Title
        title = "Podcast Sentiment Analysis with Engagement Heatmap"
        if video_id:
            title += f"\nVideo: {video_id}"
        fig.suptitle(title, y=1.02, fontsize=12)
        
        plt.tight_layout()
        
        # Save
        filename = f"{video_id}_analysis.png" if video_id else "analysis.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        return plot_path