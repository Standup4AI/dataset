"""
Script made by user to download and cut the videos using the timestamps of the candidate laughters. 
I think this can be adapted by using the videos already downloaded by our past pipeline....
"""
import os
import pandas as pd
import subprocess
from pathlib import Path
import argparse

def download_and_trim_videos(csv_file, video_id, output_dir="trimmed_videos"):
    """
    Process a CSV file to download and trim YouTube videos based on start and end times.
    Only processes rows with 'Added' in the source column.
    
    Args:
        csv_file (str): Path to the CSV file with t0, t1, and source columns
        video_id (str): YouTube video ID
        output_dir (str): Directory to save trimmed videos
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp_downloads")
    temp_dir.mkdir(exist_ok=True)
    
    # Download the full video once
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    temp_file = temp_dir / f"{video_id}_full.mp4"
    
    print(f"Reading CSV file: {csv_file}")
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check for required columns
        required_columns = ['t0', 't1', 'source']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in the CSV file.")
                return
        
        # Filter for rows with 'Added' as source
        added_df = df[df['source'] == 'Added']
        total_clips = len(added_df)
        
        if total_clips == 0:
            print("No rows with 'Added' source found in the CSV file.")
            return
            
        print(f"Found {total_clips} clips to process.")
        
        # Download the full video only once
        print(f"Downloading full video {video_id}...")
        download_cmd = [
            "yt-dlp", 
            "-f", "best[ext=mp4]", 
            "-o", str(temp_file),
            video_url
        ]
        
        try:
            subprocess.run(download_cmd, check=True)
        except subprocess.CalledProcessError:
            print("Error: Failed to download video.")
            return
        
        # Process each row marked as 'Added'
        for index, row in added_df.iterrows():
            start_time = float(row['t0'])
            end_time = float(row['t1'])
            
            # Create output filename
            output_filename = output_path / f"{video_id}_{start_time:.2f}_{end_time:.2f}.mp4"
            
            # Trim the video using ffmpeg
            print(f"Processing clip {index+1}/{total_clips}: {start_time:.2f}s to {end_time:.2f}s")
            duration = end_time - start_time
            trim_cmd = [
                "ffmpeg",
                "-i", str(temp_file),
                "-ss", str(start_time),
                "-t", str(duration),
                "-c:v", "libx264", 
                "-c:a", "aac",
                "-y",  # Overwrite output files without asking
                str(output_filename)
            ]
            
            try:
                subprocess.run(trim_cmd, check=True)
                print(f"✓ Created: {output_filename}")
            except subprocess.CalledProcessError:
                print(f"✗ Error: Failed to trim clip {start_time:.2f}s to {end_time:.2f}s")
        
        # Clean up the temp file
        os.remove(temp_file)
        print(f"Finished processing {total_clips} clips. Videos saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download and trim YouTube videos from CSV file')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with t0, t1, and source columns')
    parser.add_argument('--video-id', type=str, required=True, help='YouTube video ID')
    parser.add_argument('--output-dir', type=str, default="trimmed_videos", help='Output directory for trimmed videos')
    
    args = parser.parse_args()
    
    download_and_trim_videos(args.csv, args.video_id, args.output_dir)

if __name__ == "__main__":
    main()