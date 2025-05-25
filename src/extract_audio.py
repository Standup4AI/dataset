import os
import subprocess
import argparse
from pathlib import Path

def extract_audio(input_folder, output_folder):
    """
    Extract audio from all MP4 files in input_folder and save as WAV in output_folder.
    
    Args:
        input_folder (str): Path to folder containing MP4 files
        output_folder (str): Path to folder where WAV files will be saved
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all MP4 files in input folder
    mp4_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        print(f"No MP4 files found in {input_folder}")
        return
    
    # Process each MP4 file
    for mp4_file in mp4_files:
        input_path = os.path.join(input_folder, mp4_file)
        output_filename = f"{os.path.splitext(mp4_file)[0]}.wav"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"Processing {mp4_file}...")
        
        try:
            # Execute ffmpeg command
            subprocess.run([
                'ffmpeg',
                '-y',
                '-i', input_path,
                '-vn',
                '-acodec', 'pcm_s16le',  # Use PCM codec for WAV
                '-ar', '44100',  # Sample rate
                output_path
            ], check=True)
            print(f"Successfully extracted audio to {output_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing {mp4_file}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {mp4_file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Extract audio from MP4 files in a folder using ffmpeg'
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input folder containing MP4 files',
        type=str,
        dest='input_folder'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output folder for WAV files',
        type=str,
        dest='output_folder'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output',
        default=False
    )

    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        parser.error(f"Input folder '{args.input_folder}' does not exist")
    
    extract_audio(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()