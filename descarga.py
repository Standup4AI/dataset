import subprocess
import sys
import os
import logging
import pandas as pd
import glob
from normalize_names_id import normalize_names_id

def setup_logging(log_file):
    # Configure logging to write to both console and a log file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Log to a file
            logging.StreamHandler(sys.stdout)  # Log to the console
        ]
    )

def download_videos(sub_lang, urls_file, skip_alread_download=True):
    # Check if the file exists
    if not os.path.exists(urls_file):
        logging.error(f"The file {urls_file} does not exist.")
        return

    # Create the download directory if it doesn't exist
    download_dir = os.path.join("../data/standup/videos", sub_lang)
    # download_dir = os.path.join("download", sub_lang)
    os.makedirs(download_dir, exist_ok=True)   

    if skip_alread_download:
        # remove already downloaded urls from the list 
        list_videos = [os.path.splitext(os.path.basename(k))[0] for k in glob.glob(download_dir+'/*.mp4')]
        list_url = pd.read_csv(urls_file).url.tolist()
        len_ini = len(list_url)
        list_url = [k for k in list_url if k.split("?v=")[1] not in list_videos]
        len_end = len(list_url)
        # import pdb
        # pdb.set_trace()
        print(f'Skipping {len_ini-len_end} already downloaded files...')
        pd.DataFrame(list_url, columns=['url']).to_csv(urls_file, index=False)

    # sometimes I put lang_region, like es_latam
    sub_lang_yt = sub_lang.split('_')[0]

    # Base command for yt-dlp
    command = [
        "yt-dlp",
        "--merge-output-format", "mp4",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", sub_lang_yt,
        "-a", urls_file,
        "--output", os.path.join(download_dir, "%(title)s.%(ext)s"),
        "--no-overwrites"  # Skip downloading if the file already exists
    ]

    try:
        # Execute the command
        logging.info(f"Starting download for language: {sub_lang}")
        subprocess.run(command, check=True)
        logging.info("Download completed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing yt-dlp: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Check if the necessary arguments were provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <subtitle_language> <urls_file>")
        sys.exit(1)

    # Arguments
    sub_lang = sys.argv[1]  # Subtitle language (e.g., "fr")
    urls_file = sys.argv[2]  # Name of the file containing the URLs

    # Create the url file
    path_ini, ext = os.path.splitext(urls_file)
    if ext == ".csv":
        df = pd.read_csv(urls_file)
        df['url'].to_csv(path_ini + '.txt', index=False)
        urls_file = path_ini + '.txt'

    # Set up logging
    # log_file = os.path.join("download", sub_lang, "download.log")
    log_file = os.path.join("../data/standup/videos", sub_lang, "download.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure the directory exists
    setup_logging(log_file)

    # Call the download function
    download_videos(sub_lang, urls_file)

    # remove the url file
    if ext == ".csv": os.remove(path_ini + '.txt')

    if ext == ".csv":
        normalize_names_id(os.path.split(path_ini)[-1]+'.csv', sub_lang, test=False)
