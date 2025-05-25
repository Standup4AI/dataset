import yt_dlp
import pandas as pd

def get_standup_video_urls(channel_search_url, list_search_str=["Stand Up"]):
    """
    Extracts video URLs from a channel search page whose titles include the search string.

    Args:
        channel_search_url (str): The URL of the channel’s search results page.
            For example:
            "https://www.youtube.com/@ComedyCentralLA/search?query=%22%7C%20Stand%20Up%20%7C%22"
        search_str (str): The text to search for in video titles (default is "Stand Up").

    Returns:
        list: A list of full YouTube video URLs.
    """
    # Configure yt-dlp to extract a flat list (basic metadata only, no full downloads)
    ydl_opts = {
        'extract_flat': True,  # Do not extract complete metadata for each video
        'skip_download': True, # We are not downloading video files
        'quiet': True,         # Suppress extra logging output
    }

    if len(list_search_str) == 0:
      print('Retrieve all the videos...')
      list_search_str = [" "]

    data_list = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract the info dictionary from the provided URL
        info = ydl.extract_info(channel_search_url, download=False)

        # The entries key holds the list of videos found
        entries = info.get('entries', [])
        for entry in entries:
            title = entry.get('title', '')
            if sum([k.lower() in title.lower() for k in list_search_str]):

                video_data = {
                    "url": entry.get("url", None),
                    "title": entry.get("title", title),
                    "description": entry.get("description", None),
                    "duration": entry.get("duration", None),         # Duration in seconds
                    "channel_id": entry.get("uploader_id", None),      # Typically the channel's ID
                    "channel": entry.get("uploader", None),            # Channel name
                    "view_count": entry.get("view_count", None),
                }
                data_list.append(video_data)

    # Create and return a DataFrame from the collected data.
    df = pd.DataFrame(data_list)
    return df

if __name__=='__main__':
  
    channel_search_url = "https://www.youtube.com/c/ComedyCentralHungary/videos"
    list_search_str = [" | Comedy Club", "Comedy Central Bemutatja"]

    df = get_standup_video_urls(channel_search_url, list_search_str=list_search_str)

    df['channel'] = 'Comedy Central Magyarország'
    df['channel_id'] = "@ComedyCentralHungary"
    df[df.duration > 70].to_csv('ComedyCentralHungary_standup_test.csv', index=False)
