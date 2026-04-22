import urllib.parse
import streamlit as st

def generate_spotify_search_url(track_name, artist_name):
    # Strip any brackets or single quotes from artist name
    clean_artist = str(artist_name).replace('[', '').replace(']', '').replace("'", "")
    query = f"{track_name} {clean_artist}"
    encoded_query = urllib.parse.quote(query)
    return f"https://open.spotify.com/search/{encoded_query}"

def generate_youtube_music_search_url(track_name, artist_name):
    clean_artist = str(artist_name).replace('[', '').replace(']', '').replace("'", "")
    query = f"{track_name} {clean_artist}"
    encoded_query = urllib.parse.quote(query)
    return f"https://music.youtube.com/search?q={encoded_query}"

def generate_youtube_play_url(track_name, artist_name):
    clean_artist = str(artist_name).replace('[', '').replace(']', '').replace("'", "")
    query = f"{track_name} {clean_artist} official audio"

    try:
        import yt_dlp
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if result and result.get("entries"):
                video_id = result["entries"][0].get("id", "")
                if video_id:
                    return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        print(f"Error fetching YouTube video: {e}")

    return None
