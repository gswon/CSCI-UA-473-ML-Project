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
    # Shared lookup path: cache hit → HTTP scrape → yt_dlp fallback.
    # Same logic the thumbnail batch uses, so the Your-pick card, the
    # recommendation grid, and the Up Next sidebar all agree.
    from utils.thumbnails import get_video_id
    vid = get_video_id(track_name, artist_name)
    return f"https://www.youtube.com/watch?v={vid}" if vid else None
