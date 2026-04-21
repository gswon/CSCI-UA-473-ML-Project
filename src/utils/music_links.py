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

import urllib.request
import re

def generate_youtube_play_url(track_name, artist_name):
    clean_artist = str(artist_name).replace('[', '').replace(']', '').replace("'", "")
    query = f"{track_name} {clean_artist} official audio"
    encoded_query = urllib.parse.quote(query)
    
    try:
        # Search youtube and extract the first video ID
        url = "https://www.youtube.com/results?search_query=" + encoded_query
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req).read().decode('utf-8')
        
        # Try finding the videoId in the JS objects first
        video_ids = re.findall(r"\"videoId\":\"([a-zA-Z0-9_-]{11})\"", html)
        if not video_ids:
            # Fallback to older watch?v= format
            video_ids = re.findall(r"watch\?v=([a-zA-Z0-9_-]{11})", html)
            
        if video_ids:
            return f"https://www.youtube.com/watch?v={video_ids[0]}"
    except Exception as e:
        print(f"Error fetching YouTube video: {e}")
        
    return None
