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
    # Consult the shared disk-backed video-ID cache first — the thumbnail
    # layer already populated it for every visible card, so this is usually
    # an instant hit.
    try:
        from utils.thumbnails import get_video_id, set_video_id
        vid = get_video_id(track_name, artist_name)
        if vid:
            return f"https://www.youtube.com/watch?v={vid}"
    except Exception:
        set_video_id = None  # type: ignore

    # Fallback: yt_dlp (slower but more robust against YouTube edge cases).
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
                    if set_video_id:
                        set_video_id(track_name, artist_name, video_id)
                    return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        print(f"Error fetching YouTube video: {e}")

    return None
