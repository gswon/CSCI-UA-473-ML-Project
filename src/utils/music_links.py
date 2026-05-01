import urllib.parse

from utils.df_helpers import clean_artist


_SERVICE_PREFIXES = {
    "spotify": "https://open.spotify.com/search/",
    "ytmusic": "https://music.youtube.com/search?q=",
}


def _search_url(track_name, artist_name, service: str) -> str:
    query = f"{track_name} {clean_artist(artist_name)}"
    return _SERVICE_PREFIXES[service] + urllib.parse.quote(query)


def generate_spotify_search_url(track_name, artist_name):
    return _search_url(track_name, artist_name, "spotify")


def generate_youtube_music_search_url(track_name, artist_name):
    return _search_url(track_name, artist_name, "ytmusic")


def search_links(track_name, artist_name) -> dict:
    return {
        "spotify": _search_url(track_name, artist_name, "spotify"),
        "ytmusic": _search_url(track_name, artist_name, "ytmusic"),
    }


def generate_youtube_play_url(track_name, artist_name):
    # Shared lookup path: cache hit → HTTP scrape → yt_dlp fallback.
    # Same logic the thumbnail batch uses, so the Your-pick card, the
    # recommendation grid, and the Up Next sidebar all agree.
    from utils.thumbnails import get_video_id
    vid = get_video_id(track_name, artist_name)
    return f"https://www.youtube.com/watch?v={vid}" if vid else None
