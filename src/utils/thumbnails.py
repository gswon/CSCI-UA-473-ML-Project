"""
Fast YouTube video ID lookup + thumbnail URL helpers.

Strategy
--------
1. Persistent JSON cache on disk (data/video_id_cache.json) — song lookups are
   stable, so we only hit the network once per (name, artist) pair across all
   sessions.
2. Direct HTTP scraping of YouTube search HTML for a videoId — ~5-10x faster
   than spinning up yt_dlp's full extractor pipeline.
3. Parallel batch fetching via ThreadPoolExecutor — network-bound, so we can
   safely fan out 20 workers for a row of cards.

The thumbnail URL is built from the video ID:
    https://img.youtube.com/vi/{video_id}/mqdefault.jpg   (320x180, always exists)
"""

import json
import re
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_FILE = PROJECT_ROOT / "data" / "video_id_cache.json"

_VIDEO_ID_RE = re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"')
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/120.0 Safari/537.36")

_cache_lock = threading.Lock()
_memory_cache: dict | None = None


def _load_cache() -> dict:
    global _memory_cache
    if _memory_cache is not None:
        return _memory_cache
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text())
            _memory_cache = data if isinstance(data, dict) else {}
        except Exception:
            _memory_cache = {}
    else:
        _memory_cache = {}
    return _memory_cache


def _save_cache() -> None:
    cache = _load_cache()
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(CACHE_FILE)
    except Exception:
        pass


def _clean_artist(artist: str) -> str:
    return str(artist).replace("[", "").replace("]", "").replace("'", "").strip()


def _cache_key(name: str, artist: str) -> str:
    return f"{str(name).strip().lower()}||{_clean_artist(artist).lower()}"


def _fetch_http(name: str, artist: str, timeout: float = 3.0) -> str:
    query = f"{name} {_clean_artist(artist)} audio"
    q_encoded = urllib.parse.quote(query)
    url = f"https://www.youtube.com/results?search_query={q_encoded}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA,
            "Accept-Language": "en-US,en;q=0.9",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        m = _VIDEO_ID_RE.search(html)
        return m.group(1) if m else ""
    except Exception:
        return ""


def get_video_id(name: str, artist: str) -> str:
    """Single lookup — returns cached ID or fetches once."""
    if not name:
        return ""
    cache = _load_cache()
    key = _cache_key(name, artist)
    if key in cache and cache[key]:
        return cache[key]
    vid = _fetch_http(name, artist)
    # Only cache real hits; empty results stay retryable so a transient
    # YouTube failure does not poison the card thumbnail forever.
    if vid:
        with _cache_lock:
            cache[key] = vid
        _save_cache()
    return vid


def get_video_ids_batch(pairs: list, max_workers: int = 20) -> dict:
    """
    Batch lookup. pairs: list of (name, artist) tuples.
    Returns {(name, artist): video_id_or_empty}.
    """
    cache = _load_cache()
    result: dict = {}
    to_fetch: list = []
    for name, artist in pairs:
        if not name:
            result[(name, artist)] = ""
            continue
        key = _cache_key(name, artist)
        if key in cache and cache[key]:
            result[(name, artist)] = cache[key]
        else:
            to_fetch.append((name, artist, key))

    if not to_fetch:
        return result

    def _one(item):
        n, a, k = item
        return k, (n, a), _fetch_http(n, a)

    workers = max(1, min(len(to_fetch), max_workers))
    wrote_any = False
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, pair, vid in ex.map(_one, to_fetch):
            # Skip cache writes for failed scrapes so they retry next time
            # instead of permanently rendering a picsum placeholder.
            if vid:
                with _cache_lock:
                    cache[k] = vid
                wrote_any = True
            result[pair] = vid

    if wrote_any:
        _save_cache()
    return result


def set_video_id(name: str, artist: str, video_id: str) -> None:
    """Back-populate the cache (e.g., after a yt_dlp fallback succeeds)."""
    if not name or not video_id:
        return
    cache = _load_cache()
    key = _cache_key(name, artist)
    with _cache_lock:
        cache[key] = video_id
    _save_cache()


def thumb_url(video_id: str, size: str = "mq", fallback_seed: str = "") -> str:
    """YouTube thumbnail URL, with picsum fallback when no video ID was found."""
    if video_id:
        return f"https://img.youtube.com/vi/{video_id}/{size}default.jpg"
    if fallback_seed:
        return f"https://picsum.photos/seed/{fallback_seed}/200/200"
    return ""
