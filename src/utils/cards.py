"""
Shared UI helpers for the Streamlit app.

Centralizes card markup that was previously duplicated across Tab 1's
"Your pick" + recommendation grid and Tab 2's vibe results.
"""

import streamlit as st

from utils.thumbnails import thumb_url


_SPOTIFY_ICON = (
    "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg"
)
_YT_MUSIC_ICON = (
    "https://upload.wikimedia.org/wikipedia/commons/6/6a/Youtube_Music_icon.svg"
)


def render_song_card(
    name: str,
    artist: str,
    sp_url: str,
    yt_url: str,
    seed: str = "",
    video_id: str = "",
    badge: str | None = None,
    border_gold: bool = False,
) -> None:
    """
    Render a single square song card with thumbnail, title, artist, and
    Spotify / YouTube Music icon links.

    Args:
        name:        track title
        artist:      artist name(s)
        sp_url:      Spotify search URL
        yt_url:      YouTube Music search URL
        seed:        fallback seed for placeholder thumbnails
        video_id:    YouTube video ID; empty string falls back to picsum seed
        badge:       small label pinned to the top-left (e.g. "YOUR SONG")
        border_gold: gold accent border (used for the "Your pick" highlight)
    """
    border = (
        "2px solid #FFD700"
        if border_gold
        else "1px solid rgba(255,255,255,0.12)"
    )
    badge_html = (
        f"<div style='position:absolute;top:6px;left:6px;z-index:2;"
        f"background:#FFD700;color:#000;font-size:0.62rem;font-weight:700;"
        f"padding:2px 6px;border-radius:4px;'>{badge}</div>"
        if badge
        else ""
    )
    img_src = thumb_url(video_id, size="mq", fallback_seed=str(seed))
    st.markdown(
        f"<div style='border:{border};border-radius:10px;"
        f"overflow:hidden;background:#1a1a1a;margin-bottom:4px;position:relative;'>"
        f"{badge_html}"
        f"<img src='{img_src}'"
        f" style='width:100%;aspect-ratio:1/1;object-fit:cover;display:block;'>"
        f"<div style='padding:7px 9px 8px;'>"
        f"<p style='font-size:0.8rem;font-weight:700;margin:0 0 2px;color:#FFFFFF;"
        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'"
        f" title='{name}'>{name}</p>"
        f"<p style='font-size:0.68rem;color:#999;margin:0 0 6px;"
        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
        f"{artist}</p>"
        f"<div style='display:flex;gap:6px;'>"
        f"<a href='{sp_url}' target='_blank'>"
        f"<img src='{_SPOTIFY_ICON}' width='14' style='opacity:.75'></a>"
        f"<a href='{yt_url}' target='_blank'>"
        f"<img src='{_YT_MUSIC_ICON}' width='14' style='opacity:.75'></a>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )


def render_stat(label: str, value: str) -> None:
    """Small two-line stat: muted label on top, bold value below."""
    st.markdown(
        f"<p style='font-size:0.85rem;color:#888;margin-bottom:0;'>{label}</p>"
        f"<p style='font-size:1.2rem;font-weight:600;margin-top:0;'>{value}</p>",
        unsafe_allow_html=True,
    )
