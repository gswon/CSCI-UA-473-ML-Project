import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from models.transformer import TextTransformer
from utils.music_links import (
    generate_spotify_search_url,
    generate_youtube_music_search_url,
)
from utils.thumbnails import get_video_ids_batch, thumb_url


PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed_songs.parquet"
VECTORS_PATH = PROJECT_ROOT / "data" / "song_vectors.npy"
TEXT_MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "text_model.pth"


class CharTokenizer:
    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 -,;'\"()[]{}&|!@#$%^*+=_~`.<>?/\\"
        self.char2idx = {c: i + 1 for i, c in enumerate(chars)}
        self.char2idx["<UNK>"] = len(self.char2idx) + 1
        self.vocab_size = len(self.char2idx) + 1

    def encode(self, text, max_len=64):
        text = str(text).lower()
        indices = [self.char2idx.get(c, self.char2idx["<UNK>"]) for c in text[:max_len]]
        padding = [0] * (max_len - len(indices))
        return indices + padding


@st.cache_resource
def load_vibe_data():
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    return pd.DataFrame(
        columns=[
            "id", "name", "artists", "album",
            "danceability", "energy", "valence",
            "acousticness", "speechiness", "tempo"
        ]
    )


@st.cache_resource
def load_vibe_vectors():
    if VECTORS_PATH.exists():
        return np.load(VECTORS_PATH)
    return None


@st.cache_resource
def load_vibe_transformer():
    tokenizer = CharTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TextTransformer(vocab_size=tokenizer.vocab_size).to(device)

    if TEXT_MODEL_PATH.exists():
        encoder.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device))
    encoder.eval()
    return encoder, tokenizer, device


def vibe_assets_exist():
    return PARQUET_PATH.exists() and VECTORS_PATH.exists() and TEXT_MODEL_PATH.exists()


def _render_rec_card(name, artist, seed, sp_url, yt_url,
                     border_gold=False, badge=None, video_id=""):
    """Render one recommendation card matching the Find-Similar-Songs layout."""
    border = "2px solid #FFD700" if border_gold else "1px solid rgba(255,255,255,0.12)"
    badge_html = (
        f"<div style='position:absolute;top:6px;left:6px;z-index:2;"
        f"background:#FFD700;color:#000;font-size:0.62rem;font-weight:700;"
        f"padding:2px 6px;border-radius:4px;'>{badge}</div>"
        if badge else ""
    )
    img_src = thumb_url(video_id, size="mq", fallback_seed=str(seed))
    st.markdown(
        f"<div style='border:{border};border-radius:10px;"
        f"overflow:hidden;background:#1a1a1a;margin-bottom:4px;position:relative;'>"
        f"{badge_html}"
        f"<img src='{img_src}'"
        f" style='width:100%;aspect-ratio:1/1;object-fit:cover;display:block;'>"
        f"<div style='padding:7px 9px 8px;'>"
        f"<p style='font-size:0.8rem;font-weight:700;margin:0 0 2px;"
        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'"
        f" title='{name}'>{name}</p>"
        f"<p style='font-size:0.68rem;color:#999;margin:0 0 6px;"
        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
        f"{artist}</p>"
        f"<div style='display:flex;gap:6px;'>"
        f"<a href='{sp_url}' target='_blank'><img src='https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg' width='14' style='opacity:.75'></a>"
        f"<a href='{yt_url}' target='_blank'><img src='https://upload.wikimedia.org/wikipedia/commons/6/6a/Youtube_Music_icon.svg' width='14' style='opacity:.75'></a>"
        f"</div></div></div>",
        unsafe_allow_html=True
    )


def render_vibe_extension(top_n=10):
    st.subheader("📝 Describe the vibe you want")
    st.caption("Type a phrase that captures the mood — semantic text search + slider reranking finds matching songs.")

    # ── Query input ──────────────────────────────────────────────────────────
    query = st.text_input(
        "Describe the vibe you want:",
        placeholder="e.g. Late night driving...",
        key="vibe_query",
    )

    # ── Slider expander (reranking knobs) ────────────────────────────────────
    with st.expander("🎛️ Reranking knobs (optional — push results toward specific audio traits)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            slider_energy = st.slider("Vibe Energy", -1.0, 1.0, 0.0, step=0.01, key="vibe_energy")
            slider_dance = st.slider("Danceability", -1.0, 1.0, 0.0, step=0.01, key="vibe_dance")
        with col2:
            slider_mood = st.slider("Mood (Valence)", -1.0, 1.0, 0.0, step=0.01, key="vibe_mood")
            slider_acoustic = st.slider("Acoustic Feel", -1.0, 1.0, 0.0, step=0.01, key="vibe_acoustic")
        with col3:
            slider_speech = st.slider("Vocal Focus (Speechiness)", -1.0, 1.0, 0.0, step=0.01, key="vibe_speech")
            slider_tempo = st.slider("Intensity (Tempo)", -1.0, 1.0, 0.0, step=0.01, key="vibe_tempo")

    if not query:
        st.info("Enter a sentence above to search by mood, vibe, or listening context.")
        return

    if not vibe_assets_exist():
        missing = []
        if not PARQUET_PATH.exists():    missing.append(str(PARQUET_PATH))
        if not VECTORS_PATH.exists():    missing.append(str(VECTORS_PATH))
        if not TEXT_MODEL_PATH.exists(): missing.append(str(TEXT_MODEL_PATH))
        st.warning(
            "The transformer extension is missing one or more required files:\n\n"
            + "\n".join(f"- {m}" for m in missing)
        )
        return

    df = load_vibe_data()
    vectors = load_vibe_vectors()
    encoder, tokenizer, device = load_vibe_transformer()

    if df.empty or vectors is None:
        st.error("Data or vectors are missing. Cannot perform vibe search.")
        return

    if len(df) != len(vectors):
        st.error(
            f"Metadata length ({len(df)}) does not match vector length ({len(vectors)}). "
            "Rebuild processed_songs.parquet and song_vectors.npy from the same filtered dataset."
        )
        return

    # ── Encode query + score candidates ──────────────────────────────────────
    with st.spinner("Searching for the perfect vibe..."):
        tokens = tokenizer.encode(query)
        token_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

        with torch.no_grad():
            query_vector = encoder(token_tensor).cpu().numpy().astype(np.float32)
            query_vector = np.ascontiguousarray(query_vector)

        sim_scores = np.dot(vectors, query_vector.T).flatten()

        top_500_idx = np.argsort(sim_scores)[-500:][::-1]
        top_500_scores = sim_scores[top_500_idx]

        if len(top_500_idx) == 0:
            st.warning("No results found.")
            return

        candidate_df = df.iloc[top_500_idx].copy()
        candidate_df["cosine_sim"] = top_500_scores

        feat_arr = candidate_df[
            ["energy", "danceability", "valence", "acousticness", "speechiness", "tempo"]
        ].values

        slider_weights = np.array([
            slider_energy, slider_dance, slider_mood,
            slider_acoustic, slider_speech, slider_tempo,
        ])

        shifted_features = feat_arr - 0.5
        slider_contributions = np.sum(shifted_features * slider_weights, axis=1)

        candidate_df["final_score"] = (0.7 * candidate_df["cosine_sim"]) + (0.05 * slider_contributions)

        top_results = candidate_df.sort_values(by="final_score", ascending=False).head(top_n).reset_index(drop=True)
        top_results["clean_artists"] = top_results["artists"].astype(str).str.replace(r"\[|\]|'", "", regex=True)

    st.subheader(f"Top {top_n} matching songs for *'{query}'*")

    # ── Optional advanced toggle (show raw scores) ──────────────────────────
    show_scores = st.checkbox(
        "🔍 Show score breakdown (semantic sim + final score)",
        value=False, key="vibe_show_scores",
    )

    # ── 5-column card grid ──────────────────────────────────────────────────
    st.write("")
    COLS = 5
    rec_list = [
        {"name": str(row["name"]), "artist": str(row["clean_artists"])}
        for _, row in top_results.iterrows()
    ]

    # Batch-fetch YouTube video IDs for all cards in parallel (disk-cached,
    # so repeat queries are instant). Used for real thumbnails + faster Listen.
    _pairs = [(t["name"], t["artist"]) for t in rec_list]
    with st.spinner("Loading covers..."):
        _vid_map = get_video_ids_batch(_pairs)

    for _row_start in range(0, len(top_results), COLS):
        _row_recs = top_results.iloc[_row_start:_row_start + COLS]
        _cols = st.columns(COLS)
        for _ci in range(len(_row_recs)):
            _pos = _row_start + _ci
            _row = top_results.iloc[_pos]
            _seed = str(_row.get("id", _pos))
            _name = str(_row["name"])
            _artist = str(_row["clean_artists"])
            _sp_url = generate_spotify_search_url(_name, _artist)
            _yt_url = generate_youtube_music_search_url(_name, _artist)
            _vid = _vid_map.get((_name, _artist), "")
            with _cols[_ci]:
                _render_rec_card(
                    name=_name, artist=_artist, seed=_seed,
                    sp_url=_sp_url, yt_url=_yt_url,
                    video_id=_vid,
                )

                if show_scores:
                    _fs = float(_row["final_score"])
                    _cs = float(_row["cosine_sim"])
                    st.markdown(
                        f"<div style='background:rgba(29,185,84,0.12);"
                        f"border:1px solid rgba(29,185,84,0.35);"
                        f"border-radius:6px;padding:4px 8px;margin:4px 0;"
                        f"font-size:0.7rem;color:#ccc;'>"
                        f"score <b style='color:#1DB954'>{_fs:.3f}</b> · "
                        f"sim <b style='color:#1DB954'>{_cs:.3f}</b>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                if st.button("▶ Listen", key=f"vibe_listen_{_seed}_{_pos}",
                             use_container_width=True):
                    st.session_state["now_playing"] = {
                        "name":   _name,
                        "artist": _artist,
                        "ts":     _time.time(),
                    }
                    st.session_state["now_playing_idx"] = _pos
                    st.session_state["rec_list"] = rec_list
                    for _k in ("_track_data", "_track_data_key"):
                        st.session_state.pop(_k, None)
                    st.rerun()
