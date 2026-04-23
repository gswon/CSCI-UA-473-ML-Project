"""
app.py — Spotify Mood Playlist Generator (Personalized)

Run with: streamlit run src/app.py
"""

import sys
import json as _json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from preprocess import (preprocess, normalize_query, apply_feature_weights,
                        AUDIO_FEATURES, FEATURE_LABELS)
from kmeans import KMeans, elbow_method
from recommend import get_recommendations, song_to_vector, fuzzy_search
from reduce import pca_reduce
from mood_labels import label_all_clusters
from vibe_extension import render_vibe_extension

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="🎵 Spotify Mood Clusters", page_icon="🎵", layout="wide")
st.title("🎵 Spotify Mood Playlist Generator")
st.caption("Personalized k-means clustering — focus on the audio features YOU care about most.")

DATA_PATH = Path(__file__).parent.parent / "data" / "tracks_features.csv"

if not DATA_PATH.exists():
    st.error(
        "**Dataset not found.**\n\n"
        f"Expected: `{DATA_PATH}`\n\n"
        "Download the dataset and place `tracks_features.csv` in the `data/` folder."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading 1.2M songs... (first load only)")
def load_data():
    return preprocess(DATA_PATH)

try:
    X_norm, X_min, X_max, df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Search lookup — built once at startup
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_name_lookup(_df):
    name_col   = "name"    if "name"    in _df.columns else "track_name"
    artist_col = "artists" if "artists" in _df.columns else "track_artist"
    names   = _df[name_col].fillna("").astype(str)
    artists = _df[artist_col].fillna("").astype(str).str.replace(r"[\[\]']", "", regex=True).str.strip()
    lower   = names.str.lower()
    return names.to_numpy(), artists.to_numpy(), lower

_names_arr, _artists_arr, _lower_series = build_name_lookup(df)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎵 Find Similar Songs",
    "📝 Describe a Vibe",
    "📊 Evaluation",
    "📖 Project Overview",
])

# ============================================================================
# TAB 1 — Find Similar Songs
# ============================================================================
with tab1:
    # Detect player state at the very top so CSS is injected before anything renders
    _is_playing = "now_playing" in st.session_state

    # Only override Streamlit's layout when the player panel is visible.
    # When not playing, inject nothing — let Streamlit use its default
    # centering and max-width so the page looks normal.
    if _is_playing:
        st.markdown("""<style>
        [data-testid="stMainBlockContainer"],
        .main .block-container {
            padding-right: 295px !important;
            padding-bottom: 110px !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        @media (max-width: 960px) {
            [data-testid="stMainBlockContainer"],
            .main .block-container {
                padding-right: unset !important;
            }
            #yt-playlist-panel { display: none !important; }
        }
        </style>""", unsafe_allow_html=True)

    st.subheader("Find songs that sound like one you already love")

    # ── Inline controls ──────────────────────────────────────────────────────
    with st.expander("⚙️ Clustering & Recommendation Settings", expanded=False):
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            k = st.slider("Number of mood clusters (k)", 2, 20, 8,
                          help="How many distinct mood groups to create.")
        with ctrl_col2:
            top_n = st.slider("Recommendations to show", 3, 30, 10)

        st.markdown("**🎛️ Feature Weights** — drag to emphasize what matters most (0 = ignore)")

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            dance_preset = st.button("💃 Dance", help="Focus on danceability + tempo + energy")
        with pc2:
            chill_preset = st.button("😌 Chill", help="Focus on acousticness + valence + low energy")
        with pc3:
            reset_preset = st.button("↺ Reset", help="Equal weights on all features")

        if "weights" not in st.session_state:
            st.session_state.weights = {f: 1.0 for f in AUDIO_FEATURES}

        def _apply_preset(vals: dict):
            st.session_state.weights = vals
            # Must also update each slider's own session_state key; otherwise
            # Streamlit ignores the value= parameter and keeps the old slider position.
            for feat, val in vals.items():
                st.session_state[f"w_{feat}"] = float(val)

        if dance_preset:
            _apply_preset({
                "danceability": 3.0, "energy": 3.0, "tempo": 3.0,
                "loudness": 1.0, "speechiness": 0.5, "acousticness": 0.5,
                "instrumentalness": 0.5, "liveness": 0.5, "valence": 1.0,
            })
        if chill_preset:
            _apply_preset({
                "danceability": 0.5, "energy": 0.5, "tempo": 0.5,
                "loudness": 0.5, "speechiness": 0.5, "acousticness": 3.0,
                "instrumentalness": 2.0, "liveness": 0.5, "valence": 3.0,
            })
        if reset_preset:
            _apply_preset({f: 1.0 for f in AUDIO_FEATURES})

        user_weights = {}
        feature_groups = {
            "Mood & Vibe 😊": ["valence", "energy", "danceability"],
            "Acoustics & Setup 🎸": ["acousticness", "instrumentalness", "liveness"],
            "Audio Properties 🔊": ["loudness", "tempo", "speechiness"],
        }
        w_cols = st.columns(3)
        for col_idx, (group_name, feats) in enumerate(feature_groups.items()):
            with w_cols[col_idx]:
                st.markdown(f"**{group_name}**")
                for feat in feats:
                    if feat in AUDIO_FEATURES:
                        user_weights[feat] = st.slider(
                            FEATURE_LABELS.get(feat, feat),
                            min_value=0.0, max_value=5.0,
                            value=float(st.session_state.weights.get(feat, 1.0)),
                            step=0.5, key=f"w_{feat}"
                        )
        for feat in AUDIO_FEATURES:
            if feat not in user_weights:
                user_weights[feat] = st.session_state.weights.get(feat, 1.0)
        st.session_state.weights = user_weights

    user_weights = st.session_state.weights
    weight_vector = np.array([user_weights[f] for f in AUDIO_FEATURES], dtype=np.float32)
    weight_key = tuple(round(w, 1) for w in weight_vector.tolist()) + (k,)

    # ── Fit k-means ──────────────────────────────────────────────────────────
    @st.cache_resource(show_spinner="Fitting k-means with your feature weights...")
    def fit_weighted_model(weight_key):
        _k = weight_key[-1]
        _weights = np.array(weight_key[:-1], dtype=np.float32)
        X_w = apply_feature_weights(X_norm, _weights)
        model = KMeans(k=_k, random_seed=42)
        model.fit(X_w)
        return model

    @st.cache_data(show_spinner="Projecting to 2D...")
    def get_2d_projection(weight_key):
        _weights = np.array(weight_key[:-1], dtype=np.float32)
        X_w = apply_feature_weights(X_norm, _weights)
        return pca_reduce(X_w)

    try:
        model = fit_weighted_model(weight_key)
    except Exception as e:
        st.error(f"K-means failed: {e}")
        st.stop()

    df_tab1 = df.iloc[:len(X_norm)].copy().reset_index(drop=True)
    df_tab1["cluster_id"] = model.labels_
    mood_map = label_all_clusters(model.centroids, feature_names=AUDIO_FEATURES)
    df_tab1["mood"] = df_tab1["cluster_id"].map(mood_map)

    name_col   = "name"    if "name"    in df_tab1.columns else "track_name"
    artist_col = "artists" if "artists" in df_tab1.columns else "track_artist"

    # Active weights summary
    active = [(FEATURE_LABELS.get(f, f), w) for f, w in user_weights.items() if w > 0]
    active.sort(key=lambda x: -x[1])
    top_active = [f"{label} ({w:.1f}x)" for label, w in active[:4] if w != 1.0]
    if top_active:
        st.info(f"**Current focus:** {' · '.join(top_active)}")

    # ── Song search ──────────────────────────────────────────────────────────
    _scol1, _scol2 = st.columns(2)
    with _scol1:
        search_input = st.text_input(
            "Song name:",
            placeholder="e.g. Bohemian Rhapsody",
            key="song_search_input",
        )
    with _scol2:
        artist_input = st.text_input(
            "Artist name: (optional — narrows results)",
            placeholder="e.g. Queen",
            key="artist_search_input",
        )

    query_name = ""
    query_artist_filter = None
    if search_input and search_input.strip():
        q_name   = search_input.strip().lower()
        q_artist = artist_input.strip().lower() if artist_input else ""

        match_idxs = _lower_series[_lower_series.str.contains(q_name, regex=False, na=False)].index

        # Filter by artist substring if provided
        seen, options = set(), []
        for i in match_idxs:
            name   = _names_arr[i]
            artist = _artists_arr[i]
            if q_artist and q_artist not in artist.lower():
                continue
            key = (name.lower(), artist.lower())
            if key not in seen:
                seen.add(key)
                options.append((name, artist))

        if not options:
            msg = f'No songs matching "{search_input}"'
            if q_artist:
                msg += f' by "{artist_input}"'
            st.warning(msg + " found.")
        elif len(options) == 1:
            query_name, query_artist_filter = options[0]
        else:
            # Multiple matches — let user pick from a clean selectbox
            labels = [f"{n} — {a}" for n, a in options]
            chosen_label = st.selectbox(
                f"{len(options)} matches — select one:",
                labels,
                key="song_select",
            )
            chosen_idx = labels.index(chosen_label)
            query_name, query_artist_filter = options[chosen_idx]

    # Reset player when song, k, top_n, or any feature weight changes
    _player_ctx = (query_name.strip(), query_artist_filter, k, top_n, weight_key)
    if query_name.strip() and _player_ctx != st.session_state.get("_last_player_ctx"):
        st.session_state["_last_player_ctx"] = _player_ctx
        for _k in ["now_playing", "now_playing_idx", "rec_list", "_track_data", "_track_data_key"]:
            st.session_state.pop(_k, None)
        components.html("""<script>
(function(){
  var p;try{p=window.parent.document;}catch(e){return;}
  ['yt-playlist-panel','yt-fixed-player','yt-pl-style','yt-fixed-style'].forEach(function(id){
    var el=p.getElementById(id); if(el) el.remove();
  });
  var mc=p.querySelector('[data-testid="stMainBlockContainer"]')||p.querySelector('.main .block-container');
  if(mc){mc.style.removeProperty('padding-right');mc.style.removeProperty('padding-bottom');}
})();
</script>""", height=0, scrolling=False)

    if not query_name.strip():
        st.info("Type a song name above to get started. Adjust settings in the expander above to personalize results.")
    else:
        # If the user picked a specific artist version, find that exact row.
        if query_artist_filter:
            _name_col_t  = "name"    if "name"    in df_tab1.columns else "track_name"
            _art_col_t   = "artists" if "artists" in df_tab1.columns else "track_artist"
            _name_mask   = df_tab1[_name_col_t].str.lower() == query_name.strip().lower()
            _clean_arts  = df_tab1[_art_col_t].astype(str).str.replace(r"[\[\]']", "", regex=True).str.strip()
            _art_mask    = _clean_arts.str.lower() == query_artist_filter.lower()
            _exact       = df_tab1[_name_mask & _art_mask]
            if not _exact.empty:
                query_idx = _exact.index[0]
                query_vec = X_norm[query_idx]
            else:
                query_vec, query_idx = song_to_vector(query_name.strip(), df_tab1, X_norm, AUDIO_FEATURES)
        else:
            query_vec, query_idx = song_to_vector(query_name.strip(), df_tab1, X_norm, AUDIO_FEATURES)

        if query_vec is None:
            st.error(f'**"{query_name}"** not found in the dataset.')
            suggestions = fuzzy_search(query_name, df_tab1)
            if suggestions:
                st.write("**Did you mean one of these?**")
                for s in suggestions:
                    st.write(f"  • {s}")
        else:
            query_w    = apply_feature_weights(query_vec[np.newaxis, :], weight_vector)[0]
            cluster_id = int(model.predict(query_w)[0])
            mood       = mood_map[cluster_id]

            # ── Cluster stat cards ────────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"<p style='font-size:0.85rem;color:#888;margin-bottom:0;'>Mood Cluster</p>"
                    f"<p style='font-size:1.2rem;font-weight:600;margin-top:0;'>{mood}</p>",
                    unsafe_allow_html=True
                )
            with col2:
                cluster_size = int((model.labels_ == cluster_id).sum())
                st.markdown(
                    f"<p style='font-size:0.85rem;color:#888;margin-bottom:0;'>Songs in this cluster</p>"
                    f"<p style='font-size:1.2rem;font-weight:600;margin-top:0;'>{cluster_size:,}</p>",
                    unsafe_allow_html=True
                )
            with col3:
                year = df_tab1.iloc[query_idx].get("year", "—") if query_idx is not None else "—"
                st.markdown(
                    f"<p style='font-size:0.85rem;color:#888;margin-bottom:0;'>Release Year</p>"
                    f"<p style='font-size:1.2rem;font-weight:600;margin-top:0;'>{year}</p>",
                    unsafe_allow_html=True
                )

            # ── Audio profile expander ────────────────────────────────────────
            with st.expander("🔬 Audio profile of this song vs. cluster average"):
                song_feats     = X_norm[query_idx] if query_idx is not None else query_vec
                centroid_feats = model.centroids[cluster_id]
                feat_df = pd.DataFrame({
                    "Feature":         [FEATURE_LABELS.get(f, f) for f in AUDIO_FEATURES],
                    "This Song":       song_feats.round(3),
                    "Cluster Average": centroid_feats.round(3),
                })
                fig_bar = px.bar(
                    feat_df.melt(id_vars="Feature", var_name="Source", value_name="Value"),
                    x="Feature", y="Value", color="Source", barmode="group",
                    title="Song vs. Cluster Audio Profile", height=350,
                )
                fig_bar.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()
            st.subheader(f"Top {top_n} similar songs (weighted by your preferences)")

            try:
                recs = get_recommendations(
                    query_vec, X_norm, df_tab1, model,
                    weights=weight_vector, top_n=top_n
                )

                from utils.music_links import (generate_spotify_search_url,
                                               generate_youtube_music_search_url,
                                               generate_youtube_play_url)
                import time as _time

                recs[artist_col] = recs[artist_col].astype(str).str.replace(r"\[|\]|'", "", regex=True)
                recs["YouTube Music"] = recs.apply(
                    lambda row: generate_youtube_music_search_url(row[name_col], row[artist_col]), axis=1)
                recs["Spotify"] = recs.apply(
                    lambda row: generate_spotify_search_url(row[name_col], row[artist_col]), axis=1)

                show_cols     = [name_col, artist_col, "year", "mood"]
                show_advanced = st.toggle("🔍 Show advanced details (Euclidean Distance, Audio Features, PCA, Elbow)")
                if show_advanced:
                    if "euclidean_distance" in recs:
                        recs["euclidean_distance"] = recs["euclidean_distance"].round(4)
                        show_cols += ["euclidean_distance"]
                    show_cols += AUDIO_FEATURES
                show_cols += ["YouTube Music", "Spotify"]
                display_df = recs[show_cols]

                _rec_list_full = [
                    {"name": recs.iloc[_i][name_col], "artist": recs.iloc[_i][artist_col]}
                    for _i in range(len(recs))
                ]

                st.write("")
                COLS = 5
                for _row_start in range(0, len(recs), COLS):
                    _row_recs = recs.iloc[_row_start:_row_start + COLS]
                    _cols = st.columns(COLS)
                    for _ci, _row_s in enumerate(_row_recs.itertuples()):
                        _pos = _row_start + _ci
                        _seed = str(getattr(_row_s, "id", str(_pos)))
                        _sp = str(recs.iloc[_pos]["Spotify"])
                        _yt = str(recs.iloc[_pos]["YouTube Music"])
                        _rname = str(getattr(_row_s, name_col, ""))
                        _rartist = str(getattr(_row_s, artist_col, ""))
                        with _cols[_ci]:
                            st.markdown(
                                f"<div style='border:1px solid rgba(255,255,255,0.12);border-radius:10px;"
                                f"overflow:hidden;background:#1a1a1a;margin-bottom:4px;position:relative;'>"
                                f"<img src='https://picsum.photos/seed/{_seed}/200/200'"
                                f" style='width:100%;display:block;'>"
                                f"<div style='padding:7px 9px 8px;'>"
                                f"<p style='font-size:0.8rem;font-weight:700;margin:0 0 2px;"
                                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'"
                                f" title='{_rname}'>{_rname}</p>"
                                f"<p style='font-size:0.68rem;color:#999;margin:0 0 6px;"
                                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
                                f"{_rartist}</p>"
                                f"<div style='display:flex;gap:6px;'>"
                                f"<a href='{_sp}' target='_blank'><img src='https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg' width='14' style='opacity:.75'></a>"
                                f"<a href='{_yt}' target='_blank'><img src='https://upload.wikimedia.org/wikipedia/commons/6/6a/Youtube_Music_icon.svg' width='14' style='opacity:.75'></a>"
                                f"</div></div></div>",
                                unsafe_allow_html=True
                            )
                            if st.button("▶ Listen", key=f"listen_{_seed}_{_pos}",
                                         use_container_width=True):
                                st.session_state["now_playing"] = {
                                    "name": _rname,
                                    "artist": _rartist,
                                    "ts": _time.time()
                                }
                                st.session_state["now_playing_idx"] = _pos
                                st.session_state["rec_list"] = _rec_list_full
                                st.rerun()

                # ── Advanced details ──────────────────────────────────────────
                if show_advanced:
                    st.divider()
                    st.subheader("📊 Detailed Data View")
                    st.dataframe(
                        display_df, use_container_width=True, hide_index=True,
                        column_config={
                            "YouTube Music": st.column_config.LinkColumn("YouTube Music", display_text="▶ YouTube"),
                            "Spotify":       st.column_config.LinkColumn("Spotify",       display_text="🎵 Spotify"),
                        }
                    )

                    st.divider()

                    # ── PCA visualization ─────────────────────────────────────
                    import math as _math
                    import plotly.graph_objects as _go
                    st.subheader("🗺️ Cluster Map (PCA)")
                    st.caption("PCA projects the 9-dimensional weighted feature space to 2D. Each cluster has its own color; your input and its recommendations pop out on top.")
                    st.caption("★ Gold star = your input song · 🔴 Red numbered dots = top recommendations · Colored dots = cluster members.")
                    try:
                        X_2d = get_2d_projection(weight_key)
                        df_tab1["pca_x"] = X_2d[:, 0]
                        df_tab1["pca_y"] = X_2d[:, 1]

                        # Sample background points (per-cluster coloring)
                        df_sample = df_tab1.sample(min(5000, len(df_tab1)), random_state=42).copy()

                        # Recommendations preserve their ranked order (1..N)
                        rec_ordered = [i for i in recs.index.tolist() if i != query_idx]
                        rank_map = {idx: rank + 1 for rank, idx in enumerate(rec_ordered)}

                        # Exclude input + recs from background so they don't render twice
                        _hl_set = set(rec_ordered) | ({query_idx} if query_idx is not None else set())
                        df_sample = df_sample[~df_sample.index.isin(_hl_set)]

                        # Build distinct palette — one color per cluster
                        _palette = (
                            px.colors.qualitative.Bold
                            + px.colors.qualitative.Vivid
                            + px.colors.qualitative.Pastel
                            + px.colors.qualitative.Safe
                        )

                        fig_pca = _go.Figure()

                        # One trace per cluster (background)
                        for _c in sorted(df_tab1["cluster_id"].unique()):
                            _sub = df_sample[df_sample["cluster_id"] == _c]
                            if len(_sub) == 0:
                                continue
                            _mood = mood_map.get(int(_c), "?")
                            _color = _palette[int(_c) % len(_palette)]
                            _hover = [
                                f"<b>{n}</b><br>{a}<br>Cluster {_c}: {_mood}"
                                for n, a in zip(
                                    _sub[name_col].astype(str),
                                    _sub[artist_col].astype(str),
                                )
                            ]
                            fig_pca.add_trace(_go.Scatter(
                                x=_sub["pca_x"], y=_sub["pca_y"],
                                mode="markers",
                                marker=dict(size=4, color=_color, opacity=0.55,
                                            line=dict(width=0)),
                                name=f"C{_c}: {_mood}",
                                hovertext=_hover, hoverinfo="text",
                                legendgroup="clusters",
                                legendgrouptitle_text="Clusters",
                            ))

                        # Recommendation dots (red, drawn above clusters)
                        if rec_ordered:
                            _rx = [float(df_tab1.loc[i, "pca_x"]) for i in rec_ordered]
                            _ry = [float(df_tab1.loc[i, "pca_y"]) for i in rec_ordered]
                            _rh = [
                                f"<b>#{rank_map[i]} — {df_tab1.loc[i, name_col]}</b>"
                                f"<br>{df_tab1.loc[i, artist_col]}"
                                f"<br>Cluster {int(df_tab1.loc[i, 'cluster_id'])}: "
                                f"{mood_map.get(int(df_tab1.loc[i, 'cluster_id']), '?')}"
                                for i in rec_ordered
                            ]
                            fig_pca.add_trace(_go.Scatter(
                                x=_rx, y=_ry, mode="markers",
                                marker=dict(symbol="circle", size=14, color="#FF3333",
                                            line=dict(width=1.5, color="white")),
                                name="🔴 Recommendation",
                                hovertext=_rh, hoverinfo="text",
                                legendgroup="highlights",
                                legendgrouptitle_text="Your pick + recs",
                            ))

                        # Input star (drawn on top of everything)
                        if query_idx is not None and query_idx in df_tab1.index:
                            _qx = float(df_tab1.loc[query_idx, "pca_x"])
                            _qy = float(df_tab1.loc[query_idx, "pca_y"])
                            _qh = (
                                f"<b>{df_tab1.loc[query_idx, name_col]}</b>"
                                f"<br>{df_tab1.loc[query_idx, artist_col]}"
                                f"<br>Cluster {int(df_tab1.loc[query_idx, 'cluster_id'])}: "
                                f"{mood_map.get(int(df_tab1.loc[query_idx, 'cluster_id']), '?')}"
                            )
                            fig_pca.add_trace(_go.Scatter(
                                x=[_qx], y=[_qy], mode="markers",
                                marker=dict(symbol="star", size=26, color="#FFD700",
                                            line=dict(width=2, color="white")),
                                name="★ Your Song",
                                hovertext=[_qh], hoverinfo="text",
                                legendgroup="highlights",
                            ))

                        # Rank annotations (#1..#N) on a small ring around each rec
                        _N = max(len(rec_ordered), 1)
                        for idx in rec_ordered:
                            rank = rank_map[idx]
                            _px = float(df_tab1.loc[idx, "pca_x"])
                            _py = float(df_tab1.loc[idx, "pca_y"])
                            _ang = 2 * _math.pi * ((rank - 1) / _N) - _math.pi / 2
                            _ax = 26 * _math.cos(_ang)
                            _ay = -26 * _math.sin(_ang)
                            fig_pca.add_annotation(
                                x=_px, y=_py, ax=_ax, ay=_ay,
                                xref="x", yref="y", axref="pixel", ayref="pixel",
                                text=f"#{rank}",
                                showarrow=True, arrowhead=2, arrowsize=0.8,
                                arrowwidth=1, arrowcolor="#FF3333",
                                font=dict(color="#FFFFFF", size=11, family="Arial Black"),
                                bgcolor="rgba(255,51,51,0.85)",
                                bordercolor="#FFFFFF", borderwidth=1, borderpad=2,
                            )

                        # Input star label
                        if query_idx is not None and query_idx in df_tab1.index:
                            fig_pca.add_annotation(
                                x=_qx, y=_qy, ax=0, ay=-36,
                                xref="x", yref="y", axref="pixel", ayref="pixel",
                                text="YOUR SONG",
                                showarrow=True, arrowhead=2, arrowsize=0.9,
                                arrowwidth=1.2, arrowcolor="#FFD700",
                                font=dict(color="#000000", size=11, family="Arial Black"),
                                bgcolor="#FFD700", bordercolor="#FFFFFF",
                                borderwidth=1, borderpad=3,
                            )

                        fig_pca.update_layout(
                            title=f"Song Space — k={k} clusters (PCA 2D, {len(df_sample) + len(_hl_set):,} songs shown)",
                            xaxis_title="PC 1", yaxis_title="PC 2",
                            height=640,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            legend=dict(
                                orientation="v",
                                yanchor="top", y=1.0,
                                xanchor="left", x=1.02,
                                font=dict(size=11),
                                bgcolor="rgba(20,20,20,0.6)",
                                bordercolor="#444", borderwidth=1,
                                itemsizing="constant",
                                groupclick="toggleitem",
                            ),
                            margin=dict(l=40, r=170, t=60, b=40),
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)

                        # ── Coordinate table for chosen + recommended songs ────
                        with st.expander("📍 Coordinates of your song + recommendations"):
                            coord_rows = []
                            if query_idx is not None and query_idx in df_tab1.index:
                                coord_rows.append({
                                    "Rank":   "★ Input",
                                    "Name":   str(df_tab1.loc[query_idx, name_col]),
                                    "Artist": str(df_tab1.loc[query_idx, artist_col]),
                                    "PC 1":   round(float(df_tab1.loc[query_idx, "pca_x"]), 3),
                                    "PC 2":   round(float(df_tab1.loc[query_idx, "pca_y"]), 3),
                                    "Cluster": int(df_tab1.loc[query_idx, "cluster_id"]),
                                })
                            for idx in rec_ordered:
                                coord_rows.append({
                                    "Rank":   f"#{rank_map[idx]}",
                                    "Name":   str(df_tab1.loc[idx, name_col]),
                                    "Artist": str(df_tab1.loc[idx, artist_col]),
                                    "PC 1":   round(float(df_tab1.loc[idx, "pca_x"]), 3),
                                    "PC 2":   round(float(df_tab1.loc[idx, "pca_y"]), 3),
                                    "Cluster": int(df_tab1.loc[idx, "cluster_id"]),
                                })
                            coord_df = pd.DataFrame(coord_rows)
                            st.dataframe(coord_df, use_container_width=True, hide_index=True)

                        st.subheader("Cluster Summary")
                        summary = (
                            df_tab1.groupby(["cluster_id", "mood"])
                            .size().reset_index(name="# Songs")
                            .sort_values("cluster_id")
                        )
                        def _summary_rows(df):
                            out = []
                            for r in df.itertuples():
                                n = f"{int(getattr(r, '_3', 0)):,}"
                                out.append(
                                    f"<tr>"
                                    f"<td style='text-align:center;padding:6px 12px;border-bottom:1px solid #333;'>{int(r.cluster_id)}</td>"
                                    f"<td style='text-align:left;padding:6px 12px;border-bottom:1px solid #333;'>{r.mood}</td>"
                                    f"<td style='text-align:left;padding:6px 12px;border-bottom:1px solid #333;'>{n}</td>"
                                    f"</tr>"
                                )
                            return "".join(out)
                        rows = _summary_rows(summary)
                        st.markdown(f"""
                        <table style='width:100%;border-collapse:collapse;font-size:0.9rem;'>
                          <thead>
                            <tr>
                              <th style='text-align:center;padding:6px 12px;border-bottom:2px solid #555;'>cluster_id</th>
                              <th style='text-align:left;padding:6px 12px;border-bottom:2px solid #555;'>mood</th>
                              <th style='text-align:left;padding:6px 12px;border-bottom:2px solid #555;'># Songs</th>
                            </tr>
                          </thead>
                          <tbody>{rows}</tbody>
                        </table>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"PCA visualization error: {e}")

                    st.divider()

                    # ── Elbow plot ────────────────────────────────────────────
                    st.subheader("📉 Elbow Plot — Choosing k")
                    st.caption("Runs k-means for k=2–15 to help pick the optimal number of clusters.")

                    elbow_cache_key = f"elbow_{weight_key}"
                    eb_col1, eb_col2 = st.columns([1, 5])
                    with eb_col1:
                        run_elbow = st.button("▶ Run Elbow Analysis", type="primary",
                                              use_container_width=True, key="elbow_btn")
                    with eb_col2:
                        if elbow_cache_key in st.session_state:
                            if st.button("🔄 Re-run", key="elbow_rerun"):
                                del st.session_state[elbow_cache_key]
                                st.rerun()

                    if run_elbow and elbow_cache_key not in st.session_state:
                        K_RANGE = list(range(2, 16))
                        _weights_e = np.array(weight_key[:-1], dtype=np.float32)
                        X_w_e      = apply_feature_weights(X_norm, _weights_e)
                        progress_bar = st.progress(0, text="Computing k=2 / 15... (0%)")
                        inertias = {}
                        try:
                            for i, ki in enumerate(K_RANGE):
                                pct = int((i / len(K_RANGE)) * 100)
                                progress_bar.progress(
                                    i / len(K_RANGE),
                                    text=f"Computing k={ki} / {K_RANGE[-1]}... ({pct}%)"
                                )
                                m = KMeans(k=ki, random_seed=42)
                                m.fit(X_w_e)
                                inertias[ki] = m.inertia_
                            progress_bar.progress(1.0, text="✅ Done! (100%)")
                            st.session_state[elbow_cache_key] = inertias
                        except Exception as e:
                            progress_bar.empty()
                            st.error(f"Elbow plot failed: {e}")

                    if elbow_cache_key in st.session_state:
                        inertias = st.session_state[elbow_cache_key]
                        st.success("✅ Analysis complete! (Click Re-run if you change feature weights)")
                        elbow_df = pd.DataFrame({
                            "k":       list(inertias.keys()),
                            "Inertia": list(inertias.values()),
                        })
                        fig_elbow = px.line(elbow_df, x="k", y="Inertia", markers=True,
                                            title="Elbow Plot: Within-Cluster Inertia vs. k",
                                            height=400)
                        fig_elbow.add_vline(x=k, line_dash="dash", line_color="#1DB954",
                                            annotation_text=f"Current k={k}",
                                            annotation_font_color="#1DB954")
                        fig_elbow.update_traces(line_color="#1DB954", marker_color="#1DB954")
                        fig_elbow.update_layout(
                            xaxis=dict(tickmode="linear", dtick=1),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_elbow, use_container_width=True)
                    elif not run_elbow:
                        st.info("Click **▶ Run Elbow Analysis** to start.")

            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

            # ── Player injection — OUTSIDE the try/except, full-width ─────────
            # This block runs whenever now_playing is set (regardless of show_advanced).
            if "now_playing" in st.session_state:
                playing     = st.session_state["now_playing"]
                current_idx = st.session_state.get("now_playing_idx", 0)
                rec_list    = st.session_state.get("rec_list", [
                    {"name": playing["name"], "artist": playing["artist"]}
                ])

                rec_cache_key = "|".join(f"{t['name']}::{t['artist']}" for t in rec_list)

                if st.session_state.get("_track_data_key") != rec_cache_key:
                    with st.spinner("Fetching audio links..."):
                        def _fetch_one(args):
                            t_idx, track = args
                            try:
                                url = generate_youtube_play_url(track["name"], track["artist"])
                                vid = url.split("v=")[1].split("&")[0] if url and "v=" in url else ""
                            except Exception:
                                vid = ""
                            try:
                                seed_val = recs.iloc[t_idx].get("id", str(t_idx)) if t_idx < len(recs) else str(t_idx)
                                sp_url   = recs.iloc[t_idx]["Spotify"]       if t_idx < len(recs) else ""
                                yt_url   = recs.iloc[t_idx]["YouTube Music"] if t_idx < len(recs) else ""
                            except Exception:
                                seed_val, sp_url, yt_url = str(t_idx), "", ""
                            return t_idx, {
                                "name":     track["name"],
                                "artist":   track["artist"],
                                "video_id": vid,
                                "seed":     str(seed_val),
                                "spotify":  sp_url,
                                "ytmusic":  yt_url,
                            }

                        results = [None] * len(rec_list)
                        with ThreadPoolExecutor(max_workers=min(len(rec_list), 10)) as executor:
                            for t_idx, entry in executor.map(_fetch_one, enumerate(rec_list)):
                                results[t_idx] = entry
                        st.session_state["_track_data"]     = results
                        st.session_state["_track_data_key"] = rec_cache_key

                track_data    = st.session_state["_track_data"]
                current_track = track_data[current_idx] if track_data else None

                if not current_track or not current_track["video_id"]:
                    st.warning(
                        f"⚠️ Could not fetch a playable clip for "
                        f"**{current_track['name'] if current_track else 'this track'}** "
                        f"(YouTube may have blocked the request). "
                        "Recommendations above are still available — use the Spotify / YouTube Music icons to open links directly."
                    )
                    if st.button("✕ Dismiss & return to full view", key="dismiss_player"):
                        for _k in ["now_playing", "now_playing_idx", "rec_list",
                                   "_track_data", "_track_data_key"]:
                            st.session_state.pop(_k, None)
                        components.html("""<script>
(function(){
  var p;try{p=window.parent.document;}catch(e){return;}
  ['yt-playlist-panel','yt-fixed-player','yt-pl-style','yt-fixed-style'].forEach(function(id){
    var el=p.getElementById(id); if(el) el.remove();
  });
  var mc=p.querySelector('[data-testid="stMainBlockContainer"]')||p.querySelector('.main .block-container');
  if(mc){mc.style.removeProperty('padding-right');mc.style.removeProperty('padding-bottom');}
})();
</script>""", height=0, scrolling=False)
                        st.rerun()
                else:
                    tracks_json = _json.dumps(track_data)
                    components.html(f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://www.youtube.com/iframe_api"></script>
</head>
<body style="margin:0;background:transparent;">
<div id="yt-hidden" style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;overflow:hidden;"></div>
<script>
(function() {{
  var TRACKS = {tracks_json};
  var currentIdx = {current_idx};
  var ytPlayer, ytPlaying = true, ytMuted = false, ytDragging = false;

  function esc(s) {{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }}

  var par;
  try {{ par = window.parent.document; }} catch(e) {{ par = null; }}

  // Helper: find Streamlit's main block container (selector differs by version)
  function _mc() {{
    return par && (
      par.querySelector('[data-testid="stMainBlockContainer"]') ||
      par.querySelector('.main .block-container')
    );
  }}

  // Push main content left so it never slides under the 260px fixed right panel.
  // 260px panel + 16px right margin + 19px safety = 295px. Reset on <960px (panel hidden).
  function _applyContentPadding() {{
    var el = _mc(); if (!el) return;
    var narrow = window.parent.innerWidth < 960;
    el.style.setProperty('padding-right', narrow ? '' : '295px', 'important');
    el.style.setProperty('padding-bottom', '110px', 'important');
    el.style.setProperty('max-width', '100%', 'important');
    el.style.setProperty('box-sizing', 'border-box', 'important');
  }}
  function _resetContentPadding() {{
    var el = _mc(); if (!el) return;
    el.style.removeProperty('padding-right');
    el.style.removeProperty('padding-bottom');
  }}

  if (par) {{
    _applyContentPadding();
    window.parent.addEventListener('resize', _applyContentPadding);

    var eps = par.getElementById('yt-pl-style'); if (eps) eps.remove();
    var plStyle = par.createElement('style');
    plStyle.id = 'yt-pl-style';
    plStyle.textContent = `
      #yt-playlist-panel {{
        position:fixed; top:60px; right:16px; width:260px; bottom:96px;
        background:#181818; border-radius:10px; z-index:99998;
        display:flex; flex-direction:column; overflow:hidden;
        box-shadow:0 4px 24px rgba(0,0,0,0.5);
        font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      }}
      @media (max-width: 900px) {{ #yt-playlist-panel {{ display: none !important; }} }}
      #yt-playlist-panel .pl-header {{
        padding:10px 14px 6px; font-size:0.78rem; font-weight:700; color:#b3b3b3;
        letter-spacing:0.05em; text-transform:uppercase; flex-shrink:0; border-bottom:1px solid #282828;
      }}
      #yt-playlist-panel .pl-scroll {{
        flex:1; overflow-y:auto; scrollbar-width:thin; scrollbar-color:#535353 transparent; padding:4px 0;
      }}
      #yt-playlist-panel .pl-scroll::-webkit-scrollbar {{ width:4px; }}
      #yt-playlist-panel .pl-scroll::-webkit-scrollbar-thumb {{ background:#535353; border-radius:2px; }}
      #yt-playlist-panel .pl-row {{
        display:flex; align-items:center; gap:9px; padding:6px 12px;
        cursor:pointer; transition:background 0.15s; min-width:0; border-left:3px solid transparent;
      }}
      #yt-playlist-panel .pl-row:hover {{ background:#282828; }}
      #yt-playlist-panel .pl-row.active {{ background:#242424; border-left-color:#1DB954; }}
      #yt-playlist-panel .pl-thumb {{ width:42px; height:42px; border-radius:4px; object-fit:cover; flex-shrink:0; }}
      #yt-playlist-panel .pl-info {{ flex:1; min-width:0; display:flex; flex-direction:column; gap:1px; }}
      #yt-playlist-panel .pl-title {{
        font-size:0.78rem; font-weight:600; color:#fff;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
      }}
      #yt-playlist-panel .pl-row.active .pl-title {{ color:#1DB954; }}
      #yt-playlist-panel .pl-artist {{
        font-size:0.68rem; color:#b3b3b3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
      }}
      #yt-playlist-panel .pl-links {{ display:flex; gap:5px; margin-top:3px; align-items:center; }}
      #yt-playlist-panel .pl-links a {{ line-height:0; opacity:0.65; transition:opacity 0.15s; }}
      #yt-playlist-panel .pl-links a:hover {{ opacity:1; }}
    `;
    par.head.appendChild(plStyle);

    var epp = par.getElementById('yt-playlist-panel'); if (epp) epp.remove();
    var panel = par.createElement('div');
    panel.id = 'yt-playlist-panel';
    panel.innerHTML = '<div class="pl-header">Up Next</div><div class="pl-scroll" id="pl-scroll"></div>';
    par.body.appendChild(panel);

    var scroll = par.getElementById('pl-scroll');
    TRACKS.forEach(function(t, i) {{
      var row = par.createElement('div');
      row.className = 'pl-row' + (i === currentIdx ? ' active' : '');
      row.dataset.idx = i;
      row.innerHTML =
        '<img class="pl-thumb" src="https://picsum.photos/seed/' + esc(t.seed) + '/80/80" alt="">' +
        '<div class="pl-info">' +
          '<div class="pl-title" title="' + esc(t.name) + '">' + esc(t.name) + '</div>' +
          '<div class="pl-artist">' + esc(t.artist) + '</div>' +
          '<div class="pl-links">' +
            '<a href="' + esc(t.spotify) + '" target="_blank" onclick="event.stopPropagation()">' +
              '<img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" width="14" height="14"></a>' +
            '<a href="' + esc(t.ytmusic) + '" target="_blank" onclick="event.stopPropagation()">' +
              '<img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/Youtube_Music_icon.svg" width="14" height="14"></a>' +
          '</div>' +
        '</div>';
      row.addEventListener('click', function() {{ loadTrack(i); }});
      scroll.appendChild(row);
    }});
    var activeRow = scroll.querySelector('.pl-row.active');
    if (activeRow) activeRow.scrollIntoView({{block:'nearest'}});
  }}

  if (par) {{
    var es = par.getElementById('yt-fixed-style'); if (es) es.remove();
    var styleEl = par.createElement('style');
    styleEl.id = 'yt-fixed-style';
    styleEl.textContent = [
      '#yt-fixed-player {{ position:fixed; bottom:0; left:0; right:0; height:80px; background:#181818; border-top:1px solid #282828; display:flex; align-items:center; padding:0 clamp(12px,2vw,28px); z-index:999999; box-sizing:border-box; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}',
      '#yt-fixed-player * {{ box-sizing:border-box; }}',
      '#yt-fixed-player .np-left {{ display:flex; align-items:center; gap:10px; flex:0 0 200px; min-width:0; overflow:hidden; }}',
      '#yt-fixed-player .np-thumb {{ width:44px;height:44px;border-radius:5px;object-fit:cover;flex-shrink:0;background:#333; }}',
      '#yt-fixed-player .np-info {{ display:flex;flex-direction:column;justify-content:center;min-width:0;overflow:hidden; }}',
      '#yt-fixed-player .np-title {{ font-weight:700;font-size:clamp(.7rem,.85vw,.9rem);color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:0; }}',
      '#yt-fixed-player .np-artist {{ font-size:clamp(.6rem,.7vw,.78rem);color:#b3b3b3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:0; }}',
      '#yt-fixed-player .np-center {{ display:flex;flex-direction:column;align-items:center;gap:4px;position:absolute;left:50%;transform:translateX(-50%);width:clamp(280px,40vw,560px); }}',
      '#yt-fixed-player .np-controls {{ display:flex;align-items:center;gap:clamp(4px,0.8vw,10px); }}',
      '#yt-fixed-player .np-btn {{ background:none;border:none;cursor:pointer;border-radius:50%;display:flex;align-items:center;justify-content:center;padding:6px;transition:background .15s;flex-shrink:0; }}',
      '#yt-fixed-player .np-btn:hover {{ background:rgba(255,255,255,.15); }}',
      '#yt-fixed-player .np-btn-play {{ background:#1DB954;width:36px;height:36px; }}',
      '#yt-fixed-player .np-btn-play:hover {{ background:#1ed760;transform:scale(1.06); }}',
      '#yt-fixed-player .np-progress {{ display:flex;align-items:center;gap:8px;width:100%; }}',
      '#yt-fixed-player .np-time {{ font-size:.68rem;color:#b3b3b3;min-width:36px;text-align:center;flex-shrink:0; }}',
      '#yt-fixed-player .np-bar-wrap {{ flex:1;height:4px;background:#535353;border-radius:2px;position:relative;cursor:pointer;transition:height .12s;min-width:0; }}',
      '#yt-fixed-player .np-bar-wrap:hover {{ height:7px; }}',
      '#yt-fixed-player .np-bar-fill {{ height:100%;width:0%;background:#1DB954;border-radius:2px;pointer-events:none; }}',
      '#yt-fixed-player .np-bar-wrap:hover .np-bar-fill {{ background:#1ed760; }}',
      '#yt-fixed-player .np-bar-thumb {{ position:absolute;top:50%;right:-5px;width:11px;height:11px;background:#fff;border-radius:50%;transform:translateY(-50%);opacity:0;transition:opacity .15s;pointer-events:none; }}',
      '#yt-fixed-player .np-bar-wrap:hover .np-bar-thumb {{ opacity:1; }}',
      '#yt-fixed-player .np-btn-skip {{ opacity:0.7; }}',
      '#yt-fixed-player .np-btn-skip:hover:not(:disabled) {{ opacity:1;background:rgba(255,255,255,.15); }}',
      '#yt-fixed-player .np-btn-skip:disabled {{ opacity:0.25;cursor:default; }}',
      '@media (max-width: 640px) {{ #yt-fixed-player .np-left {{ flex: 0 0 auto; }} #yt-fixed-player .np-center {{ width: calc(100vw - 120px); }} }}',
    ].join('\\n');
    par.head.appendChild(styleEl);

    var ep = par.getElementById('yt-fixed-player'); if (ep) ep.remove();
    var pdiv = par.createElement('div');
    pdiv.id = 'yt-fixed-player';
    var t0 = TRACKS[currentIdx];
    pdiv.innerHTML =
      '<div class="np-left">' +
        '<img class="np-thumb" id="np-thumb" src="https://img.youtube.com/vi/' + t0.video_id + '/mqdefault.jpg" alt="">' +
        '<div class="np-info"><p class="np-title" id="np-title">' + esc(t0.name) + '</p><p class="np-artist" id="np-artist">' + esc(t0.artist) + '</p></div>' +
      '</div>' +
      '<div class="np-center">' +
        '<div class="np-controls">' +
          '<button class="np-btn np-btn-skip" id="np-btn-prev"' + (currentIdx===0?' disabled':'') + '><svg width="16" height="16" viewBox="0 0 24 24" fill="white"><polygon points="19,5 9,12 19,19"/><rect x="5" y="5" width="3" height="14" rx="1"/></svg></button>' +
          '<button class="np-btn np-btn-play" id="np-btn-play"><svg id="np-icon-pause" width="16" height="16" viewBox="0 0 24 24" fill="white"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg><svg id="np-icon-play" width="16" height="16" viewBox="0 0 24 24" fill="white" style="display:none"><polygon points="5,3 20,12 5,21"/></svg></button>' +
          '<button class="np-btn np-btn-skip" id="np-btn-next"' + (currentIdx===TRACKS.length-1?' disabled':'') + '><svg width="16" height="16" viewBox="0 0 24 24" fill="white"><polygon points="5,5 15,12 5,19"/><rect x="16" y="5" width="3" height="14" rx="1"/></svg></button>' +
          '<button class="np-btn" id="np-btn-mute"><svg id="np-icon-sound" width="16" height="16" viewBox="0 0 24 24" fill="white"><polygon points="11,5 6,9 2,9 2,15 6,15 11,19"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07" stroke="white" stroke-width="2" fill="none"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14" stroke="white" stroke-width="2" fill="none"/></svg><svg id="np-icon-mute" width="16" height="16" viewBox="0 0 24 24" fill="white" style="display:none"><polygon points="11,5 6,9 2,9 2,15 6,15 11,19"/><line x1="23" y1="9" x2="17" y2="15" stroke="white" stroke-width="2.2" stroke-linecap="round"/><line x1="17" y1="9" x2="23" y2="15" stroke="white" stroke-width="2.2" stroke-linecap="round"/></svg></button>' +
        '</div>' +
        '<div class="np-progress"><span class="np-time" id="np-cur">0:00</span><div class="np-bar-wrap" id="np-bar"><div class="np-bar-fill" id="np-fill"><div class="np-bar-thumb"></div></div></div><span class="np-time" id="np-dur">0:00</span></div>' +
      '</div>';
    par.body.appendChild(pdiv);

    function fmt(s) {{ s=Math.floor(s||0); return Math.floor(s/60)+':'+((s%60)<10?'0':'')+s%60; }}
    function tick() {{
      if (!ytPlayer||!ytPlayer.getCurrentTime||ytDragging) return;
      var cur=ytPlayer.getCurrentTime(), dur=ytPlayer.getDuration();
      if (!dur) return;
      par.getElementById('np-fill').style.width=(cur/dur*100)+'%';
      par.getElementById('np-cur').textContent=fmt(cur);
      par.getElementById('np-dur').textContent=fmt(dur);
    }}
    function seekAt(e) {{
      if (!ytPlayer||!ytPlayer.getDuration) return;
      var bar=par.getElementById('np-bar'), r=bar.getBoundingClientRect();
      var pct=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));
      ytPlayer.seekTo(pct*ytPlayer.getDuration(),true);
      par.getElementById('np-fill').style.width=(pct*100)+'%';
    }}
    par.getElementById('np-bar').addEventListener('mousedown',function(e){{ytDragging=true;seekAt(e);}});
    par.addEventListener('mousemove',function(e){{if(ytDragging)seekAt(e);}});
    par.addEventListener('mouseup',function(){{ytDragging=false;}});
    par.getElementById('np-btn-play').addEventListener('click',function(){{
      if(!ytPlayer) return;
      if(ytPlaying){{ytPlayer.pauseVideo();par.getElementById('np-icon-pause').style.display='none';par.getElementById('np-icon-play').style.display='block';}}
      else{{ytPlayer.playVideo();par.getElementById('np-icon-pause').style.display='block';par.getElementById('np-icon-play').style.display='none';}}
      ytPlaying=!ytPlaying;
    }});
    par.getElementById('np-btn-prev').addEventListener('click',function(){{ if(currentIdx>0) loadTrack(currentIdx-1); }});
    par.getElementById('np-btn-next').addEventListener('click',function(){{ if(currentIdx<TRACKS.length-1) loadTrack(currentIdx+1); }});
    par.getElementById('np-btn-mute').addEventListener('click',function(){{
      if(!ytPlayer) return;
      if(ytMuted){{ytPlayer.unMute();par.getElementById('np-icon-sound').style.display='block';par.getElementById('np-icon-mute').style.display='none';}}
      else{{ytPlayer.mute();par.getElementById('np-icon-sound').style.display='none';par.getElementById('np-icon-mute').style.display='block';}}
      ytMuted=!ytMuted;
    }});
  }}

  function loadTrack(idx) {{
    if (idx < 0 || idx >= TRACKS.length) return;
    currentIdx = idx;
    var t = TRACKS[idx];
    if (par) {{
      var scroll2=par.getElementById('pl-scroll');
      if(scroll2){{
        scroll2.querySelectorAll('.pl-row').forEach(function(r){{r.classList.toggle('active',parseInt(r.dataset.idx)===idx);}});
        var ar=scroll2.querySelector('.pl-row.active'); if(ar) ar.scrollIntoView({{block:'nearest'}});
      }}
      var thumb=par.getElementById('np-thumb'); if(thumb&&t.video_id) thumb.src='https://img.youtube.com/vi/'+t.video_id+'/mqdefault.jpg';
      var ti=par.getElementById('np-title'); if(ti) ti.textContent=t.name;
      var art=par.getElementById('np-artist'); if(art) art.textContent=t.artist;
      var bp=par.getElementById('np-btn-prev'); if(bp) bp.disabled=(idx===0);
      var bn=par.getElementById('np-btn-next'); if(bn) bn.disabled=(idx===TRACKS.length-1);
      var fill=par.getElementById('np-fill'); if(fill) fill.style.width='0%';
      var cur=par.getElementById('np-cur'); if(cur) cur.textContent='0:00';
      var dur=par.getElementById('np-dur'); if(dur) dur.textContent='0:00';
      var ip=par.getElementById('np-icon-pause'); if(ip) ip.style.display='block';
      var ipl=par.getElementById('np-icon-play'); if(ipl) ipl.style.display='none';
    }}
    if(ytPlayer&&ytPlayer.loadVideoById&&t.video_id){{ytPlayer.loadVideoById(t.video_id);ytPlaying=true;}}
  }}

  function initYT() {{
    ytPlayer = new YT.Player('yt-hidden', {{
      videoId: TRACKS[currentIdx].video_id,
      playerVars: {{autoplay:1, controls:0, modestbranding:1, playsinline:1}},
      events: {{ onReady: function(e) {{ e.target.playVideo(); setInterval(tick, 500); }} }}
    }});
  }}

  if (window.YT && window.YT.Player) {{ initYT(); }}
  else {{
    var _prev = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = function() {{ if(typeof _prev==='function') _prev(); initYT(); }};
  }}
}})();
</script>
</body>
</html>""", height=0, scrolling=False)

# ============================================================================
# TAB 2 — Describe a Vibe
# ============================================================================
with tab2:
    render_vibe_extension(top_n=10)


# ============================================================================
# TAB 3 — Evaluation
# ============================================================================
with tab3:
    st.header("📊 Evaluation")
    st.caption("Quantitative checks on the clustering and recommendation quality (uses the same k + feature weights as Tab 1).")

    # Reuse the clustering produced in Tab 1 — `weight_key` and `fit_weighted_model`
    # are module-level because `with tab1:` does not introduce a new scope.
    try:
        _weight_key_eval = weight_key
        _k_eval = _weight_key_eval[-1]
        _weights_eval = np.array(_weight_key_eval[:-1], dtype=np.float32)
        _model_eval = fit_weighted_model(_weight_key_eval)
    except NameError:
        st.info("Open the **Find Similar Songs** tab first so the clustering gets built, then come back.")
        st.stop()
    except Exception as e:
        st.error(f"Could not load model for evaluation: {e}")
        st.stop()

    X_w_eval = apply_feature_weights(X_norm, _weights_eval)
    labels_eval = _model_eval.labels_
    centroids_eval = _model_eval.centroids
    inertia_eval = _model_eval.inertia_

    # ── Top-line metrics ───────────────────────────────────────────────────
    st.subheader("Cluster Quality at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("k (clusters)", _k_eval)
    col2.metric("Songs clustered", f"{len(X_w_eval):,}")
    col3.metric("Inertia (WCSS)", f"{inertia_eval:,.0f}")
    col4.metric("Iterations to converge", _model_eval.n_iters_)

    st.divider()

    # ── Silhouette on a subsample (full 1.2M is too slow) ──────────────────
    st.subheader("Silhouette Score (approximate, on 3k sample)")
    st.caption(
        "Silhouette ∈ [-1, 1]. Higher = samples are well-matched to their own cluster "
        "and far from others. Computed on a random 3,000-song sample for tractability."
    )

    @st.cache_data(show_spinner="Computing silhouette...")
    def _silhouette_sample(weight_key, sample_size=3000, seed=42):
        _w = np.array(weight_key[:-1], dtype=np.float32)
        X_w = apply_feature_weights(X_norm, _w)
        _k = weight_key[-1]
        model = fit_weighted_model(weight_key)
        labels = model.labels_
        rng = np.random.default_rng(seed)
        n = len(X_w)
        idx = rng.choice(n, size=min(sample_size, n), replace=False)
        Xs = X_w[idx]
        ls = labels[idx]

        # Full pairwise distance matrix on the sample
        diff = Xs[:, None, :] - Xs[None, :, :]
        D = np.sqrt(np.sum(diff ** 2, axis=2))

        sil = np.zeros(len(Xs), dtype=np.float32)
        for i in range(len(Xs)):
            own = ls == ls[i]
            own_idx = np.where(own)[0]
            own_idx = own_idx[own_idx != i]
            if len(own_idx) == 0:
                sil[i] = 0.0
                continue
            a = D[i, own_idx].mean()
            b = np.inf
            for c in range(_k):
                if c == ls[i]:
                    continue
                other = np.where(ls == c)[0]
                if len(other) == 0:
                    continue
                b = min(b, D[i, other].mean())
            sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        return float(sil.mean()), sil

    run_sil = st.button("▶ Compute Silhouette", type="primary", key="run_sil_btn")
    if run_sil:
        try:
            sil_mean, sil_arr = _silhouette_sample(_weight_key_eval)
            st.session_state[f"_sil_{_weight_key_eval}"] = (sil_mean, sil_arr)
        except Exception as e:
            st.error(f"Silhouette failed: {e}")

    if f"_sil_{_weight_key_eval}" in st.session_state:
        sil_mean, sil_arr = st.session_state[f"_sil_{_weight_key_eval}"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean silhouette", f"{sil_mean:.3f}")
        c2.metric("% samples > 0", f"{(sil_arr > 0).mean() * 100:.1f}%")
        c3.metric("% samples > 0.25", f"{(sil_arr > 0.25).mean() * 100:.1f}%")

        fig_sil = px.histogram(
            sil_arr, nbins=40, title="Silhouette distribution (per sample)",
            labels={"value": "silhouette", "count": "# samples"},
            height=340,
        )
        fig_sil.update_traces(marker_color="#1DB954")
        fig_sil.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_sil, use_container_width=True)
    else:
        st.info("Click the button to compute silhouette on a 3k-song sample (~3 sec).")

    st.divider()

    # ── Cluster balance ────────────────────────────────────────────────────
    st.subheader("Cluster Balance")
    st.caption("Imbalanced clusters (one huge, many tiny) often signal poor k or noisy features.")
    _counts = pd.Series(labels_eval).value_counts().sort_index()
    _mood_map = label_all_clusters(centroids_eval, feature_names=AUDIO_FEATURES)
    _balance_df = pd.DataFrame({
        "Cluster": _counts.index,
        "Mood":    [_mood_map.get(int(c), "?") for c in _counts.index],
        "Songs":   _counts.values,
        "Share":   (_counts.values / _counts.sum() * 100).round(2),
    })
    fig_bal = px.bar(
        _balance_df, x="Cluster", y="Songs",
        hover_data=["Mood", "Share"],
        title="Songs per cluster",
        height=320,
    )
    fig_bal.update_traces(marker_color="#1DB954")
    fig_bal.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(fig_bal, use_container_width=True)
    st.dataframe(_balance_df, use_container_width=True, hide_index=True)


# ============================================================================
# TAB 4 — Project Overview
# ============================================================================
with tab4:
    st.header("📖 Project Overview")

    st.markdown("""
### What this app does
Given a song you already like, this tool recommends similar songs from a
**1.2 million-track Spotify dataset** — but with a twist: **you** choose which
audio features matter (e.g. danceability vs. acousticness), and the clustering
+ ranking re-run in real time under those weights.

---

### Pipeline
1. **Load** `data/tracks_features.csv` (1.2M rows, 9 continuous audio features).
2. **Normalize** each feature to `[0, 1]` with min–max scaling (preserves the
   native interpretation of Spotify features and mood-label thresholds).
3. **Weight** each feature axis by the sidebar slider values.
4. **Cluster** with from-scratch NumPy K-Means (k-means++ init, squared
   Euclidean distance) into `k` mood clusters.
5. **Label** each cluster by inspecting its centroid against mood thresholds
   in `mood_labels.py` (e.g. high energy + high valence → "Party 🎉").
6. **Recommend** the nearest songs to the query within its cluster (weighted
   Euclidean distance), falling back to global search if the cluster is tiny.
7. **Visualize** the 9D space in 2D via from-scratch PCA (SVD-based).

---

### Algorithms implemented from scratch (NumPy only)
- **K-Means** with k-means++ seeding and Frobenius-norm convergence check —
  `src/kmeans.py`
- **PCA** via centered SVD, keeping top 2 components — `src/reduce.py`
- **Weighted Euclidean distance** for nearest-neighbor recommendation —
  `src/recommend.py`
- **Silhouette score** (sample-based) — computed in the Evaluation tab
- **Character-level Transformer** for free-text vibe search (Tab 2) —
  `src/models/transformer.py`, trained separately

---

### Tab-by-tab
| Tab | What it does |
|-----|--------------|
| 🎵 Find Similar Songs | Search a song → get top-N recs; click ▶ to play in-app with YouTube; PCA map shows where your pick + recs live |
| 📝 Describe a Vibe | Free-text query ("late night driving…") → transformer embedding → cosine-similarity search |
| 📊 Evaluation | Inertia, silhouette histogram, cluster balance — tells you if your current weights produce a healthy clustering |
| 📖 Project Overview | This page |

---

### Why min–max (not standardization)?
- Spotify's `danceability`, `energy`, `valence`, etc. are already natively
  bounded in `[0, 1]` — min–max preserves that meaning.
- `src/mood_labels.py` uses hard thresholds like `if valence > 0.65` — those
  only make sense in the original `[0, 1]` space.
- Slider weights (0–5) stay interpretable: doubling a slider really does
  double that axis's contribution to distance.

---

### Dataset
- **File:** `data/tracks_features.csv` (~1.2M rows, not checked into git)
- **Features used:** danceability, energy, loudness, speechiness,
  acousticness, instrumentalness, liveness, valence, tempo
- **Excluded:** key, mode, time_signature (categorical), duration_ms (metadata)

---

### Listen feature
- YouTube IFrame Player API (official, no key required) for in-browser audio
- `yt-dlp` scrapes the first search result's `videoId` for each recommended
  track; IDs are pre-fetched in parallel on the first ▶ click, then cached
  in `session_state` so switching tracks never re-fetches.

---

### File layout
```
src/
├── app.py              ← Streamlit entry point (this app)
├── preprocess.py       ← feature extraction + normalization
├── kmeans.py           ← from-scratch KMeans + elbow
├── recommend.py        ← weighted nearest-neighbor
├── reduce.py           ← from-scratch PCA
├── mood_labels.py      ← centroid → human-readable mood
├── vibe_extension.py   ← Tab 2 (text → embedding → similarity)
├── models/
│   └── transformer.py  ← char-level Transformer for vibe search
└── utils/
    └── music_links.py  ← Spotify / YT Music / YouTube-video search URLs
```
""")
