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
tab1, tab2 = st.tabs([
    "🎵 Find Similar Songs",
    "📝 Describe a Vibe",
])

# ============================================================================
# TAB 1 — Find Similar Songs
# ============================================================================
with tab1:
    # Detect player state at the very top so CSS is injected before anything renders
    _is_playing = "now_playing" in st.session_state

    # When the right-side playlist panel is visible, push main content left with
    # CSS padding so the fixed 260px panel never overlaps page elements.
    st.markdown(f"""<style>
    .main .block-container {{
        {"padding-right: 290px !important;" if _is_playing else ""}
        padding-bottom: {"110px" if _is_playing else "20px"} !important;
        max-width: 100% !important;
    }}
    @media (max-width: 900px) {{
        #yt-playlist-panel {{ display: none !important; }}
        .main .block-container {{ padding-right: 12px !important; }}
    }}
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
    from streamlit_searchbox import st_searchbox

    def search_song(searchterm: str) -> list:
        if not searchterm or not searchterm.strip():
            return []
        q = searchterm.strip().lower()
        # Scan up to 200 candidates; deduplicate by (name, artist) to preserve covers
        idxs = _lower_series[_lower_series.str.contains(q, regex=False, na=False)].index[:200]
        seen, results = set(), []
        for i in idxs:
            name   = _names_arr[i]
            artist = _artists_arr[i]
            key = (name.lower(), artist.lower())
            if key not in seen:
                seen.add(key)
                results.append((f"{name} — {artist}", name))
            if len(results) >= 20:
                break
        return results

    query_name = st_searchbox(
        search_song,
        key="song_search",
        placeholder="Enter a song name (e.g. Bohemian Rhapsody)...",
        label="Enter a song name:"
    )
    if not query_name:
        query_name = ""

    # Reset player when song, k, top_n, or any feature weight changes
    _player_ctx = (query_name.strip(), k, top_n, weight_key)
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
})();
</script>""", height=0, scrolling=False)

    if not query_name.strip():
        st.info("Type a song name above to get started. Adjust settings in the expander above to personalize results.")
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

                # ── Carousel grid ─────────────────────────────────────────────
                st.write("")
                cols_per_row = 5
                for i in range(0, len(recs), cols_per_row):
                    card_cols = st.columns(cols_per_row)
                    for j, col in enumerate(card_cols):
                        if i + j < len(recs):
                            row = recs.iloc[i + j]
                            with col:
                                seed = row.get("id", str(i + j))
                                st.image(f"https://picsum.photos/seed/{seed}/400/400",
                                         use_container_width=True)
                                st.markdown(
                                    f"<div style='margin-top:-10px;margin-bottom:5px;'>"
                                    f"<p style='font-size:1rem;font-weight:600;margin-bottom:0;"
                                    f"text-overflow:ellipsis;white-space:nowrap;overflow:hidden;'"
                                    f" title=\"{row[name_col]}\">{row[name_col]}</p>"
                                    f"<p style='font-size:0.82rem;color:#aaa;margin-top:0;"
                                    f"text-overflow:ellipsis;white-space:nowrap;overflow:hidden;'>"
                                    f"{row[artist_col]}</p></div>",
                                    unsafe_allow_html=True
                                )
                                btn_col1, btn_col2 = st.columns([1.5, 1])
                                with btn_col1:
                                    if st.button("▶ Listen", key=f"listen_{seed}",
                                                 use_container_width=True):
                                        st.session_state["now_playing"] = {
                                            "name": row[name_col],
                                            "artist": row[artist_col],
                                            "ts": _time.time()
                                        }
                                        st.session_state["now_playing_idx"] = i + j
                                        st.session_state["rec_list"] = [
                                            {"name": recs.iloc[idx][name_col],
                                             "artist": recs.iloc[idx][artist_col]}
                                            for idx in range(len(recs))
                                        ]
                                        st.rerun()
                                with btn_col2:
                                    st.markdown(
                                        f"<div style='text-align:right;padding-top:4px;'>"
                                        f"<a href='{row['Spotify']}' target='_blank'>"
                                        f"<img src='https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg'"
                                        f" width='22' style='margin-right:6px;vertical-align:middle;'></a>"
                                        f"<a href='{row['YouTube Music']}' target='_blank'>"
                                        f"<img src='https://upload.wikimedia.org/wikipedia/commons/6/6a/Youtube_Music_icon.svg'"
                                        f" width='22' style='vertical-align:middle;'></a></div>",
                                        unsafe_allow_html=True
                                    )
                                st.write("")

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
                    st.subheader("🗺️ Cluster Map (PCA)")
                    st.caption(
                        "PCA projects the 9-dimensional weighted feature space to 2D. "
                        "Songs near each other sound similar under your current weights. "
                        "⭐ Gold = your input song · 🔴 Red = top recommendations · ● Gray = everything else."
                    )
                    try:
                        X_2d = get_2d_projection(weight_key)
                        df_tab1["pca_x"] = X_2d[:, 0]
                        df_tab1["pca_y"] = X_2d[:, 1]

                        # Sample background points
                        df_sample = df_tab1.sample(min(5000, len(df_tab1)), random_state=42).copy()
                        df_sample["point_type"] = "Other"

                        rec_index_set = set(recs.index.tolist())
                        highlight_idx = (
                            ([query_idx] if query_idx is not None else []) +
                            [i for i in rec_index_set if i != query_idx]
                        )
                        highlight_rows = df_tab1.loc[highlight_idx].copy()
                        highlight_rows["point_type"] = pd.Series({
                            i: ("Input Song ⭐" if i == query_idx else "Recommendation 🔴")
                            for i in highlight_rows.index
                        })

                        # Concat, then keep highlight version where indices overlap
                        df_pca = pd.concat([df_sample, highlight_rows])
                        df_pca = df_pca[~df_pca.index.duplicated(keep="last")]

                        # Sort so "Other" renders first (behind highlights)
                        _order = {"Other": 0, "Recommendation 🔴": 1, "Input Song ⭐": 2}
                        df_pca = df_pca.sort_values(
                            "point_type", key=lambda s: s.map(_order).fillna(0)
                        )

                        color_map = {
                            "Input Song ⭐":     "#FFD700",
                            "Recommendation 🔴": "#FF3333",
                            "Other":             "#555555",
                        }

                        fig_pca = px.scatter(
                            df_pca, x="pca_x", y="pca_y",
                            color="point_type",
                            color_discrete_map=color_map,
                            hover_data=[c for c in [name_col, artist_col, "mood"] if c in df_pca.columns],
                            title=f"Song Space — k={k} clusters (PCA 2D, {len(df_pca):,} songs shown)",
                            labels={"pca_x": "PC 1", "pca_y": "PC 2", "point_type": ""},
                            height=620,
                        )

                        # Per-trace marker size — iterating traces is the only correct way
                        for trace in fig_pca.data:
                            if "Input Song" in trace.name:
                                trace.marker.size = 18
                                trace.marker.opacity = 1.0
                                trace.marker.line = dict(width=2, color="white")
                            elif "Recommendation" in trace.name:
                                trace.marker.size = 11
                                trace.marker.opacity = 1.0
                                trace.marker.line = dict(width=1, color="white")
                            else:
                                trace.marker.size = 4
                                trace.marker.opacity = 0.45

                        fig_pca.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            legend=dict(
                                orientation="h", yanchor="bottom", y=1.01,
                                xanchor="right", x=1, font=dict(size=13)
                            ),
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)

                        st.subheader("Cluster Summary")
                        summary = (
                            df_tab1.groupby(["cluster_id", "mood"])
                            .size().reset_index(name="# Songs")
                            .sort_values("cluster_id")
                        )
                        st.dataframe(summary, use_container_width=True, hide_index=True)

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

  if (par) {{
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
