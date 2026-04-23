"""
app.py — Spotify Mood Playlist Generator (Personalized)

Run with: streamlit run src/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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
st.set_page_config(page_title="Spotify Mood Clusters", page_icon="🎵", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit header chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* App title */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 1.25rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.5rem;
}
.spotify-dot {
    width: 38px; height: 38px;
    background: #1DB954;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.app-title { font-size: 18px; font-weight: 500; margin: 0; }
.app-subtitle { font-size: 12px; opacity: 0.55; margin: 2px 0 0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Sidebar section labels */
.sidebar-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin: 0 0 8px;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: #1DB954 !important;
}
.stSlider [data-testid="stThumbValue"] {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
}

/* Preset buttons */
.stButton > button {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 400;
    border-radius: 20px;
    padding: 5px 14px;
    border: 1px solid rgba(255,255,255,0.15);
    background: transparent;
    color: rgba(255,255,255,0.65);
    transition: all 0.15s;
}
.stButton > button:hover {
    border-color: #1DB954;
    color: #1DB954;
    background: rgba(29,185,84,0.08);
}
.stButton > button:focus {
    box-shadow: none;
    border-color: #1DB954;
    color: #1DB954;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 400;
    padding: 8px 18px;
    border-radius: 6px 6px 0 0;
    color: rgba(255,255,255,0.45);
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    color: white !important;
    background: rgba(255,255,255,0.05) !important;
    border-bottom: 2px solid #1DB954 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Text input */
.stTextInput > div > div > input {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    color: white;
    padding: 10px 14px;
}
.stTextInput > div > div > input:focus {
    border-color: #1DB954;
    box-shadow: 0 0 0 1px #1DB954;
}
.stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.25); }

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 400;
    opacity: 0.5;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-size: 20px !important;
    font-weight: 500;
}

/* Info / warning boxes */
.stAlert {
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 13px;
}

/* Dataframe */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
.stDataFrame thead th {
    font-size: 11px !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(255,255,255,0.04) !important;
    color: rgba(255,255,255,0.4) !important;
}
.stDataFrame tbody td {
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 13px;
    font-weight: 400;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06); }

/* Focus tag chips */
.focus-chip {
    display: inline-block;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(29,185,84,0.12);
    border: 1px solid rgba(29,185,84,0.3);
    color: #1DB954;
    margin: 0 4px 4px 0;
}

/* Song result rows */
.song-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
    margin-bottom: 4px;
    transition: background 0.12s;
}
.song-row:hover { background: rgba(255,255,255,0.05); }
.song-num {
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    color: rgba(255,255,255,0.25);
    width: 18px;
    text-align: right;
}
.song-name { font-size: 13px; font-weight: 500; }
.song-artist { font-size: 11px; color: rgba(255,255,255,0.45); margin-top: 1px; }
.song-mood {
    font-size: 11px;
    padding: 3px 9px;
    border-radius: 20px;
    background: rgba(55,138,221,0.15);
    color: #378ADD;
    white-space: nowrap;
}
.song-dist {
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    color: rgba(255,255,255,0.3);
    white-space: nowrap;
}

/* Section heading */
.section-heading {
    font-size: 13px;
    font-weight: 500;
    color: rgba(255,255,255,0.85);
    margin: 0 0 12px;
}

/* Scatter caption */
.scatter-caption {
    font-size: 12px;
    color: rgba(255,255,255,0.35);
    margin-bottom: 12px;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
  <div class="spotify-dot">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="white">
      <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
    </svg>
  </div>
  <div>
    <p class="app-title">Mood Playlist Generator</p>
    <p class="app-subtitle">Personalized k-means clustering · 1.2M Spotify songs</p>
  </div>
</div>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent.parent / "data" / "tracks_features.csv"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading 1.2M songs... (first load only)")
def load_data():
    return preprocess(DATA_PATH)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="sidebar-label">Clusters</p>', unsafe_allow_html=True)
    k = st.slider("Number of mood clusters (k)", 2, 20, 8, label_visibility="collapsed",
                  help="How many distinct mood playlists to create.")
    st.caption(f"k = {k} clusters")

    st.markdown("---")
    st.markdown('<p class="sidebar-label">Results</p>', unsafe_allow_html=True)
    top_n = st.slider("Recommendations to show", 3, 30, 10, label_visibility="collapsed")
    st.caption(f"Show top {top_n} songs")

    st.markdown("---")
    st.markdown('<p class="sidebar-label">What matters to you?</p>', unsafe_allow_html=True)
    st.caption("Drag to weight features. 0 = ignore, 5 = prioritize.")

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        dance_preset = st.button("💃 Dance", use_container_width=True)
    with preset_col2:
        chill_preset = st.button("😌 Chill", use_container_width=True)
    with preset_col3:
        reset_preset = st.button("↺ Reset", use_container_width=True)

    if "weights" not in st.session_state:
        st.session_state.weights = {f: 1.0 for f in AUDIO_FEATURES}

    if dance_preset:
        st.session_state.weights = {
            "danceability": 3.0, "energy": 3.0, "tempo": 3.0,
            "loudness": 1.0, "speechiness": 0.5, "acousticness": 0.5,
            "instrumentalness": 0.5, "liveness": 0.5, "valence": 1.0,
        }
    if chill_preset:
        st.session_state.weights = {
            "danceability": 0.5, "energy": 0.5, "tempo": 0.5,
            "loudness": 0.5, "speechiness": 0.5, "acousticness": 3.0,
            "instrumentalness": 2.0, "liveness": 0.5, "valence": 3.0,
        }
    if reset_preset:
        st.session_state.weights = {f: 1.0 for f in AUDIO_FEATURES}

    user_weights = {}
    for feat in AUDIO_FEATURES:
        label = FEATURE_LABELS.get(feat, feat)
        user_weights[feat] = st.slider(
            label,
            min_value=0.0, max_value=5.0,
            value=float(st.session_state.weights.get(feat, 1.0)),
            step=0.5,
            key=f"w_{feat}"
        )

    st.session_state.weights = user_weights
    weight_vector = np.array([user_weights[f] for f in AUDIO_FEATURES], dtype=np.float32)

    st.markdown("---")
    show_elbow = st.checkbox("Show elbow plot", value=False)
    st.markdown('<p class="sidebar-label" style="margin-top:8px">Dataset</p>', unsafe_allow_html=True)
    st.caption("1.2M Spotify songs · K-Means from scratch (NumPy)")

# ---------------------------------------------------------------------------
# Guard: check CSV exists
# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(
        "**Dataset not found.**\n\n"
        f"Expected: `{DATA_PATH}`\n\n"
        "Download the dataset and place `tracks_features.csv` in the `data/` folder."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    X_norm, X_min, X_max, df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Weighted k-means
# ---------------------------------------------------------------------------
weight_key = tuple(round(w, 1) for w in weight_vector.tolist()) + (k,)

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

df = df.iloc[:len(X_norm)].copy().reset_index(drop=True)
df["cluster_id"] = model.labels_
mood_map = label_all_clusters(model.centroids, feature_names=AUDIO_FEATURES)
df["mood"] = df["cluster_id"].map(mood_map)

name_col = "name" if "name" in df.columns else "track_name"
artist_col = "artists" if "artists" in df.columns else "track_artist"

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎵  Find similar songs",
    "✨  Describe a vibe",
    "🗺️  Explore clusters",
    "📉  Elbow plot",
])

# ── Tab 1: Recommendations ──────────────────────────────────────────────────
with tab1:
    # Active weights summary chips
    active = [(FEATURE_LABELS.get(f, f), w) for f, w in user_weights.items() if w != 1.0 and w > 0]
    active.sort(key=lambda x: -x[1])
    if active:
        chips = " ".join(
            f'<span class="focus-chip">{label} {w:.1f}×</span>'
            for label, w in active[:4]
        )
        st.markdown(f'<div style="margin-bottom:12px">{chips}</div>', unsafe_allow_html=True)

    query_name = st.text_input(
        "Song name",
        placeholder="e.g. Bohemian Rhapsody, Blinding Lights, Levitating...",
        label_visibility="collapsed",
    )

    if not query_name.strip():
        st.markdown("""
        <div style="padding:32px 0;text-align:center;color:rgba(255,255,255,0.25)">
            <div style="font-size:32px;margin-bottom:10px">🎵</div>
            <div style="font-size:13px">Type a song name above to find similar tracks</div>
            <div style="font-size:11px;margin-top:4px">Use the sidebar sliders to personalize results by audio features</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        query_vec, query_idx = song_to_vector(query_name.strip(), df, X_norm, AUDIO_FEATURES)

        if query_vec is None:
            st.error(f'**"{query_name}"** not found in the dataset.')
            suggestions = fuzzy_search(query_name, df)
            if suggestions:
                st.markdown("**Did you mean one of these?**")
                for s in suggestions:
                    st.markdown(f"&nbsp;&nbsp;· {s}")
        else:
            query_w = apply_feature_weights(query_vec[np.newaxis, :], weight_vector)[0]
            cluster_id = int(model.predict(query_w)[0])
            mood = mood_map[cluster_id]

            # Metrics row
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Mood cluster", mood)
            with c2:
                cluster_size = int((model.labels_ == cluster_id).sum())
                st.metric("Songs in cluster", f"{cluster_size:,}")
            with c3:
                year = df.iloc[query_idx].get("year", "—") if query_idx is not None else "—"
                st.metric("Release year", year)

            st.markdown("---")

            # Audio profile expander
            with st.expander("🔬 Audio profile — this song vs. cluster average"):
                song_feats = X_norm[query_idx] if query_idx is not None else query_vec
                centroid_feats = model.centroids[cluster_id]
                feat_df = pd.DataFrame({
                    "Feature": [FEATURE_LABELS.get(f, f) for f in AUDIO_FEATURES],
                    "This Song": song_feats.round(3),
                    "Cluster Avg": centroid_feats.round(3),
                })
                fig_bar = px.bar(
                    feat_df.melt(id_vars="Feature", var_name="Source", value_name="Value"),
                    x="Feature", y="Value", color="Source", barmode="group",
                    height=320,
                    color_discrete_map={"This Song": "#1DB954", "Cluster Avg": "rgba(255,255,255,0.2)"},
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", size=12, color="rgba(255,255,255,0.6)"),
                    xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    legend=dict(orientation="h", y=1.1),
                    margin=dict(t=20, b=0, l=0, r=0),
                    showlegend=True,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Results as styled rows
            st.markdown(f'<p class="section-heading">Top {top_n} similar songs</p>', unsafe_allow_html=True)
            try:
                recs = get_recommendations(
                    query_vec, X_norm, df, model,
                    weights=weight_vector, top_n=top_n
                )
                rows_html = ""
                for i, (_, row) in enumerate(recs.iterrows(), 1):
                    song = row.get(name_col, "Unknown")
                    artist = row.get(artist_col, "")
                    yr = row.get("year", "")
                    mood_tag = row.get("mood", "")
                    dist = row.get("euclidean_distance", 0)
                    rows_html += f"""
                    <div class="song-row">
                        <div class="song-num">{i}</div>
                        <div style="flex:1;min-width:0">
                            <div class="song-name">{song}</div>
                            <div class="song-artist">{artist}{f" · {yr}" if yr else ""}</div>
                        </div>
                        <div class="song-mood">{mood_tag}</div>
                        <div class="song-dist">{dist:.4f}</div>
                    </div>"""
                st.markdown(rows_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

# ── Tab 2: Vibe extension ────────────────────────────────────────────────────
with tab2:
    render_vibe_extension(top_n=top_n)

# ── Tab 3: Cluster scatter ────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="scatter-caption">PCA projects the weighted high-dimensional space to 2D. Songs near each other sound similar under your current settings.</p>', unsafe_allow_html=True)

    try:
        X_2d = get_2d_projection(weight_key)
        df["pca_x"] = X_2d[:, 0]
        df["pca_y"] = X_2d[:, 1]

        df_sample = df.sample(min(8000, len(df)), random_state=42)
        hover_cols = [c for c in [name_col, artist_col, "year"] if c in df_sample.columns]

        fig = px.scatter(
            df_sample, x="pca_x", y="pca_y",
            color="mood",
            hover_data=hover_cols or None,
            labels={"pca_x": "PC 1", "pca_y": "PC 2", "mood": "Mood"},
            opacity=0.55,
            height=520,
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,13,13,1)",
            font=dict(family="DM Sans", size=12, color="rgba(255,255,255,0.5)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)"),
            legend=dict(title="", font=dict(size=11)),
            margin=dict(t=10, b=0, l=0, r=0),
            title=f"Mood clusters in weighted audio space (k={k})",
            title_font=dict(size=13, color="rgba(255,255,255,0.7)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-heading">Cluster summary</p>', unsafe_allow_html=True)
        summary = (
            df.groupby(["cluster_id", "mood"])
            .size().reset_index(name="Songs")
            .sort_values("cluster_id")
            .rename(columns={"cluster_id": "Cluster", "mood": "Mood"})
        )
        summary["Songs"] = summary["Songs"].apply(lambda x: f"{x:,}")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Visualization error: {e}")

# ── Tab 4: Elbow plot ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <p style="font-size:13px;color:rgba(255,255,255,0.5);line-height:1.6;margin-bottom:16px">
    Runs k-means for k=2–15 in the <strong style="color:rgba(255,255,255,0.7)">current weighted space</strong>
    and plots inertia (within-cluster distance). Choose k at the elbow —
    where the curve bends and stops dropping sharply.
    </p>
    """, unsafe_allow_html=True)

    if show_elbow:
        with st.spinner("Running k=2..15 (may take a minute on 1.2M songs)..."):
            try:
                X_w = apply_feature_weights(X_norm, weight_vector)
                inertias = elbow_method(X_w, k_range=range(2, 16))
                elbow_df = pd.DataFrame({
                    "k": list(inertias.keys()),
                    "Inertia": list(inertias.values())
                })
                fig2 = px.line(
                    elbow_df, x="k", y="Inertia", markers=True,
                    height=380,
                    color_discrete_sequence=["#1DB954"],
                )
                fig2.add_vline(
                    x=k, line_dash="dash",
                    line_color="rgba(255,255,255,0.3)",
                    annotation_text=f"current k={k}",
                    annotation_font_color="rgba(255,255,255,0.4)",
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(13,13,13,1)",
                    font=dict(family="DM Sans", size=12, color="rgba(255,255,255,0.5)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    margin=dict(t=10, b=0, l=0, r=0),
                )
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Elbow plot failed: {e}")
    else:
        st.markdown("""
        <div style="padding:40px 0;text-align:center;color:rgba(255,255,255,0.2)">
            <div style="font-size:28px;margin-bottom:8px">📉</div>
            <div style="font-size:13px">Enable <strong style="color:rgba(255,255,255,0.35)">"Show elbow plot"</strong> in the sidebar to run this analysis</div>
        </div>
        """, unsafe_allow_html=True)