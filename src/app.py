"""
app.py — Spotify Mood Playlist Generator (Personalized)

Run with: streamlit run src/app.py

Key feature: users select WHICH audio dimensions matter most to them
(e.g. "I care about BPM and energy, not acousticness") and k-means
re-clusters the songs in that weighted embedding space on the fly.
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
st.set_page_config(page_title="🎵 Spotify Mood Clusters", page_icon="🎵", layout="wide")
st.title("🎵 Spotify Mood Playlist Generator")
st.caption("Personalized k-means clustering — focus on the audio features YOU care about most.")

DATA_PATH = Path(__file__).parent.parent / "data" / "tracks_features.csv"

# ---------------------------------------------------------------------------
# Data loading (cached — only runs once)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading 1.2M songs... (first load only)")
def load_data():
    """Loads and scales the Spotify dataset using src/preprocess.py."""
    return preprocess(DATA_PATH)


# ---------------------------------------------------------------------------
# Sidebar — ALL user controls live here
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    k = st.slider("Number of mood clusters (k)", 2, 20, 8,
                  help="How many distinct mood playlists to create.")
    top_n = st.slider("Recommendations to show", 3, 30, 10)

    st.divider()

    # ── Feature weight controls ──────────────────────────────────────────
    st.subheader("🎛️ What matters to YOU?")
    st.caption(
        "Drag each slider to tell the algorithm how much to emphasize "
        "that feature when grouping and recommending songs. "
        "Setting something to 0 ignores it completely."
    )

    # Quick preset buttons
    st.write("") # slight spacing
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        dance_preset = st.button("Dance", width='stretch', help="Focus on danceability + tempo + energy")
    with preset_col2:
        chill_preset = st.button("Chill", width='stretch', help="Focus on acousticness + valence + low energy")
    with preset_col3:
        reset_preset = st.button("Reset", width='stretch', help="Equal weights on all features")

    st.write("") # Extra spacing before expanders

    # Initialize session state for weights
    if "weights" not in st.session_state:
        st.session_state.weights = {f: 1.0 for f in AUDIO_FEATURES}

    # Apply presets
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

    # Grouped sliders for a cleaner UI
    user_weights = {}
    
    feature_groups = {
        "Mood & Vibe 😊": ["valence", "energy", "danceability"],
        "Acoustics & Setup 🎸": ["acousticness", "instrumentalness", "liveness"],
        "Audio Properties 🔊": ["loudness", "tempo", "speechiness"]
    }
    
    for group_name, feats in feature_groups.items():
        with st.expander(group_name, expanded=True):
            for feat in feats:
                if feat in AUDIO_FEATURES:
                    label = FEATURE_LABELS.get(feat, feat)
                    user_weights[feat] = st.slider(
                        label,
                        min_value=0.0, max_value=5.0,
                        value=float(st.session_state.weights.get(feat, 1.0)),
                        step=0.5,
                        key=f"w_{feat}"
                    )
    
    # In case any AUDIO_FEATURES were not in the groups (future proofing)
    for feat in AUDIO_FEATURES:
        if feat not in user_weights:
            user_weights[feat] = st.session_state.weights.get(feat, 1.0)

    # Update session state
    st.session_state.weights = user_weights
    weight_vector = np.array([user_weights[f] for f in AUDIO_FEATURES], dtype=np.float32)

    st.divider()
    show_elbow = st.checkbox("Show elbow plot", value=False)
    st.markdown("**Dataset:** 1.2M Spotify songs")
    st.markdown("**Algorithm:** K-Means from scratch (NumPy)")

# ---------------------------------------------------------------------------
# Guard: check CSV exists
# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(
        "**Dataset not found.**\n\n"
        f"Expected: `{DATA_PATH}`\n\n"
        "Download the dataset and place `track_features.csv` in the `data/` folder. "
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
# Weighted feature matrix + k-means (re-run when weights or k change)
# ---------------------------------------------------------------------------
# We cache by the weight vector + k so Streamlit only reruns k-means
# when something actually changes.
weight_key = tuple(round(w, 1) for w in weight_vector.tolist()) + (k,)

@st.cache_resource(show_spinner="Fitting k-means with your feature weights...")
def fit_weighted_model(weight_key):
    """Fit k-means on the weighted embedding space."""
    _k = weight_key[-1]
    _weights = np.array(weight_key[:-1], dtype=np.float32)
    X_w = apply_feature_weights(X_norm, _weights)
    model = KMeans(k=_k, random_seed=42)
    model.fit(X_w)
    return model

@st.cache_data(show_spinner="Projecting to 2D...")
def get_2d_projection(weight_key):
    """PCA on weighted space so the scatter matches the clustering."""
    _weights = np.array(weight_key[:-1], dtype=np.float32)
    X_w = apply_feature_weights(X_norm, _weights)
    return pca_reduce(X_w)

try:
    model = fit_weighted_model(weight_key)
except Exception as e:
    st.error(f"K-means failed: {e}")
    st.stop()

# Attach cluster labels and mood names
df = df.iloc[:len(X_norm)].copy().reset_index(drop=True)
df["cluster_id"] = model.labels_
mood_map = label_all_clusters(model.centroids, feature_names=AUDIO_FEATURES)
df["mood"] = df["cluster_id"].map(mood_map)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎵 Find Similar Songs",
    "📝 Describe a Vibe",
    "🗺️ Explore Clusters",
    "📉 Elbow Plot",
])

# ── Tab 1: Recommendations ──────────────────────────────────────────────────
with tab1:
    st.subheader("Find songs that sound like one you already love")

    # Show active weights summary
    active = [(FEATURE_LABELS.get(f, f), w) for f, w in user_weights.items() if w > 0]
    active.sort(key=lambda x: -x[1])
    top_active = [f"{label} ({w:.1f}x)" for label, w in active[:4] if w != 1.0]
    if top_active:
        st.info(f"**Current focus:** {' · '.join(top_active)}")

    # Autocomplete function using streamlit_searchbox
    from streamlit_searchbox import st_searchbox
    
    def search_song(searchterm: str) -> list:
        if not searchterm:
            return []
        mask = df["name"].str.contains(searchterm, case=False, na=False)
        matches = df[mask].head(10)
        # Returns a list of tuples (Display Label, Return Value)
        # Clean up the artist name for display
        return [
            (f"{row['name']} by {str(row['artists']).replace('[', '').replace(']', '').replace('\'', '')}", row['name']) 
            for idx, row in matches.iterrows()
        ]

    query_name = st_searchbox(
        search_song,
        key="song_search",
        placeholder="Enter a song name (e.g. Bohemian Rhapsody)...",
        label="Enter a song name:"
    )
    
    if not query_name:
        query_name = ""

    if not query_name.strip():
        st.info("Type a song name above to get started. Use the sidebar sliders to personalize results.")
    else:
        query_vec, query_idx = song_to_vector(query_name.strip(), df, X_norm, AUDIO_FEATURES)

        if query_vec is None:
            st.error(f'**"{query_name}"** not found in the dataset.')
            suggestions = fuzzy_search(query_name, df)
            if suggestions:
                st.write("**Did you mean one of these?**")
                for s in suggestions:
                    st.write(f"  • {s}")
        else:
            # Apply weights to query vector before predict
            query_w = apply_feature_weights(query_vec[np.newaxis, :], weight_vector)[0]
            cluster_id = int(model.predict(query_w)[0])
            mood = mood_map[cluster_id]

            # Song info + cluster info
            col1, col2, col3 = st.columns(3)
            name_col = "name" if "name" in df.columns else "track_name"
            artist_col = "artists" if "artists" in df.columns else "track_artist"

            with col1:
                st.markdown(f"<p style='font-size:0.85rem; color:#888; margin-bottom:0;'>Mood Cluster</p><p style='font-size:1.2rem; font-weight:600; margin-top:0;'>{mood}</p>", unsafe_allow_html=True)
            with col2:
                cluster_size = int((model.labels_ == cluster_id).sum())
                st.markdown(f"<p style='font-size:0.85rem; color:#888; margin-bottom:0;'>Songs in this cluster</p><p style='font-size:1.2rem; font-weight:600; margin-top:0;'>{cluster_size:,}</p>", unsafe_allow_html=True)
            with col3:
                year = df.iloc[query_idx].get("year", "—") if query_idx is not None else "—"
                st.markdown(f"<p style='font-size:0.85rem; color:#888; margin-bottom:0;'>Release Year</p><p style='font-size:1.2rem; font-weight:600; margin-top:0;'>{year}</p>", unsafe_allow_html=True)

            # Feature radar / bar chart for this song
            with st.expander("🔬 Audio profile of this song vs. cluster average"):
                song_feats = X_norm[query_idx] if query_idx is not None else query_vec
                centroid_feats = model.centroids[cluster_id]

                # Unweight centroid back to original space for display
                feat_df = pd.DataFrame({
                    "Feature": [FEATURE_LABELS.get(f, f) for f in AUDIO_FEATURES],
                    "This Song": song_feats.round(3),
                    "Cluster Average": centroid_feats.round(3),
                })
                fig_bar = px.bar(
                    feat_df.melt(id_vars="Feature", var_name="Source", value_name="Value"),
                    x="Feature", y="Value", color="Source", barmode="group",
                    title="Song vs. Cluster Audio Profile",
                    height=350,
                )
                fig_bar.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_bar, width='stretch')

            st.divider()
            st.subheader(f"Top {top_n} similar songs (weighted by your preferences)")
            try:
                recs = get_recommendations(
                    query_vec, X_norm, df, model,
                    weights=weight_vector, top_n=top_n
                )
                
                # Import link generators here
                from utils.music_links import generate_spotify_search_url, generate_youtube_music_search_url, generate_youtube_play_url
                
                # Clean up artist names: remove brackets and single quotes
                recs[artist_col] = recs[artist_col].astype(str).str.replace(r"\[|\]|'", "", regex=True)
                
                # Create hyperlink columns
                recs["YouTube Music"] = recs.apply(
                    lambda row: generate_youtube_music_search_url(row[name_col], row[artist_col]), axis=1
                )
                recs["Spotify"] = recs.apply(
                    lambda row: generate_spotify_search_url(row[name_col], row[artist_col]), axis=1
                )
                
                # Select only the specific columns
                show_cols = [name_col, artist_col, "year", "mood"]
                
                # Dynamic toggle for advanced details
                show_advanced = st.toggle("🔍 Show advanced details (Euclidean Distance & Audio Features)")
                if show_advanced:
                    if "euclidean_distance" in recs:
                        recs["euclidean_distance"] = recs["euclidean_distance"].round(4)
                        show_cols += ["euclidean_distance"]
                    show_cols += AUDIO_FEATURES
                    
                show_cols += ["YouTube Music", "Spotify"]
                display_df = recs[show_cols]
                
                st.write("") # spacer
                
                # ========== NEW CAROUSEL-STYLE GRID UI ==========
                cols_per_row = 5
                
                for i in range(0, len(recs), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(recs):
                            row = recs.iloc[i + j]
                            with col:
                                # Use unique ID as seed so album covers stay persistent per song
                                seed = row.get("id", str(i+j))
                                st.image(f"https://picsum.photos/seed/{seed}/400/400", use_container_width=True)
                                
                                # UI Card contents
                                st.markdown(
                                    f"<div style='margin-top:-10px; margin-bottom: 5px;'>"
                                    f"<p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0px; text-overflow: ellipsis; white-space: nowrap; overflow: hidden;' title=\"{row[name_col]}\">{row[name_col]}</p>"
                                    f"<p style='font-size: 0.9rem; color: #aaa; margin-top: 0px; text-overflow: ellipsis; white-space: nowrap; overflow: hidden;'>{row[artist_col]}</p>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                                
                                # Buttons and links
                                btn_col1, btn_col2 = st.columns([1.5, 1])
                                with btn_col1:
                                    if st.button("▶ Listen", key=f"listen_{seed}", use_container_width=True):
                                        st.session_state["now_playing"] = {"name": row[name_col], "artist": row[artist_col]}
                                with btn_col2:
                                    st.markdown(
                                        f"<div style='text-align: right; padding-top: 4px;'>"
                                        f"<a href='{row['Spotify']}' target='_blank' title='Open in Spotify'>"
                                        f"<img src='https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg' width='24' style='margin-right: 8px; vertical-align: middle; transition: transform 0.2s;'></a>"
                                        f"<a href='{row['YouTube Music']}' target='_blank' title='Open in YouTube Music'>"
                                        f"<img src='https://upload.wikimedia.org/wikipedia/commons/6/6a/Youtube_Music_icon.svg' width='24' style='vertical-align: middle; transition: transform 0.2s;'></a>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                                st.write("") # Pad between rows
                
                # Show old dataframe view only if "advanced details" is toggled on
                if show_advanced:
                    st.divider()
                    st.subheader("📊 Detailed Data View")
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "YouTube Music": st.column_config.LinkColumn("YouTube Music", display_text="▶ YouTube"),
                            "Spotify": st.column_config.LinkColumn("Spotify", display_text="🎵 Spotify")
                        }
                    )
                
                # Dedicated Spotify-style sticky player utilizing session_state from the click
                if "now_playing" in st.session_state:
                    playing = st.session_state["now_playing"]
                    with st.spinner("Fetching audio..."):
                        play_url = generate_youtube_play_url(playing['name'], playing['artist'])
                        if play_url:
                            video_id = play_url.split("v=")[1].split("&")[0]
                            player_html = f"""
                            <style>
                                .sticky-player {{
                                    position: fixed;
                                    bottom: 0px;
                                    left: 0px;
                                    width: 100vw;
                                    background-color: #181818;
                                    border-top: 1px solid #282828;
                                    z-index: 999999;
                                    display: flex;
                                    justify-content: center;
                                    align-items: center;
                                    padding: 15px;
                                    box-shadow: 0px -5px 25px rgba(0,0,0,0.8);
                                }}
                                /* Add padding to body to prevent content from hiding behind the player */
                                .block-container {{ padding-bottom: 150px !important; }}
                            </style>
                            <div class="sticky-player">
                                <div style="display: flex; align-items: center; justify-content: center; gap: 25px; width: 100%; max-width: 800px;">
                                    <iframe width="280" height="110" style="border-radius:10px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);" src="https://www.youtube.com/embed/{video_id}?autoplay=1&controls=1&modestbranding=1" frameborder="0" allow="autoplay; encrypted-media"></iframe>
                                    <div style="color: white; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; flex-direction: column; justify-content: center; min-width: 250px;">
                                        <span style="font-weight: 700; font-size: 1.25rem; margin-bottom: 4px; color: #fff;">{playing['name']}</span>
                                        <span style="color: #b3b3b3; font-size: 0.95rem;">{playing['artist']}</span>
                                    </div>
                                </div>
                            </div>
                            """
                            st.markdown(player_html, unsafe_allow_html=True)
                        else:
                            st.error("Could not find a video for this track.")
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

# ── Tab 2: Vibe extension ────────────────────────────────────────────────
with tab2:
    render_vibe_extension(top_n=top_n, app_df=df)

# ── Tab 3: Cluster scatter ────────────────────────────────────────────────
with tab3:
    st.subheader("Audio Feature Space — colored by mood cluster")
    st.caption("PCA projects the weighted high-dimensional space to 2D. Songs near each other sound similar under your current settings.")

    try:
        X_2d = get_2d_projection(weight_key)
        df["pca_x"] = X_2d[:, 0]
        df["pca_y"] = X_2d[:, 1]

        df_sample = df.sample(min(8000, len(df)), random_state=42)
        name_col = "name" if "name" in df.columns else "track_name"
        artist_col = "artists" if "artists" in df.columns else "track_artist"
        hover_cols = [c for c in [name_col, artist_col, "year"] if c in df_sample.columns]

        fig = px.scatter(
            df_sample, x="pca_x", y="pca_y",
            color="mood",
            hover_data=hover_cols or None,
            title=f"Mood Clusters in Weighted Audio Space (k={k})",
            labels={"pca_x": "PC 1", "pca_y": "PC 2", "mood": "Mood"},
            opacity=0.5, height=600,
        )
        fig.update_traces(marker=dict(size=3))
        st.plotly_chart(fig, width='stretch')

        # Cluster summary table
        st.subheader("Cluster Summary")
        summary = (
            df.groupby(["cluster_id", "mood"])
            .size().reset_index(name="# Songs")
            .sort_values("cluster_id")
        )
        st.dataframe(summary, width='stretch', hide_index=True)

    except Exception as e:
        st.error(f"Visualization error: {e}")

# ── Tab 4: Elbow plot ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Elbow Method — Choosing k")
    st.write(
        "Runs k-means for k=2..15 in the **current weighted space** and plots "
        "inertia (within-cluster distance). Choose k at the elbow — where the "
        "curve bends and stops dropping sharply. This is how we justify our k "
        "without relying on genre labels."
    )
    if show_elbow:
        with st.spinner("Running k=2..15 (may take a minute on 1.2M songs)..."):
            try:
                X_w = apply_feature_weights(X_norm, weight_vector)
                inertias = elbow_method(X_w, k_range=range(2, 16))
                elbow_df = pd.DataFrame({
                    "k": list(inertias.keys()),
                    "Inertia": list(inertias.values())
                })
                fig2 = px.line(elbow_df, x="k", y="Inertia", markers=True,
                               title="Elbow Plot: Within-Cluster Inertia vs. k")
                fig2.add_vline(x=k, line_dash="dash", line_color="red",
                               annotation_text=f"Current k={k}")
                st.plotly_chart(fig2, width='stretch')
            except Exception as e:
                st.error(f"Elbow plot failed: {e}")
    else:
        st.info("Enable **'Show elbow plot'** in the sidebar to run this analysis.")
