"""
app.py — Streamlit Web Dashboard

Run with: streamlit run src/app.py

Features:
    - Search any song by name → get top-N similar songs from the same mood cluster
    - 2D scatter plot of all songs colored by mood cluster (PCA projection)
    - Elbow plot to justify choice of k
    - Graceful error handling for missing CSV, unknown songs, bad input
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from preprocess import preprocess, normalize_query, AUDIO_FEATURES
from kmeans import KMeans, elbow_method
from recommend import get_recommendations, song_to_vector
from reduce import pca_reduce
from mood_labels import label_all_clusters

st.set_page_config(page_title="🎵 Spotify Mood Clusters", page_icon="🎵", layout="wide")
st.title("🎵 Spotify Mood Playlist Generator")
st.caption("K-Means clustering on audio features — no genre labels, no popularity bias.")

DATA_PATH = Path(__file__).parent.parent / "data" / "spotify_songs.csv"

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return preprocess(DATA_PATH)

@st.cache_resource(show_spinner="Fitting k-means clusters...")
def fit_model(k: int):
    X_norm, _, _, _ = load_data()
    model = KMeans(k=k, random_seed=42)
    model.fit(X_norm)
    return model

@st.cache_data(show_spinner="Projecting to 2D...")
def get_2d(k: int):
    X_norm, _, _, _ = load_data()
    return pca_reduce(X_norm)

with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of mood clusters (k)", min_value=2, max_value=20, value=8)
    top_n = st.slider("Recommendations to return", min_value=3, max_value=30, value=10)
    show_elbow = st.checkbox("Show elbow plot", value=False)
    st.divider()
    st.markdown("**Algorithm:** K-Means from scratch (NumPy)")
    st.markdown("**Dataset:** [Spotify Tracks (Kaggle)](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)")

if not DATA_PATH.exists():
    st.error("Dataset not found. Download `spotify_songs.csv` from Kaggle and place it in `data/`. See `data/README.md`.")
    st.stop()

try:
    X_norm, X_min, X_max, df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

try:
    model = fit_model(k)
except Exception as e:
    st.error(f"Failed to fit k-means: {e}")
    st.stop()

df_display = df.iloc[:len(X_norm)].copy().reset_index(drop=True)
df_display["cluster_id"] = model.labels_
mood_map = label_all_clusters(model.centroids, feature_names=AUDIO_FEATURES)
df_display["mood"] = df_display["cluster_id"].map(mood_map)

tab1, tab2, tab3 = st.tabs(["🔍 Find Similar Songs", "🗺️ Explore Clusters", "📐 Elbow Plot"])

with tab1:
    st.subheader("Find songs that sound like one you already love")
    query_name = st.text_input("Enter a song name:", placeholder="e.g. Blinding Lights")

    if query_name.strip() == "":
        st.info("Type a song name above to get started.")
    else:
        query_vec = song_to_vector(query_name.strip(), df_display, X_norm, AUDIO_FEATURES)
        if query_vec is None:
            st.error(f'Song "{query_name}" not found. Check spelling or try the full track name.')
            name_col = "track_name" if "track_name" in df_display.columns else None
            if name_col:
                mask = df_display[name_col].str.lower().str.contains(query_name.lower(), na=False)
                suggestions = df_display[mask][name_col].unique()[:5]
                if len(suggestions):
                    st.write("Did you mean:")
                    for s in suggestions:
                        st.write(f"  • {s}")
        else:
            cluster_id = int(model.predict(query_vec)[0])
            mood = mood_map[cluster_id]
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Mood Cluster", mood)
                st.metric("Cluster ID", f"#{cluster_id}")
            with col2:
                st.write("**Cluster centroid audio profile:**")
                feat_df = pd.DataFrame({"Feature": AUDIO_FEATURES, "Cluster Average": model.centroids[cluster_id].round(3)})
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
            st.divider()
            st.subheader(f"Top {top_n} similar songs")
            try:
                recs = get_recommendations(query_vec, X_norm, df_display, model, top_n=top_n)
                display_cols = [c for c in ["track_name", "track_artist", "playlist_genre", "similarity", "mood"] if c in recs.columns]
                recs["similarity"] = recs["similarity"].round(4)
                st.dataframe(recs[display_cols], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Recommendation failed: {e}")

with tab2:
    st.subheader("All songs projected to 2D (PCA) — colored by mood cluster")
    st.caption("Each point is a song. Songs near each other sound similar.")
    try:
        X_2d = get_2d(k)
        df_display["pca_x"] = X_2d[:, 0]
        df_display["pca_y"] = X_2d[:, 1]
        df_sample = df_display.sample(min(6000, len(df_display)), random_state=42)
        hover_cols = [c for c in ["track_name", "track_artist", "playlist_genre"] if c in df_sample.columns]
        fig = px.scatter(df_sample, x="pca_x", y="pca_y", color="mood",
                         hover_data=hover_cols or None,
                         title=f"Mood Clusters in Audio Feature Space (k={k}, PCA)",
                         labels={"pca_x": "PC 1", "pca_y": "PC 2", "mood": "Mood"},
                         opacity=0.55, height=600)
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
        st.subheader("Cluster Summary")
        summary = df_display.groupby(["cluster_id", "mood"]).size().reset_index(name="songs").sort_values("cluster_id")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Visualization failed: {e}")

with tab3:
    st.subheader("Elbow Method — Choosing k")
    st.write("We run k-means for k=2..15 and look for the 'elbow' where inertia stops dropping sharply.")
    if show_elbow:
        with st.spinner("Running k-means for k=2..15..."):
            try:
                inertias = elbow_method(X_norm, k_range=range(2, 16))
                elbow_df = pd.DataFrame({"k": list(inertias.keys()), "Inertia": list(inertias.values())})
                fig2 = px.line(elbow_df, x="k", y="Inertia", markers=True, title="Elbow Plot")
                fig2.add_vline(x=k, line_dash="dash", line_color="red", annotation_text=f"Current k={k}")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Elbow plot failed: {e}")
    else:
        st.info("Enable 'Show elbow plot' in the sidebar to run this analysis.")