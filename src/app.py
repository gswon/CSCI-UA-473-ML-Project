"""
app.py — Streamlit Web Dashboard

Users can:
    1. Search for a song by name
    2. See its cluster (mood group) and top-N similar songs
    3. Explore the 2D scatter plot of all songs colored by cluster

Run with: streamlit run src/app.py
"""

import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from preprocess import preprocess, normalize_query, AUDIO_FEATURES
from kmeans import KMeans, elbow_method
from recommend import get_recommendations, song_to_vector
from reduce import pca_reduce

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="🎵 Spotify Mood Clusters", layout="wide")
st.title("🎵 Spotify Mood Playlist Generator")
st.caption("K-Means clustering on audio features — no genre labels, no popularity bias.")

# ---------------------------------------------------------------------------
# Load & cache data  (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    X_norm, X_min, X_max, df = preprocess()
    return X_norm, X_min, X_max, df

@st.cache_resource
def fit_model(k: int):
    X_norm, _, _, _ = load_data()
    model = KMeans(k=k, random_seed=42)
    model.fit(X_norm)
    return model

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of mood clusters (k)", min_value=2, max_value=20, value=8)
    top_n = st.slider("Recommendations to show", min_value=5, max_value=30, value=10)
    show_elbow = st.checkbox("Show elbow plot (slow — runs k=2..15)", value=False)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
try:
    X_norm, X_min, X_max, df = load_data()
    model = fit_model(k)

    # Add cluster labels to the dataframe for display
    df = df.iloc[:len(X_norm)].copy()
    df["cluster"] = model.labels_

    # 2D projection for visualization
    X_2d = pca_reduce(X_norm)
    df["pca_x"] = X_2d[:, 0]
    df["pca_y"] = X_2d[:, 1]

    # -----------------------------------------------------------------------
    # Song search
    # -----------------------------------------------------------------------
    st.subheader("🔍 Find Similar Songs")
    query_name = st.text_input("Enter a song name:", placeholder="e.g. Blinding Lights")

    if query_name:
        query_vec = song_to_vector(query_name, df, X_norm, AUDIO_FEATURES)
        if query_vec is None:
            st.error(f"Song '{query_name}' not found in the dataset. Try another title.")
        else:
            recs = get_recommendations(query_vec, X_norm, df, model, top_n=top_n)
            cluster_id = int(model.predict(query_vec)[0])
            st.success(f"Found! This song belongs to **Mood Cluster {cluster_id}**.")
            st.dataframe(recs[["track_name", "track_artist", "similarity", "cluster"]],
                         use_container_width=True)

    # -----------------------------------------------------------------------
    # 2D Cluster Scatter Plot
    # -----------------------------------------------------------------------
    st.subheader("🗺️ Audio Feature Space (PCA → 2D)")
    fig = px.scatter(
        df.sample(min(5000, len(df))),   # sample for rendering speed
        x="pca_x", y="pca_y",
        color=df["cluster"].astype(str),
        hover_data=["track_name", "track_artist"] if "track_name" in df.columns else None,
        title=f"Songs colored by Mood Cluster (k={k})",
        labels={"color": "Cluster"},
        opacity=0.6,
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Elbow plot (optional, slow)
    # -----------------------------------------------------------------------
    if show_elbow:
        st.subheader("📐 Elbow Method — Choosing k")
        with st.spinner("Running k-means for k=2..15..."):
            inertias = elbow_method(X_norm, k_range=range(2, 16))
        elbow_df = pd.DataFrame({"k": list(inertias.keys()),
                                 "inertia": list(inertias.values())})
        st.line_chart(elbow_df.set_index("k"))
        st.caption("Choose k at the 'elbow' — where inertia stops dropping sharply.")

except FileNotFoundError:
    st.error("Dataset not found. Please download `spotify_songs.csv` from Kaggle "
             "and place it in the `data/` folder. See README for instructions.")