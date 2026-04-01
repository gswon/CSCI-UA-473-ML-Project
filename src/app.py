"""
app.py — Streamlit Web Dashboard for Spotify Mood Clusters

Run with: streamlit run src/app.py

Features:
    - Search song by name → retrieve top-N Euclidean nearest neighbors
    - Interactive 2D scatter plot (PCA) of acoustic feature space
    - Elbow plot for data-driven choice of k (number of clusters)

Course concepts used:
    - Vector representations (Week 2)
    - PCA Dimensionality Reduction (Week 6)
    - K-Means & Euclidean Distance (Week 7)
"""

import sys
from pathlib import Path

# Ensure the app can find local modules in the /src directory
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Local project imports
from preprocess import preprocess, AUDIO_FEATURES
from kmeans import KMeans, elbow_method
from recommend import get_recommendations, song_to_vector
from reduce import pca_reduce
from mood_labels import label_all_clusters

# --- UI Configuration ---
st.set_page_config(page_title="Proud Cheetahs | Spotify Mood Clusters", page_icon="🎵", layout="wide")
st.title("🎵 Spotify Mood Playlist Generator")
st.caption("Custom K-Means implementation using standard Euclidean distance for acoustic discovery.")

# Path to the dataset (should be in /data/spotify_songs.csv)
DATA_PATH = Path(__file__).parent.parent / "data" / "spotify_songs.csv"

# --- Data & Model Caching ---
# We use st.cache to ensure the model doesn't re-train every time the user clicks a button.

@st.cache_data(show_spinner="Preparing dataset...")
def load_data():
    """Loads and scales the Spotify dataset using src/preprocess.py."""
    return preprocess(DATA_PATH)

@st.cache_resource(show_spinner="Fitting k-means clusters...")
def fit_model(k: int, X_norm: np.ndarray):
    """Initializes and fits your scratch-built K-Means model."""
    model = KMeans(k=k, random_seed=42)
    model.fit(X_norm)
    return model

@st.cache_data(show_spinner="Computing PCA projection...")
def get_2d_projection(X_norm: np.ndarray):
    """Reduces 12D features to 2D for the map visualization."""
    return pca_reduce(X_norm)

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    k_value = st.slider("Number of mood clusters (k)", min_value=2, max_value=20, value=8)
    n_recs = st.slider("Recommendations to return", min_value=3, max_value=20, value=10)
    show_elbow = st.checkbox("Enable Elbow Plot analysis", value=False)
    st.divider()
    st.markdown("**Team:** Proud Cheetahs")
    st.markdown("**Algorithm:** Euclidean K-Means (NumPy)")

# --- Main Execution Flow ---
if not DATA_PATH.exists():
    st.error("Missing `spotify_songs.csv`. Please place the Kaggle dataset in the `/data` folder.")
    st.stop()

# 1. Load Data
X_norm, X_min, X_max, df = load_data()

# 2. Fit K-Means
model = fit_model(k_value, X_norm)

# 3. Label Clusters (Dynamic 'Mood' assignment based on centroids)
df_display = df.iloc[:len(X_norm)].copy().reset_index(drop=True)
df_display["cluster_id"] = model.labels_
mood_map = label_all_clusters(model.centroids, feature_names=AUDIO_FEATURES)
df_display["mood"] = df_display["cluster_id"].map(mood_map)

# --- App Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Song Discovery", "🗺️ Acoustic Map", "📊 Model Evaluation"])

with tab1:
    st.subheader("Find similar sounding tracks")
    query_name = st.text_input("Enter a song you like:", placeholder="e.g., Starboy")

    if query_name.strip():
        # Convert user text to its feature vector from the dataset
        query_vec = song_to_vector(query_name.strip(), df_display, X_norm, AUDIO_FEATURES)
        
        if query_vec is None:
            st.warning(f"Could not find '{query_name}' in the library. Try checking your spelling!")
        else:
            # Show the user which 'mood' their song falls into
            current_cluster = int(model.predict(query_vec)[0])
            st.info(f"This song is in the **{mood_map[current_cluster]}** cluster (# {current_cluster}).")
            
            # Retrieve recommendations using Euclidean Distance
            st.write(f"#### Top {n_recs} Recommendations")
            try:
                recs = get_recommendations(query_vec, X_norm, df_display, model, top_n=n_recs)
                
                # Format table for display
                cols_to_show = ["track_name", "track_artist", "euclidean_distance", "mood"]
                recs["euclidean_distance"] = recs["euclidean_distance"].round(4)
                
                st.dataframe(recs[cols_to_show], use_container_width=True, hide_index=True)
                st.caption("Note: Lower Euclidean distance indicates higher acoustic similarity.")
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
    else:
        st.write("Enter a track name to see the recommendation engine in action.")

with tab2:
    st.subheader("Visualizing the Music Landscape")
    st.caption("High-dimensional audio features compressed into 2D via PCA.")
    
    try:
        X_2d = get_2d_projection(X_norm)
        df_display["PC1"] = X_2d[:, 0]
        df_display["PC2"] = X_2d[:, 1]
        
        # Sampling for performance (Plotly can be slow with 30k points)
        df_sample = df_display.sample(min(5000, len(df_display)))
        
        fig = px.scatter(
            df_sample, x="PC1", y="PC2", color="mood",
            hover_data=["track_name", "track_artist"],
            template="plotly_dark",
            height=600,
            title="Acoustic Clusters (PCA Projection)"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Visualization error: {e}")

with tab3:
    st.subheader("The Elbow Method")
    st.write("This plot shows the Within-Cluster Sum of Squares (Inertia) for different values of $k$.")
    
    if show_elbow:
        with st.spinner("Calculating inertia for k=2..15..."):
            inertias = elbow_method(X_norm, k_range=range(2, 16))
            elbow_df = pd.DataFrame({"k": list(inertias.keys()), "Inertia": list(inertias.values())})
            
            fig_elbow = px.line(elbow_df, x="k", y="Inertia", markers=True)
            fig_elbow.add_vline(x=k_value, line_dash="dash", line_color="red", annotation_text="Current k")
            st.plotly_chart(fig_elbow, use_container_width=True)
    else:
        st.info("Check 'Enable Elbow Plot' in the sidebar to see the mathematical justification for your cluster count.")