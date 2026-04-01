"""
recommend.py — Euclidean Distance Nearest-Neighbor Retrieval (Fixed for Duplicates)
"""

from typing import Union, List
import numpy as np
import pandas as pd
from kmeans import euclidean_distance_matrix, KMeans

def get_recommendations(
    query_vector: np.ndarray,
    X_norm: np.ndarray,
    df: pd.DataFrame,
    model: KMeans,
    top_n: int = 10,
    cluster_only: bool = True,
) -> pd.DataFrame:
    """
    Recommend unique tracks using Euclidean distance, 
    filtering out duplicates and the query song itself.
    """
    
    # Ensure 2D for math
    query_vec_2d = query_vector[np.newaxis, :] if query_vector.ndim == 1 else query_vector

    # 1. Find the cluster
    cluster_id = int(model.predict(query_vec_2d)[0])
    cluster_mask = model.labels_ == cluster_id

    # 2. Narrow the search space (or use global if cluster is tiny)
    if cluster_only and cluster_mask.sum() >= 50: # Bigger buffer for duplicates
        search_X = X_norm[cluster_mask]
        search_idx = np.where(cluster_mask)[0]
    else:
        search_X = X_norm
        search_idx = np.arange(len(X_norm))

    # 3. Calculate Euclidean distances
    dists = euclidean_distance_matrix(search_X, query_vec_2d).flatten()

    # 4. Create a temporary DataFrame to handle the unique filtering logic
    # We map the distances back to the metadata
    potential_recs = df.iloc[search_idx].copy()
    potential_recs["euclidean_distance"] = dists
    potential_recs["cluster"] = cluster_id

    # --- THE FIX: DROP DUPLICATES & SELF ---
    
    # Sort by distance first so we keep the "closest" version of a duplicate
    potential_recs = potential_recs.sort_values("euclidean_distance", ascending=True)

    # Drop tracks with the same name and artist
    potential_recs = potential_recs.drop_duplicates(subset=["track_name", "track_artist"])

    # Filter out the query song (distance ~ 0) 
    # We use a tiny threshold instead of == 0 to catch floating point noise
    potential_recs = potential_recs[potential_recs["euclidean_distance"] > 1e-5]

    return potential_recs.head(top_n).reset_index(drop=True)

def song_to_vector(
    track_name: str, 
    df: pd.DataFrame,
    X_norm: np.ndarray, 
    feature_cols: List[str]
) -> Union[np.ndarray, None]:
    """
    Look up a song title and return its normalized feature vector.
    """
    # Use lowercase for case-insensitive matching
    matches = df[df["track_name"].str.lower() == track_name.lower()]
    
    if matches.empty:
        return None
        
    # Just take the first one if there are duplicates
    idx = matches.index[0]
    return X_norm[idx]