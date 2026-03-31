"""
recommend.py — Cosine Similarity Nearest-Neighbor Retrieval

Given a query song, find the most acoustically similar tracks using
cosine similarity. Searches within the query's cluster first for speed,
then falls back to a global search if needed.

Course concepts used:
    - Similarity metrics (Week 4): cosine similarity
    - Nearest neighbor search (Week 4/7): retrieve top-N closest vectors
"""
from typing import Union, List
import numpy as np
import pandas as pd
from kmeans import cosine_similarity, KMeans


def get_recommendations(
    query_vector: np.ndarray,
    X_norm: np.ndarray,
    df: pd.DataFrame,
    model: KMeans,
    top_n: int = 10,
    cluster_only: bool = True,
) -> pd.DataFrame:
    """
    Recommend the top_n most acoustically similar songs to a query track.

    Strategy:
        1. Predict which cluster the query belongs to.
        2. Search within that cluster first (faster, more relevant).
        3. If the cluster has fewer than top_n songs, broaden to global.

    Args:
        query_vector:  (n_features,) normalized feature vector of the query song
        X_norm:        (n_songs, n_features) full normalized feature matrix
        df:            original DataFrame with track metadata
        model:         fitted KMeans model
        top_n:         number of recommendations to return
        cluster_only:  if True, restrict search to query's cluster

    Returns:
        DataFrame with top_n recommended tracks and their similarity scores.
    """
    # --- Step 1: find the query's cluster ---
    cluster_id = int(model.predict(query_vector)[0])
    cluster_mask = model.labels_ == cluster_id

    # --- Step 2: search within cluster or globally ---
    if cluster_only and cluster_mask.sum() >= top_n + 1:
        search_X = X_norm[cluster_mask]
        search_idx = np.where(cluster_mask)[0]
    else:
        search_X = X_norm
        search_idx = np.arange(len(X_norm))

    # --- Step 3: compute cosine similarity to every candidate ---
    sims = cosine_similarity(query_vector, search_X)   # (n_candidates,)

    # --- Step 4: rank and return top_n (excluding the query itself if present) ---
    ranked = np.argsort(sims)[::-1]          # descending similarity
    top_indices = search_idx[ranked[:top_n + 1]]

    results = df.iloc[top_indices].copy()
    results["similarity"] = sims[ranked[:top_n + 1]]
    results["cluster"] = cluster_id

    return results.head(top_n).reset_index(drop=True)


def song_to_vector(track_name: str, df: pd.DataFrame,
                   X_norm: np.ndarray, feature_cols: List[str]) -> Union[np.ndarray, None]:
    """
    Look up a song by name and return its normalized feature vector.

    Args:
        track_name:   song title to search for (case-insensitive)
        df:           original DataFrame
        X_norm:       normalized feature matrix (rows aligned with df)
        feature_cols: list of feature column names (same order as X_norm)

    Returns:
        (n_features,) vector, or None if song not found.
    """
    matches = df[df["track_name"].str.lower() == track_name.lower()]
    if matches.empty:
        return None
    idx = matches.index[0]
    return X_norm[idx]