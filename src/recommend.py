"""
<<<<<<< HEAD
recommend.py — Weighted Cosine Similarity Nearest-Neighbor Retrieval

Finds the most acoustically similar songs to a query, respecting
user-defined feature weights so that "find me something with the
same BPM and energy" works differently from "find something with
the same mood and acousticness."

Course concepts used:
    - Similarity metrics (Week 4): cosine similarity
    - Nearest neighbor search: top-N retrieval in weighted embedding space
"""

import numpy as np
import pandas as pd
from kmeans import cosine_similarity, KMeans
from preprocess import apply_feature_weights, AUDIO_FEATURES

=======
recommend.py — Euclidean Distance Nearest-Neighbor Retrieval (Fixed for Duplicates)
"""

from typing import Union, List
import numpy as np
import pandas as pd
from kmeans import euclidean_distance_matrix, KMeans
>>>>>>> 192fc07e374a6c12f2992bd22db2ea97230770aa

def get_recommendations(
    query_vector: np.ndarray,
    X_norm: np.ndarray,
    df: pd.DataFrame,
    model: KMeans,
    weights: np.ndarray | None = None,
    top_n: int = 10,
    cluster_only: bool = True,
) -> pd.DataFrame:
    """
<<<<<<< HEAD
    Recommend the top_n most similar songs to a query track.

    When weights are provided, both the query vector and the candidate
    vectors are scaled the same way before computing similarity — so
    cosine distance naturally emphasizes the user's chosen dimensions.

    Args:
        query_vector:  (n_features,) normalized feature vector
        X_norm:        (n_songs, n_features) full normalized matrix (unweighted)
        df:            DataFrame aligned with X_norm
        model:         fitted KMeans model (trained on weighted X)
        weights:       (n_features,) feature importance weights, or None for uniform
        top_n:         number of results to return
        cluster_only:  search within query's cluster first (faster, more relevant)

    Returns:
        DataFrame with top_n recommendations and similarity scores
    """
    n_features = X_norm.shape[1]
    if weights is None:
        weights = np.ones(n_features, dtype=np.float32)

    # Apply same weighting to both query and candidates
    query_w = apply_feature_weights(query_vector[np.newaxis, :], weights)[0]
    X_w = apply_feature_weights(X_norm, weights)

    # Find which cluster the query belongs to
    cluster_id = int(model.predict(query_w)[0])
    cluster_mask = model.labels_ == cluster_id

    # Search within cluster if large enough, else globally
    if cluster_only and cluster_mask.sum() >= top_n + 1:
        search_X = X_w[cluster_mask]
=======
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
>>>>>>> 192fc07e374a6c12f2992bd22db2ea97230770aa
        search_idx = np.where(cluster_mask)[0]
    else:
        search_X = X_w
        search_idx = np.arange(len(X_w))

<<<<<<< HEAD
    # Cosine similarity in weighted space
    sims = cosine_similarity(query_w, search_X)

    # Rank descending, skip the query song itself if present
    ranked = np.argsort(sims)[::-1]
    top_local = ranked[:top_n + 1]
    top_global_idx = search_idx[top_local]

    results = df.iloc[top_global_idx].copy()
    results["similarity"] = sims[top_local]
    results["cluster_id"] = cluster_id
    results = results[results.index != df.index[search_idx[ranked[0]]]]  # drop exact match
=======
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
>>>>>>> 192fc07e374a6c12f2992bd22db2ea97230770aa

    # Drop tracks with the same name and artist
    potential_recs = potential_recs.drop_duplicates(subset=["track_name", "track_artist"])

    # Filter out the query song (distance ~ 0) 
    # We use a tiny threshold instead of == 0 to catch floating point noise
    potential_recs = potential_recs[potential_recs["euclidean_distance"] > 1e-5]

<<<<<<< HEAD
def song_to_vector(song_name: str,
                   df: pd.DataFrame,
                   X_norm: np.ndarray,
                   features: list[str] = AUDIO_FEATURES) -> tuple[np.ndarray | None, int | None]:
    """
    Look up a song by name and return its normalized feature vector and row index.

    Args:
        song_name: track name to search (case-insensitive)
        df:        DataFrame aligned with X_norm
        X_norm:    normalized feature matrix
        features:  feature column names

    Returns:
        (vector, row_index) or (None, None) if not found
    """
    name_col = "name" if "name" in df.columns else "track_name"
    if name_col not in df.columns:
        return None, None

    matches = df[df[name_col].str.lower() == song_name.lower()]
    if matches.empty:
        return None, None

=======
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
>>>>>>> 192fc07e374a6c12f2992bd22db2ea97230770aa
    idx = matches.index[0]
    return X_norm[idx], idx


def fuzzy_search(query: str, df: pd.DataFrame, max_results: int = 8) -> list[str]:
    """
    Return song names that contain the query string (case-insensitive).
    Used to show 'did you mean?' suggestions in the UI.
    """
    name_col = "name" if "name" in df.columns else "track_name"
    if name_col not in df.columns:
        return []
    mask = df[name_col].str.lower().str.contains(query.lower(), na=False)
    return df[mask][name_col].unique()[:max_results].tolist()