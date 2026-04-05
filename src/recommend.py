import numpy as np
import pandas as pd

from kmeans import KMeans
from preprocess import apply_feature_weights, AUDIO_FEATURES


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
    Recommend the top_n nearest songs to a query track using weighted
    squared Euclidean distance.

    Lower distance = more similar.
    """
    n_features = X_norm.shape[1]
    if weights is None:
        weights = np.ones(n_features, dtype=np.float32)

    # Apply the same feature weights to the query and the dataset
    query_w = apply_feature_weights(query_vector[np.newaxis, :], weights)[0]
    X_w = apply_feature_weights(X_norm, weights)

    # Predict the query's cluster in weighted space
    cluster_id = int(model.predict(query_w)[0])
    cluster_mask = model.labels_ == cluster_id

    # Search within the cluster first if it is large enough
    if cluster_only and cluster_mask.sum() >= top_n + 1:
        search_X = X_w[cluster_mask]
        search_idx = np.where(cluster_mask)[0]
    else:
        search_X = X_w
        search_idx = np.arange(len(X_w))

    # Squared Euclidean distance from the query to all candidates
    dists = np.sum((search_X - query_w) ** 2, axis=1)

    # Rank ascending (smaller distance = more similar)
    ranked = np.argsort(dists)
    top_local = ranked[: top_n + 15]  # a few extras in case we drop duplicates
    top_global_idx = search_idx[top_local]

    results = df.iloc[top_global_idx].copy()
    results["euclidean_distance"] = dists[top_local]
    results["cluster_id"] = cluster_id

    # Drop exact match if present
    results = results[results["euclidean_distance"] > 1e-10]

    # Drop duplicate song/artist pairs if those columns exist
    subset_cols = [c for c in ["track_name", "track_artist"] if c in results.columns]
    if subset_cols:
        results = results.drop_duplicates(subset=subset_cols)

    return results.head(top_n)


def song_to_vector(
    song_name: str,
    df: pd.DataFrame,
    X_norm: np.ndarray,
    features: list[str] = AUDIO_FEATURES,
) -> tuple[np.ndarray | None, int | None]:
    """
    Look up a song by name and return its normalized feature vector and row index.
    """
    name_col = "name" if "name" in df.columns else "track_name"
    if name_col not in df.columns:
        return None, None

    matches = df[df[name_col].str.lower() == song_name.lower()]
    if matches.empty:
        return None, None

    idx = matches.index[0]
    return X_norm[idx], idx


def fuzzy_search(query: str, df: pd.DataFrame, max_results: int = 8) -> list[str]:
    """
    Return song names that contain the query string (case-insensitive).
    """
    name_col = "name" if "name" in df.columns else "track_name"
    if name_col not in df.columns:
        return []

    mask = df[name_col].str.lower().str.contains(query.lower(), na=False)
    return df[mask][name_col].unique()[:max_results].tolist()