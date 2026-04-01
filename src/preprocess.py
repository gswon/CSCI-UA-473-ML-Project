"""
preprocess.py — Feature Extraction and Normalization

Adapted for the 1.2M Spotify songs dataset with columns:
    id, name, album, album_id, artists, artist_ids, track_number,
    disc_number, explicit, danceability, energy, key, loudness, mode,
    speechiness, acousticness, instrumentalness, liveness, valence,
    tempo, duration_ms, time_signature, year, release_date

Course concepts used:
    - Vector representations (Week 2): songs become fixed-length numeric vectors
    - Weighted feature spaces: user-defined weights partition the embedding space
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# All continuous audio features available in the 1.2M dataset.
# We exclude: key, mode, time_signature (categorical/ordinal, not continuous)
# and duration_ms (metadata, not acoustic character).
# ---------------------------------------------------------------------------
AUDIO_FEATURES = [
    "danceability",     # 0–1: how suitable for dancing
    "energy",           # 0–1: intensity and activity
    "loudness",         # dB, typically -60 to 0
    "speechiness",      # 0–1: presence of spoken words
    "acousticness",     # 0–1: confidence the track is acoustic
    "instrumentalness", # 0–1: predicts no vocals
    "liveness",         # 0–1: presence of live audience
    "valence",          # 0–1: musical positiveness (sad → happy)
    "tempo",            # BPM, typically 50–200
]

# Human-readable labels shown in the UI feature selector
FEATURE_LABELS = {
    "danceability":     "Danceability 💃",
    "energy":           "Energy ⚡",
    "loudness":         "Loudness 🔊",
    "speechiness":      "Speechiness 🎤",
    "acousticness":     "Acousticness 🎸",
    "instrumentalness": "Instrumentalness 🎼",
    "liveness":         "Liveness 🎪",
    "valence":          "Mood / Valence 😊",
    "tempo":            "Tempo / BPM 🥁",
}

DATA_PATH = Path(__file__).parent.parent / "data" / "spotify_songs.csv"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the 1.2M Spotify songs CSV.

    For very large files this uses chunked reading to avoid memory issues.
    Set max_rows to a smaller number during development.

    Args:
        path:     path to CSV file
    Returns:
        DataFrame with all columns
    """
    print(f"Loading dataset from {path}...")
    # Read in chunks to handle 1.2M rows gracefully
    chunks = []
    chunk_size = 100_000
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} tracks.")
    return df


def extract_features(df: pd.DataFrame,
                     features: list[str] = AUDIO_FEATURES) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract audio feature columns and drop rows with missing values.
    Returns both the feature matrix AND the cleaned DataFrame (rows aligned).

    Args:
        df:       raw DataFrame
        features: feature column names to extract

    Returns:
        X:        (n_songs, n_features) float32 array
        df_clean: DataFrame with same row alignment as X
    """
    present = [f for f in features if f in df.columns]
    missing = set(features) - set(present)
    if missing:
        print(f"Warning: features not found, skipping: {missing}")

    df_clean = df[present + [c for c in ["id", "name", "album", "artists", "year", "explicit"]
                             if c in df.columns]].dropna(subset=present).reset_index(drop=True)

    X = df_clean[present].values.astype(np.float32)
    print(f"Feature matrix: {X.shape[0]:,} songs × {X.shape[1]} features")
    return X, df_clean

def min_max_normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale each feature column to [0, 1].

    Returns:
        X_norm, X_min, X_max  — save X_min/X_max to normalize query vectors later
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    
    # Avoid division by zero if a feature has no variance
    denom = (X_max - X_min)
    denom[denom == 0] = 1.0
    X_norm = (X - X_min) / denom
    return X_norm.astype(np.float32), X_min, X_max

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features by removing the mean and scaling to unit variance.
    Standardization is often preferred for Euclidean K-Means.
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero
    X_norm = (X - means) / stds
    return X_norm, means, stds


def apply_feature_weights(X_norm: np.ndarray,
                          weights: np.ndarray) -> np.ndarray:
    """
    Scale each feature dimension by a user-supplied weight.

    This is the core of personalization: by multiplying dimension i by
    weights[i], we stretch or shrink that axis in the embedding space.
    K-means and cosine similarity then naturally emphasize dimensions
    with higher weight when forming clusters and finding neighbors.

    Example:
        weights = [3, 3, 1, 1, 1, 1, 1, 1, 3]  # focus on danceability, energy, tempo
        → songs will cluster primarily by those three features

    Args:
        X_norm:  (n_songs, n_features) normalized feature matrix
        weights: (n_features,) non-negative weight per feature

    Returns:
        X_weighted: (n_songs, n_features) scaled matrix
    """
    w = np.array(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-9) * len(w)   # normalize so total energy is preserved
    return X_norm * w[np.newaxis, :]


def normalize_query(query: np.ndarray, X_min: np.ndarray, X_max: np.ndarray) -> np.ndarray:
    """Apply the same min-max scaling to a single query vector."""
    denom = (X_max - X_min)
    denom[denom == 0] = 1.0
    return ((query - X_min) / denom).astype(np.float32)


def preprocess(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Full pipeline: load → extract → normalize.

    Returns:
        X_norm:   (n_songs, n_features) normalized, unweighted feature matrix
        X_min:    per-feature min for query normalization
        X_max:    per-feature max for query normalization
        df_clean: DataFrame aligned row-for-row with X_norm
    """
    df = load_dataset(path)
    X, df_clean = extract_features(df)
    X_norm, X_min, X_max = min_max_normalize(X)
    return X_norm, X_min, X_max, df_clean
