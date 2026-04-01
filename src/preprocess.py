"""
preprocess.py — Feature Extraction and Normalization

Loads the Spotify Tracks Dataset and converts each track into a
normalized numerical feature vector suitable for k-means clustering.

Course concepts used:
    - Vector representations (Week 2): songs become fixed-length numeric vectors
"""

import numpy as np
import pandas as pd
from pathlib import Path

# The 12 continuous audio features we use as our vector dimensions.
# We exclude categorical fields (track_id, track_name, etc.) and
# the genre/subgenre labels — those are only used for EDA and evaluation.
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "track_popularity",
]

DATA_PATH = Path(__file__).parent.parent / "data" / "spotify_songs.csv"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Spotify Tracks CSV into a DataFrame.

    Args:
        path: path to spotify_songs.csv

    Returns:
        Raw DataFrame with all columns.
    """
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} tracks with columns: {list(df.columns)}")
    return df


def extract_features(df: pd.DataFrame,
                     features: list[str] = AUDIO_FEATURES) -> np.ndarray:
    """
    Pull the audio feature columns from the DataFrame and return as a
    float32 NumPy matrix. Rows with any missing value are dropped.

    Args:
        df:       raw DataFrame from load_dataset()
        features: list of column names to use as dimensions

    Returns:
        X: (n_songs, n_features) float32 array
    """
    present = [f for f in features if f in df.columns]
    missing = set(features) - set(present)
    if missing:
        print(f"Warning: these feature columns not found and will be skipped: {missing}")

    X = df[present].dropna().values.astype(np.float32)
    print(f"Feature matrix shape: {X.shape}")
    return X

def min_max_normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale each feature column to the range [0, 1].
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    
    # Avoid division by zero if a feature has no variance
    denom = (X_max - X_min)
    denom[denom == 0] = 1.0          
    
    X_norm = (X - X_min) / denom
    return X_norm, X_min, X_max

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


def normalize_query(query: np.ndarray,
                    X_min: np.ndarray,
                    X_max: np.ndarray) -> np.ndarray:
    """
    Apply the same min-max scaling (fitted on the full dataset) to a
    single query song vector, so it lives in the same space as the
    training data.

    Args:
        query: (n_features,) raw feature vector for one song
        X_min: per-feature min from min_max_normalize()
        X_max: per-feature max from min_max_normalize()

    Returns:
        Normalized query vector of shape (n_features,)
    """
    denom = (X_max - X_min)
    denom[denom == 0] = 1.0
    return (query - X_min) / denom


def preprocess(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Full pipeline: load → extract → standardize.
    """
    df = load_dataset(path)
    X = extract_features(df)
    
    # Use the standardize function here
    X_norm, means, stds = standardize(X)
    
    return X_norm, means, stds, df