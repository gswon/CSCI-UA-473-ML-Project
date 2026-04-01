"""
mood_labels.py — Automatic Mood Naming for K-Means Clusters

Inspects each cluster centroid's audio profile and assigns a
human-readable mood label. Works with both uniform and weighted
clustering — the label always reflects what the centroid actually
looks like in the original normalized space.
"""

import numpy as np

FEATURE_NAMES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "track_popularity"
]


def label_cluster(centroid: np.ndarray, feature_names: list[str] = FEATURE_NAMES) -> str:
    """
    Assign a mood label based on dominant centroid audio characteristics.

    Args:
        centroid:      (n_features,) normalized centroid in [0,1] per feature
        feature_names: ordered feature name list matching centroid dims

    Returns:
        Human-readable mood string with emoji
    """
    feat = {name: float(val) for name, val in zip(feature_names, centroid)}

    energy           = feat.get("energy", 0.5)
    danceability     = feat.get("danceability", 0.5)
    valence          = feat.get("valence", 0.5)
    acousticness     = feat.get("acousticness", 0.5)
    instrumentalness = feat.get("instrumentalness", 0.5)
    speechiness      = feat.get("speechiness", 0.5)
    tempo            = feat.get("tempo", 0.5)
    liveness         = feat.get("liveness", 0.5)

    if instrumentalness > 0.6:
        return "Instrumental / Focus 🎼"
    if speechiness > 0.5:
        return "Spoken Word / Rap 🎤"
    if energy > 0.75 and danceability > 0.7:
        return "Party / Dance 🔥"
    if energy > 0.75 and valence > 0.6:
        return "Energetic & Upbeat ⚡"
    if energy > 0.75 and valence < 0.4:
        return "Intense / Aggressive 💢"
    if acousticness > 0.6 and energy < 0.4:
        return "Acoustic & Calm 🌿"
    if valence > 0.65 and energy < 0.55:
        return "Happy & Mellow 😊"
    if valence < 0.35 and energy < 0.5:
        return "Sad / Melancholic 🌧️"
    if tempo > 0.65 and danceability > 0.55:
        return "Fast & Groovy 🎶"
    if liveness > 0.6:
        return "Live / Concert Feel 🎪"
    if energy < 0.35:
        return "Chill / Lo-fi 😌"
    return "Balanced / Mixed 🎵"


def label_all_clusters(centroids: np.ndarray,
                       feature_names: list[str] = FEATURE_NAMES) -> dict[int, str]:
    """Generate mood labels for all k clusters, handling duplicates."""
    labels = {}
    seen = {}
    for i, centroid in enumerate(centroids):
        label = label_cluster(centroid, feature_names)
        if label in seen:
            label = f"{label} ({i})"
        seen[label] = True
        labels[i] = label
    return labels