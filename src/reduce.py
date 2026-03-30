"""
reduce.py — Dimensionality Reduction for Visualization

Projects the high-dimensional song vectors down to 2D so we can plot
the k-means clusters on the web dashboard.

Primary method: PCA (fast, interpretable).
Optional method: shallow autoencoder with PyTorch (non-linear, richer).

Course concepts used:
    - Dimensionality reduction / visualization (Week 8)
"""

import numpy as np


def pca_reduce(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Project X onto its top principal components using NumPy SVD.
    No scikit-learn — implemented from scratch.

    PCA finds the directions of maximum variance in the data.
    The first two principal components capture the most spread,
    giving the best 2D "view" of the high-dimensional cluster structure.

    Args:
        X:            (n_songs, n_features) normalized feature matrix
        n_components: number of dimensions to project to (default 2)

    Returns:
        X_reduced: (n_songs, n_components) projected matrix
    """
    # Center the data (subtract mean of each feature)
    X_centered = X - X.mean(axis=0)

    # SVD: X = U @ S @ Vt
    # V columns are the principal component directions
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Project onto top n_components principal components
    components = Vt[:n_components]           # (n_components, n_features)
    X_reduced = X_centered @ components.T    # (n_songs, n_components)
    return X_reduced


def build_autoencoder(n_features: int, latent_dim: int = 2):
    """
    Shallow autoencoder for non-linear dimensionality reduction.
    Encoder: n_features → 64 → latent_dim
    Decoder: latent_dim → 64 → n_features

    Requires PyTorch. Use pca_reduce() if torch is unavailable.

    Args:
        n_features: input/output dimension
        latent_dim: size of the bottleneck (2 for 2D visualization)

    Returns:
        (encoder, autoencoder) — train autoencoder, then use encoder to embed.
    """
    try:
        import torch.nn as nn

        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_features),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z

        model = Autoencoder()
        return model.encoder, model

    except ImportError:
        raise ImportError(
            "PyTorch is required for the autoencoder. "
            "Use pca_reduce() instead, or: pip install torch"
        )