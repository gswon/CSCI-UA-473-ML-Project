"""
kmeans.py — From-scratch K-Means Clustering (NumPy only, no scikit-learn)

Algorithm:
    1. Initialize k centroids using k-means++ seeding
    2. Assign each song to its nearest centroid (cosine similarity)
    3. Recompute each centroid as the mean of its assigned songs
    4. Repeat steps 2–3 until centroids stop moving (convergence)

Course concepts used:
    - Vector representations (Week 2): each song is a feature vector
    - Similarity metrics (Week 4): cosine similarity as our distance
    - K-means clustering (Week 7): the core algorithm implemented here
"""

import numpy as np


# ---------------------------------------------------------------------------
# Distance / similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between vector a and every row of matrix b.

    Cosine similarity measures the angle between two vectors, ignoring
    magnitude. It's preferred here because audio features like loudness
    span very different numeric ranges even after normalization, and we
    care more about the *shape* of a song's profile than its raw scale.

    Args:
        a: 1-D feature vector of shape (n_features,)
        b: 2-D matrix of shape (n_songs, n_features)

    Returns:
        1-D array of shape (n_songs,) with similarity scores in [-1, 1].
        Higher = more similar.
    """
    # Normalize a to unit length
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    # Normalize each row of b to unit length
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    b_unit = b / b_norms
    return b_unit @ a_norm  # dot product of unit vectors = cosine similarity


def cosine_distance_matrix(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine DISTANCE between every song and every centroid.
    Distance = 1 - similarity, so smaller = more similar.

    Args:
        X:         (n_songs, n_features)
        centroids: (k, n_features)

    Returns:
        (n_songs, k) distance matrix
    """
    # Normalize rows to unit length
    X_unit = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    C_unit = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    # Similarity matrix via matrix multiply, then convert to distance
    similarity = X_unit @ C_unit.T          # (n_songs, k)
    return 1.0 - similarity                 # (n_songs, k) distances


# ---------------------------------------------------------------------------
# K-Means++ initialization
# ---------------------------------------------------------------------------

def kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    K-Means++ centroid initialization.

    Instead of choosing k random songs as starting centroids (which can
    cause slow convergence or bad clusters), k-means++ spreads initial
    centroids apart by probabilistically choosing each next centroid in
    proportion to its squared distance from the nearest existing centroid.

    Args:
        X:   (n_songs, n_features) normalized feature matrix
        k:   number of clusters
        rng: numpy random generator for reproducibility

    Returns:
        centroids: (k, n_features) initial centroid matrix
    """
    n_songs = X.shape[0]
    # Step 1: pick the first centroid uniformly at random
    first_idx = rng.integers(0, n_songs)
    centroids = [X[first_idx]]

    for _ in range(1, k):
        # Step 2: compute distance from each song to its nearest centroid so far
        dists = cosine_distance_matrix(X, np.array(centroids))   # (n, len(centroids))
        min_dists = dists.min(axis=1)                             # (n,) nearest distance

        # Step 3: sample next centroid with probability proportional to distance²
        # Songs far from all current centroids are more likely to be chosen,
        # spreading centroids across the feature space.
        weights = min_dists ** 2
        weights /= weights.sum()
        next_idx = rng.choice(n_songs, p=weights)
        centroids.append(X[next_idx])

    return np.array(centroids)   # (k, n_features)


# ---------------------------------------------------------------------------
# Core K-Means
# ---------------------------------------------------------------------------

class KMeans:
    """
    K-Means clustering implemented from scratch with NumPy.

    Uses cosine distance (1 - cosine similarity) so that the algorithm
    groups songs by the *direction* of their audio profiles — two songs
    with the same tempo-energy-danceability ratio end up in the same
    cluster even if their raw values differ by a global loudness offset.

    Parameters:
        k          : number of clusters (mood playlists to create)
        max_iters  : safety cap on iterations (default 300)
        tol        : convergence threshold — stop when centroids move
                     less than this amount between iterations (default 1e-4)
        random_seed: for reproducibility

    Example:
        >>> model = KMeans(k=8)
        >>> labels = model.fit(X_normalized)
        >>> cluster_id = model.predict(query_vector)
    """

    def __init__(self, k: int = 8, max_iters: int = 300,
                 tol: float = 1e-4, random_seed: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_seed = random_seed

        # Set after fit()
        self.centroids: np.ndarray | None = None   # (k, n_features)
        self.labels_: np.ndarray | None = None     # (n_songs,) cluster index per song
        self.inertia_: float | None = None         # within-cluster sum of squared distances
        self.n_iters_: int = 0                     # iterations until convergence

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Run k-means on the feature matrix X until convergence.

        Args:
            X: (n_songs, n_features) — should be min-max normalized
               before calling fit() (see preprocess.py)

        Returns:
            self, so you can chain: labels = KMeans(k=8).fit(X).labels_
        """
        rng = np.random.default_rng(self.random_seed)

        # --- Step 1: Initialize centroids with k-means++ ---
        self.centroids = kmeans_plus_plus_init(X, self.k, rng)

        for iteration in range(self.max_iters):
            # --- Step 2: Assignment — each song goes to nearest centroid ---
            labels = self._assign(X)

            # --- Step 3: Update — recompute centroids as cluster means ---
            new_centroids = self._update(X, labels)

            # --- Step 4: Check convergence ---
            # Convergence = centroids barely moved since last iteration
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.n_iters_ = iteration + 1

            if centroid_shift < self.tol:
                print(f"[KMeans] Converged after {self.n_iters_} iterations "
                      f"(shift={centroid_shift:.6f} < tol={self.tol})")
                break
        else:
            print(f"[KMeans] Reached max_iters={self.max_iters} without full convergence.")

        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels)
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new songs to their nearest cluster centroid.

        Args:
            X: (n_songs, n_features) or (n_features,) for a single song

        Returns:
            Integer cluster label(s)
        """
        if self.centroids is None:
            raise RuntimeError("Call fit() before predict().")
        if X.ndim == 1:
            X = X[np.newaxis, :]           # single song → (1, n_features)
        return self._assign(X)

    # ------------------------------------------------------------------
    def _assign(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each song to the cluster with the nearest centroid.

        Returns integer label array of shape (n_songs,).
        """
        dist = cosine_distance_matrix(X, self.centroids)   # (n_songs, k)
        return np.argmin(dist, axis=1)                      # (n_songs,)

    # ------------------------------------------------------------------
    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Recompute each centroid as the mean of its assigned songs.

        If a cluster is empty (no songs assigned), reinitialize its
        centroid to a random song to avoid dead centroids.
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.k, n_features))

        for cluster_id in range(self.k):
            members = X[labels == cluster_id]
            if len(members) == 0:
                # Dead centroid: reinitialize randomly to avoid NaN
                print(f"[KMeans] Warning: cluster {cluster_id} is empty — reinitializing.")
                new_centroids[cluster_id] = X[np.random.randint(0, len(X))]
            else:
                new_centroids[cluster_id] = members.mean(axis=0)

        return new_centroids

    # ------------------------------------------------------------------
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Within-cluster sum of squared cosine distances.
        Used by the elbow method to choose k.
        """
        total = 0.0
        for cluster_id in range(self.k):
            members = X[labels == cluster_id]
            if len(members) == 0:
                continue
            centroid = self.centroids[cluster_id]
            dists = cosine_distance_matrix(members, centroid[np.newaxis, :])
            total += (dists ** 2).sum()
        return float(total)


# ---------------------------------------------------------------------------
# Elbow method — choosing the right k
# ---------------------------------------------------------------------------

def elbow_method(X: np.ndarray, k_range: range = range(2, 16),
                 random_seed: int = 42) -> dict[int, float]:
    """
    Run k-means for each k in k_range and record the inertia.

    Plot the result: the "elbow" — where inertia stops dropping sharply —
    is a good choice for k (number of mood playlists).

    Args:
        X:           normalized feature matrix (n_songs, n_features)
        k_range:     values of k to try (default 2–15)
        random_seed: for reproducibility

    Returns:
        dict mapping k → inertia
    """
    results = {}
    for k in k_range:
        print(f"Fitting k={k}...")
        model = KMeans(k=k, random_seed=random_seed)
        model.fit(X)
        results[k] = model.inertia_
    return results
