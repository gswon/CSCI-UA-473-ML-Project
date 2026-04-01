"""
kmeans.py — From-scratch K-Means Clustering (NumPy only, no scikit-learn)

Algorithm:
    1. Initialize k centroids using k-means++ seeding
    2. Assign each song to its nearest centroid (Squared Euclidean Distance)
    3. Recompute each centroid as the mean of its assigned songs
    4. Repeat steps 2–3 until centroids stop moving (convergence)

Course concepts used:
    - Vector representations (Week 2): each song is a feature vector
    - Distance metrics (Week 4): Squared Euclidean Distance
    - K-means clustering (Week 7): the core algorithm implemented here
"""

import numpy as np

# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def euclidean_distance_matrix(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distance between every song and every centroid.
    
    Using the expansion: (a - b)^2 = a^2 + b^2 - 2ab
    This is much faster in NumPy than looping or using np.linalg.norm.

    Args:
        X:         (n_songs, n_features)
        centroids: (k, n_features)

    Returns:
        (n_songs, k) distance matrix where entry [i, j] is distance between song i and centroid j.
    """
    # X_sq: (n_songs, 1), C_sq: (1, k)
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    C_sq = np.sum(centroids**2, axis=1, keepdims=True).T
    # Cross term: (n_songs, k)
    cross_term = 2.0 * (X @ centroids.T)
    
    # Ensure distances aren't negative due to floating point errors
    dists = X_sq + C_sq - cross_term
    return np.maximum(dists, 0)


# ---------------------------------------------------------------------------
# K-Means++ initialization
# ---------------------------------------------------------------------------

def kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    K-Means++ centroid initialization using Euclidean distance.
    """
    n_songs = X.shape[0]
    first_idx = rng.integers(0, n_songs)
    centroids = [X[first_idx]]

    for _ in range(1, k):
        # Compute distance from each song to the nearest existing centroid
        dists = euclidean_distance_matrix(X, np.array(centroids))   
        min_dists = dists.min(axis=1)                             

        # Probabilistic selection: far points are more likely to be chosen
        weights = min_dists  # min_dists is already squared distance from our helper
        weights_sum = weights.sum()
        if weights_sum == 0:
            # Fallback if all points are identical
            next_idx = rng.integers(0, n_songs)
        else:
            weights /= weights_sum
            next_idx = rng.choice(n_songs, p=weights)
        
        centroids.append(X[next_idx])

    return np.array(centroids)


# ---------------------------------------------------------------------------
# Core K-Means
# ---------------------------------------------------------------------------

class KMeans:
    """
    K-Means clustering implemented from scratch with NumPy using Euclidean Distance.

    Parameters:
        k          : number of clusters (mood playlists)
        max_iters  : safety cap on iterations (default 300)
        tol        : convergence threshold (default 1e-4)
        random_seed: for reproducibility
    """

    def __init__(self, k: int = 8, max_iters: int = 300,
                 tol: float = 1e-4, random_seed: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_seed = random_seed

        self.centroids: np.ndarray | None = None   
        self.labels_: np.ndarray | None = None     
        self.inertia_: float | None = None         
        self.n_iters_: int = 0                     

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Run k-means on the feature matrix X. 
        CRITICAL: X must be scaled (StandardScalar) before calling fit.
        """
        rng = np.random.default_rng(self.random_seed)

        # Step 1: Initialize
        self.centroids = kmeans_plus_plus_init(X, self.k, rng)

        for iteration in range(self.max_iters):
            # Step 2: Assignment
            labels = self._assign(X)

            # Step 3: Update (Mean of cluster members)
            new_centroids = self._update(X, labels)

            # Step 4: Check convergence
            # Use Frobenius norm to see how much the centroid matrix shifted
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.n_iters_ = iteration + 1

            if centroid_shift < self.tol:
                print(f"[KMeans] Converged after {self.n_iters_} iterations.")
                break
        
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("Call fit() before predict().")
        if X.ndim == 1:
            X = X[np.newaxis, :]           
        return self._assign(X)

    def _assign(self, X: np.ndarray) -> np.ndarray:
        """Assign to nearest centroid by minimizing squared Euclidean distance."""
        dist = euclidean_distance_matrix(X, self.centroids)   
        return np.argmin(dist, axis=1)                      

    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Standard centroid update: mean of all points in the cluster."""
        n_features = X.shape[1]
        new_centroids = np.zeros((self.k, n_features))

        for cluster_id in range(self.k):
            members = X[labels == cluster_id]
            if len(members) == 0:
                # Handle dead centroids by picking a random point from X
                new_centroids[cluster_id] = X[np.random.randint(0, len(X))]
            else:
                new_centroids[cluster_id] = members.mean(axis=0)

        return new_centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """The Within-Cluster Sum of Squares (WCSS)."""
        total = 0.0
        for cluster_id in range(self.k):
            members = X[labels == cluster_id]
            if len(members) == 0: continue
            centroid = self.centroids[cluster_id]
            # Sum of squared distances to the centroid
            dists = np.sum((members - centroid)**2, axis=1)
            total += dists.sum()
        return float(total)

# ---------------------------------------------------------------------------
# Elbow Method
# ---------------------------------------------------------------------------

def elbow_method(X: np.ndarray, k_range: range = range(2, 16),
                 random_seed: int = 42) -> dict[int, float]:
    results = {}
    for k in k_range:
        model = KMeans(k=k, random_seed=random_seed)
        model.fit(X)
        results[k] = model.inertia_
    return results