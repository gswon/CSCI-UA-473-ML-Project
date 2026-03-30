"""
test_kmeans.py — Unit tests for the from-scratch K-Means implementation

Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from kmeans import KMeans, cosine_similarity, cosine_distance_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_X():
    """
    Three obvious clusters: top-right, bottom-left, and far right.
    K-means should separate them cleanly.
    """
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[1.0, 0.0], scale=0.05, size=(50, 2))
    cluster_b = rng.normal(loc=[0.0, 1.0], scale=0.05, size=(50, 2))
    cluster_c = rng.normal(loc=[0.7, 0.7], scale=0.05, size=(50, 2))
    return np.vstack([cluster_a, cluster_b, cluster_c]).astype(np.float32)


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical():
    """A vector compared to itself should have similarity = 1.0."""
    v = np.array([0.3, 0.6, 0.1])
    sims = cosine_similarity(v, v[np.newaxis, :])
    assert np.isclose(sims[0], 1.0, atol=1e-5)


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should have similarity = 0.0."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    sims = cosine_similarity(a, b[np.newaxis, :])
    assert np.isclose(sims[0], 0.0, atol=1e-5)


def test_cosine_distance_matrix_shape():
    """Distance matrix should be (n_songs, k)."""
    X = np.random.rand(100, 12).astype(np.float32)
    centroids = np.random.rand(5, 12).astype(np.float32)
    D = cosine_distance_matrix(X, centroids)
    assert D.shape == (100, 5)


def test_cosine_distance_nonnegative():
    """All distances should be in [0, 2] for cosine distance."""
    X = np.random.rand(50, 8).astype(np.float32)
    centroids = np.random.rand(4, 8).astype(np.float32)
    D = cosine_distance_matrix(X, centroids)
    assert D.min() >= -1e-6
    assert D.max() <= 2.0 + 1e-6


# ---------------------------------------------------------------------------
# K-Means fit tests
# ---------------------------------------------------------------------------

def test_kmeans_fit_returns_labels(simple_X):
    """fit() should populate labels_ with integer cluster assignments."""
    model = KMeans(k=3, random_seed=0)
    model.fit(simple_X)
    assert model.labels_ is not None
    assert model.labels_.shape == (len(simple_X),)
    assert set(model.labels_).issubset(set(range(3)))


def test_kmeans_correct_number_of_clusters(simple_X):
    """All k clusters should be non-empty on a well-separated dataset."""
    model = KMeans(k=3, random_seed=0)
    model.fit(simple_X)
    unique_clusters = np.unique(model.labels_)
    assert len(unique_clusters) == 3, (
        f"Expected 3 non-empty clusters, got {len(unique_clusters)}: {unique_clusters}"
    )


def test_kmeans_inertia_decreases_with_more_k(simple_X):
    """More clusters → lower inertia (within-cluster distances shrink)."""
    inertia_3 = KMeans(k=3, random_seed=0).fit(simple_X).inertia_
    inertia_5 = KMeans(k=5, random_seed=0).fit(simple_X).inertia_
    assert inertia_5 <= inertia_3, (
        f"Inertia should decrease as k grows. k=3: {inertia_3:.4f}, k=5: {inertia_5:.4f}"
    )


def test_kmeans_centroid_shape(simple_X):
    """Centroids should have shape (k, n_features)."""
    k = 3
    model = KMeans(k=k, random_seed=0).fit(simple_X)
    assert model.centroids.shape == (k, simple_X.shape[1])


def test_kmeans_predict_consistency(simple_X):
    """predict() on training data should match labels_ from fit()."""
    model = KMeans(k=3, random_seed=0).fit(simple_X)
    predicted = model.predict(simple_X)
    np.testing.assert_array_equal(predicted, model.labels_)


def test_kmeans_predict_single_vector(simple_X):
    """predict() should accept a single 1-D vector."""
    model = KMeans(k=3, random_seed=0).fit(simple_X)
    label = model.predict(simple_X[0])
    assert label.shape == (1,)
    assert 0 <= int(label[0]) < 3


def test_kmeans_raises_before_fit():
    """predict() before fit() should raise RuntimeError."""
    model = KMeans(k=3)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.array([0.1, 0.2]))


def test_kmeans_reproducibility(simple_X):
    """Same random_seed → same labels every time."""
    labels_1 = KMeans(k=3, random_seed=42).fit(simple_X).labels_
    labels_2 = KMeans(k=3, random_seed=42).fit(simple_X).labels_
    np.testing.assert_array_equal(labels_1, labels_2)