"""Tests for entity representation methods: k-means, GMM (diagonal/full), KDE."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))

from common.representations import (
    fit_kmeans,
    fit_gmm_diagonal,
    fit_gmm_full,
    fit_kde,
    score_kmeans,
    score_gmm,
    score_kde,
)


@pytest.fixture
def sample_embeddings():
    """Two clusters in 10-dim space."""
    rng = np.random.RandomState(42)
    cluster1 = rng.randn(20, 10) + np.array([5] * 10)
    cluster2 = rng.randn(20, 10) + np.array([-5] * 10)
    return np.vstack([cluster1, cluster2]).astype(np.float32)


@pytest.fixture
def test_point_near_cluster1():
    """A point near cluster 1."""
    return np.array([[5.1] * 10], dtype=np.float32)


@pytest.fixture
def test_point_far():
    """A point far from both clusters, in an orthogonal direction."""
    # Use a direction orthogonal to both clusters so cosine similarity is low
    vec = np.zeros(10, dtype=np.float32)
    vec[0] = 50.0
    vec[1] = -50.0
    return vec.reshape(1, -1)


class TestKMeans:
    def test_fit_returns_centroids(self, sample_embeddings):
        model = fit_kmeans(sample_embeddings, n_prototypes=2)
        assert model["centroids"].shape == (2, 10)

    def test_score_near_is_higher(self, sample_embeddings, test_point_near_cluster1, test_point_far):
        model = fit_kmeans(sample_embeddings, n_prototypes=2)
        score_near = score_kmeans(model, test_point_near_cluster1)[0]
        score_far = score_kmeans(model, test_point_far)[0]
        assert score_near > score_far


class TestGMMDiagonal:
    def test_fit_selects_components_by_bic(self, sample_embeddings):
        model = fit_gmm_diagonal(sample_embeddings, max_components=5)
        assert 1 <= model["n_components"] <= 5

    def test_score_near_is_higher(self, sample_embeddings, test_point_near_cluster1, test_point_far):
        model = fit_gmm_diagonal(sample_embeddings, max_components=3)
        score_near = score_gmm(model, test_point_near_cluster1)[0]
        score_far = score_gmm(model, test_point_far)[0]
        assert score_near > score_far


class TestGMMFull:
    def test_fit_with_pca(self, sample_embeddings):
        model = fit_gmm_full(sample_embeddings, max_components=3, pca_dims=5)
        assert model["pca_dims"] == 5

    def test_score_near_is_higher(self, sample_embeddings, test_point_near_cluster1, test_point_far):
        model = fit_gmm_full(sample_embeddings, max_components=3, pca_dims=5)
        score_near = score_gmm(model, test_point_near_cluster1)[0]
        score_far = score_gmm(model, test_point_far)[0]
        assert score_near > score_far


class TestKDE:
    def test_fit_stores_training_data(self, sample_embeddings):
        model = fit_kde(sample_embeddings)
        assert "kde" in model

    def test_score_near_is_higher(self, sample_embeddings, test_point_near_cluster1, test_point_far):
        model = fit_kde(sample_embeddings)
        score_near = score_kde(model, test_point_near_cluster1)[0]
        score_far = score_kde(model, test_point_far)[0]
        assert score_near > score_far
