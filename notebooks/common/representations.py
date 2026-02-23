"""Entity representation methods: k-means, GMM (diagonal/full), KDE."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity


# -- K-means (baseline) ------------------------------------------------------

def fit_kmeans(embeddings, n_prototypes=3):
    """Fit k-means centroids. L2-normalize each centroid.

    Args:
        embeddings: (N, D) array of L2-normalized entry embeddings
        n_prototypes: number of centroids (1-3)

    Returns:
        dict with 'centroids': (n_prototypes, D) L2-normalized array
    """
    if len(embeddings) <= n_prototypes:
        n_prototypes = 1

    if n_prototypes == 1:
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return {"centroids": centroid.reshape(1, -1), "method": "kmeans"}

    km = KMeans(n_clusters=n_prototypes, n_init=10, random_state=42)
    km.fit(embeddings)
    centroids = km.cluster_centers_
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    centroids = centroids / norms
    return {"centroids": centroids.astype(np.float32), "method": "kmeans"}


def score_kmeans(model, query_embeddings):
    """Score query embeddings against k-means centroids.

    Returns: (N,) array of max cosine similarity across centroids.
    """
    centroids = model["centroids"]
    # Normalize queries
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    q_norm = query_embeddings / norms
    sim = q_norm @ centroids.T  # (N, K)
    return sim.max(axis=1)


# -- GMM diagonal ------------------------------------------------------------

def fit_gmm_diagonal(embeddings, max_components=5, pca_dims=100):
    """Fit GMM with diagonal covariance on PCA-reduced embeddings,
    select components via BIC.

    Args:
        embeddings: (N, D) array
        max_components: max components to try (1 to max_components)
        pca_dims: target PCA dimensionality (avoids degenerate fits in high-D)

    Returns:
        dict with 'gmm': fitted GaussianMixture, 'n_components': int,
        'method': 'gmm_diag', 'pca': fitted PCA, 'pca_dims': int
    """
    # GMM requires at least 2 samples; fall back to kmeans centroid for 1
    if len(embeddings) < 2:
        return fit_kmeans(embeddings, n_prototypes=1)

    # PCA reduction to avoid degenerate covariance in high dimensions
    actual_dims = min(pca_dims, embeddings.shape[1], embeddings.shape[0] - 1)
    pca = PCA(n_components=actual_dims, random_state=42)
    reduced = pca.fit_transform(embeddings)

    best_bic = float("inf")
    best_gmm = None
    best_k = 1

    max_k = min(max_components, len(embeddings) - 1)
    max_k = max(max_k, 1)

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type="diag",
            n_init=1, random_state=42, max_iter=100,
        )
        gmm.fit(reduced)
        bic = gmm.bic(reduced)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    return {
        "gmm": best_gmm, "n_components": best_k,
        "method": "gmm_diag", "pca": pca, "pca_dims": actual_dims,
    }


def fit_gmm_full(embeddings, max_components=5, pca_dims=100):
    """Fit GMM with full covariance on PCA-reduced embeddings.

    Args:
        embeddings: (N, D) array
        max_components: max components to try
        pca_dims: target dimensionality for PCA reduction

    Returns:
        dict with 'gmm', 'n_components', 'method': 'gmm_full',
        'pca': fitted PCA, 'pca_dims': int
    """
    # GMM requires at least 2 samples; fall back to kmeans centroid for 1
    if len(embeddings) < 2:
        return fit_kmeans(embeddings, n_prototypes=1)

    actual_dims = min(pca_dims, embeddings.shape[1], embeddings.shape[0] - 1)
    pca = PCA(n_components=actual_dims, random_state=42)
    reduced = pca.fit_transform(embeddings)

    best_bic = float("inf")
    best_gmm = None
    best_k = 1

    max_k = min(max_components, len(embeddings) - 1)
    max_k = max(max_k, 1)

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            n_init=1, random_state=42, max_iter=100,
        )
        gmm.fit(reduced)
        bic = gmm.bic(reduced)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    return {
        "gmm": best_gmm, "n_components": best_k,
        "method": "gmm_full", "pca": pca, "pca_dims": actual_dims,
    }


def score_gmm(model, query_embeddings):
    """Score query embeddings against a fitted GMM (or kmeans fallback).

    Returns: (N,) array of log-likelihood scores (or cosine sim for kmeans fallback).
    """
    if model.get("method") == "kmeans":
        return score_kmeans(model, query_embeddings)
    gmm = model["gmm"]
    if model.get("pca") is not None:
        query_embeddings = model["pca"].transform(query_embeddings)
    return gmm.score_samples(query_embeddings)


# -- KDE ---------------------------------------------------------------------

def fit_kde(embeddings, bandwidth=None):
    """Fit kernel density estimation.

    Args:
        embeddings: (N, D) array
        bandwidth: kernel bandwidth. If None, uses Silverman's rule.

    Returns:
        dict with 'kde': fitted KernelDensity, 'method': 'kde'
    """
    if bandwidth is None:
        n, d = embeddings.shape
        bandwidth = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))

    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(embeddings)
    return {"kde": kde, "method": "kde"}


def score_kde(model, query_embeddings):
    """Score query embeddings against fitted KDE.

    Returns: (N,) array of log-density scores.
    """
    return model["kde"].score_samples(query_embeddings)
