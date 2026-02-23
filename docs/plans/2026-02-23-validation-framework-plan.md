# Validation Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `notebooks/08_validate_models.py` -- a stratified k-fold validation framework that compares 4 entity representation methods across 4 embedding models, with tag-level and edge-level metrics plus hyperparameter tuning.

**Architecture:** Single new script that loads cached embeddings, splits explicitly-tagged entries into stratified folds, fits entity representations (k-means/GMM/KDE) on training folds, scores held-out entries, tunes hyperparameters on validation fold, and reports tag-level + edge-level metrics. Uses existing `notebooks/common/` utilities for paths, embeddings, and graph building.

**Tech Stack:** Python 3.9, numpy, scikit-learn (GaussianMixture, KernelDensity, PCA, StratifiedKFold), scipy (spearmanr). No new dependencies.

---

### Task 1: Data Loading and Fold Splitting

**Files:**
- Create: `notebooks/common/validation.py`
- Test: `tests/test_validation.py`

**Step 1: Write the failing test for fold splitting**

```python
# tests/test_validation.py
import json
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))

from common.validation import build_stratified_folds, load_validation_data


class TestFoldSplitting:
    """Test stratified k-fold splitting of tagged entries."""

    def test_folds_cover_all_entries(self):
        """Every tagged entry appears in exactly one test fold."""
        # Minimal synthetic data: 10 entries, 3 entities
        entry_explicit_tags = {
            1: {"kaladin", "dalinar"},
            2: {"kaladin"},
            3: {"dalinar", "shallan"},
            4: {"kaladin", "shallan"},
            5: {"dalinar"},
            6: {"shallan"},
            7: {"kaladin", "dalinar"},
            8: {"shallan"},
            9: {"dalinar", "shallan"},
            10: {"kaladin"},
        }
        folds = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42)
        all_test_eids = set()
        for fold in folds:
            all_test_eids.update(fold["test"])
        assert all_test_eids == set(entry_explicit_tags.keys())

    def test_folds_no_overlap(self):
        """No entry appears in more than one test fold."""
        entry_explicit_tags = {
            i: {"entity_a"} for i in range(20)
        }
        folds = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42)
        seen = set()
        for fold in folds:
            overlap = seen & set(fold["test"])
            assert len(overlap) == 0, f"Overlap: {overlap}"
            seen.update(fold["test"])

    def test_train_val_test_disjoint(self):
        """Train, val, and test sets are disjoint in each fold."""
        entry_explicit_tags = {
            i: {"entity_a"} for i in range(25)
        }
        folds = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42)
        for fold in folds:
            train = set(fold["train"])
            val = set(fold["val"])
            test = set(fold["test"])
            assert len(train & val) == 0
            assert len(train & test) == 0
            assert len(val & test) == 0

    def test_fold_structure(self):
        """Each fold has train, val, test keys."""
        entry_explicit_tags = {i: {"a"} for i in range(15)}
        folds = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42)
        assert len(folds) == 5
        for fold in folds:
            assert "train" in fold
            assert "val" in fold
            assert "test" in fold
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.validation'`

**Step 3: Write minimal implementation**

```python
# notebooks/common/validation.py
"""Stratified k-fold splitting and data loading for validation framework."""

import json
from collections import Counter, defaultdict

import numpy as np

from .paths import DATA_DIR, WOB_PATH, TAG_CLASS_PATH
from .models import EXCLUDE_TYPES
from .embeddings import load_embeddings, normalize_embeddings


def load_validation_data():
    """Load all data needed for validation.

    Returns:
        entry_explicit_tags: dict {eid: set of entity names}
        entity_tags: set of all entity tag names (excl meta/book)
        entry_ids: list of all entry IDs (matching embedding rows)
        eid_to_idx: dict {eid: row index in embeddings}
    """
    with open(TAG_CLASS_PATH) as f:
        tag_class = json.load(f)

    with open(WOB_PATH) as f:
        raw_entries = json.load(f)

    entity_tags = {t for t, info in tag_class.items() if info["type"] not in EXCLUDE_TYPES}

    entry_explicit_tags = {}
    for e in raw_entries:
        eid = e["id"]
        explicit = {t for t in e["tags"] if t in entity_tags}
        if explicit:
            entry_explicit_tags[eid] = explicit

    ids_path = DATA_DIR / "embeddings_cache" / "entry_ids.json"
    with open(ids_path) as f:
        entry_ids = json.load(f)
    eid_to_idx = {eid: idx for idx, eid in enumerate(entry_ids)}

    return entry_explicit_tags, entity_tags, entry_ids, eid_to_idx


def build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42):
    """Build stratified k-fold splits for multi-label entity tags.

    Uses iterative stratification: assigns entries to folds one at a time,
    prioritizing the least-represented entity in each step to maintain
    proportional representation.

    Args:
        entry_explicit_tags: dict {eid: set of entity names}
        n_folds: number of folds
        seed: random seed

    Returns:
        list of dicts, each with:
            train: list of eids (3 folds)
            val: list of eids (1 fold)
            test: list of eids (1 fold)
    """
    rng = np.random.RandomState(seed)

    # Assign each entry to a fold using iterative stratification
    eids = list(entry_explicit_tags.keys())
    rng.shuffle(eids)

    # Count entity frequency
    entity_counts = Counter()
    for tags in entry_explicit_tags.values():
        for t in tags:
            entity_counts[t] += 1

    # Initialize fold assignments
    fold_assignment = {}
    fold_entity_counts = [Counter() for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    # Sort entries by rarest entity first (hardest to stratify)
    def rarest_entity_freq(eid):
        tags = entry_explicit_tags[eid]
        return min(entity_counts[t] for t in tags)

    eids_sorted = sorted(eids, key=rarest_entity_freq)

    for eid in eids_sorted:
        tags = entry_explicit_tags[eid]
        # Pick the fold that most needs this entry's entities
        best_fold = None
        best_score = float("inf")
        for f in range(n_folds):
            # Score: how over-represented are this entry's entities in fold f?
            score = sum(
                fold_entity_counts[f][t] / max(1, entity_counts[t])
                for t in tags
            )
            # Break ties by fold size (prefer smaller folds)
            score += fold_sizes[f] * 0.001
            if score < best_score:
                best_score = score
                best_fold = f
        fold_assignment[eid] = best_fold
        for t in tags:
            fold_entity_counts[best_fold][t] += 1
        fold_sizes[best_fold] += 1

    # Build fold lists
    fold_eids = [[] for _ in range(n_folds)]
    for eid, f in fold_assignment.items():
        fold_eids[f].append(eid)

    # Create train/val/test splits (rotating which fold is val and test)
    folds = []
    for test_idx in range(n_folds):
        val_idx = (test_idx + 1) % n_folds
        train_indices = [i for i in range(n_folds) if i != test_idx and i != val_idx]
        train = []
        for i in train_indices:
            train.extend(fold_eids[i])
        folds.append({
            "train": train,
            "val": fold_eids[val_idx],
            "test": fold_eids[test_idx],
        })

    return folds
```

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_validation.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add notebooks/common/validation.py tests/test_validation.py
git commit -m "feat: add stratified k-fold splitting for validation framework"
```

---

### Task 2: Entity Representation Methods

**Files:**
- Create: `notebooks/common/representations.py`
- Test: `tests/test_representations.py`

**Step 1: Write the failing tests**

```python
# tests/test_representations.py
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))

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
    """A point far from both clusters."""
    return np.array([[50.0] * 10], dtype=np.float32)


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
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_representations.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# notebooks/common/representations.py
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

def fit_gmm_diagonal(embeddings, max_components=5):
    """Fit GMM with diagonal covariance, select components via BIC.

    Args:
        embeddings: (N, D) array
        max_components: max components to try (1 to max_components)

    Returns:
        dict with 'gmm': fitted GaussianMixture, 'n_components': int,
        'method': 'gmm_diag', 'pca': None
    """
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
        gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    return {
        "gmm": best_gmm, "n_components": best_k,
        "method": "gmm_diag", "pca": None,
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
    """Score query embeddings against a fitted GMM.

    Returns: (N,) array of log-likelihood scores.
    """
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
```

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_representations.py -v`
Expected: PASS (all 8 tests)

**Step 5: Commit**

```bash
git add notebooks/common/representations.py tests/test_representations.py
git commit -m "feat: add entity representation methods (kmeans, GMM, KDE)"
```

---

### Task 3: Metrics Module

**Files:**
- Create: `notebooks/common/metrics.py`
- Test: `tests/test_metrics.py`

**Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))

from common.metrics import compute_tag_metrics, compute_edge_metrics


class TestTagMetrics:
    def test_perfect_prediction(self):
        """Perfect predictions yield F1 = 1.0."""
        # Entry 1 is tagged with entity_a, entry 2 with entity_b
        ground_truth = {1: {"entity_a"}, 2: {"entity_b"}}
        # Predictions: scores per entity per entry
        predictions = {
            "entity_a": {1: 0.9, 2: 0.1},
            "entity_b": {1: 0.1, 2: 0.9},
        }
        threshold = 0.5
        metrics = compute_tag_metrics(ground_truth, predictions, threshold)
        assert metrics["overall"]["f1"] == pytest.approx(1.0)

    def test_no_predictions(self):
        """No predictions above threshold yields recall = 0."""
        ground_truth = {1: {"entity_a"}}
        predictions = {"entity_a": {1: 0.1}}
        metrics = compute_tag_metrics(ground_truth, predictions, threshold=0.5)
        assert metrics["overall"]["recall"] == 0.0

    def test_map_perfect_ranking(self):
        """Perfect ranking yields MAP = 1.0."""
        ground_truth = {1: {"entity_a"}}
        predictions = {
            "entity_a": {1: 0.9},
            "entity_b": {1: 0.1},
        }
        metrics = compute_tag_metrics(ground_truth, predictions, threshold=0.5)
        assert metrics["overall"]["map"] == pytest.approx(1.0)


class TestEdgeMetrics:
    def test_perfect_edge_recovery(self):
        """If implicit tags recreate all original edges, edge recall = 1.0."""
        # Original graph has edge (a, b) from entry 1 being tagged with both
        true_edges = {("a", "b"): [1]}
        predicted_edges = {("a", "b"): [1]}
        metrics = compute_edge_metrics(true_edges, predicted_edges)
        assert metrics["edge_recall"] == pytest.approx(1.0)

    def test_novel_edges_counted(self):
        """Edges not in true graph are counted as novel."""
        true_edges = {("a", "b"): [1]}
        predicted_edges = {("a", "b"): [1], ("a", "c"): [2]}
        metrics = compute_edge_metrics(true_edges, predicted_edges)
        assert metrics["novel_edges"] == 1
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# notebooks/common/metrics.py
"""Metrics for tag-level and edge-level evaluation."""

from collections import defaultdict

import numpy as np


def compute_tag_metrics(ground_truth, predictions, threshold):
    """Compute precision, recall, F1, and MAP for tag predictions.

    Args:
        ground_truth: dict {eid: set of entity names} -- true tags for held-out entries
        predictions: dict {entity_name: {eid: score}} -- predicted scores
        threshold: score threshold for positive prediction

    Returns:
        dict with 'per_entity' and 'overall' metrics
    """
    # Build reverse index: eid -> {entity: score}
    scores_by_entry = defaultdict(dict)
    for entity, entry_scores in predictions.items():
        for eid, score in entry_scores.items():
            scores_by_entry[eid][entity] = score

    total_tp, total_fp, total_fn = 0, 0, 0
    ap_scores = []

    per_entity = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for eid, true_tags in ground_truth.items():
        entry_scores = scores_by_entry.get(eid, {})

        # Binary predictions at threshold
        predicted_tags = {e for e, s in entry_scores.items() if s >= threshold}

        tp_tags = predicted_tags & true_tags
        fp_tags = predicted_tags - true_tags
        fn_tags = true_tags - predicted_tags

        total_tp += len(tp_tags)
        total_fp += len(fp_tags)
        total_fn += len(fn_tags)

        for t in tp_tags:
            per_entity[t]["tp"] += 1
        for t in fp_tags:
            per_entity[t]["fp"] += 1
        for t in fn_tags:
            per_entity[t]["fn"] += 1

        # Average Precision for this entry
        if true_tags and entry_scores:
            ranked = sorted(entry_scores.items(), key=lambda x: -x[1])
            hits = 0
            precision_sum = 0.0
            for rank, (entity, score) in enumerate(ranked, 1):
                if entity in true_tags:
                    hits += 1
                    precision_sum += hits / rank
            if hits > 0:
                ap_scores.append(precision_sum / len(true_tags))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_ap = float(np.mean(ap_scores)) if ap_scores else 0.0

    return {
        "per_entity": dict(per_entity),
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map": mean_ap,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
    }


def compute_edge_metrics(true_edges, predicted_edges):
    """Compute edge-level precision, recall, F1, and novel edge count.

    Args:
        true_edges: dict {(entity_a, entity_b): [entry_ids]} -- edges from full graph
        predicted_edges: dict {(entity_a, entity_b): [entry_ids]} -- edges from predictions

    Returns:
        dict with edge_precision, edge_recall, edge_f1, novel_edges, weight_correlation
    """
    true_set = set(true_edges.keys())
    pred_set = set(predicted_edges.keys())

    recovered = true_set & pred_set
    novel = pred_set - true_set
    missed = true_set - pred_set

    tp = len(recovered)
    fp = len(novel)
    fn = len(missed)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Weight correlation for recovered edges
    weight_corr = 0.0
    if len(recovered) > 1:
        from scipy.stats import spearmanr
        true_weights = [len(true_edges[e]) for e in recovered]
        pred_weights = [len(predicted_edges[e]) for e in recovered]
        corr, _ = spearmanr(true_weights, pred_weights)
        weight_corr = float(corr) if not np.isnan(corr) else 0.0

    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "novel_edges": len(novel),
        "recovered_edges": tp,
        "missed_edges": fn,
        "weight_correlation": weight_corr,
    }
```

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add notebooks/common/metrics.py tests/test_metrics.py
git commit -m "feat: add tag-level and edge-level evaluation metrics"
```

---

### Task 4: Main Validation Script

**Files:**
- Create: `notebooks/08_validate_models.py`

**Step 1: Write the script**

This is the orchestration layer that wires together Tasks 1-3. No separate test file -- it's validated by running it and checking output.

```python
# notebooks/08_validate_models.py
"""
Validate embedding models and entity representation methods.

Runs stratified k-fold cross-validation on explicitly-tagged entries to compare:
  - 4 representation methods: k-means, GMM diagonal, GMM full (PCA), KDE
  - 4 embedding models: Azure OpenAI, Cohere, Gemini, Voyage

Outputs tag-level metrics (precision, recall, F1, MAP) and edge-level metrics
(edge precision, recall, F1, weight correlation) plus optimal hyperparameters.

Usage:
    python 08_validate_models.py
    python 08_validate_models.py --models azure_openai,voyage
    python 08_validate_models.py --methods kmeans,gmm_diag
    python 08_validate_models.py --folds 5
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from common.paths import DATA_DIR, CACHE_DIR
from common.models import ALL_MODELS, MIN_EDGE_WEIGHT
from common.embeddings import load_embeddings, normalize_embeddings
from common.validation import load_validation_data, build_stratified_folds
from common.representations import (
    fit_kmeans, fit_gmm_diagonal, fit_gmm_full, fit_kde,
    score_kmeans, score_gmm, score_kde,
)
from common.metrics import compute_tag_metrics, compute_edge_metrics

# -- CLI ---------------------------------------------------------------------

ALL_METHODS = ["kmeans", "gmm_diag", "gmm_full", "kde"]

parser = argparse.ArgumentParser(
    description="Validate embedding models and entity representation methods",
)
parser.add_argument(
    "--models", type=str, default=None,
    help="Comma-separated models (default: all available)",
)
parser.add_argument(
    "--methods", type=str, default=",".join(ALL_METHODS),
    help="Comma-separated methods (default: all 4)",
)
parser.add_argument(
    "--folds", type=int, default=5,
    help="Number of folds (default: 5)",
)
parser.add_argument(
    "--min-entity-entries", type=int, default=1,
    help="Minimum entries for an entity to be evaluated (default: 1)",
)
args = parser.parse_args()

requested_methods = [m.strip() for m in args.methods.split(",")]

# -- Output buffer -----------------------------------------------------------

output_buf = StringIO()


def out(text=""):
    print(text)
    output_buf.write(text + "\n")


# -- Load data ---------------------------------------------------------------

out("=" * 70)
out("EMBEDDING VALIDATION FRAMEWORK")
out("=" * 70)

entry_explicit_tags, entity_tags, entry_ids, eid_to_idx = load_validation_data()

out(f"\nTagged entries: {len(entry_explicit_tags)}")
out(f"Entity tags: {len(entity_tags)}")
out(f"Total entries (embedded): {len(entry_ids)}")

# Discover available models
available_models = []
for model in ALL_MODELS:
    if (CACHE_DIR / f"{model}.npy").exists():
        available_models.append(model)

if args.models:
    requested_models = [m.strip() for m in args.models.split(",")]
    available_models = [m for m in requested_models if m in available_models]

out(f"Models: {', '.join(available_models)}")
out(f"Methods: {', '.join(requested_methods)}")
out(f"Folds: {args.folds}")

# -- Build folds -------------------------------------------------------------

folds = build_stratified_folds(entry_explicit_tags, n_folds=args.folds, seed=42)
out(f"\nFold sizes: {[len(f['test']) for f in folds]}")

# -- Hyperparameter grid -----------------------------------------------------

CALIBRATION_PERCENTILES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
# For grid search, we use a coarser grid of the tagging-engine params.
# The representation method is fixed per run; we tune the scoring threshold.
# We test a range of thresholds and pick the one that maximizes F1.
THRESHOLD_GRID = np.arange(0.0, 1.01, 0.05)

# -- Method dispatch ----------------------------------------------------------

FIT_FUNCTIONS = {
    "kmeans": lambda emb, **kw: fit_kmeans(emb, n_prototypes=kw.get("n_proto", 3)),
    "gmm_diag": lambda emb, **kw: fit_gmm_diagonal(emb, max_components=kw.get("max_comp", 5)),
    "gmm_full": lambda emb, **kw: fit_gmm_full(emb, max_components=kw.get("max_comp", 5), pca_dims=kw.get("pca_dims", 100)),
    "kde": lambda emb, **kw: fit_kde(emb),
}

SCORE_FUNCTIONS = {
    "kmeans": score_kmeans,
    "gmm_diag": score_gmm,
    "gmm_full": score_gmm,
    "kde": score_kde,
}


# -- Main evaluation loop ----------------------------------------------------

results = {}

for model_name in available_models:
    embeddings, _ = load_embeddings(model_name)
    entries_norm = normalize_embeddings(embeddings).astype(np.float32)

    out(f"\n{'='*70}")
    out(f"MODEL: {model_name} ({embeddings.shape[1]}-dim)")
    out(f"{'='*70}")

    for method_name in requested_methods:
        out(f"\n  Method: {method_name}")
        start_time = time.time()

        fit_fn = FIT_FUNCTIONS[method_name]
        score_fn = SCORE_FUNCTIONS[method_name]

        fold_tag_metrics = []
        fold_edge_metrics = []

        for fold_idx, fold in enumerate(folds):
            train_eids = set(fold["train"])
            val_eids = set(fold["val"])
            test_eids = set(fold["test"])

            # Build entity representations from training entries
            entity_models = {}
            for entity in entity_tags:
                # Collect training entries for this entity
                tagged_eids = [
                    eid for eid, tags in entry_explicit_tags.items()
                    if entity in tags and eid in train_eids and eid in eid_to_idx
                ]
                if len(tagged_eids) < args.min_entity_entries:
                    continue

                indices = [eid_to_idx[eid] for eid in tagged_eids]
                entity_emb = entries_norm[indices]

                n_proto = 1 if len(tagged_eids) < 5 else (2 if len(tagged_eids) < 10 else 3)
                entity_models[entity] = fit_fn(
                    entity_emb, n_proto=n_proto, max_comp=5, pca_dims=100,
                )

            # Score test entries against all entity models
            predictions = {}
            for entity, emodel in entity_models.items():
                test_entry_eids = [eid for eid in test_eids if eid in eid_to_idx]
                if not test_entry_eids:
                    continue
                test_indices = [eid_to_idx[eid] for eid in test_entry_eids]
                test_emb = entries_norm[test_indices]
                scores = score_fn(emodel, test_emb)
                predictions[entity] = {
                    eid: float(scores[i])
                    for i, eid in enumerate(test_entry_eids)
                }

            # Ground truth for test fold
            test_ground_truth = {
                eid: tags for eid, tags in entry_explicit_tags.items()
                if eid in test_eids
            }

            # Find best threshold on validation set
            val_ground_truth = {
                eid: tags for eid, tags in entry_explicit_tags.items()
                if eid in val_eids
            }

            best_threshold = 0.5
            best_val_f1 = -1.0
            for threshold in THRESHOLD_GRID:
                val_metrics = compute_tag_metrics(val_ground_truth, predictions, threshold)
                if val_metrics["overall"]["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["overall"]["f1"]
                    best_threshold = threshold

            # Evaluate on test set with best threshold
            tag_metrics = compute_tag_metrics(test_ground_truth, predictions, best_threshold)
            fold_tag_metrics.append({
                "threshold": best_threshold,
                **tag_metrics["overall"],
            })

            # Edge-level evaluation
            # Build true edges from full explicit tags on test entries
            test_entries_for_edges = [
                {"id": eid, "tags": list(entry_explicit_tags[eid])}
                for eid in test_eids if eid in entry_explicit_tags
            ]
            true_node_entries = defaultdict(list)
            true_edge_entries = defaultdict(list)
            for e in test_entries_for_edges:
                tags_sorted = sorted(e["tags"])
                for t in tags_sorted:
                    true_node_entries[t].append(e["id"])
                for i in range(len(tags_sorted)):
                    for j in range(i + 1, len(tags_sorted)):
                        pair = (tags_sorted[i], tags_sorted[j])
                        true_edge_entries[pair].append(e["id"])
            true_edges = {
                pair: eids for pair, eids in true_edge_entries.items()
                if len(eids) >= MIN_EDGE_WEIGHT
            }

            # Build predicted edges from implicit + explicit tags
            pred_tags_by_entry = defaultdict(set)
            for eid in test_eids:
                if eid in entry_explicit_tags:
                    pred_tags_by_entry[eid] = set(entry_explicit_tags[eid])
            # Add implicit tags above threshold
            for entity, entry_scores in predictions.items():
                for eid, score in entry_scores.items():
                    if eid in test_eids and score >= best_threshold:
                        if entity not in pred_tags_by_entry.get(eid, set()):
                            pred_tags_by_entry[eid].add(entity)

            pred_edge_entries = defaultdict(list)
            for eid, tags in pred_tags_by_entry.items():
                tags_sorted = sorted(tags)
                for i in range(len(tags_sorted)):
                    for j in range(i + 1, len(tags_sorted)):
                        pair = (tags_sorted[i], tags_sorted[j])
                        pred_edge_entries[pair].append(eid)
            pred_edges = {
                pair: eids for pair, eids in pred_edge_entries.items()
                if len(eids) >= MIN_EDGE_WEIGHT
            }

            edge_metrics = compute_edge_metrics(true_edges, pred_edges)
            fold_edge_metrics.append(edge_metrics)

        # Aggregate across folds
        elapsed = time.time() - start_time

        avg_tag = {
            "f1": np.mean([m["f1"] for m in fold_tag_metrics]),
            "precision": np.mean([m["precision"] for m in fold_tag_metrics]),
            "recall": np.mean([m["recall"] for m in fold_tag_metrics]),
            "map": np.mean([m["map"] for m in fold_tag_metrics]),
            "threshold": np.mean([m["threshold"] for m in fold_tag_metrics]),
        }
        avg_edge = {
            "edge_f1": np.mean([m["edge_f1"] for m in fold_edge_metrics]),
            "edge_recall": np.mean([m["edge_recall"] for m in fold_edge_metrics]),
            "edge_precision": np.mean([m["edge_precision"] for m in fold_edge_metrics]),
            "novel_edges": np.mean([m["novel_edges"] for m in fold_edge_metrics]),
            "weight_correlation": np.mean([m["weight_correlation"] for m in fold_edge_metrics]),
        }

        key = f"{model_name}+{method_name}"
        results[key] = {
            "model": model_name,
            "method": method_name,
            "tag": avg_tag,
            "edge": avg_edge,
            "elapsed_s": elapsed,
        }

        out(f"    Tag F1={avg_tag['f1']:.3f}  MAP={avg_tag['map']:.3f}  "
            f"Threshold={avg_tag['threshold']:.2f}  "
            f"Edge F1={avg_edge['edge_f1']:.3f}  "
            f"Novel={avg_edge['novel_edges']:.0f}  ({elapsed:.1f}s)")


# -- Summary table -----------------------------------------------------------

out(f"\n{'='*70}")
out("SUMMARY")
out(f"{'='*70}")

header = f"{'Config':<30} {'Tag F1':>8} {'MAP':>8} {'Thresh':>8} {'Edge F1':>8} {'E.Recall':>8} {'Novel':>8}"
out(f"\n{header}")
out("-" * len(header))

for key in sorted(results.keys()):
    r = results[key]
    t = r["tag"]
    e = r["edge"]
    out(f"{key:<30} {t['f1']:>8.3f} {t['map']:>8.3f} {t['threshold']:>8.2f} "
        f"{e['edge_f1']:>8.3f} {e['edge_recall']:>8.3f} {e['novel_edges']:>8.0f}")


# -- Save results ------------------------------------------------------------

results_path = DATA_DIR / "validation_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
out(f"\nFull results saved to {results_path}")

summary_path = DATA_DIR / "validation_summary.txt"
with open(summary_path, "w") as f:
    f.write(output_buf.getvalue())
out(f"Summary saved to {summary_path}")
```

**Step 2: Run the script with a quick sanity check (1 model, 1 method, 3 folds)**

Run: `venv/bin/python notebooks/08_validate_models.py --models azure_openai --methods kmeans --folds 3`
Expected: Output with Tag F1 and Edge F1 numbers, completes in < 2 minutes.

**Step 3: Run full evaluation**

Run: `venv/bin/python notebooks/08_validate_models.py`
Expected: Completes in ~10-15 minutes. Output saved to `data/validation_results.json` and `data/validation_summary.txt`.

**Step 4: Commit**

```bash
git add notebooks/08_validate_models.py
git commit -m "feat: add validation framework for embedding model comparison"
```

---

### Task 5: Integration Test

**Files:**
- Modify: `tests/test_validation.py` (add integration test class)

**Step 1: Add integration test using real embeddings**

Append to `tests/test_validation.py`:

```python
# At the top, add:
# pytestmark_real = pytest.mark.skipif(
#     not (Path(__file__).parent.parent / "data" / "embeddings_cache" / "azure_openai.npy").exists(),
#     reason="Requires azure_openai embeddings cache",
# )

class TestIntegration:
    """Integration tests using real embeddings (skipped if cache absent)."""

    pytestmark = pytest.mark.skipif(
        not (Path(__file__).parent.parent / "data" / "embeddings_cache" / "azure_openai.npy").exists(),
        reason="Requires azure_openai embeddings cache",
    )

    def test_full_pipeline_one_fold(self):
        """Run one fold of validation and check output structure."""
        from common.validation import load_validation_data, build_stratified_folds
        from common.embeddings import load_embeddings, normalize_embeddings
        from common.representations import fit_kmeans, score_kmeans
        from common.metrics import compute_tag_metrics

        entry_explicit_tags, entity_tags, entry_ids, eid_to_idx = load_validation_data()
        embeddings, _ = load_embeddings("azure_openai")
        entries_norm = normalize_embeddings(embeddings).astype(np.float32)

        folds = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42)
        fold = folds[0]

        train_eids = set(fold["train"])
        test_eids = set(fold["test"])

        # Fit one entity
        kaladin_eids = [
            eid for eid, tags in entry_explicit_tags.items()
            if "kaladin" in tags and eid in train_eids and eid in eid_to_idx
        ]
        assert len(kaladin_eids) > 5, "Not enough Kaladin entries in training fold"

        indices = [eid_to_idx[eid] for eid in kaladin_eids]
        model = fit_kmeans(entries_norm[indices], n_prototypes=3)

        # Score test entries
        test_entry_eids = [eid for eid in test_eids if eid in eid_to_idx]
        test_indices = [eid_to_idx[eid] for eid in test_entry_eids]
        scores = score_kmeans(model, entries_norm[test_indices])

        assert len(scores) == len(test_entry_eids)
        assert scores.max() > 0.5, "Expected some entries to score above 0.5 for Kaladin"
```

**Step 2: Run integration test**

Run: `venv/bin/python -m pytest tests/test_validation.py::TestIntegration -v`
Expected: PASS (or SKIP if embeddings not cached)

**Step 3: Commit**

```bash
git add tests/test_validation.py
git commit -m "test: add integration test for validation pipeline"
```

---

### Task 6: Run Full Evaluation and Analyze Results

**Step 1: Run the full validation**

Run: `venv/bin/python notebooks/08_validate_models.py`

**Step 2: Review `data/validation_summary.txt`**

Check the comparison table. Identify:
- Which model + method combination has the highest Tag F1
- Whether GMM-diagonal beats k-means
- Whether the model ranking changes with better representations
- Optimal thresholds for the frontend

**Step 3: Commit results**

```bash
git add data/validation_results.json data/validation_summary.txt
git commit -m "data: add validation results for 4 models x 4 methods"
```

---

Plan complete and saved to `docs/plans/2026-02-23-validation-framework-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** -- I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** -- Open new session with executing-plans, batch execution with checkpoints

Which approach?