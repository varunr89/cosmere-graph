"""Tests for stratified k-fold splitting in the validation framework."""

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

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical folds."""
        entry_explicit_tags = {
            i: {"entity_a", "entity_b"} if i % 2 == 0 else {"entity_a"}
            for i in range(30)
        }
        folds_a = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42)
        folds_b = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42)
        for a, b in zip(folds_a, folds_b):
            assert a["train"] == b["train"]
            assert a["val"] == b["val"]
            assert a["test"] == b["test"]

    def test_different_seeds_differ(self):
        """Different seeds produce different folds."""
        entry_explicit_tags = {
            i: {"entity_a", "entity_b"} if i % 2 == 0 else {"entity_a"}
            for i in range(30)
        }
        folds_a = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42)
        folds_b = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=99)
        # At least one fold's test set should differ
        any_different = any(
            set(a["test"]) != set(b["test"])
            for a, b in zip(folds_a, folds_b)
        )
        assert any_different, "Different seeds should produce different folds"

    def test_union_of_train_val_test_is_all_entries(self):
        """In each fold, train + val + test = all entries."""
        entry_explicit_tags = {i: {"a"} for i in range(25)}
        all_eids = set(entry_explicit_tags.keys())
        folds = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42)
        for fold in folds:
            combined = set(fold["train"]) | set(fold["val"]) | set(fold["test"])
            assert combined == all_eids

    def test_stratification_rough_balance(self):
        """Entity representation should be roughly balanced across folds.

        With iterative stratification, each fold's test set should have
        proportional entity representation (not perfect, but not wildly off).
        """
        entry_explicit_tags = {}
        # 50 entries tagged with entity_a, 10 tagged with entity_b
        for i in range(50):
            entry_explicit_tags[i] = {"entity_a"}
        for i in range(50, 60):
            entry_explicit_tags[i] = {"entity_b"}

        folds = build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42)

        entity_b_per_fold = []
        for fold in folds:
            count = sum(
                1 for eid in fold["test"]
                if "entity_b" in entry_explicit_tags[eid]
            )
            entity_b_per_fold.append(count)

        # entity_b has 10 entries across 5 folds -> expect 2 per fold
        # Allow some slack but no fold should have 0 or all 10
        assert all(c > 0 for c in entity_b_per_fold), (
            f"Some fold has 0 entity_b entries: {entity_b_per_fold}"
        )
        assert all(c < 10 for c in entity_b_per_fold), (
            f"Some fold has all entity_b entries: {entity_b_per_fold}"
        )


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
