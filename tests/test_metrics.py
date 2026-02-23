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
