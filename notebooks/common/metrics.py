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
