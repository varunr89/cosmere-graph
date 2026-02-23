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

# Cosine-sim methods (kmeans): scores in [0, 1], useful thresholds ~0.3-0.9
COSINE_THRESHOLD_GRID = np.arange(0.30, 0.91, 0.05)

# Log-likelihood methods (GMM, KDE): scores can be very negative.
# We use percentile-based thresholds computed per fold (see below).
LOGLIK_PERCENTILES = np.arange(5, 96, 5)  # 5th to 95th percentile

METHOD_USES_COSINE = {"kmeans"}

# -- Method dispatch ----------------------------------------------------------

FIT_FUNCTIONS = {
    "kmeans": lambda emb, **kw: fit_kmeans(emb, n_prototypes=kw.get("n_proto", 3)),
    "gmm_diag": lambda emb, **kw: fit_gmm_diagonal(emb, max_components=kw.get("max_comp", 5), pca_dims=kw.get("pca_dims", 100)),
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
            # For tag-level metrics, only score entries with explicit tags
            # (avoids counting all predictions on untagged entries as false positives)
            tagged_test_eids = [
                eid for eid in test_eids
                if eid in eid_to_idx and eid in entry_explicit_tags
            ]
            tagged_test_indices = [eid_to_idx[eid] for eid in tagged_test_eids]
            tagged_test_emb = entries_norm[tagged_test_indices]

            # Also prepare all test entries for edge-level evaluation
            all_test_eids = [eid for eid in test_eids if eid in eid_to_idx]
            all_test_indices = [eid_to_idx[eid] for eid in all_test_eids]
            all_test_emb = entries_norm[all_test_indices]

            # Tag-level predictions: only for entries with ground-truth tags
            tag_predictions = {}
            # Edge-level predictions: for all test entries
            edge_predictions = {}
            for entity, emodel in entity_models.items():
                tag_scores = score_fn(emodel, tagged_test_emb)
                tag_predictions[entity] = {
                    eid: float(tag_scores[i])
                    for i, eid in enumerate(tagged_test_eids)
                }
                edge_scores = score_fn(emodel, all_test_emb)
                edge_predictions[entity] = {
                    eid: float(edge_scores[i])
                    for i, eid in enumerate(all_test_eids)
                }

            # Ground truth for test fold (only entries with explicit tags)
            test_ground_truth = {
                eid: tags for eid, tags in entry_explicit_tags.items()
                if eid in test_eids
            }

            # Find best threshold on validation set
            tagged_val_eids = [
                eid for eid in val_eids
                if eid in eid_to_idx and eid in entry_explicit_tags
            ]
            tagged_val_indices = [eid_to_idx[eid] for eid in tagged_val_eids]
            tagged_val_emb = entries_norm[tagged_val_indices]

            val_predictions = {}
            for entity, emodel in entity_models.items():
                val_scores = score_fn(emodel, tagged_val_emb)
                val_predictions[entity] = {
                    eid: float(val_scores[i])
                    for i, eid in enumerate(tagged_val_eids)
                }

            val_ground_truth = {
                eid: tags for eid, tags in entry_explicit_tags.items()
                if eid in val_eids
            }

            # Build method-appropriate threshold grid
            if method_name in METHOD_USES_COSINE:
                threshold_grid = COSINE_THRESHOLD_GRID
            else:
                # For log-likelihood methods, derive thresholds from score distribution
                all_scores = []
                for entity_scores in val_predictions.values():
                    all_scores.extend(entity_scores.values())
                if all_scores:
                    threshold_grid = np.percentile(all_scores, LOGLIK_PERCENTILES)
                else:
                    threshold_grid = COSINE_THRESHOLD_GRID

            best_threshold = float(threshold_grid[len(threshold_grid) // 2])
            best_val_f1 = -1.0
            for threshold in threshold_grid:
                val_metrics = compute_tag_metrics(val_ground_truth, val_predictions, float(threshold))
                if val_metrics["overall"]["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["overall"]["f1"]
                    best_threshold = float(threshold)

            # Evaluate on test set with best threshold
            tag_metrics = compute_tag_metrics(test_ground_truth, tag_predictions, best_threshold)
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
            true_edge_entries = defaultdict(list)
            for e in test_entries_for_edges:
                tags_sorted = sorted(e["tags"])
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
            for entity, entry_scores in edge_predictions.items():
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
