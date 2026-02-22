"""
Compare embedding models for Cosmere entity disambiguation and tagging.

Runs 3 evaluation tests across all cached embedding models and prints
comparison results:
  1. Disambiguation Accuracy -- precision/recall/F1 on hand-labeled data
  2. Similarity Quality -- top-5 most similar entities for well-known entities
  3. Discovery Power -- how many new implicit tags each model produces

Inputs:
- data/embeddings_cache/<model>.npy    -- cached embeddings for each model
- data/embeddings_cache/entry_ids.json -- entry ID alignment
- data/disambiguation_ground_truth.json -- hand-labeled test data (56 entries)
- data/tag_classifications.json        -- entity types
- ../../words-of-brandon/wob_entries.json -- for explicit tags

Usage:
    python 06_compare_models.py
    python 06_compare_models.py --threshold 0.45
    python 06_compare_models.py --thresholds 0.3,0.4,0.5,0.6,0.7
"""

import argparse
import json
import sys
from collections import Counter
from io import StringIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common.paths import DATA_DIR as data_dir, CACHE_DIR as cache_dir, WOB_PATH as wob_path
from common.models import ALL_MODELS, EXCLUDE_TYPES, MIN_EDGE_WEIGHT
from common.embeddings import (
    load_embeddings as _load_embeddings,
    normalize_embeddings,
    compute_entity_refs as _compute_refs,
)
from common.graph_builder import build_cooccurrence_edges

gt_path = data_dir / "disambiguation_ground_truth.json"
tag_class_path = data_dir / "tag_classifications.json"

# -- CLI ---------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compare embedding models for Cosmere entity disambiguation",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Cosine similarity threshold for tagging (default: 0.5)",
)
parser.add_argument(
    "--thresholds",
    type=str,
    default=None,
    help="Comma-separated thresholds for sweep, e.g. '0.3,0.4,0.5,0.6,0.7'",
)
args = parser.parse_args()

# -- Output buffer (write to stdout AND file) --------------------------------

output_buf = StringIO()


def out(text=""):
    """Print to stdout and capture for file output."""
    print(text)
    output_buf.write(text + "\n")


# -- Load shared data -------------------------------------------------------

out("=" * 70)
out("COSMERE EMBEDDING MODEL COMPARISON")
out("=" * 70)

# Ground truth
if not gt_path.exists():
    out(f"\nERROR: Ground truth not found at {gt_path}")
    sys.exit(1)

with open(gt_path) as f:
    ground_truth = json.load(f)

gt_labels = ground_truth["labels"]
gt_entities = ground_truth["entities_tested"]
out(f"\nGround truth: {len(gt_labels)} labeled entries across {len(gt_entities)} entities")
out(f"Entities: {', '.join(gt_entities)}")

# Tag classifications
if not tag_class_path.exists():
    out(f"\nERROR: Tag classifications not found at {tag_class_path}")
    sys.exit(1)

with open(tag_class_path) as f:
    tag_class = json.load(f)

entity_tags = {t for t, info in tag_class.items() if info["type"] not in EXCLUDE_TYPES}
out(f"Entity tags (excl meta/book): {len(entity_tags)}")

# Entry IDs
ids_path = cache_dir / "entry_ids.json"
if not ids_path.exists():
    out(f"\nERROR: Entry IDs not found at {ids_path}")
    out("Run 04_embed_entries.py first.")
    sys.exit(1)

with open(ids_path) as f:
    entry_ids = json.load(f)

eid_to_idx = {eid: idx for idx, eid in enumerate(entry_ids)}
out(f"Cached entry IDs: {len(entry_ids)}")

# Raw WoB entries (for explicit tags)
if not wob_path.exists():
    out(f"\nERROR: WoB entries not found at {wob_path}")
    sys.exit(1)

with open(wob_path) as f:
    raw_entries = json.load(f)

# Build explicit entity tags per entry
entry_explicit_tags = {}
for e in raw_entries:
    eid = e["id"]
    explicit = [t for t in e["tags"] if t in entity_tags]
    if explicit:
        entry_explicit_tags[eid] = set(explicit)

out(f"Raw WoB entries: {len(raw_entries)}")
out(f"Entries with explicit entity tags: {len(entry_explicit_tags)}")

# -- Discover cached models --------------------------------------------------

available_models = []
for model in ALL_MODELS:
    npy_path = cache_dir / f"{model}.npy"
    if npy_path.exists():
        available_models.append(model)

if not available_models:
    out("\nERROR: No cached embedding models found in data/embeddings_cache/")
    out("Run 04_embed_entries.py first to generate .npy files.")
    sys.exit(1)

skipped_models = [m for m in ALL_MODELS if m not in available_models]
out(f"\nAvailable models: {', '.join(available_models)}")
if skipped_models:
    out(f"Skipped (no cache): {', '.join(skipped_models)}")

# -- Helpers -----------------------------------------------------------------

MIN_ENTRIES_FOR_REF = 5


def load_model_embeddings(model_name):
    """Load embeddings for a model and return the numpy array."""
    embeddings, _ = _load_embeddings(model_name)
    return embeddings


def compute_entity_refs_for_model(embeddings):
    """
    Compute entity reference embeddings by averaging explicitly-tagged entries.

    Returns:
        entity_names: list of entity tag names (parallel to ref rows)
        entity_ref_matrix: np.array of shape (n_entities, dims), L2-normalized
    """
    refs_dict = _compute_refs(embeddings, entity_tags, entry_explicit_tags, eid_to_idx,
                              min_entries=MIN_ENTRIES_FOR_REF)
    entity_names = list(refs_dict.keys())
    entity_ref_matrix = np.array(list(refs_dict.values()), dtype=np.float32)
    return entity_names, entity_ref_matrix


def compute_sim_matrix(entries_norm, entity_refs):
    """Compute similarity matrix: (n_entries, n_entities)."""
    sim = entries_norm @ entity_refs.T
    return np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)


# -- Precompute per-model data -----------------------------------------------

out("\nLoading and precomputing per-model data...")

model_data = {}
for model in available_models:
    embeddings = load_model_embeddings(model)
    entity_names, entity_refs = compute_entity_refs_for_model(embeddings)
    entries_norm = normalize_embeddings(embeddings)
    sim_matrix = compute_sim_matrix(entries_norm, entity_refs)

    # Map entity name -> column index in sim_matrix
    entity_col = {name: i for i, name in enumerate(entity_names)}

    model_data[model] = {
        "embeddings": embeddings,
        "entity_names": entity_names,
        "entity_refs": entity_refs,
        "entries_norm": entries_norm,
        "sim_matrix": sim_matrix,
        "entity_col": entity_col,
        "dims": embeddings.shape[1],
    }
    out(f"  {model}: {embeddings.shape[1]}-dim, {len(entity_names)} entity refs")


# ============================================================================
# TEST 1: Disambiguation Accuracy
# ============================================================================

def run_disambiguation(threshold):
    """
    Run disambiguation accuracy test at a given threshold.

    Returns:
        results: dict of model -> {
            per_entity: {entity -> {tp, fp, fn, tn, precision, recall, f1}},
            overall: {tp, fp, fn, tn, precision, recall, f1}
        }
    """
    results = {}

    for model in available_models:
        md = model_data[model]
        entity_col = md["entity_col"]
        sim_matrix = md["sim_matrix"]

        per_entity = {}
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

        for entity in gt_entities:
            if entity not in entity_col:
                # Entity does not have a reference embedding for this model
                per_entity[entity] = {
                    "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                    "precision": None, "recall": None, "f1": None,
                    "note": "no ref embedding",
                }
                continue

            col = entity_col[entity]
            tp, fp, fn, tn = 0, 0, 0, 0

            for eid_str, label in gt_labels.items():
                if label["entity"] != entity:
                    continue

                eid = int(eid_str)
                if eid not in eid_to_idx:
                    continue

                idx = eid_to_idx[eid]
                sim = float(sim_matrix[idx, col])
                predicted_positive = sim > threshold
                actual_positive = label["is_about_entity"]

                if predicted_positive and actual_positive:
                    tp += 1
                elif predicted_positive and not actual_positive:
                    fp += 1
                elif not predicted_positive and actual_positive:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = None

            per_entity[entity] = {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": recall, "f1": f1,
            }
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        # Overall
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
        if (
            overall_precision is not None
            and overall_recall is not None
            and (overall_precision + overall_recall) > 0
        ):
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        else:
            overall_f1 = None

        results[model] = {
            "per_entity": per_entity,
            "overall": {
                "tp": total_tp, "fp": total_fp, "fn": total_fn, "tn": total_tn,
                "precision": overall_precision, "recall": overall_recall, "f1": overall_f1,
            },
        }

    return results


def fmt_metric(val):
    """Format a metric value for display."""
    if val is None:
        return "  -- "
    return f"{val:.2f}"


def print_disambiguation_table(threshold, results):
    """Print the per-entity + overall disambiguation table."""
    entities_display = gt_entities + ["OVERALL"]
    col_width = max(len(e) for e in entities_display) + 2
    col_width = max(col_width, 8)
    model_label_width = max(len(m) for m in available_models) + 4

    out(f"\nDisambiguation Accuracy (threshold={threshold})")
    out("-" * 70)

    # Header
    header = " " * model_label_width
    for entity in entities_display:
        header += f"{entity:>{col_width}}"
    out(header)

    for model in available_models:
        r = results[model]
        out(model)

        for metric_name in ["Precision", "Recall", "F1"]:
            row = f"  {metric_name:<{model_label_width - 2}}"
            for entity in gt_entities:
                val = r["per_entity"][entity].get(metric_name.lower())
                row += f"{fmt_metric(val):>{col_width}}"
            # Overall
            val = r["overall"].get(metric_name.lower())
            row += f"{fmt_metric(val):>{col_width}}"
            out(row)
        out("")


out("\n" + "=" * 70)
out("TEST 1: DISAMBIGUATION ACCURACY")
out("=" * 70)

# Run at the default threshold
default_results = run_disambiguation(args.threshold)
print_disambiguation_table(args.threshold, default_results)

# Threshold sweep if requested
if args.thresholds:
    sweep_thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    sweep_thresholds.sort()

    out(f"\nThreshold sweep (overall F1):")
    out("-" * 70)

    # Header
    model_col_width = max(len(m) for m in available_models) + 2
    model_col_width = max(model_col_width, 15)
    header = f"{'Threshold':<12}"
    for model in available_models:
        header += f"{model:>{model_col_width}}"
    out(header)

    for threshold in sweep_thresholds:
        sweep_results = run_disambiguation(threshold)
        row = f"{threshold:<12.1f}"
        for model in available_models:
            f1 = sweep_results[model]["overall"]["f1"]
            row += f"{fmt_metric(f1):>{model_col_width}}"
        out(row)

    out("")


# ============================================================================
# TEST 2: Similarity Quality
# ============================================================================

out("=" * 70)
out("TEST 2: SIMILARITY QUALITY")
out("=" * 70)

SIMILARITY_ENTITIES = [
    "kaladin", "hoid", "kelsier", "allomancy", "roshar",
    "shallan", "honor", "vin", "stormfather", "hemalurgy",
]

TOP_K_SIMILAR = 5

for entity in SIMILARITY_ENTITIES:
    out(f"\nEntity: {entity}")
    for model in available_models:
        md = model_data[model]
        entity_names = md["entity_names"]
        entity_refs = md["entity_refs"]
        entity_col = md["entity_col"]

        if entity not in entity_col:
            out(f"  {model:<20} (no reference embedding)")
            continue

        # Compute entity-entity similarities
        col = entity_col[entity]
        ref_vec = entity_refs[col]  # already L2-normalized
        sims = entity_refs @ ref_vec

        # Sort descending, skip self
        ranked = np.argsort(sims)[::-1]
        similar = []
        for j in ranked:
            if j == col:
                continue
            if len(similar) >= TOP_K_SIMILAR:
                break
            similar.append((entity_names[j], float(sims[j])))

        parts = [f"{name} ({score:.2f})" for name, score in similar]
        out(f"  {model:<20} {', '.join(parts)}")

out("")


# ============================================================================
# TEST 3: Discovery Power
# ============================================================================

out("=" * 70)
out("TEST 3: DISCOVERY POWER")
out("=" * 70)

threshold = args.threshold
out(f"\nThreshold: {threshold}")

discovery_results = {}

for model in available_models:
    md = model_data[model]
    entity_names = md["entity_names"]
    entity_col = md["entity_col"]
    sim_matrix = md["sim_matrix"]

    # Run full embedding-first tagging
    implicit_tag_counts = Counter()
    total_implicit_additions = 0
    entries_newly_tagged = 0
    entry_all_tags = {}

    for eid in entry_ids:
        explicit = entry_explicit_tags.get(eid, set())

        idx = eid_to_idx[eid]
        sims = sim_matrix[idx]
        predicted = set()
        for col_idx, ent_name in enumerate(entity_names):
            if sims[col_idx] > threshold and ent_name not in explicit:
                predicted.add(ent_name)
                implicit_tag_counts[ent_name] += 1

        combined = explicit | predicted
        if combined:
            entry_all_tags[eid] = combined
            if predicted:
                total_implicit_additions += len(predicted)
                if not explicit:
                    entries_newly_tagged += 1

    # Build co-occurrence graph to count isolated nodes
    cooc_entries = [{"id": eid, "tags": list(tags)} for eid, tags in entry_all_tags.items()]
    node_entries, filtered_edges = build_cooccurrence_edges(cooc_entries, entity_tags)

    # Count isolated nodes
    connected = set()
    for src, tgt in filtered_edges:
        connected.add(src)
        connected.add(tgt)
    isolated = sum(1 for n in node_entries if n not in connected)

    # Top 10 entities by implicit additions
    top_implicit = implicit_tag_counts.most_common(10)

    discovery_results[model] = {
        "new_implicit_tags": total_implicit_additions,
        "entries_newly_tagged": entries_newly_tagged,
        "isolated_nodes": isolated,
        "total_nodes": len(node_entries),
        "total_edges": len(filtered_edges),
        "top_implicit": top_implicit,
    }

# Print comparison table
model_col_width = max(len(m) for m in available_models) + 2
model_col_width = max(model_col_width, 15)
metric_label_width = 24

out(f"\n{'Discovery Power (threshold=' + str(threshold) + ')'}")
out("-" * 70)

header = f"{'':>{metric_label_width}}"
for model in available_models:
    header += f"{model:>{model_col_width}}"
out(header)

metrics = [
    ("New implicit tags", "new_implicit_tags"),
    ("Entries newly tagged", "entries_newly_tagged"),
    ("Total nodes", "total_nodes"),
    ("Total edges", "total_edges"),
    ("Isolated nodes", "isolated_nodes"),
]

for label, key in metrics:
    row = f"{label:>{metric_label_width}}"
    for model in available_models:
        val = discovery_results[model][key]
        row += f"{val:>{model_col_width}}"
    out(row)

out("")

# Print top 10 entities by implicit additions per model
for model in available_models:
    out(f"Top 10 entities by implicit tags ({model}):")
    for tag, count in discovery_results[model]["top_implicit"]:
        out(f"  {tag:<30} +{count}")
    out("")


# -- Save summary to file ---------------------------------------------------

summary_path = data_dir / "model_comparison.txt"
with open(summary_path, "w") as f:
    f.write(output_buf.getvalue())

out(f"Summary saved to {summary_path}")
